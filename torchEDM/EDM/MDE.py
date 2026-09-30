"""
Select variables greedily: candidate input variables are added one at a time to the state
that predicts each target, keeping the candidate whose addition performs best, optionally
gated by a cross-map convergence test.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .ConvergentCrossMap import ConvergentCrossMap
from .Predictors import SimplexPredict, SMapPredict, ResolveDevice
from .Results import BatchedCCMResult, MDEResult
from .Setup import ArrayOrRuns, AsRuns, IsListOfRuns, PreparePrediction, TestTargets
from ._core import Correlation as TorchCorrelation, R2 as TorchR2, batch_simplex_predict_and_score
from ..Scoring import Correlation


def MDE(X_train: ArrayOrRuns,
		Y_train: ArrayOrRuns,
		X_test: Optional[ArrayOrRuns] = None,
		Y_test: Optional[ArrayOrRuns] = None,
		embedDimensions: int = 0,
		step: int = -1,
		predictionHorizon: int = 1,
		knn: int = 0,
		exclusionRadius: int = 0,
		trainRowMask: Optional[ArrayOrRuns] = None,
		maxVariables: int = 5,
		candidateColumns = None,
		isTargetIncluded: bool = False,
		convergenceCheck: Union[str, bool] = 'post',
		minPredictionScore: float = 0.0,
		minCandidateScore: float = 0.5,
		minSelectedZScore: Optional[float] = None,
		extraStepsBelowZScore: int = 0,
		stdThreshold: float = 1e-3,
		convergenceSubsetPercentiles = numpy.linspace(10, 90, 5),
		convergenceRepeats: int = 10,
		convergenceSlopeThreshold: float = 0.01,
		convergenceSeed = None,
		convergenceMaxEmbedDimensions: int = 15,
		isIterativeDimensionSearch: bool = False,
		isUsingSMap: bool = False,
		theta: float = 0.0,
		candidateMetric: str = 'correlation',
		batchSize: int = 1000,
		isVerbose: bool = False,
		hasProgressBar: bool = True,
		scoringFunction: Callable = Correlation,
		device = None,
		dtype: torch.dtype = torch.float32) -> MDEResult:
	"""
	Select, for each target independently, up to maxVariables input variables, then predict
	the test samples with them. The state of a candidate set is those variables as given (no
	lagged copies); the prediction horizon applies. Several targets are handled together,
	sharing the candidate distance computations.

	:param X_train:	candidate input data, [nTrain, nFeatures] or a list of such arrays with one per run
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; candidates are scored and the final prediction made on these samples. None scores the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test, and a NaN sample is predicted but not scored
	:param embedDimensions:	fixed embedding dimensions of the target's lagged history in the convergence check; 0 searches each candidate's own
	:param step:	sample offset between consecutive lagged copies in the convergence check and the embedding-dimension search; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param knn:	number of nearest neighbors in the convergence check and in a final SMapPredict, 0 meaning each one's default (the target's embedding dimensions + 1, every training state); the final SimplexPredict always uses the number of selected variables plus one
	:param exclusionRadius:	training states within this many samples of a test state are excluded from its neighbors; always applied in the convergence check, which runs on the training samples, and to selection and prediction only in-sample, since a separate test array shares no sample axis
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param maxVariables:	number of variables to select per target; the target's own variable counts when isTargetIncluded
	:param candidateColumns:	indices of the candidate input variables; None uses all
	:param isTargetIncluded:	True starts with the target series itself selected; it then appears in selected_variables as index nFeatures + targetIndex
	:param convergenceCheck:	'pre' screens every candidate for convergence before selection, 'post' checks candidates in performance order at each step, False skips the check
	:param minPredictionScore:	minimum performance a candidate must reach at a step to be selectable
	:param minCandidateScore:	minimum peak performance a candidate alone (at its best embedding dimension) must reach predicting the target to stay in the pool; applied when the convergence check is on and the embedding dimension is searched. 0 disables
	:param minSelectedZScore:	stop rule on how far the selected candidate stands out from the other candidates: at each step the performance of every evaluated candidate is z-scored, and a target stops expanding once its selected candidate's z-score falls below this value; None disables the rule
	:param extraStepsBelowZScore:	number of further selection steps run after a selected candidate first falls below minSelectedZScore, to confirm the drop; the target stops once this many further steps have also fallen below it, and a step back above it restarts the count. The candidates selected during these steps stay selected
	:param stdThreshold:	candidates whose standard deviation over the training samples is below this are dropped from the pool
	:param convergenceSubsetPercentiles:	training-subset sizes of the convergence check, as percentages of the number of training states
	:param convergenceRepeats:	number of random subsets drawn per size
	:param convergenceSlopeThreshold:	minimum slope of cross-map performance against subset fraction for a candidate to count as convergent
	:param convergenceSeed:	seed of the convergence check's subset draws; None draws fresh subsets
	:param convergenceMaxEmbedDimensions:	largest embedding dimension tried in the per-candidate embedding-dimension search
	:param isIterativeDimensionSearch:	True evaluates each embedding dimension on its own complete samples (slower, reproduces the reference); False shares the samples complete at the largest embedding dimension in one pass
	:param isUsingSMap:	True makes the final prediction with SMapPredict instead of SimplexPredict
	:param theta:	localization strength of that final SMapPredict; 0 fits one global linear map
	:param candidateMetric:	'correlation' or 'r2': the performance metric that ranks candidates at each step
	:param batchSize:	number of candidate variables whose distance matrices are held on the device at once
	:param isVerbose:	True prints when a target stops expanding and other progress details
	:param hasProgressBar:	True shows a progress bar over the selection steps
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) for the final score
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the selection tensors
	:return: MDEResult; the candidate_* fields index the candidate view [input variables | target variables]
	"""
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score candidates on X_test')
	if candidateMetric == 'correlation':
		candidateScoreFunction = TorchCorrelation
	elif candidateMetric in ['R2', 'r2', 'rsquared']:
		candidateScoreFunction = TorchR2
	else:
		raise ValueError('candidateMetric {} not supported'.format(candidateMetric))

	xRuns = AsRuns(X_train)
	yRuns = AsRuns(Y_train)
	isInSample = X_test is None
	xTestRuns = xRuns if isInSample else AsRuns(X_test)
	yTestRuns = yRuns if isInSample else AsRuns(Y_test)
	numFeatures = xRuns[0].shape[1]
	numTargets = yRuns[0].shape[1]
	# the target columns sit after the feature columns in the candidate view so that the
	# target series can start selected (isTargetIncluded) and feed the convergence check
	allTrainRuns = [numpy.column_stack([x, y]) for x, y in zip(xRuns, yRuns)]
	allTestRuns = [numpy.column_stack([x, y]) for x, y in zip(xTestRuns, yTestRuns)]
	problem = _SelectionProblem(
		xRuns = xRuns, yRuns = yRuns, xTestRuns = xTestRuns, yTestRuns = yTestRuns,
		allTrainRuns = allTrainRuns, allTestRuns = allTestRuns,
		# the columns that can enter a state: the target columns only when they start selected,
		# so that a NaN target (a row excluded from scoring) never voids the row's state otherwise
		stateTrainRuns = allTrainRuns if isTargetIncluded else xRuns,
		stateTestRuns = allTestRuns if isTargetIncluded else xTestRuns,
		isInSample = isInSample,
		isSingleTrainRun = not IsListOfRuns(X_train),
		isSingleTestRun = not IsListOfRuns(X_train if isInSample else X_test),
		numFeatures = numFeatures, numTargets = numTargets,
		targets = [numFeatures + j for j in range(numTargets)],
		embedDimensions = embedDimensions, isEmbedDimensionsGiven = embedDimensions != 0,
		step = step, predictionHorizon = predictionHorizon, knn = knn,
		exclusionRadius = exclusionRadius,
		# separate test arrays share no sample axis with the training arrays, so the radius
		# applies to the selection and the final prediction only in-sample; the convergence
		# check runs on the training rows and honors it always
		selectionExclusionRadius = exclusionRadius if isInSample else 0,
		trainRowMask = trainRowMask,
		maxVariables = maxVariables, candidateColumns = candidateColumns, isTargetIncluded = isTargetIncluded,
		convergenceCheck = convergenceCheck, minPredictionScore = minPredictionScore,
		minCandidateScore = minCandidateScore, minSelectedZScore = minSelectedZScore,
		extraStepsBelowZScore = extraStepsBelowZScore, stdThreshold = stdThreshold,
		convergenceSubsetPercentiles = convergenceSubsetPercentiles, convergenceRepeats = convergenceRepeats,
		convergenceSlopeThreshold = convergenceSlopeThreshold, convergenceSeed = convergenceSeed,
		convergenceMaxEmbedDimensions = convergenceMaxEmbedDimensions,
		isIterativeDimensionSearch = isIterativeDimensionSearch,
		isUsingSMap = isUsingSMap, theta = theta, candidateScoreFunction = candidateScoreFunction,
		batchSize = batchSize, isVerbose = isVerbose, hasProgressBar = hasProgressBar,
		device = ResolveDevice(device), dtype = dtype)

	state = _SelectVariables(problem)
	Y_pred, scores = _PredictSelected(problem, state, scoringFunction)

	selectedVariables = numpy.full([numTargets, maxVariables], -1, dtype = int)
	performance = numpy.full([numTargets, maxVariables], numpy.nan)
	selectedSlopes = numpy.full([numTargets, maxVariables], numpy.nan)
	selectedZScores = numpy.full([numTargets, maxVariables], numpy.nan)
	for j in range(numTargets):
		selectedVariables[j, :len(state.selectedVariables[j])] = state.selectedVariables[j]
		performance[j, :len(state.scores[j])] = state.scores[j]
		selectedSlopes[j, :len(state.slopes[j])] = state.slopes[j]
		selectedZScores[j, :len(state.zScores[j])] = state.zScores[j]

	return MDEResult(
		Y_pred = Y_pred,
		selected_variables = selectedVariables,
		performance = performance,
		ccm_values = selectedSlopes,
		stepwise_performance = state.stepwiseScores,
		candidate_embed_dimensions = state.candidateEmbedDimensions,
		candidate_peak_scores = state.candidatePeakScores,
		candidate_slopes = state.slopeCache,
		selected_z_scores = selectedZScores,
		score = scores)


@dataclass(frozen = True)
class _SelectionProblem:
	"""
	Hold the data views and settings shared by the selection helpers. Variable indices refer
	to the candidate view [input variables | target variables].
	"""
	xRuns: List[numpy.ndarray]
	yRuns: List[numpy.ndarray]
	xTestRuns: List[numpy.ndarray]
	yTestRuns: List[numpy.ndarray]
	allTrainRuns: List[numpy.ndarray]
	allTestRuns: List[numpy.ndarray]
	stateTrainRuns: List[numpy.ndarray]
	stateTestRuns: List[numpy.ndarray]
	isInSample: bool
	isSingleTrainRun: bool
	isSingleTestRun: bool
	numFeatures: int
	numTargets: int
	targets: List[int]
	embedDimensions: int
	isEmbedDimensionsGiven: bool
	step: int
	predictionHorizon: int
	knn: int
	exclusionRadius: int
	selectionExclusionRadius: int
	trainRowMask: Optional[ArrayOrRuns]
	maxVariables: int
	candidateColumns: Optional[List[int]]
	isTargetIncluded: bool
	convergenceCheck: Union[str, bool]
	minPredictionScore: float
	minCandidateScore: float
	minSelectedZScore: Optional[float]
	extraStepsBelowZScore: int
	stdThreshold: float
	convergenceSubsetPercentiles: numpy.ndarray
	convergenceRepeats: int
	convergenceSlopeThreshold: float
	convergenceSeed: Optional[int]
	convergenceMaxEmbedDimensions: int
	isIterativeDimensionSearch: bool
	isUsingSMap: bool
	theta: float
	candidateScoreFunction: Callable
	batchSize: int
	isVerbose: bool
	hasProgressBar: bool
	device: torch.device
	dtype: torch.dtype


@dataclass
class _SelectionState:
	"""
	Hold the working state of the greedy loop. zScores holds, per target and step, how many
	standard deviations the selected candidate's performance lay above the mean of every
	candidate evaluated at that step. The per-candidate arrays are [nTargets,
	nCandidates] over the candidate view: candidateEmbedDimensions is -1 and candidatePeakScores
	NaN where the search did not run; slopeCache is NaN for a candidate never checked and -inf
	for a computed NaN slope, so a rejected candidate stays rejected at later steps.
	"""
	numTrainStates: int
	selectedVariables: List[List[int]]
	scores: List[List[float]]
	slopes: List[List[float]]
	zScores: List[List[float]]
	stepwiseScores: numpy.ndarray
	candidateEmbedDimensions: numpy.ndarray
	candidatePeakScores: numpy.ndarray
	slopeCache: numpy.ndarray


def _RunsOfColumns(runs: List[numpy.ndarray], columns, isSingle: bool):
	"""
	Extract the given variables from every run.

	:param runs:	list of [nSamples, nVariables] arrays
	:param columns:	indices of the variables to keep
	:param isSingle:	True returns the one run as an array, False the list
	"""
	selected = [run[:, list(columns)] for run in runs]
	return selected[0] if isSingle else selected


def _SelectVariables(problem: _SelectionProblem) -> _SelectionState:
	"""
	Run the greedy selection for all targets together.

	:param problem:	the data views and settings
	:return: the finished _SelectionState: selected variables, their performance, slopes and z-scores, and the per-candidate diagnostics
	"""
	nTargets = problem.numTargets
	nVars = problem.numFeatures + problem.numTargets

	inputs = PreparePrediction(problem.stateTrainRuns, problem.yRuns, None if problem.isInSample else problem.stateTestRuns,
							   1, problem.step, problem.predictionHorizon, problem.selectionExclusionRadius, problem.trainRowMask)
	testTargets = TestTargets(problem.yTestRuns, inputs)
	# a test row whose target is NaN is predicted later but never scores a candidate
	isScored = numpy.isfinite(testTargets).all(axis = 1)

	trainData = inputs.trainStates
	testData = inputs.testStates[isScored]
	nTrain = trainData.shape[0]
	nTest = testData.shape[0]

	state = _SelectionState(
		numTrainStates = nTrain,
		selectedVariables = [[] for _ in range(nTargets)],
		scores = [[] for _ in range(nTargets)],
		slopes = [[] for _ in range(nTargets)],
		zScores = [[] for _ in range(nTargets)],
		stepwiseScores = numpy.zeros([nTargets, problem.maxVariables, nVars]),
		candidateEmbedDimensions = numpy.full([nTargets, nVars], -1, dtype = int),
		candidatePeakScores = numpy.full([nTargets, nVars], numpy.nan),
		slopeCache = numpy.full([nTargets, nVars], numpy.nan))

	if problem.isTargetIncluded:
		for j, t in enumerate(problem.targets):
			state.selectedVariables[j].append(t)

	allColumns = list(problem.candidateColumns) if problem.candidateColumns is not None else list(range(problem.numFeatures))

	excludedBase = set(problem.targets)
	allTrain = numpy.concatenate(problem.allTrainRuns)
	lowStd = set(numpy.argwhere(numpy.std(allTrain, axis = 0) < problem.stdThreshold).squeeze(axis = 1).tolist())
	excludedBase |= lowStd

	remainingVariables = []
	for j in range(nTargets):
		excluded = excludedBase | set(state.selectedVariables[j])
		remainingVariables.append(numpy.array([c for c in allColumns if c not in excluded], dtype = int))

	# per-candidate embedding-dimension search and solo-predictability gate: one batched sweep in which
	# each candidate, stacked to every embedding dimension up to convergenceMaxEmbedDimensions, predicts each
	# target; the best embedding dimension and its peak are kept per (target, candidate), and candidates
	# below minCandidateScore leave the pool before any convergence check
	if problem.convergenceCheck is not False and not problem.isEmbedDimensionsGiven:
		_SearchCandidateEmbedDimensions(problem, state, remainingVariables)
		if problem.minCandidateScore > 0:
			passesCandidateGate = state.candidatePeakScores >= problem.minCandidateScore
			remainingVariables = [pool[passesCandidateGate[j, pool]] for j, pool in enumerate(remainingVariables)]

	if problem.convergenceCheck == 'pre':
		for j, t in enumerate(problem.targets):
			remainingVariables[j] = _FilterConvergentVariables(problem, state, remainingVariables[j], t)

	device, dtype = problem.device, problem.dtype
	trainDataTensor = torch.tensor(trainData, device = device, dtype = dtype)
	testDataTensor = torch.tensor(testData, device = device, dtype = dtype)

	# [nTargets, nTrain, nTest] accumulated squared distances of the selected columns
	selectedDistances = torch.zeros([nTargets, nTrain, nTest], device = device, dtype = dtype)
	if inputs.exclusionMask is not None:
		maskTensor = torch.tensor(inputs.exclusionMask[:, isScored], device = device)
		selectedDistances[:, maskTensor] = float('inf')
	for j in range(nTargets):
		for var in state.selectedVariables[j]:
			trainColumn = trainDataTensor[:, var]
			testColumn = testDataTensor[:, var]
			selectedDistances[j] += (trainColumn.unsqueeze(1) - testColumn.unsqueeze(0)) ** 2

	trainTargetTensor = torch.tensor(inputs.trainTargets, device = device, dtype = dtype)	# [nTrain, nTargets]
	testTargetTensor = torch.tensor(testTargets[isScored], device = device, dtype = dtype)	# [nTest, nTargets]

	numCandidateColumns = len(set().union(*[set(pool.tolist()) for pool in remainingVariables]) or {0})
	batchSize = max(1, min(problem.batchSize, numCandidateColumns))
	batchDistances = torch.zeros([batchSize, nTrain, nTest], device = device, dtype = dtype)
	candidateDistances = torch.empty([batchSize, nTrain, nTest], device = device, dtype = dtype)
	candidateScores = torch.zeros([nTargets, batchSize], device = device, dtype = dtype)

	progressBar = ProgressBar(total = problem.maxVariables, desc = 'Selecting variables', leave = False, disable = not problem.hasProgressBar)

	# a target that selects nothing in a round can never select anything later
	activeTargets = [True] * nTargets
	# consecutive steps whose selected candidate fell below minSelectedZScore, per target
	stepsBelowZScore = [0] * nTargets

	for stepIndex in range(problem.maxVariables):
		currentNeighborCounts = [len(state.selectedVariables[j]) + 2 for j in range(nTargets)]

		allRemaining = sorted(set().union(*[set(remainingVariables[j]) for j in range(nTargets) if activeTargets[j]] or [set()]))
		if len(allRemaining) == 0:
			break

		candidatePerformance = [[] for _ in range(nTargets)]

		for batchStart in range(0, len(allRemaining), batchSize):
			batchEnd = min(batchStart + batchSize, len(allRemaining))
			batchVars = allRemaining[batchStart:batchEnd]
			for k, var in enumerate(batchVars):
				diff = trainDataTensor[:, var].unsqueeze(1) - testDataTensor[:, var].unsqueeze(0)
				batchDistances[k, :, :] = diff * diff

			for j in range(nTargets):
				if not activeTargets[j]:
					continue
				theseRemainingVars = set(remainingVariables[j])
				theseIndices = [k for k, var in enumerate(batchVars) if var in theseRemainingVars]
				theseCandidates = [batchVars[k] for k in theseIndices]
				if len(theseCandidates) == 0:
					continue

				numCandidates = len(theseCandidates)
				torch.add(batchDistances[theseIndices], selectedDistances[j].unsqueeze(0),
						  out = candidateDistances[:numCandidates])

				batch_simplex_predict_and_score(candidateDistances[:numCandidates], currentNeighborCounts[j],
												trainTargetTensor[:, j], testTargetTensor[:, j],
												problem.candidateScoreFunction, performanceOut = candidateScores[j, :numCandidates])

				scoresNumpy = candidateScores[j, :numCandidates].cpu().numpy()
				for v, var in enumerate(theseCandidates):
					candidatePerformance[j].append((var, float(scoresNumpy[v])))

		for j in range(nTargets):
			if not activeTargets[j]:
				continue
			candidatePerformance[j].sort(key = lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf, reverse = True)
			# the population the selected candidate is z-scored against: every candidate evaluated at this step
			populationScores = numpy.array([score for _, score in candidatePerformance[j]], dtype = float)
			populationScores = populationScores[numpy.isfinite(populationScores)]

			if problem.minPredictionScore > 0:
				candidatePerformance[j] = [(var, score) for var, score in candidatePerformance[j]
										   if not numpy.isnan(score) and score >= problem.minPredictionScore]

			ranked = numpy.array(candidatePerformance[j]) if len(candidatePerformance[j]) > 0 else numpy.array([]).reshape(0, 2)
			if len(ranked) > 0:
				state.stepwiseScores[j, stepIndex, ranked[:, 0].astype(int)] = ranked[:, 1]

			bestVar = None
			bestScore = None

			if problem.convergenceCheck == 'post':
				for candidateVar, candidateScore in candidatePerformance[j]:
					if numpy.isnan(candidateScore):
						continue
					isConvergent, slope = _CandidateConvergence(problem, state, int(candidateVar), problem.targets[j])
					if isConvergent:
						bestVar = candidateVar
						bestScore = candidateScore
						state.slopes[j].append(slope)
						break
			else:
				if candidatePerformance[j] and not numpy.isnan(candidatePerformance[j][0][1]):
					bestVar = candidatePerformance[j][0][0]
					bestScore = candidatePerformance[j][0][1]

			if bestVar is not None:
				state.selectedVariables[j].append(bestVar)
				remainingVariables[j] = remainingVariables[j][remainingVariables[j] != bestVar]
				state.scores[j].append(bestScore)

				trainColumn = trainDataTensor[:, bestVar]
				testColumn = testDataTensor[:, bestVar]
				selectedDistances[j] += (trainColumn.unsqueeze(1) - testColumn.unsqueeze(0)) ** 2

				zScore = _ZScore(bestScore, populationScores)
				state.zScores[j].append(zScore)
				if problem.minSelectedZScore is not None and numpy.isfinite(zScore):
					if zScore < problem.minSelectedZScore:
						stepsBelowZScore[j] += 1
						if stepsBelowZScore[j] > problem.extraStepsBelowZScore:
							activeTargets[j] = False
							if problem.isVerbose:
								print('Step {}: the selected candidate for target {} has z-score {} below {} for {} consecutive steps; terminating its expansion'.format(
									stepIndex + 1, problem.targets[j], zScore, problem.minSelectedZScore, stepsBelowZScore[j]))
					else:
						stepsBelowZScore[j] = 0
			else:
				activeTargets[j] = False
				if problem.isVerbose:
					print('Step {}: no acceptable candidate for target {}; terminating its expansion'.format(stepIndex + 1, problem.targets[j]))

		progressBar.update(1)

		if not any(activeTargets):
			break

	progressBar.close()
	del trainDataTensor, testDataTensor, trainTargetTensor, testTargetTensor
	del batchDistances, candidateDistances, candidateScores, selectedDistances
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return state


def _ZScore(value: float, population: numpy.ndarray) -> float:
	"""
	Compute how many standard deviations a value lies from the mean of a population.

	:param value:	the value scored
	:param population:	the finite values it is compared against, [n]
	:return: (value - mean) / standard deviation; NaN when the population has fewer than two values or no spread
	"""
	if len(population) < 2:
		return numpy.nan
	spread = float(numpy.std(population))
	if spread == 0:
		return numpy.nan
	return float((value - numpy.mean(population)) / spread)


def _PredictSelected(problem: _SelectionProblem, state: _SelectionState, scoringFunction: Callable):
	"""
	Predict every test sample of every target from its selected variables with SimplexPredict
	(or SMapPredict). A target that selected nothing has NaN predictions and performance.

	:param problem:	the data views and settings
	:param state:	the finished selection
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData), applied per target
	:return: (Y_pred with the same shape as Y_test and one variable per target, performance [nTargets])
	"""
	nTargets = problem.numTargets
	testLengths = [run.shape[0] for run in problem.yTestRuns]
	Y_pred = [numpy.full((length, nTargets), numpy.nan) for length in testLengths]
	scores = numpy.full(nTargets, numpy.nan)

	for j in range(nTargets):
		variables = list(state.selectedVariables[j])
		if len(variables) == 0:
			continue
		X_train = _RunsOfColumns(problem.allTrainRuns, variables, problem.isSingleTrainRun)
		Y_train = _RunsOfColumns(problem.yRuns, [j], problem.isSingleTrainRun)
		X_test = None if problem.isInSample else _RunsOfColumns(problem.allTestRuns, variables, problem.isSingleTestRun)
		Y_test = _RunsOfColumns(problem.yTestRuns, [j], problem.isSingleTestRun)
		common = dict(embedDimensions = 1, step = problem.step, predictionHorizon = problem.predictionHorizon,
					  exclusionRadius = problem.selectionExclusionRadius, trainRowMask = problem.trainRowMask,
					  scoringFunction = scoringFunction, device = problem.device, dtype = problem.dtype)
		if problem.isUsingSMap:
			result = SMapPredict(X_train, Y_train, X_test, Y_test, knn = problem.knn, theta = problem.theta, **common)
		else:
			result = SimplexPredict(X_train, Y_train, X_test, Y_test, knn = len(variables) + 1, **common)
		predicted = result.Y_pred if isinstance(result.Y_pred, list) else [result.Y_pred]
		for run, values in zip(Y_pred, predicted):
			run[:, j] = values[:, 0]
		scores[j] = result.score[0]

	return (Y_pred[0] if problem.isSingleTestRun else Y_pred), scores


def _SearchCandidateEmbedDimensions(problem: _SelectionProblem, state: _SelectionState, remainingVariables) -> None:
	"""
	Search each candidate's embedding dimension: each candidate variable, stacked at every
	embedding dimension up to convergenceMaxEmbedDimensions, predicts each target over the test
	samples. The best embedding dimension and its peak performance are written into the state
	per (target, candidate) for the convergence check.

	:param problem:	the data views and settings
	:param state:	the selection state whose candidateEmbedDimensions and candidatePeakScores are filled
	:param remainingVariables:	candidate pool per target; the union is swept
	"""
	from ..Hyperparameters import FindOptimalEmbeddingDimensionality

	sweepColumns = numpy.unique(numpy.concatenate(remainingVariables))
	if len(sweepColumns) == 0:
		return

	scores = FindOptimalEmbeddingDimensionality(
		_RunsOfColumns(problem.allTrainRuns, sweepColumns, problem.isSingleTrainRun),
		problem.yRuns[0] if problem.isSingleTrainRun else problem.yRuns,
		None if problem.isInSample else _RunsOfColumns(problem.allTestRuns, sweepColumns, problem.isSingleTestRun),
		None if problem.isInSample else (problem.yTestRuns[0] if problem.isSingleTestRun else problem.yTestRuns),
		maxDims = problem.convergenceMaxEmbedDimensions,
		step = problem.step, predictionHorizon = problem.predictionHorizon,
		exclusionRadius = problem.selectionExclusionRadius, trainRowMask = problem.trainRowMask,
		isBatched = not problem.isIterativeDimensionSearch, isJoint = False, device = problem.device, dtype = problem.dtype)

	scores = numpy.asarray(scores)
	if scores.ndim == 2:  # single target squeezed to [nVars, maxDims]
		scores = scores[None, :, :]

	bestDimensions = numpy.argmax(scores, axis = 2)
	peaks = numpy.take_along_axis(scores, bestDimensions[:, :, None], axis = 2)[:, :, 0]
	state.candidateEmbedDimensions[:, sweepColumns] = bestDimensions + 1
	state.candidatePeakScores[:, sweepColumns] = peaks


def _SubsetSizes(problem: _SelectionProblem, state: _SelectionState) -> List[int]:
	"""
	Compute the training-subset sizes of the convergence check as percentiles of the number of
	training states.

	:param problem:	the settings, for the percentiles
	:param state:	the selection state, for the number of training states
	:return: list of subset sizes in samples
	"""
	return [int(percentile / 100 * state.numTrainStates) for percentile in problem.convergenceSubsetPercentiles]


def _ConvergenceCheck(problem: _SelectionProblem, candidateColumns, target: int, embeddingDimensions, subsetSizes,
					  sourceBatchSize: int = 1000) -> BatchedCCMResult:
	"""
	Cross-map each candidate from the target's lagged history on the training samples.

	:param problem:	the data views and settings
	:param candidateColumns:	indices in the candidate view of the variables the target's history cross-maps
	:param target:	index of the target in the candidate view
	:param embeddingDimensions:	embedding dimensions of the target's history: an int, [1, nCandidates] with one per candidate, or None to search
	:param subsetSizes:	training-subset sizes at which the performance is measured
	:param sourceBatchSize:	batch size handed to ConvergentCrossMap
	:return: BatchedCCMResult whose forward_performance is [nSizes, nCandidates] with singleton axes squeezed
	"""
	return ConvergentCrossMap(
		X_train = _RunsOfColumns(problem.allTrainRuns, [target], problem.isSingleTrainRun),
		Y_train = _RunsOfColumns(problem.allTrainRuns, candidateColumns, problem.isSingleTrainRun),
		embedDimensions = embeddingDimensions,
		step = problem.step,
		predictionHorizon = problem.predictionHorizon,
		knn = problem.knn if problem.knn > 0 else None,
		exclusionRadius = problem.exclusionRadius,
		trainRowMask = problem.trainRowMask,
		trainSizes = subsetSizes,
		repeats = problem.convergenceRepeats,
		maxEmbedDimensions = problem.convergenceMaxEmbedDimensions,
		seed = problem.convergenceSeed,
		batchMode = 'sample',
		sourceBatchSize = sourceBatchSize,
		hasProgressBar = False,
		device = problem.device,
		dtype = problem.dtype)


def _FilterConvergentVariables(problem: _SelectionProblem, state: _SelectionState, candidateColumns, target: int):
	"""
	Keep the candidates whose cross-map performance grows with the training-subset size.

	:param problem:	the data views and settings
	:param state:	the selection state, for the candidate embedding dimensions and the number of training states
	:param candidateColumns:	candidate pool of one target
	:param target:	index of the target in the candidate view
	:return: the convergent subset of candidateColumns as an int array
	"""
	if len(candidateColumns) == 0:
		return numpy.asarray(candidateColumns, dtype = int)

	subsetSizes = _SubsetSizes(problem, state)
	if len(subsetSizes) < 2:
		return candidateColumns

	# slope per fraction of the training states, so the threshold does not depend on the percentile grid
	normalizedSizes = numpy.array(subsetSizes, dtype = float) / state.numTrainStates

	if problem.isEmbedDimensionsGiven:
		embeddingDimensions = problem.embedDimensions
	else:
		targetPosition = problem.targets.index(target)
		candidateColumnArray = numpy.asarray(candidateColumns, dtype = int)
		embeddingDimensions = state.candidateEmbedDimensions[targetPosition, candidateColumnArray][None, :]

	result = _ConvergenceCheck(problem, list(candidateColumns), target, embeddingDimensions, subsetSizes)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	x = torch.tensor(normalizedSizes, dtype = torch.float32, device = problem.device)
	# a single candidate comes back squeezed; restore [nSizes, nCandidates]
	y = torch.tensor(result.forward_performance, dtype = torch.float32, device = problem.device).reshape(len(subsetSizes), -1)

	xMean = x.mean()
	yMean = y.mean(dim = 0)
	xyMean = (x.unsqueeze(1) * y).mean(dim = 0)
	xVariance = (x ** 2).mean() - xMean ** 2
	slopes = (xyMean - xMean * yMean) / xVariance

	isConvergent = (slopes > problem.convergenceSlopeThreshold).cpu().numpy()
	return numpy.asarray(candidateColumns, dtype = int)[isConvergent]


def _CandidateConvergence(problem: _SelectionProblem, state: _SelectionState, candidate: int, target: int) -> Tuple[bool, float]:
	"""
	Test the cross-map convergence of one candidate, cached per run so that a rejected candidate
	stays rejected at later steps.

	:param problem:	the data views and settings
	:param state:	the selection state holding the slope cache and candidate embedding dimensions
	:param candidate:	index of the candidate in the candidate view
	:param target:	index of the target in the candidate view
	:return: (isConvergent, slope)
	"""
	targetPosition = problem.targets.index(target)

	cachedSlope = state.slopeCache[targetPosition, candidate]
	if not numpy.isnan(cachedSlope):
		return (cachedSlope > problem.convergenceSlopeThreshold, float(cachedSlope))

	subsetSizes = _SubsetSizes(problem, state)
	if len(subsetSizes) < 2:
		if problem.isVerbose:
			print('Warning: fewer than two training-subset sizes; the convergence check on column {} is skipped'.format(candidate))
		return (True, 0.5)

	normalizedSizes = numpy.array(subsetSizes, dtype = float) / state.numTrainStates

	if problem.isEmbedDimensionsGiven:
		embeddingDimensions = problem.embedDimensions
	else:
		embeddingDimensions = int(state.candidateEmbedDimensions[targetPosition, candidate])
		if embeddingDimensions < 1:
			embeddingDimensions = None

	result = _ConvergenceCheck(problem, [candidate], target, embeddingDimensions, subsetSizes, sourceBatchSize = 1)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	x = torch.tensor(normalizedSizes, dtype = torch.float32, device = problem.device)
	y = torch.tensor(result.forward_performance, dtype = torch.float32, device = problem.device)

	xMean = x.mean()
	yMean = y.mean()
	xyMean = (x * y).mean()
	xVariance = (x ** 2).mean() - xMean ** 2
	slope = float((xyMean - xMean * yMean) / xVariance)
	if numpy.isnan(slope):
		slope = -numpy.inf

	state.slopeCache[targetPosition, candidate] = slope
	return (slope > problem.convergenceSlopeThreshold, slope)
