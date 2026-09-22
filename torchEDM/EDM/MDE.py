"""
Greedy variable selection: candidate columns of X are added one at a time to the state that
predicts each target, keeping the candidate whose addition scores best, optionally gated by a
cross-map convergence test.
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
		scoringFunction: Callable = Correlation,
		device = None,
		dtype: torch.dtype = torch.float32) -> MDEResult:
	"""
	Each target independently selects up to maxVariables columns of X. The state of a
	candidate set is those columns as given (no history stacking); the horizon shift applies.
	Several targets are handled together, sharing the candidate distance computations. The
	selection then predicts the test rows.

	:param X_train:		[nTrain, nFeatures] or a list of runs: the candidate columns
	:param Y_train:		[nTrain, nTargets] or a list of runs
	:param X_test:		[nTest, nFeatures] or a list of runs; candidates are scored on these rows.
		None scores the training rows in-sample.
	:param Y_test:		targets for X_test; required with X_test. A NaN target row is predicted but never scored.
	:param embedDimensions:	fixed embedding dimensions for the convergence check; 0 searches each candidate's embedding dimension
	:param step:		row offset between stacked copies in the convergence check and embedding-dimension search
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors for the final prediction and the convergence check; 0 means the default
	:param exclusionRadius:	training states this close in rows are not neighbors. Always applied in the
		convergence check, which runs on the training rows; applied to the selection and the final
		prediction only when X_test is omitted, since a separate test array shares no sample axis
	:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
	:param maxVariables:	columns to select per target (the target itself counts when isTargetIncluded)
	:param candidateColumns:	candidate X columns; None uses all
	:param isTargetIncluded:	start with the target series itself selected; it then appears in
		selected_variables as column index nFeatures + targetIndex
	:param convergenceCheck:	'pre' screens every candidate for convergence before selection, 'post' checks
		candidates in score order at each step, False skips the check
	:param minPredictionScore:	minimum candidate score to be selectable
	:param minCandidateScore:	minimum score a candidate alone (at its best embedding dimension) must reach predicting
		the target to stay in the pool; applied when the convergence check is on and the embedding dimension is searched. 0 disables.
	:param stdThreshold:	candidates with standard deviation below this are dropped
	:param convergenceSubsetPercentiles:	training-subset sizes of the convergence check, percent of training states
	:param convergenceRepeats:	random subsets per size
	:param convergenceSlopeThreshold:	minimum skill-versus-size slope to count as convergent
	:param convergenceSeed:	random seed for the convergence check
	:param convergenceMaxEmbedDimensions:	largest embedding dimension tried in the per-candidate embedding-dimension search
	:param isIterativeDimensionSearch:	True evaluates each embedding dimension on its own complete rows (slower, reproduces the
		reference); False shares the rows complete at the largest embedding dimension in one pass
	:param isUsingSMap:	final prediction with SMapPredict instead of SimplexPredict
	:param theta:		localization for isUsingSMap
	:param candidateMetric:	'correlation' or 'r2' for candidate scoring
	:param batchSize:	candidates per batch
	:param isVerbose:	print progress details
	:param scoringFunction:	scoringFunction(actual, predicted) -> float for the final score
	:param device:		torch device; None picks cuda when available
	:param dtype:		torch dtype for the selection tensors
	:return: MDEResult; the candidate_* fields index the candidate view [X columns | target columns]
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
		minCandidateScore = minCandidateScore, stdThreshold = stdThreshold,
		convergenceSubsetPercentiles = convergenceSubsetPercentiles, convergenceRepeats = convergenceRepeats,
		convergenceSlopeThreshold = convergenceSlopeThreshold, convergenceSeed = convergenceSeed,
		convergenceMaxEmbedDimensions = convergenceMaxEmbedDimensions,
		isIterativeDimensionSearch = isIterativeDimensionSearch,
		isUsingSMap = isUsingSMap, theta = theta, candidateScoreFunction = candidateScoreFunction,
		batchSize = batchSize, isVerbose = isVerbose, device = ResolveDevice(device), dtype = dtype)

	state = _SelectVariables(problem)
	Y_pred, scores = _PredictSelected(problem, state, scoringFunction)

	selectedVariables = numpy.full([numTargets, maxVariables], -1, dtype = int)
	performance = numpy.full([numTargets, maxVariables], numpy.nan)
	selectedSlopes = numpy.full([numTargets, maxVariables], numpy.nan)
	for j in range(numTargets):
		selectedVariables[j, :len(state.selectedVariables[j])] = state.selectedVariables[j]
		performance[j, :len(state.scores[j])] = state.scores[j]
		selectedSlopes[j, :len(state.slopes[j])] = state.slopes[j]

	return MDEResult(
		Y_pred = Y_pred,
		selected_variables = selectedVariables,
		performance = performance,
		ccm_values = selectedSlopes,
		stepwise_performance = state.stepwiseScores,
		candidate_embed_dimensions = state.candidateEmbedDimensions,
		candidate_peak_scores = state.candidatePeakScores,
		candidate_slopes = state.slopeCache,
		score = scores)


@dataclass(frozen = True)
class _SelectionProblem:
	"""The data views and settings the selection helpers share. Column indices refer to the
	candidate view [X columns | target columns]."""
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
	device: torch.device
	dtype: torch.dtype


@dataclass
class _SelectionState:
	"""
	Working state of the greedy loop. The per-candidate arrays are [nTargets, nColumns] over
	the candidate view: candidateEmbedDimensions is -1 and candidatePeakScores NaN where the
	search did not run; slopeCache is NaN for a candidate never checked and -inf for a computed
	NaN slope, so a rejected candidate stays rejected at later steps.
	"""
	numTrainStates: int
	selectedVariables: List[List[int]]
	scores: List[List[float]]
	slopes: List[List[float]]
	stepwiseScores: numpy.ndarray
	candidateEmbedDimensions: numpy.ndarray
	candidatePeakScores: numpy.ndarray
	slopeCache: numpy.ndarray


def _RunsOfColumns(runs: List[numpy.ndarray], columns, isSingle: bool):
	selected = [run[:, list(columns)] for run in runs]
	return selected[0] if isSingle else selected


def _SelectVariables(problem: _SelectionProblem) -> _SelectionState:
	"""Greedy selection for all targets together."""
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

	progressBar = ProgressBar(total = problem.maxVariables, desc = 'Selecting variables', leave = False, disable = not problem.isVerbose)

	# a target that selects nothing in a round can never select anything later
	activeTargets = [True] * nTargets

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


def _PredictSelected(problem: _SelectionProblem, state: _SelectionState, scoringFunction: Callable):
	"""
	Predict every test row of every target from its selected columns with SimplexPredict
	(or SMapPredict). A target that selected nothing has NaN predictions and score.

	:return: (Y_pred shaped like Y_test with one column per target, scores [nTargets])
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
	Per-candidate embedding-dimension search: each candidate column, stacked at every embedding
	dimension up to convergenceMaxEmbedDimensions, predicts each target over the test rows. The
	best embedding dimension and its peak score are stored per (target, candidate) for the
	convergence check.
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
	"""Training-subset sizes of the convergence check: percentiles of the training states."""
	return [int(percentile / 100 * state.numTrainStates) for percentile in problem.convergenceSubsetPercentiles]


def _ConvergenceCheck(problem: _SelectionProblem, candidateColumns, target: int, embeddingDimensions, subsetSizes,
					  sourceBatchSize: int = 1000) -> BatchedCCMResult:
	"""The target's stacked history cross-maps each candidate on the training rows."""
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
	"""Keep the candidates whose cross-map skill grows with the training-subset size."""
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
	Cross-map convergence of one candidate, cached per run: a rejected candidate stays
	rejected at later steps.
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
