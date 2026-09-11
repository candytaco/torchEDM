"""
Greedy variable selection (manifold dimensional expansion): candidate columns of X are
added one at a time to the state that predicts each target, keeping the candidate whose
addition scores best, optionally gated by a cross-map convergence test.
"""
from typing import List, Optional, Tuple, Union

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .ConvergentCrossMap import ConvergentCrossMap
from .Predictors import SimplexPredict, SMapPredict, ResolveDevice
from .Results import MDEResult
from .Setup import ArrayOrRuns, AsRuns, IsListOfRuns, PreparePrediction, TestTargets
from ._core import Correlation, R2, batch_simplex_predict_and_score
from ..Scoring import Correlation as ScoringCorrelation


class MDE:
	"""
	Each target independently selects up to maxD columns of X. The state of a candidate set
	is those columns as given (no history stacking); the horizon shift applies. Several
	targets are handled together, sharing the candidate distance computations.
	"""

	def __init__(self,
				 X_train: ArrayOrRuns,
				 Y_train: ArrayOrRuns,
				 X_test: Optional[ArrayOrRuns] = None,
				 Y_test: Optional[ArrayOrRuns] = None,
				 maxD: int = 5,
				 include_target: bool = False,
				 convergent = 'post',
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 dtype: torch.dtype = torch.float32,
				 columns = None,
				 embedDimensions: int = 0,
				 predictionHorizon: int = 1,
				 knn: int = 0,
				 step: int = -1,
				 exclusionRadius: int = 0,
				 trainRowMask: Optional[ArrayOrRuns] = None,
				 verbose: bool = False,
				 useSMap: bool = False,
				 theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5, ),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 IterativeDimensionSearch: bool = False,
				 TimeDelay: int = 0,
				 device = None):
		"""
		:param X_train:		[nTrain, nFeatures] or a list of runs: the candidate columns
		:param Y_train:		[nTrain, nTargets] or a list of runs
		:param X_test:		[nTest, nFeatures] or a list of runs; candidates are scored on these rows.
			None scores the training rows in-sample.
		:param Y_test:		targets for X_test; required with X_test. A NaN target row is predicted but never scored.
		:param maxD:		columns to select per target (the target itself counts when include_target)
		:param include_target:	start with the target series itself selected; it then appears in
			selected_variables as column index nFeatures + targetIndex
		:param convergent:	'pre' screens every candidate for convergence before selection, 'post' checks
			candidates in score order at each step, False skips the check
		:param metric:		'correlation' or 'r2' for candidate scoring
		:param batch_size:	candidates per batch
		:param dtype:		torch dtype for the selection tensors
		:param columns:		candidate X columns; None uses all
		:param embedDimensions:	fixed embedding dimensions for the convergence check; 0 searches each candidate's embedding dimension
		:param predictionHorizon:	rows between a state and the target it predicts
		:param knn:			neighbors for the final prediction and the convergence check; 0 means the default
		:param step:		row offset between stacked copies in the convergence check and embedding-dimension search
		:param exclusionRadius:	training states this close in rows are not neighbors. Always applied in the
			convergence check, which runs on the training rows; applied to the selection and the final
			prediction only when X_test is omitted, since a separate test array shares no sample axis
		:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
		:param verbose:		print progress details
		:param useSMap:		final prediction with SMapPredict instead of SimplexPredict
		:param theta:		localization for useSMap
		:param stdThreshold:	candidates with standard deviation below this are dropped
		:param CCMLibraryPercentiles:	training-subset sizes of the convergence check, percent of training rows
		:param CCMNumSamples:	random subsets per size
		:param CCMConvergenceThreshold:	minimum skill-versus-size slope to count as convergent
		:param CCMSeed:		random seed for the convergence check
		:param CCMMaxEmbeddingDimensions:	largest embedding dimension tried in the per-candidate embedding-dimension search
		:param MinPredictionThreshold:	minimum candidate score to be selectable
		:param MinCandidatePerformance:	minimum score a candidate alone (at its best embedding dimension) must reach predicting
			the target to stay in the pool; applied when convergence checking is on and the embedding dimension is searched. 0 disables.
		:param IterativeDimensionSearch:	True evaluates each embedding dimension on its own complete rows (slower, reproduces the
			reference); False shares the rows complete at the largest embedding dimension in one pass
		:param TimeDelay:	time delay analysis embedding dimensions; 0 disables
		:param device:		torch device; None picks cuda when available
		"""
		if X_test is not None and Y_test is None:
			raise ValueError('Y_test is needed to score candidates on X_test')
		self.xRuns = AsRuns(X_train)
		self.yRuns = AsRuns(Y_train)
		self.isInSample = X_test is None
		self.xTestRuns = self.xRuns if self.isInSample else AsRuns(X_test)
		self.yTestRuns = self.yRuns if self.isInSample else AsRuns(Y_test)
		self.isSingleTestRun = not IsListOfRuns(X_train if self.isInSample else X_test)
		self.isSingleTrainRun = not IsListOfRuns(X_train)
		self.numFeatures = self.xRuns[0].shape[1]
		self.numTargets = self.yRuns[0].shape[1]
		# the target columns sit after the feature columns in the candidate view so that the
		# target series can start selected (include_target) and feed the convergence check
		self.targets = [self.numFeatures + j for j in range(self.numTargets)]
		self.allTrainRuns = [numpy.column_stack([x, y]) for x, y in zip(self.xRuns, self.yRuns)]
		self.allTestRuns = [numpy.column_stack([x, y]) for x, y in zip(self.xTestRuns, self.yTestRuns)]
		# the columns that can enter a state: the target columns only when they start selected,
		# so that a NaN target (a row excluded from scoring) never voids the row's state otherwise
		self.stateTrainRuns = self.allTrainRuns if include_target else self.xRuns
		self.stateTestRuns = self.allTestRuns if include_target else self.xTestRuns

		self.maxD = maxD
		self.include_target = include_target
		self.convergent = convergent
		self.metric = metric
		self.batch_size = batch_size
		self.columns = columns
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		# separate test arrays share no sample axis with the training arrays, so the radius
		# applies to the selection and the final prediction only in-sample; the convergence
		# check runs on the training rows and honors it always
		self.selectionExclusionRadius = exclusionRadius if self.isInSample else 0
		self.trainRowMask = trainRowMask
		self.verbose = verbose
		self.useSMap = useSMap
		self.theta = theta
		self.stdThreshold = stdThreshold
		self.dtype = dtype
		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxEmbedDimensions = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.MinCandidatePerformance = MinCandidatePerformance
		self.iterativeDimensionSearch = IterativeDimensionSearch
		self.TimeDelay = TimeDelay

		# per-run candidate embedding-dimension-search results, arrays [nTargets, nColumns]
		self.candidateEmbedDimensions = None
		self.candidatePeakPerformance = None
		self._ccmSlopeCache = None
		self._userProvidedEmbedDimensions = embedDimensions != 0
		self.device = ResolveDevice(device)

		self.stepwise_performance = None
		self.selectedVariables = None
		self.results_ = None
		self.trainData = None
		self.testData = None
		self.timeDelayResults = None

		if metric == 'correlation':
			self.ScoreFunction = Correlation
		elif metric in ['R2', 'r2', 'rsquared']:
			self.ScoreFunction = R2
		else:
			raise ValueError('Metric {} not supported'.format(metric))

	@property
	def target(self) -> int:
		"""First target column index in the candidate view."""
		return self.targets[0]

	def _RunsOfColumns(self, runs: List[numpy.ndarray], columns, isSingle: bool):
		selected = [run[:, list(columns)] for run in runs]
		return selected[0] if isSingle else selected

	def Run(self, return_predictions: bool = True, scoring_function = ScoringCorrelation) -> MDEResult:
		"""
		Select variables for every target, then predict the test rows with the selection.

		:param return_predictions:	False leaves Y_pred empty
		:param scoring_function:	scoring_function(actual, predicted) -> float for the final score
		"""
		nTargets = self.numTargets

		self._select_variables()

		Y_pred, scores = self._predict(scoring_function)

		selected_variables_arr = numpy.zeros([nTargets, self.maxD], dtype = int) - 1
		accuracy_arr = numpy.full([nTargets, self.maxD], numpy.nan)
		ccm_values_arr = numpy.full([nTargets, self.maxD], numpy.nan)

		for j in range(nTargets):
			n = len(self._selected_variables[j])
			selected_variables_arr[j, :n] = self._selected_variables[j]
			n_acc = len(self._accuracy[j])
			accuracy_arr[j, :n_acc] = self._accuracy[j]
			n_ccm = len(self._ccm_values[j])
			ccm_values_arr[j, :n_ccm] = self._ccm_values[j]

		self.selectedVariables = self._selected_variables

		self.results_ = MDEResult(
			Y_pred = Y_pred if return_predictions else None,
			selected_variables = selected_variables_arr,
			performance = accuracy_arr,
			ccm_values = ccm_values_arr,
			stepwise_performance = self.stepwise_performance,
			timeDelayResults = self.timeDelayResults,
			score = scores)
		return self.results_

	def _select_variables(self) -> None:
		"""Greedy selection for all targets together."""
		nTargets = self.numTargets

		self._selected_variables = [[] for _ in range(nTargets)]
		self._accuracy = [[] for _ in range(nTargets)]
		self._ccm_values = [[] for _ in range(nTargets)]

		if self.include_target:
			for j, t in enumerate(self.targets):
				self._selected_variables[j].append(t)

		inputs = PreparePrediction(self.stateTrainRuns, self.yRuns, None if self.isInSample else self.stateTestRuns,
								   1, self.step, self.predictionHorizon, self.selectionExclusionRadius, self.trainRowMask)
		testTargets = TestTargets(self.yTestRuns, inputs)
		# a test row whose target is NaN is predicted later but never scores a candidate
		isScored = numpy.isfinite(testTargets).all(axis = 1)

		trainData = inputs.trainStates
		testData = inputs.testStates[isScored]
		self.trainData = trainData
		self.testData = testData

		nTrain = trainData.shape[0]
		nTest = testData.shape[0]
		nVars = self.numFeatures + self.numTargets

		all_columns = list(self.columns) if self.columns is not None else list(range(self.numFeatures))

		excluded_base = set(self.targets)
		allTrain = numpy.concatenate(self.allTrainRuns)
		low_std = set(numpy.argwhere(numpy.std(allTrain, axis = 0) < self.stdThreshold).squeeze(axis = 1).tolist())
		excluded_base |= low_std

		remaining_variables = []
		for j in range(nTargets):
			excluded_j = excluded_base | set(self._selected_variables[j])
			pool = [c for c in all_columns if c not in excluded_j]
			remaining_variables.append(pool)

		# growth-slope cache [targetPosition, column]: NaN means not yet computed; a computed
		# NaN slope is stored as -inf so it stays cached as rejected
		self._ccmSlopeCache = numpy.full([nTargets, nVars], numpy.nan)

		# per-candidate embedding-dimension search and solo-predictability gate: one batched sweep in which
		# each candidate, stacked to every embedding dimension up to CCMMaxEmbedDimensions, predicts each
		# target; the best embedding dimension and its peak are kept per (target, candidate), and candidates
		# below MinCandidatePerformance leave the pool before any convergence check
		self.candidateEmbedDimensions = numpy.full([nTargets, nVars], -1, dtype = int)
		self.candidatePeakPerformance = numpy.full([nTargets, nVars], numpy.nan)
		remaining_variables = [numpy.array(pool, dtype = int) for pool in remaining_variables]
		if self.convergent is not False and not self._userProvidedEmbedDimensions:
			self._search_candidate_embedding_dimensions(remaining_variables)
			if self.MinCandidatePerformance > 0:
				passesCandidateGate = self.candidatePeakPerformance >= self.MinCandidatePerformance
				remaining_variables = [pool[passesCandidateGate[j, pool]] for j, pool in enumerate(remaining_variables)]

		if self.convergent == 'pre':
			for j, t in enumerate(self.targets):
				remaining_variables[j] = self._filter_convergent_variables(remaining_variables[j], t)

		self.stepwise_performance = numpy.zeros([nTargets, self.maxD, nVars])

		trainData_tensor = torch.tensor(trainData, device = self.device, dtype = self.dtype)
		testData_tensor = torch.tensor(testData, device = self.device, dtype = self.dtype)

		# [nTargets, nTrain, nTest] accumulated squared distances of the selected columns
		current_best_distance_matrix = torch.zeros([nTargets, nTrain, nTest], device = self.device, dtype = self.dtype)
		if inputs.exclusionMask is not None:
			mask_tensor = torch.tensor(inputs.exclusionMask[:, isScored], device = self.device)
			current_best_distance_matrix[:, mask_tensor] = float('inf')
		for j in range(nTargets):
			for var in self._selected_variables[j]:
				train_col = trainData_tensor[:, var]
				test_col = testData_tensor[:, var]
				current_best_distance_matrix[j] += (train_col.unsqueeze(1) - test_col.unsqueeze(0)) ** 2

		train_y_tensor = torch.tensor(inputs.trainTargets, device = self.device, dtype = self.dtype)	# [nTrain, nTargets]
		test_y_tensor = torch.tensor(testTargets[isScored], device = self.device, dtype = self.dtype)	# [nTest, nTargets]

		numCandidateColumns = len(set().union(*[set(pool.tolist()) for pool in remaining_variables]) or {0})
		batchSize = max(1, min(self.batch_size, numCandidateColumns))
		batch_distances = torch.zeros([batchSize, nTrain, nTest], device = self.device, dtype = self.dtype)
		candidateDistances = torch.empty([batchSize, nTrain, nTest], device = self.device, dtype = self.dtype)
		perfs = torch.zeros([nTargets, batchSize], device = self.device, dtype = self.dtype)

		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False, disable = not self.verbose)

		# a target that selects nothing in a round can never select anything later
		activeTargets = [True] * nTargets

		for i in range(self.maxD):
			current_knns = [len(self._selected_variables[j]) + 2 for j in range(nTargets)]

			all_remaining = sorted(set().union(*[set(remaining_variables[j]) for j in range(nTargets) if activeTargets[j]] or [set()]))
			if len(all_remaining) == 0:
				break

			candidate_performance = [[] for _ in range(nTargets)]

			for batch_start in range(0, len(all_remaining), batchSize):
				batch_end = min(batch_start + batchSize, len(all_remaining))
				batch_vars = all_remaining[batch_start:batch_end]
				for k, var in enumerate(batch_vars):
					diff = trainData_tensor[:, var].unsqueeze(1) - testData_tensor[:, var].unsqueeze(0)
					batch_distances[k, :, :] = diff * diff

				for j in range(nTargets):
					if not activeTargets[j]:
						continue
					theseRemainingVars = set(remaining_variables[j])
					theseIndices = [k for k, var in enumerate(batch_vars) if var in theseRemainingVars]
					theseCandidates = [batch_vars[k] for k in theseIndices]
					if len(theseCandidates) == 0:
						continue

					numCandidates = len(theseCandidates)
					knn = current_knns[j]

					torch.add(batch_distances[theseIndices], current_best_distance_matrix[j].unsqueeze(0),
							  out = candidateDistances[:numCandidates])

					batch_simplex_predict_and_score(candidateDistances[:numCandidates], knn,
													train_y_tensor[:, j], test_y_tensor[:, j],
													self.ScoreFunction, performanceOut = perfs[j, :numCandidates])

					perfs_numpy = perfs[j, :numCandidates].cpu().numpy()
					for v, var in enumerate(theseCandidates):
						candidate_performance[j].append((var, float(perfs_numpy[v])))

			for j in range(nTargets):
				if not activeTargets[j]:
					continue
				candidate_performance[j].sort(key = lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf, reverse = True)

				if self.MinPredictionThreshold > 0:
					candidate_performance[j] = [(var, score) for var, score in candidate_performance[j]
												if not numpy.isnan(score) and score >= self.MinPredictionThreshold]

				r = numpy.array(candidate_performance[j]) if len(candidate_performance[j]) > 0 else numpy.array([]).reshape(0, 2)
				if len(r) > 0:
					self.stepwise_performance[j, i, r[:, 0].astype(int)] = r[:, 1]

				best_var = None
				best_score = None

				if self.convergent == 'post':
					for candidate_var, candidate_score in candidate_performance[j]:
						if numpy.isnan(candidate_score):
							continue
						is_convergent, ccm_slope = self._check_single_candidate_convergence(int(candidate_var), self.targets[j])
						if is_convergent:
							best_var = candidate_var
							best_score = candidate_score
							self._ccm_values[j].append(ccm_slope)
							break
				else:
					if candidate_performance[j] and not numpy.isnan(candidate_performance[j][0][1]):
						best_var = candidate_performance[j][0][0]
						best_score = candidate_performance[j][0][1]

				if best_var is not None:
					self._selected_variables[j].append(best_var)
					remaining_variables[j] = remaining_variables[j][remaining_variables[j] != best_var]
					self._accuracy[j].append(best_score)

					train_col = trainData_tensor[:, best_var]
					test_col = testData_tensor[:, best_var]
					current_best_distance_matrix[j] += (train_col.unsqueeze(1) - test_col.unsqueeze(0)) ** 2
				else:
					activeTargets[j] = False
					if self.verbose:
						print('Dimension {}: no acceptable candidate for target {}; terminating its expansion'.format(i + 1, self.targets[j]))

			progressBar.update(1)

			if not any(activeTargets):
				break

		progressBar.close()
		del trainData_tensor, testData_tensor, train_y_tensor, test_y_tensor
		del batch_distances, candidateDistances, perfs, current_best_distance_matrix
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _predict(self, scoring_function = ScoringCorrelation):
		"""
		Predict every test row of every target from its selected columns with SimplexPredict
		(or SMapPredict). A target that selected nothing has NaN predictions and score.

		:return: (Y_pred shaped like Y_test with one column per target, scores [nTargets])
		"""
		nTargets = self.numTargets
		testLengths = [run.shape[0] for run in self.yTestRuns]
		Y_pred = [numpy.full((length, nTargets), numpy.nan) for length in testLengths]
		scores = numpy.full(nTargets, numpy.nan)

		for j in range(nTargets):
			variables = list(self._selected_variables[j])
			if len(variables) == 0:
				continue
			X_train = self._RunsOfColumns(self.allTrainRuns, variables, self.isSingleTrainRun)
			Y_train = self._RunsOfColumns(self.yRuns, [j], self.isSingleTrainRun)
			X_test = None if self.isInSample else self._RunsOfColumns(self.allTestRuns, variables, self.isSingleTestRun)
			Y_test = self._RunsOfColumns(self.yTestRuns, [j], self.isSingleTestRun)
			common = dict(embedDimensions = 1, step = self.step, predictionHorizon = self.predictionHorizon,
						  exclusionRadius = self.selectionExclusionRadius, trainRowMask = self.trainRowMask,
						  scoringFunction = scoring_function, device = self.device, dtype = self.dtype)
			if self.useSMap:
				result = SMapPredict(X_train, Y_train, X_test, Y_test, knn = self.knn, theta = self.theta, **common)
			else:
				result = SimplexPredict(X_train, Y_train, X_test, Y_test, knn = len(variables) + 1, **common)
			predicted = result.Y_pred if isinstance(result.Y_pred, list) else [result.Y_pred]
			for run, values in zip(Y_pred, predicted):
				run[:, j] = values[:, 0]
			scores[j] = result.score[0]

		return (Y_pred[0] if self.isSingleTestRun else Y_pred), scores

	def _search_candidate_embedding_dimensions(self, remaining_variables) -> None:
		"""
		Per-candidate embedding-dimension search: each candidate column, stacked at every embedding dimension up to
		CCMMaxEmbedDimensions, predicts each target over the test rows. The best embedding dimension and
		its peak score are stored per (target, candidate) for the convergence check.
		"""
		from ..Hyperparameters import FindOptimalEmbeddingDimensionality

		sweepColumns = numpy.unique(numpy.concatenate(remaining_variables))
		if len(sweepColumns) == 0:
			return

		scores = FindOptimalEmbeddingDimensionality(
			self._RunsOfColumns(self.allTrainRuns, sweepColumns, self.isSingleTrainRun),
			self.yRuns[0] if self.isSingleTrainRun else self.yRuns,
			None if self.isInSample else self._RunsOfColumns(self.allTestRuns, sweepColumns, self.isSingleTestRun),
			None if self.isInSample else (self.yTestRuns[0] if self.isSingleTestRun else self.yTestRuns),
			maxDims = self.CCMMaxEmbedDimensions,
			predictionHorizon = self.predictionHorizon,
			step = self.step, exclusionRadius = self.selectionExclusionRadius, trainRowMask = self.trainRowMask,
			batched = not self.iterativeDimensionSearch, joint = False, dtype = self.dtype, device = self.device)

		scores = numpy.asarray(scores)
		if scores.ndim == 2:  # single target squeezed to [nVars, maxDims]
			scores = scores[None, :, :]

		bestDimensions = numpy.argmax(scores, axis = 2)
		peaks = numpy.take_along_axis(scores, bestDimensions[:, :, None], axis = 2)[:, :, 0]
		self.candidateEmbedDimensions[:, sweepColumns] = bestDimensions + 1
		self.candidatePeakPerformance[:, sweepColumns] = peaks

	def _LibrarySizes(self):
		return [int(percentile / 100 * self.trainData.shape[0]) for percentile in self.CCMLibraryPercentiles]

	def _ConvergenceCheck(self, candidateColumns, target, embeddingDimensions, libSizes, x_batch = 1000) -> ConvergentCrossMap:
		"""The target's stacked history cross-maps each candidate on the training rows."""
		return ConvergentCrossMap(
			X_train = self._RunsOfColumns(self.allTrainRuns, [target], self.isSingleTrainRun),
			Y_train = self._RunsOfColumns(self.allTrainRuns, candidateColumns, self.isSingleTrainRun),
			trainSizes = libSizes,
			repeats = self.CCMNumSamples,
			embedDimensions = embeddingDimensions,
			maxEmbedDimensions = self.CCMMaxEmbedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn if self.knn > 0 else None,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			trainRowMask = self.trainRowMask,
			device = self.device,
			x_batch = x_batch,
			batchMode = 'sample',
			dtype = self.dtype,
			seed = self.CCMSeed,
			showProgress = False)

	def _filter_convergent_variables(self, candidate_columns, target: int):
		"""Keep the candidates whose cross-map skill grows with the training-subset size."""
		if len(candidate_columns) == 0:
			return numpy.asarray(candidate_columns, dtype = int)

		lib_sizes = self._LibrarySizes()
		if len(lib_sizes) < 2:
			return candidate_columns

		# slope per fraction of the training rows, so the threshold does not depend on the percentile grid
		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float) / self.trainData.shape[0]

		if self._userProvidedEmbedDimensions:
			embeddingDimensions = self.embedDimensions
		else:
			targetPosition = self.targets.index(target)
			candidateColumnArray = numpy.asarray(candidate_columns, dtype = int)
			embeddingDimensions = self.candidateEmbedDimensions[targetPosition, candidateColumnArray][None, :]

		result = self._ConvergenceCheck(list(candidate_columns), target, embeddingDimensions, lib_sizes).Run()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		x = torch.tensor(lib_sizes_normalized, dtype = torch.float32, device = self.device)
		# a single candidate comes back squeezed; restore [nSizes, nCandidates]
		y = torch.tensor(result.forward_performance, dtype = torch.float32, device = self.device).reshape(len(lib_sizes), -1)

		x_mean = x.mean()
		y_mean = y.mean(dim = 0)
		xy_mean = (x.unsqueeze(1) * y).mean(dim = 0)
		x_var = (x ** 2).mean() - x_mean ** 2
		slopes = (xy_mean - x_mean * y_mean) / x_var

		isConvergent = (slopes > self.CCMConvergenceThreshold).cpu().numpy()
		return numpy.asarray(candidate_columns, dtype = int)[isConvergent]

	def _check_single_candidate_convergence(self, candidate: int, target: int) -> Tuple[bool, float]:
		"""
		Cross-map convergence of one candidate, cached per run: a rejected candidate stays
		rejected at later steps.
		:return: (isConvergent, slope)
		"""
		targetPosition = self.targets.index(target)

		if self._ccmSlopeCache is not None:
			cachedSlope = self._ccmSlopeCache[targetPosition, candidate]
			if not numpy.isnan(cachedSlope):
				return (cachedSlope > self.CCMConvergenceThreshold, float(cachedSlope))

		lib_sizes = self._LibrarySizes()
		if len(lib_sizes) < 2:
			if self.verbose:
				print('Warning: Not enough library sizes for CCM convergence check on column {}'.format(candidate))
			return (True, 0.5)

		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float) / self.trainData.shape[0]

		if self._userProvidedEmbedDimensions:
			embeddingDimensions = self.embedDimensions
		else:
			embeddingDimensions = int(self.candidateEmbedDimensions[targetPosition, candidate])
			if embeddingDimensions < 1:
				embeddingDimensions = None

		result = self._ConvergenceCheck([candidate], target, embeddingDimensions, lib_sizes, x_batch = 1).Run()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		x = torch.tensor(lib_sizes_normalized, dtype = torch.float32, device = self.device)
		y = torch.tensor(result.forward_performance, dtype = torch.float32, device = self.device)

		x_mean = x.mean()
		y_mean = y.mean()
		xy_mean = (x * y).mean()
		x_var = (x ** 2).mean() - x_mean ** 2
		slope = float((xy_mean - x_mean * y_mean) / x_var)
		if numpy.isnan(slope):
			slope = -numpy.inf

		if self._ccmSlopeCache is not None:
			self._ccmSlopeCache[targetPosition, candidate] = slope
		return (slope > self.CCMConvergenceThreshold, slope)
