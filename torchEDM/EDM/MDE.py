from typing import List, Tuple, Union

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .CCM_batch import BatchedCCM
from .Results import MDEResult, SimplexResult
from .SMap import SMap
from .Simplex import Simplex
from ._MDE import RowwiseCorrelation, RowwiseR2, FloorArray
from ..Hyperparameters import FindOptimalEmbeddingDimensionality
from ..Scoring import Correlation


class MDE:
	"""Manifold dimensional expansion for variable selection.

	This class implements the iterative variable selection algorithm that
	evaluates combinations of variables using EDM methods and selects the
	best performing variables based on convergence criteria.

	Supports multiple simultaneous target variables. Each target independently
	selects the best combination of X variables to predict it.
	"""

	def __init__(self,
				 data: numpy.ndarray,
				 target: Union[int, List[int]],
				 maxD: int = 5,
				 include_target: bool = False,
				 convergent = 'post',
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 use_half_precision: bool = False,
				 columns = None,
				 train = None,
				 test = None,
				 embedDimensions = 0,
				 predictionHorizon = 0,
				 knn = 0,
				 step = -1,
				 exclusionRadius = 0,
				 embedded = False,
				 validLib = None,
				 noTime = False,
				 ignoreNan = True,
				 verbose = False,
				 useSMap: bool = False,
				 theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5, ),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0):
		"""Initialize MDE with data and parameters.

		:param data: 	2D numpy array where column 0 is time (unless noTime=True)
		:param target: 	Column index or list of column indices of the target column(s) to forecast
		:param maxD: 	Maximum number of variables to select per target (including target if include_target=True)
		:param include_target: 	Whether to start with target in variable list
		:param convergent: 	Convergence checking mode: 'pre' runs batch CCM on all variables before selection, 'post' checks convergence within each selection loop iteration, False disables convergence checking
		:param metric: 	Metric to use: "correlation" or "r2"
		:param batch_size: 	Number of variables to process in each batch
		:param use_half_precision: 	Use float16 instead of float32 for GPU tensors to save memory
		:param columns: 	Column indices to use for embedding (defaults to all except time)
		:param train: 	Training set indices [start, end]
		:param test: 	Test set indices [start, end]
		:param embedDimensions: 	Embedding dimension (E). If 0, will be set by Validate()
		:param predictionHorizon: 	Prediction time horizon (Tp)
		:param knn: 	Number of nearest neighbors. If 0, will be set to E+1 by Validate()
		:param step: 	Time delay step size (tau). Negative values indicate lag
		:param exclusionRadius: 	Temporal exclusion radius for neighbors
		:param embedded: 	Whether data is already embedded
		:param validLib: 	Boolean mask for valid library points
		:param noTime: 	Whether first column is time or data
		:param ignoreNan: 	Remove NaN values from embedding
		:param verbose: 	Print diagnostic messages
		:param useSMap: 	Whether to use SMap instead of Simplex
		:param theta: 	S-Map localization parameter. theta=0 is global linear map, larger values increase localization
		:param stdThreshold: 	Minimum standard deviation threshold
		:param CCMLibraryPercentiles: 	Library sizes for CCM testing as percent of train data size
		:param CCMNumSamples: 	Number of random samples per library size for CCM
		:param CCMConvergenceThreshold: 	Minimum slope threshold for CCM convergence
		:param CCMSeed: 	Random seed for reproducible CCM sampling (None for non-reproducible)
		:param CCMMaxEmbeddingDimensions: 	Maximum embedding dimension for per-variable E search in CCM convergence check
		:param MinPredictionThreshold: 	Minimum correlation threshold for candidate filtering
		:param EmbedDimCorrelationMin: 	Minimum correlation for E selection
		:param FirstEMax: 	Use first local maximum in E-rho curve instead of global max
		:param TimeDelay: 	Time delay analysis depth. If 0, time delay analysis is disabled
		"""
		self.data = data
		self.targets = [target] if isinstance(target, int) else list(target)
		self.maxD = maxD
		self.include_target = include_target
		self.convergent = convergent
		self.metric = metric
		self.batch_size = batch_size
		self.columns = columns
		self.train = train
		self.test = test
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.noTime = noTime
		self.ignoreNan = ignoreNan
		self.verbose = verbose
		self.useSMap = useSMap
		self.theta = theta
		self.stdThreshold = stdThreshold
		self.use_half_precision = use_half_precision
		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxE = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.EmbedDimCorrelationMin = EmbedDimCorrelationMin
		self.FirstEMax = FirstEMax
		self.TimeDelay = TimeDelay

		# Keyed by (column, target) to support multi-target convergence checks
		self.optimalEmbeddingDimensions = {}

		self._userProvidedE = embedDimensions != 0

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.dtype = torch.float16 if use_half_precision else torch.float32

		self.stepwise_performance = None
		self.selectedVariables = None
		self.results_ = None
		self.trainData = None
		self.testData = None
		self.timeDelayResults = None

		if metric == 'correlation':
			self.EvaluatePerformance = RowwiseCorrelation
		elif metric in ['R2', 'r2', 'rsquared']:
			self.EvaluatePerformance = RowwiseR2
		else:
			raise ValueError('Metric {} not supported'.format(metric))

	@property
	def target(self) -> int:
		"""First target column index, for backward compatibility."""
		return self.targets[0]

	def Run(self, return_predictions: bool = True, scoring_function = Correlation) -> MDEResult:
		"""Execute MDE variable selection and return results.

		:param return_predictions: If False, the predictions field of the result will not be populated
		:param scoring_function: Scoring function taking (actual, predicted) and returning a scalar. Default is Correlation.
		:return: Results containing final predictions, selected variables, accuracy, and CCM values
		:rtype: MDEResult
		"""
		nTargets = len(self.targets)

		if self.embedDimensions == 0:
			best_e = 0
			for t in self.targets:
				dims, corrs = FindOptimalEmbeddingDimensionality(
					self.data, [t], t, self.maxD,
					train = self.train, test = self.test,
					predictionHorizon = self.predictionHorizon,
					noTime = self.noTime
					)
				e = dims[numpy.argmax(corrs)]
				if e > best_e:
					best_e = e
			self.embedDimensions = best_e

		self._select_variables()

		final_forecasts, time_values, scores = self._final_prediction(scoring_function)

		# Build padded 2D arrays for selected_variables, accuracy, ccm_values
		selected_variables_arr = numpy.zeros([nTargets, self.maxD], dtype = int) - 1
		accuracy_arr = numpy.zeros([nTargets, self.maxD]) * numpy.nan
		ccm_values_arr = numpy.zeros([nTargets, self.maxD]) * numpy.nan

		for j in range(nTargets):
			n = len(self._selected_variables[j])
			selected_variables_arr[j, :n] = self._selected_variables[j]
			n_acc = len(self._accuracy[j])
			accuracy_arr[j, :n_acc] = self._accuracy[j]
			n_ccm = len(self._ccm_values[j])
			ccm_values_arr[j, :n_ccm] = self._ccm_values[j]

		self.selectedVariables = self._selected_variables

		self.results_ = MDEResult(
			time = time_values,
			predictions = final_forecasts if return_predictions else None,
			selected_variables = selected_variables_arr,
			performance = accuracy_arr,
			ccm_values = ccm_values_arr,
			stepwise_performance = self.stepwise_performance,
			timeDelayResults = self.timeDelayResults,
			score = scores
			)
		return self.results_

	def _select_variables(self) -> None:
		"""Perform iterative variable selection for all targets in parallel."""
		nTargets = len(self.targets)

		self._selected_variables = [[] for _ in range(nTargets)]
		self._accuracy = [[] for _ in range(nTargets)]
		self._ccm_values = [[] for _ in range(nTargets)]

		if self.include_target:
			for j, t in enumerate(self.targets):
				self._selected_variables[j].append(t)

		dummy = Simplex(
			data = self.data,
			columns = numpy.arange(self.data.shape[1]).tolist(),
			target = self.targets[0],
			train = self.train,
			test = self.test,
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			embedded = True,
			validLib = self.validLib,
			noTime = self.noTime,
			ignoreNan = self.ignoreNan,
			verbose = self.verbose
			)
		dummy.EmbedData()
		trainIndices = numpy.array(dummy.trainIndices, dtype = int)
		testIndices = numpy.array(dummy.testIndices, dtype = int)

		trainData = dummy.Embedding[trainIndices, :]
		testData = dummy.Embedding[testIndices, :]
		self.trainData = trainData
		self.testData = testData

		nTrain = trainData.shape[0]
		nTest = testData.shape[0]
		nVars = self.data.shape[1]

		if self.columns is not None:
			all_columns = list(self.columns)
		else:
			all_columns = list(range(nVars))

		excluded_base = set(self.targets)
		low_std = set(numpy.argwhere(numpy.std(self.data, axis = 0) < self.stdThreshold).squeeze().tolist())
		excluded_base |= low_std
		if not self.noTime:
			excluded_base.add(0)

		# Each target gets its own remaining pool
		remaining_variables = []
		for j in range(nTargets):
			excluded_j = excluded_base | set(self._selected_variables[j])
			pool = [c for c in all_columns if c not in excluded_j]
			remaining_variables.append(pool)

		# Filter convergent variables before selection if convergent='pre'
		if self.convergent == 'pre':
			for j, t in enumerate(self.targets):
				remaining_variables[j] = self._filter_convergent_variables(remaining_variables[j], t)

		self.stepwise_performance = numpy.zeros([nTargets, self.maxD, nVars])

		trainData_tensor = torch.tensor(trainData, device = self.device, dtype = self.dtype)
		testData_tensor = torch.tensor(testData, device = self.device, dtype = self.dtype)

		exclusion_mask = dummy._BuildExclusionMask()

		# 3D distance matrix: [nTargets, nTrain, nTest]
		current_best_distance_matrix = torch.zeros([nTargets, nTrain, nTest], device = self.device, dtype = self.dtype)
		if exclusion_mask.any():
			mask_tensor = torch.tensor(exclusion_mask, device = self.device)
			current_best_distance_matrix[:, mask_tensor] = float('inf')

		train_y_tensor = torch.tensor(self.data[trainIndices + self.predictionHorizon, :][:, self.targets],
									  device = self.device, dtype = self.dtype).T  # shape [nTargets, nTrain]

		test_y_tensor = torch.tensor(self.data[testIndices + self.predictionHorizon, :][:, self.targets],
									 device = self.device, dtype = self.dtype).T  # shape [nTargets, nTest]

		batch_distances = torch.zeros([self.batch_size, nTrain, nTest], device = self.device, dtype = self.dtype)
		candidateDistances = torch.empty([self.batch_size, nTrain, nTest], device = self.device, dtype = self.dtype)
		perfs = torch.zeros([nTargets, self.batch_size], device = self.device, dtype = self.dtype)

		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False)

		for i in range(self.maxD):
			current_knns = [len(self._selected_variables[j]) + 2 for j in range(nTargets)]

			all_remaining = sorted(set().union(*[set(var) for var in remaining_variables]))
			if len(all_remaining) == 0:
				break

			candidate_performance = [[] for _ in range(nTargets)]

			for batch_start in range(0, len(all_remaining), self.batch_size):
				batch_end = min(batch_start + self.batch_size, len(all_remaining))
				batch_vars = all_remaining[batch_start:batch_end]
				# Compute X candidate distances (shared across all targets)
				for k, var in enumerate(batch_vars):
					diff = trainData_tensor[:, var].unsqueeze(1) - testData_tensor[:, var].unsqueeze(0)
					batch_distances[k, :, :] = diff * diff

				# Per-target evaluation
				for j in range(nTargets):
					theseRemainingVars = set(remaining_variables[j])
					theseIndices = [k for k, var in enumerate(batch_vars) if var in theseRemainingVars]
					theseCandidates = [batch_vars[k] for k in theseIndices]
					if len(theseCandidates) == 0:
						continue

					numCandidates = len(theseCandidates)
					knn = current_knns[j]

					torch.add(batch_distances[theseIndices],
							  current_best_distance_matrix[j].unsqueeze(0),
							  out = candidateDistances[:numCandidates])

					neighborDistances, nearestNeighbors = torch.topk(candidateDistances[:numCandidates], knn, dim = 1,
																	 largest = False)
					neighborDistances.sqrt_()
					FloorArray(neighborDistances, 1e-6)

					minDistances = torch.amin(neighborDistances, dim = 1)
					weights = neighborDistances / minDistances.unsqueeze(1)
					weights.neg_().exp_()
					weightSum = torch.sum(weights, dim = 1)
					select = train_y_tensor[j][nearestNeighbors]
					predictions = torch.sum(weights * select, dim = 1) / weightSum

					perfs[j, :numCandidates].zero_()
					self.EvaluatePerformance(test_y_tensor[j], predictions, perfs[j, :numCandidates])

					perfs_numpy = perfs[j, :numCandidates].cpu().numpy()
					for v, var in enumerate(theseCandidates):
						candidate_performance[j].append((var, float(perfs_numpy[v])))

			# Per-target selection
			for j in range(nTargets):
				candidate_performance[j].sort(
					key = lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf,
					reverse = True
					)

				if self.MinPredictionThreshold > 0:
					candidate_performance[j] = [
						(var, score) for var, score in candidate_performance[j]
						if not numpy.isnan(score) and score >= self.MinPredictionThreshold
						]

				r = numpy.array(candidate_performance[j]) if len(candidate_performance[j]) > 0 else numpy.array(
					[]).reshape(0, 2)
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
					remaining_variables[j].remove(best_var)
					self._accuracy[j].append(best_score)

					train_col = trainData_tensor[:, best_var]
					test_col = testData_tensor[:, best_var]
					dist = (train_col.unsqueeze(1) - test_col.unsqueeze(0)) ** 2
					current_best_distance_matrix[j] += dist

			progressBar.update(1)

		# Clean up GPU tensors
		if torch.cuda.is_available():
			del trainData_tensor
			del testData_tensor
			del train_y_tensor
			del test_y_tensor
			del batch_distances
			del candidateDistances
			del perfs
			torch.cuda.empty_cache()

		# Move to CPU for reuse in _final_prediction, avoiding redundant distance computation
		self._finalDistanceMatrix = current_best_distance_matrix.cpu()
		self._selectionTrainIndices = trainIndices
		self._selectionTestIndices = testIndices
		del current_best_distance_matrix

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _fit_single_EDM_instance(self, variables: List[int], target: int) -> SimplexResult:
		"""
		Fit a single EDM instance with given variable indices and target.

		:param variables: Column indices to use for prediction
		:param target: Target column index
		:return: Prediction results
		:rtype: SimplexResult or SMapResult
		"""
		if self.useSMap:
			smap = SMap(
				data = self.data,
				columns = variables,
				target = target,
				train = self.train,
				test = self.test,
				embedDimensions = self.embedDimensions,
				predictionHorizon = self.predictionHorizon,
				knn = self.knn,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				theta = self.theta,
				embedded = True,
				validLib = self.validLib,
				noTime = self.noTime,
				ignoreNan = self.ignoreNan,
				verbose = self.verbose
				)
			smap.knnThreads = 1
			result = smap.Run()
			return result
		else:
			simplex = Simplex(
				data = self.data,
				columns = variables,
				target = target,
				train = self.train,
				test = self.test,
				embedDimensions = self.embedDimensions,
				predictionHorizon = self.predictionHorizon,
				knn = 0,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				embedded = True,
				validLib = self.validLib,
				noTime = self.noTime,
				ignoreNan = self.ignoreNan,
				verbose = self.verbose
				)
			return simplex.Run()

	def _final_prediction(self, scoring_function = Correlation):
		"""Run final prediction for each target with its selected variables.

		Reuses the accumulated distance matrix from _select_variables() rather than
		recomputing pairwise distances from scratch.

		For SMap, falls back to _fit_single_EDM_instance since SMap does not use
		the same distance-accumulation scheme.

		:param scoring_function: Scoring function taking (actual, predicted) and returning a scalar.
		:return: (predictions [N, K], time [N], scores [K])
		"""
		nTargets = len(self.targets)

		if self.useSMap:
			results = [self._fit_single_EDM_instance(self._selected_variables[j], self.targets[j])
					   for j in range(nTargets)]
			timeValues = results[0].time
			n = len(timeValues)
			predictions = numpy.zeros([n, nTargets])
			scores = numpy.zeros(nTargets)
			for j, result in enumerate(results):
				predictions[:, j] = result.projection[:, 2]
				scores[j] = scoring_function(result.projection[:, 1], result.projection[:, 2])
			return predictions, timeValues, scores

		trainIndices = self._selectionTrainIndices
		testIndices = self._selectionTestIndices
		nTest = len(testIndices)

		if self.noTime:
			timeValues = testIndices + self.predictionHorizon + 1
		else:
			timeValues = self.data[testIndices + self.predictionHorizon, 0]

		# Move stored squared-distance matrix back to device: shape [nTargets, nTrain, nTest]
		distanceMatrix = self._finalDistanceMatrix.to(self.device)

		predictions = numpy.zeros([nTest, nTargets])
		scores = numpy.zeros(nTargets)

		for j, target in enumerate(self.targets):
			knn = len(self._selected_variables[j]) + 1

			sqrtDists = distanceMatrix[j].sqrt()  # [nTrain, nTest]
			topkDists, topkIndices = torch.topk(sqrtDists, knn, dim = 0, largest = False)
			topkDists = topkDists.t()    # [nTest, knn]
			topkIndices = topkIndices.t()  # [nTest, knn]

			minDists = topkDists[:, 0].clamp(min = 1e-6)
			weights = torch.exp(-topkDists / minDists.unsqueeze(1))
			weightSum = weights.sum(dim = 1)

			trainY = torch.tensor(
				self.data[trainIndices + self.predictionHorizon, target],
				device = self.device, dtype = self.dtype
			)
			neighborY = trainY[topkIndices]
			predJ = (weights * neighborY).sum(dim = 1) / weightSum

			testY = self.data[testIndices + self.predictionHorizon, target]
			predictions[:, j] = predJ.cpu().numpy()
			scores[j] = scoring_function(testY, predictions[:, j])

		del distanceMatrix
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		return predictions, timeValues, scores

	def _filter_convergent_variables(self, candidate_columns: List[int], target: int) -> List[int]:
		"""Filter candidate variables to only include convergent ones using BatchedCCM.

		:param candidate_columns: Column indices to check for convergence
		:param target: Target column index
		:return: Convergent column indices
		:rtype: List[int]
		"""
		if len(candidate_columns) == 0:
			return []

		lib_sizes = [int(percentile / 100 * self.trainData.shape[0]) for percentile in self.CCMLibraryPercentiles]

		if len(lib_sizes) < 2:
			return candidate_columns

		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = lib_sizes_normalized / lib_sizes_normalized.max()

		X = self.data[:, candidate_columns]
		Y = self.data[:, target]

		batchedCCM = BatchedCCM(
			X = X,
			Y = Y,
			trainSizes = lib_sizes,
			sample = self.CCMNumSamples,
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn if self.knn > 0 else self.embedDimensions + 1,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			validLib = self.validLib,
			includeData = False,
			ignoreNan = self.ignoreNan,
			directions = 'reverse',
			trainBlockIndices = self.train,
			testBlockIndices = self.test,
			device = self.device,
			batchSize = int(self.batch_size * self.testData.shape[0] / self.trainData.shape[0]),
			useHalfPrecision = self.use_half_precision,
			seed = self.CCMSeed
			)

		result = batchedCCM.Run()

		del batchedCCM
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		x = torch.tensor(lib_sizes_normalized, dtype = torch.float32, device = self.device)
		y = torch.tensor(result.reverse_performance, dtype = torch.float32, device = self.device)

		x_mean = x.mean()
		y_mean = y.mean(dim = 0)
		xy_mean = (x.unsqueeze(1) * y).mean(dim = 0)
		x_var = (x ** 2).mean() - x_mean ** 2
		slopes = (xy_mean - x_mean * y_mean) / x_var

		convergent_mask = slopes > self.CCMConvergenceThreshold
		convergent_indices = torch.where(convergent_mask)[0].cpu().tolist()
		convergent_vars = [candidate_columns[i] for i in convergent_indices]

		return convergent_vars

	def _check_single_candidate_convergence(self, candidate: int, target: int) -> Tuple[bool, float]:
		"""Check CCM convergence for a candidate variable predicting a given target.

		:param candidate: candidate variable to check
		:param target: Target column index
		:return: (convergent, ccm_slope) tuple
		"""
		from scipy.signal import argrelextrema

		cache_key = (candidate, target)

		if self._userProvidedE:
			best_e = self.embedDimensions
		elif cache_key not in self.optimalEmbeddingDimensions:
			dims, corrs = FindOptimalEmbeddingDimensionality(
				self.data,
				[candidate],
				target,
				self.CCMMaxE,
				train = self.train,
				test = self.test,
				predictionHorizon = self.predictionHorizon,
				noTime = self.noTime
				)

			correlations = numpy.array(corrs)

			if self.FirstEMax:
				local_max_indices = argrelextrema(correlations, numpy.greater)[0]
				if len(local_max_indices) > 0:
					best_e_idx = local_max_indices[0]
				else:
					best_e_idx = len(correlations) - 1
			else:
				best_e_idx = numpy.argmax(correlations)

			best_e = int(dims[best_e_idx])
			best_e_correlation = correlations[best_e_idx]

			if best_e_correlation < self.EmbedDimCorrelationMin:
				return (False, 0.0)

			self.optimalEmbeddingDimensions[cache_key] = best_e
		else:
			best_e = self.optimalEmbeddingDimensions[cache_key]

		lib_sizes = [int(percentile / 100 * self.trainData.shape[0]) for percentile in self.CCMLibraryPercentiles]

		if len(lib_sizes) < 2:
			if self.verbose:
				print('Warning: Not enough library sizes for CCM convergence check on column {}'.format(candidate))
			return (True, 0.5)

		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = lib_sizes_normalized / lib_sizes_normalized.max()

		batchedCCM = BatchedCCM(
			X = self.data[:, [candidate]],
			Y = self.data[:, target],
			trainSizes = lib_sizes,
			sample = self.CCMNumSamples,
			embedDimensions = best_e,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn if self.knn > 0 else best_e + 1,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			validLib = self.validLib,
			includeData = False,
			ignoreNan = self.ignoreNan,
			directions = 'reverse',
			trainBlockIndices = self.train,
			testBlockIndices = self.test,
			device = self.device,
			batchSize = 1,
			batchMode = 'samples',
			useHalfPrecision = self.use_half_precision,
			showProgress = False,
			seed = self.CCMSeed
			)

		result = batchedCCM.Run()

		del batchedCCM
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		x = torch.tensor(lib_sizes_normalized, dtype = torch.float32, device = self.device)
		y = torch.tensor(result.reverse_performance, dtype = torch.float32, device = self.device)

		x_mean = x.mean()
		y_mean = y.mean()
		xy_mean = (x * y).mean()
		x_var = (x ** 2).mean() - x_mean ** 2
		slope = float((xy_mean - x_mean * y_mean) / x_var)

		return (slope > self.CCMConvergenceThreshold, slope)
