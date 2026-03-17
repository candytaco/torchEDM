from typing import List, Tuple

import numpy
from tqdm import tqdm as ProgressBar
import torch

from .CCM_batch import BatchedCCM
from .Results import MDEResult, SimplexResult
from .SMap import SMap
from .Simplex import Simplex

from ._MDE import RowwiseCorrelation, RowwiseR2, FloorArray
from ..Hyperparameters import FindOptimalEmbeddingDimensionality


class MDE:
	"""Manifold dimensional expansion for feature selection.

	This class implements the iterative feature selection algorithm that
	evaluates combinations of features using EDM methods and selects the
	best performing features based on convergence criteria.
	"""

	def __init__(self,
				 data: numpy.ndarray,
				 target: int,
				 maxD: int = 5,
				 include_target: bool = False,
				 convergent = 'post',
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 use_half_precision: bool = False,
				 columns=None,
				 train=None,
				 test=None,
				 embedDimensions=0,
				 predictionHorizon=0,
				 knn=0,
				 step=-1,
				 exclusionRadius=0,
				 embedded=False,
				 validLib=None,
				 noTime=False,
				 ignoreNan=True,
				 verbose=False,
				 useSMap: bool = False,
				 theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0):
		"""Initialize MDE with data and parameters.

		:param data: 	2D numpy array where column 0 is time (unless noTime=True)
		:param target: 	Column index of the target column to forecast
		:param maxD: 	Maximum number of features to select (including target if include_target=True)
		:param include_target: 	Whether to start with target in feature list
		:param convergent: 	Convergence checking mode: 'pre' runs batch CCM on all variables before selection, 'post' checks convergence within each selection loop iteration, False disables convergence checking
		:param metric: 	Metric to use: "correlation" or "MAE"
		:param batch_size: 	Number of features to process in each batch
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
		:param MinPredictionThreshold: 	Minimum correlation threshold for candidate filtering
		:param EmbedDimCorrelationMin: 	Minimum correlation for E selection
		:param FirstEMax: 	Use first local maximum in E-rho curve instead of global max
		:param TimeDelay: 	Time delay analysis depth. If 0, time delay analysis is disabled
		"""
		self.data = data
		self.target = target
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
		self.MinPredictionThreshold = MinPredictionThreshold
		self.EmbedDimCorrelationMin = EmbedDimCorrelationMin
		self.FirstEMax = FirstEMax
		self.TimeDelay = TimeDelay
		self.optimalEmbeddingDimensions = {}

		# True when the caller explicitly supplied embedDimensions > 0 (equivalent
		# to dimx's args.E > 0). When False, _check_convergence finds optimal E
		# per variable, matching dimx's default behavior of calling EmbedDimension()
		# for each CCM candidate.
		self._userProvidedE = embedDimensions != 0

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.dtype = torch.float16 if use_half_precision else torch.float32

		self.stepwise_performance = None # performances of adding each variable at each iteration
		self.all_distances = None
		self.current_best_distance_matrix = None

		# Initialize feature selection state
		self.selectedVariables = []
		self.accuracy = []
		self.ccm_values = []
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

	def Run(self) -> MDEResult:
		"""Execute MDE feature selection and return results.

		:return: Results containing final prediction, selected features, accuracy, and CCM values
		:rtype: MDEResult
		"""
		if self.embedDimensions == 0:
			dims, corrs = FindOptimalEmbeddingDimensionality(self.data, [self.target], self.target, self.maxD,
																	  train = self.train, test = self.test, predictionHorizon = self.predictionHorizon,
																	  noTime = self.noTime)
			self.embedDimensions = dims[numpy.argmax(corrs)]
		if self.knn == 0:
			self.knn = self.embedDimensions + 1

		# variable selection
		self._select_features()

		# Final training and testing
		finalPrediction = self._final_prediction()

		self.results_ = MDEResult(
			final_forecast = finalPrediction,
			selected_features = self.selectedVariables,
			accuracy = self.accuracy,
			ccm_values = self.ccm_values,
			stepwise_performance = self.stepwise_performance,
			timeDelayResults = self.timeDelayResults
		)
		return self.results_

	def _select_features(self) -> None:
		"""Perform iterative feature selection with parallel processing."""

		self.selectedVariables = []
		if self.include_target:
			self.selectedVariables.append(self.target)
		#
		# we use this to correctly get indices for calculating the distance tensor
		dummy = Simplex(
			data = self.data,
			columns = numpy.arange(self.data.shape[1]).tolist(),
			target = self.target,
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
		
		all_columns = numpy.arange(nVars - 1) # ignore the Y var, which is the last column
		# ignore all variables with stdev less than threshold
		excluded = numpy.argwhere(numpy.std(self.data, axis = 0) < self.stdThreshold).squeeze().tolist()
		if not self.noTime: # time is first column if true and we exclude that
			excluded.append(0)
		if not self.include_target:
			excluded.append(self.target)
		excluded += self.selectedVariables

		remaining_variables = [c for c in all_columns if c not in excluded]

		# Filter convergent variables BEFORE selection if convergent='pre'
		if self.convergent == 'pre':
			remaining_variables = self._filter_convergent_variables(remaining_variables)

		# Iteratively add variables up to maxD
		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False)

		# rankings is a numpy array because it's apparently more multithreading friendly
		# than just storing the lists that come out?
		self.stepwise_performance = numpy.zeros([self.maxD, self.data.shape[1]])

		trainData_tensor = torch.tensor(trainData, device = self.device, dtype = self.dtype)
		testData_tensor = torch.tensor(testData, device = self.device, dtype = self.dtype)
		# Initialize distance matrix from the exclusion mask. Excluded pairs must
		# be set to inf so topk always ranks them last, matching pyEDM's handling.
		# Casting the boolean mask to float would give 1.0 for excluded pairs,
		# which is not large enough to guarantee exclusion from the top-k.
		exclusion_mask = dummy._BuildExclusionMask()
		current_best_distance_matrix = torch.zeros([nTrain, nTest], device = self.device, dtype = self.dtype)
		if exclusion_mask.any():
			current_best_distance_matrix[torch.tensor(exclusion_mask, device = self.device)] = float('inf')

		# we offset the prediction horizon with the indices because this accounts for non-continuous data selection
		train_y = self.data[trainIndices + self.predictionHorizon, self.target]
		train_y_tensor = torch.tensor(train_y, device = self.device, dtype = self.dtype)
		test_y = self.data[testIndices + self.predictionHorizon, self.target]
		test_y_tensor = torch.tensor(test_y, device = self.device, dtype = self.dtype)

		# Pre-allocate tensors whose size does not vary with knn.
		# knn-dependent tensors (neighborDistances, nearestNeighbors, weights, select)
		# are not pre-allocated because current_knn grows as variables are added.
		batch_distances = torch.zeros([self.batch_size, nTrain, nTest], device = self.device, dtype = self.dtype)
		candidateDistances = torch.empty([self.batch_size, nTrain, nTest], device = self.device, dtype = self.dtype)
		minDistances = torch.empty([self.batch_size, nTest], device = self.device, dtype = self.dtype)
		weightSum = torch.empty([self.batch_size, nTest], device = self.device, dtype = self.dtype)
		predictions = torch.empty([self.batch_size, nTest], device = self.device, dtype = self.dtype)
		perfs = torch.zeros(self.batch_size, device = self.device, dtype = self.dtype)

		for i in range(self.maxD):
			# knn grows as variables are added, matching dimx where knn = E + 1
			# and E is the number of columns in the current embedding.
			# At iteration i, the embedding contains len(selectedVariables) already-
			# chosen columns plus the one candidate being evaluated, giving E =
			# len(selectedVariables) + 1 and therefore knn = len(selectedVariables) + 2.
			current_knn = len(self.selectedVariables) + 2

			# Process remaining variables in batches to avoid OOM
			metric_results = []

			for batch_start in range(0, len(remaining_variables), self.batch_size):
				batch_end = min(batch_start + self.batch_size, len(remaining_variables))
				batch_vars = remaining_variables[batch_start:batch_end]
				batch_size = len(batch_vars)

				# Compute distances for this batch of variables
				for j, var in enumerate(batch_vars):
					diff = trainData_tensor[:, var].unsqueeze(1) - testData_tensor[:, var].unsqueeze(0)
					batch_distances[j, :, :] = diff * diff

				# Add current best distances (slice to actual batch size)
				torch.add(batch_distances[:batch_size], current_best_distance_matrix.unsqueeze(0), out = candidateDistances[:batch_size])

				# Find current_knn nearest neighbors. current_knn varies per outer
				# iteration so we do not use pre-allocated out= buffers here.
				# candidateDistances holds squared Euclidean distances; topk ranking
				# is identical on squared vs. Euclidean (sqrt is monotone), so
				# neighbor selection is correct. We take sqrt afterward so that the
				# weight formula exp(-d/d_min) uses Euclidean distances, matching
				# Simplex. Without sqrt, weights would be exp(-d²/d²_min) instead.
				neighborDistances, nearestNeighbors = torch.topk(candidateDistances[:batch_size], current_knn, dim = 1, largest = False)
				neighborDistances.sqrt_()
				FloorArray(neighborDistances, 1e-6)

				# compute weights and predictions
				torch.amin(neighborDistances, dim = 1, out = minDistances[:batch_size])
				weights = neighborDistances / minDistances[:batch_size].unsqueeze(1)
				weights.neg_().exp_()
				torch.sum(weights, dim = 1, out = weightSum[:batch_size])
				select = train_y_tensor[nearestNeighbors]
				torch.sum(weights * select, dim = 1, out = predictions[:batch_size])
				predictions[:batch_size].div_(weightSum[:batch_size])

				# calculate performances (slice to actual batch size)
				perfs[:batch_size].zero_()
				self.EvaluatePerformance(test_y_tensor, predictions[:batch_size], perfs[:batch_size])

				# Convert to list of tuples
				perfs_numpy = perfs[:batch_size].cpu().numpy()
				batch_results = [(var, perfs_numpy[j]) for j, var in enumerate(batch_vars)]
				metric_results.extend(batch_results)

			metric_results.sort(key=lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf, reverse=True)

			# Apply correlation threshold filtering
			if self.MinPredictionThreshold > 0:
				original_count = len(metric_results)
				metric_results = [(var, score) for var, score in metric_results if not numpy.isnan(score) and score >= self.MinPredictionThreshold]
				if self.verbose and len(metric_results) < original_count:
					print(f"Filtered {original_count - len(metric_results)} candidates below correlation threshold {self.MinPredictionThreshold}")

			# Flatten results and sort
			r = numpy.array(metric_results) if len(metric_results) > 0 else numpy.array([]).reshape(0, 2)
			if len(r) > 0:
				self.stepwise_performance[i, r[:, 0].astype(int)] = r[:, 1]

			best_var = None
			best_score = None

			if self.convergent == 'post':
				# Iterate down ranked candidates, pick first that is convergent
				for candidate_var, candidate_score in metric_results:
					if numpy.isnan(candidate_score):
						continue
					is_convergent, ccm_slope = self._check_convergence(int(candidate_var))
					if is_convergent:
						best_var = candidate_var
						best_score = candidate_score
						self.ccm_values.append(ccm_slope)
						if self.verbose:
							print(f"Variable {int(candidate_var)} is convergent (slope={ccm_slope:.4f}), score={candidate_score:.4f}")
						break
					else:
						if self.verbose:
							print(f"Variable {int(candidate_var)} is NOT convergent (slope={ccm_slope:.4f}), skipping")
			else:
				# No post-convergence check: pick top scoring candidate
				if metric_results and not numpy.isnan(metric_results[0][1]):
					best_var = metric_results[0][0]
					best_score = metric_results[0][1]

			# Add best variable if found
			if best_var is not None:
				self.selectedVariables.append(best_var)
				remaining_variables.remove(best_var)
				self.accuracy.append(best_score)

				# calc distance matrix update
				train = trainData_tensor[:, best_var]
				test = testData_tensor[:, best_var]
				distances = (train.unsqueeze(1) - test.unsqueeze(0)) ** 2
				current_best_distance_matrix += distances
				progressBar.update(1)
			else:
				# No more valid candidates
				break

		# Clean up GPU tensors before time delay analysis
		if torch.cuda.is_available():
			del trainData_tensor
			del testData_tensor
			del train_y_tensor
			del test_y_tensor
			del batch_distances
			del candidateDistances
			del neighborDistances
			del nearestNeighbors
			del minDistances
			del weights
			del weightSum
			del select
			del predictions
			del perfs
			torch.cuda.empty_cache()

		# Time delay analysis
		# TODO: parallelize this
		if self.TimeDelay > 0:
			if self.verbose:
				print(f"Starting time delay analysis with max delay {self.TimeDelay}")

			self.timeDelayResults = []
			best_accuracy = max(self.accuracy) if len(self.accuracy) > 0 else 0

			for var in self.selectedVariables:
				for delay in range(1, self.TimeDelay + 1):
					# Create time-delayed version by shifting the column
					delayed_data = numpy.roll(self.data[:, var], delay)
					# Zero out the first delay values to avoid wrap-around
					delayed_data[:delay] = numpy.nan

					# Temporarily add delayed column to data
					augmented_data = numpy.column_stack([self.data, delayed_data])
					delayed_col_idx = augmented_data.shape[1] - 1

					# Evaluate with delayed variable added
					test_variables = self.selectedVariables + [delayed_col_idx]

					# Save original data and restore after
					original_data = self.data
					self.data = augmented_data

					try:
						result = self._fit_single_EDM_instance(test_variables)
						score = self._compute_performance(result)

						improvement = score - best_accuracy
						self.timeDelayResults.append((var, delay, improvement, score))

						if self.verbose:
							print(f"Variable {var} with delay {delay}: score={score:.4f}, improvement={improvement:.4f}")

					except Exception as e:
						if self.verbose:
							print(f"Warning: Time delay evaluation failed for var {var}, delay {delay}: {e}")
					finally:
						self.data = original_data

		# Convert and clean up distance matrix
		self.current_best_distance_matrix = current_best_distance_matrix.cpu().numpy()
		del current_best_distance_matrix

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _fit_single_EDM_instance(self, variables: List[int]) -> SimplexResult:
		"""
		Fit a single EDM instalce with given variable indices.

		:param variables: Column indices to use for prediction
		:return: Prediction results
		:rtype: SimplexResult or SMapResult
		"""
		# Run prediction
		if self.useSMap:
			smap = SMap(
				data = self.data,
				columns = variables,
				target = self.target,
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
				target = self.target,
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
			return simplex.Run()

	def _compute_performance(self, result: SimplexResult) -> float:
		"""Compute optimization metric from prediction result.

		:param result: Prediction result
		:return: Metric value (correlation or MAE)
		:rtype: float
		"""
		if self.metric == "correlation":
			return result.compute_error()
		else:
			return result.compute_error("MAE")

	def _filter_convergent_variables(self, candidate_columns: List[int]) -> List[int]:
		"""Filter candidate variables to only include convergent ones using BatchedCCM.

		:param candidate_columns: Column indices to check for convergence
		:return: Tuple of convergent column indices and their CCM slopes
		:rtype: List[int]
		"""

		if len(candidate_columns) == 0:
			return []

		lib_sizes = [int(percentile / 100 * self.data.shape[0]) for percentile in self.CCMLibraryPercentiles]

		if len(lib_sizes) < 2:
			return candidate_columns

		# Normalize by maximum (matching dimx: libSizesVec / libSizesVec[-1]).
		# Min-max normalization would shift the x-axis origin to 0 and change
		# the slope values relative to the CCMConvergenceThreshold.
		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = lib_sizes_normalized / lib_sizes_normalized.max()

		X = self.data[:, candidate_columns]
		Y = self.data[:, self.target]

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
			useHalfPrecision = self.use_half_precision
		)

		result = batchedCCM.Run()

		# Clean up BatchedCCM GPU resources
		del batchedCCM
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		# Compute linear regression slopes for all columns in parallel
		# slope = cov(x,y) / var(x) = (mean(xy) - mean(x)*mean(y)) / (mean(x^2) - mean(x)^2)
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


	def _check_convergence(self, column: int) -> Tuple[bool, float]:
		"""Check convergence for a candidate feature.

		:param column: Column index to check
		:return: (convergent, ccm_value) tuple
		:rtype: Tuple[bool, float]
		"""
		from scipy.signal import argrelextrema

		# If the user explicitly provided E, use it for all CCM checks (matching
		# dimx when args.E > 0). Otherwise find optimal E per variable (matching
		# dimx's default behavior of calling EmbedDimension() for each candidate).
		# Results are cached in self.optimalEmbeddingDimensions to avoid redundant work.
		if self._userProvidedE:
			best_e = self.embedDimensions
		elif column not in self.optimalEmbeddingDimensions:
			# FindOptimalEmbeddingDimensionality returns (dimensions_list, correlations_list).
			dims, corrs = FindOptimalEmbeddingDimensionality(
				self.data,
				[column],
				self.target,
				self.maxD,
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

			self.optimalEmbeddingDimensions[column] = best_e
		else:
			best_e = self.optimalEmbeddingDimensions[column]

		lib_sizes = [int(percentile / 100 * self.data.shape[0]) for percentile in self.CCMLibraryPercentiles]

		if len(lib_sizes) < 2:
			if self.verbose:
				print(f"Warning: Not enough library sizes for CCM convergence check on column {column}")
			return (True, 0.5)

		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = lib_sizes_normalized / lib_sizes_normalized.max()

		batchedCCM = BatchedCCM(
			X = self.data[:, [column]],
			Y = self.data[:, self.target],
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
			useHalfPrecision = self.use_half_precision,
			showProgress = False
		)

		result = batchedCCM.Run()

		del batchedCCM
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		# result.reverse_performance has shape [len(lib_sizes)] for a single column+target
		x = torch.tensor(lib_sizes_normalized, dtype = torch.float32, device = self.device)
		y = torch.tensor(result.reverse_performance, dtype = torch.float32, device = self.device)

		x_mean = x.mean()
		y_mean = y.mean()
		xy_mean = (x * y).mean()
		x_var = (x ** 2).mean() - x_mean ** 2
		slope = float((xy_mean - x_mean * y_mean) / x_var)

		return (slope > self.CCMConvergenceThreshold, slope)

	def _final_prediction(self) -> numpy.ndarray:
		"""Run final prediction with selected features.

		:return: Final prediction array [Time, Observations, Predictions]
		:rtype: numpy.ndarray
		"""
		result = self._fit_single_EDM_instance(self.selectedVariables)
		return result.projection
