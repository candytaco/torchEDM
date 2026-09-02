from typing import List
import numpy
import torch
from tqdm import tqdm as ProgressBar
from sklearn.model_selection import KFold
from .Results import MDECVResult, MDEResult
from .MDE import MDE
from ..Scoring import Correlation


class MDECV:
	"""Manifold dimensional expansion with Cross-Validation.

	This class extends MDE with cross-validation functionality, allowing
	feature selection to be performed within each fold and providing
	methods for selecting final features based on different criteria.
	"""

	def __init__(self,
				 trainData: numpy.ndarray,
				 target: int,
				 maxD: int = 5,
				 include_target: bool = True,
				 convergent: bool = True,
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 dtype: torch.dtype = torch.float32,
				 folds: int = 5,
				 test_size: float = 0.2,
				 final_feature_mode: str = "best_fold",
				 columns=None,
				 train=None,
				 test=None,
				 embedDimensions=0,
				 predictionHorizon=1,
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
				 solver=None,
				 stdThreshold = 1e-2,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5, ),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 IterativeDimensionSearch: bool = False,
				 TimeDelay: int = 0):
		"""Initialize MDECV with data and parameters.

		Parameters
		----------
		trainData : numpy.ndarray
			train data for cross-validation 0 is time (unless noTime=True)
		target : int
			Column index of the target column to forecast
		maxD : int, default=5
			Maximum number of features to select (including target if include_target=True)
		include_target : bool, default=True
			Whether to start with target in feature list
		convergent : bool, default=True
			Whether to use convergence checking for feature selection
		metric : str, default="correlation"
			Metric to use: "correlation" or "MAE"
		batch_size : int, default=1000
			Number of features to process in each parallel batch
		dtype : torch.dtype, default=torch.float32
			Torch dtype for tensors (e.g. torch.float32 or torch.float16)
		folds : int, default=5
			Number of cross-validation folds
		test_size : float, default=0.2
			Proportion of data to use for test set
		final_feature_mode : str, default="best_fold"
			Method for selecting final features:
			- "best_fold": Use features from best performing fold
			- "frequency": Use most frequent features across folds
			- "best_N": Use top N features based on incremental prediction
		columns : list of int, optional
			Column indices to use for embedding (defaults to all except time)
		train : tuple of (int, int), optional
			Training set indices [start, end]
		test : tuple of (int, int), optional
			Test set indices [start, end]
		embedDimensions : int, default=0
			Embedding dimension (E). If 0, will be set by Validate()
		predictionHorizon : int, default=1
			Prediction time horizon (Tp)
		knn : int, default=0
			Number of nearest neighbors. If 0, will be set to E+1 by Validate()
		step : int, default=-1
			Time delay step size (tau). Negative values indicate lag
		exclusionRadius : int, default=0
			Temporal exclusion radius for neighbors
		embedded : bool, default=False
			Whether data is already embedded
		validLib : list, optional
			Boolean mask for valid library points
		noTime : bool, default=False
			Whether first column is time or data
		ignoreNan : bool, default=True
			Remove NaN values from embedding
		verbose : bool, default=False
			Print diagnostic messages
		useSMap : bool, default=False
			Whether to use SMap instead of Simplex
		theta : float, default=0.0
			S-Map localization parameter. theta=0 is global linear map,
			larger values increase localization
		solver : object, optional
			Solver to use for S-Map regression. If None, uses numpy.linalg.lstsq.
			Can be any sklearn-compatible regressor.
		CCMLibraryPercentiles : array-like, default=numpy.linspace(10, 90, 5)
			Library sizes for the convergence check as percent of train data size
		CCMNumSamples : int, default=10
			Number of random samples per library size for the convergence check
		CCMConvergenceThreshold : float, default=0.01
			Minimum slope threshold for convergence
		CCMSeed : int, optional
			Random seed for reproducible convergence-check sampling
		CCMMaxEmbeddingDimensions : int, default=15
			Maximum embedding dimension for the per-variable dimension search
		MinPredictionThreshold : float, default=0.0
			Minimum performance threshold for candidate filtering
		MinCandidatePerformance : float, default=0.5
			Minimum performance a candidate alone must reach predicting the
			target to stay in the candidate pool; 0 disables
		IterativeDimensionSearch : bool, default=False
			If True, the candidate dimension search evaluates each depth on
			its own valid rows instead of one batched pass
		TimeDelay : int, default=0
			Time delay analysis depth. If 0, time delay analysis is disabled
		"""
		self.data = trainData
		self.target = target
		self.maxD = maxD
		self.include_target = include_target
		self.convergent = convergent
		self.metric = metric
		self.batch_size = batch_size
		self.dtype = dtype
		self.folds = folds
		self.test_size = test_size
		self.final_feature_mode = final_feature_mode
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
		self.solver = solver

		self.stdThreshold = stdThreshold

		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxEmbeddingDimensions = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.MinCandidatePerformance = MinCandidatePerformance
		self.IterativeDimensionSearch = IterativeDimensionSearch
		self.TimeDelay = TimeDelay

		# Cross-validation results
		self.fold_results: List[MDEResult] = []
		self.test_accuracy = []
		self.bestFold = None
		self.best_fold_features = None
		self.best_fold_accuracy = None

	def fit(self, scoring_function = Correlation) -> None:
		"""Perform cross-validation using MDE in each fold.

		:param scoring_function: Scoring function taking (actual, predicted) and returning a scalar. Default is Correlation.
		"""

		# Build fold indices
		if self.folds <= 1:
			trainIndices = numpy.arange(len(self.data))
			fold_indices = [(trainIndices, trainIndices)]
		else:
			kFolds = KFold(n_splits = self.folds, shuffle = False)
			fold_indices = [
				(train_idx, val_idx)
				for train_idx, val_idx in kFolds.split(self.data)
			]

		# Process each fold
		progressBar = ProgressBar(total = self.folds, desc = 'MDE CV Fold', leave = False)
		for fold, (trainIndices, validationIndices) in enumerate(fold_indices, start = 1):
			fold_result = self.fitSingleFold(self.data[trainIndices], self.data[validationIndices],
											 scoring_function = scoring_function)
			self.fold_results.append(fold_result)
			progressBar.update(1)

		# Identify best fold
		self.test_accuracy = [float(r.score[0]) for r in self.fold_results]
		self.bestFold = numpy.argmax(self.test_accuracy)
		self.best_fold_accuracy = self.test_accuracy[self.bestFold]
		self.best_fold_features = self.fold_results[self.bestFold].selected_features

	def fitSingleFold(self,
				   train_data: numpy.ndarray,
				   val_data: numpy.ndarray,
				   return_predictions: bool = True,
				   scoring_function = Correlation) -> MDEResult:
		"""Process a single cross-validation fold.

		Parameters
		----------
		train_data : numpy.ndarray
			Training data for this fold
		val_data : numpy.ndarray
			Validation data for this fold
		return_predictions : bool
			If False, the predictions field of the result will not be populated
		scoring_function : callable
			Scoring function taking (actual, predicted) and returning a scalar. Default is Correlation.

		Returns
		-------
		MDEResult
			Results from this fold
		"""
		# Create fold-specific data
		fold_data = numpy.vstack([train_data, val_data])

		# Run MDE on this fold
		mde = MDE(
			data = fold_data,
			target = self.target,
			maxD = self.maxD,
			include_target = self.include_target,
			convergent = self.convergent,
			metric = self.metric,
			batch_size = self.batch_size,
			dtype = self.dtype,
			columns = self.columns,
			train = [(0, len(train_data))],
			test = [(len(train_data), len(fold_data))],
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			embedded = self.embedded,
			validLib = self.validLib,
			noTime = self.noTime,
			ignoreNan = self.ignoreNan,
			verbose = self.verbose,
			useSMap = self.useSMap,
			theta = self.theta,
			stdThreshold = self.stdThreshold,
			CCMLibraryPercentiles = self.CCMLibraryPercentiles,
			CCMNumSamples = self.CCMNumSamples,
			CCMConvergenceThreshold = self.CCMConvergenceThreshold,
			CCMSeed = self.CCMSeed,
			CCMMaxEmbeddingDimensions = self.CCMMaxEmbeddingDimensions,
			MinPredictionThreshold = self.MinPredictionThreshold,
			MinCandidatePerformance = self.MinCandidatePerformance,
			IterativeDimensionSearch = self.IterativeDimensionSearch,
			TimeDelay = self.TimeDelay
		)

		return mde.Run(return_predictions = return_predictions, scoring_function = scoring_function)

	def predict(self, testData: numpy.ndarray) -> MDECVResult:
		"""Predict using the final chosen feature set on the test set.
		Note that this is different than the original implementation where the test data is
		internally generated from the data array


		:param testData:	test data with same variables as train data

		:return: Cross-validation results including final prediction
		"""
		# Decide final features based on mode
		if self.final_feature_mode == "best_fold":
			features = self.best_fold_features
		elif self.final_feature_mode == "frequency":
			features = self._get_frequency_features()
		else:
			features = self._get_best_n_features()

		stackedData = numpy.vstack((self.data, testData))

		# Run final prediction on test set
		# TODO: shouldn't this be a simplex or smap?
		mde = MDE(
			data = stackedData,
			target = self.target,
			maxD = self.maxD,
			include_target = self.include_target,
			convergent = self.convergent,
			metric = self.metric,
			batch_size = self.batch_size,
			dtype = self.dtype,
			columns = self.columns,
			train = [(0, self.data.shape[0])],
			test = [(self.data.shape[0] - 1, stackedData.shape[0])],
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			embedded = self.embedded,
			validLib = self.validLib,
			noTime = self.noTime,
			ignoreNan = self.ignoreNan,
			verbose = self.verbose,
			useSMap = self.useSMap,
			theta = self.theta,
			stdThreshold = self.stdThreshold,
			CCMLibraryPercentiles = self.CCMLibraryPercentiles,
			CCMNumSamples = self.CCMNumSamples,
			CCMConvergenceThreshold = self.CCMConvergenceThreshold,
			CCMSeed = self.CCMSeed,
			CCMMaxEmbeddingDimensions = self.CCMMaxEmbeddingDimensions,
			MinPredictionThreshold = self.MinPredictionThreshold,
			MinCandidatePerformance = self.MinCandidatePerformance,
			IterativeDimensionSearch = self.IterativeDimensionSearch,
			TimeDelay = self.TimeDelay
		)

		final_result = mde.Run()

		return MDECVResult(
			final_forecast = final_result.final_forecast,
			selected_variables = features,
			fold_results = self.fold_results,
			accuracy = self.test_accuracy,
			best_fold = self.bestFold
		)

	def _get_frequency_features(self) -> List[int]:
		"""Get most frequent features across folds.

		Returns
		-------
		list of int
			List of most frequent feature indices
		"""
		all_features = []
		for result in self.fold_results:
			all_features.extend(result.selected_features)

		# Count frequency of each feature
		feature_counts = {}
		for feat in all_features:
			feature_counts[feat] = feature_counts.get(feat, 0) + 1

		# Sort by frequency and return top features
		sorted_features = sorted(feature_counts.items(), key = lambda x: x[1], reverse = True)
		return [feat for feat, count in sorted_features[:self.maxD]]

	def _get_best_n_features(self) -> List[int]:
		"""Get top N features based on incremental prediction.

		Returns
		-------
		list of int
			List of top N feature indices
		"""
		# Get frequency features first
		freq_features = self._get_frequency_features()

		# Perform incremental prediction to find best N
		# This is a simplified version
		return freq_features[:len(freq_features) // 2]
