from typing import Optional, List, Union

import numpy
from tqdm import tqdm as ProgressBar

from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter
from .CVSplitter import EDMCVSplitter
from ..EDM.MDE import MDE
from ..EDM.Results import MDEResult, MDECVResult
from ..EDM.Simplex import Simplex


class MDEFitterCV(EDMFitter):
	"""
	MDE with cross-validation – supports both n-fold and leave-one-run-out CV.

	The fit/predict split maps naturally onto a two-stage workflow:

	* ``fit(X_train, y_train)`` – runs cross-validated MDE feature selection on
	  training data only; stores the best feature set.
	* ``predict(X_test, y_test)`` – makes a final prediction on held-out test
	  data using the features selected during ``fit``.

	Usage::

	    fitter = MDEFitterCV(MaxD=5, PredictionHorizon=1)
	    fitter.fit(X_train, y_train)
	    y_pred = fitter.predict(X_test, y_test)
	    result = fitter.result_  # MDECVResult with fold details
	"""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'pre',
				 Metric: str = "correlation",
				 BatchSize: int = 10000,
				 HalfPrecision: bool = False,
				 Folds: int = 5,
				 LeaveOneRunOut: bool = True,
				 FinalFeatureMode: str = "best_fold",
				 Embed: bool = False,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 knn: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 stdThreshold: float = 1e-2,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0,
				 progressBar: bool = True):
		"""
		:param MaxD:                     Maximum number of features to select.
		:param IncludeTarget:            Start with target variable in the feature list.
		:param Convergent:               CCM convergence mode: ``'pre'``, ``'post'``, or ``False``.
		:param Metric:                   Optimisation metric: ``'correlation'`` or ``'MAE'``.
		:param BatchSize:                Features processed per GPU batch.
		:param HalfPrecision:            Use float16 tensors on GPU.
		:param Folds:                    Number of CV folds (ignored when LeaveOneRunOut is True).
		:param LeaveOneRunOut:           Use leave-one-run-out CV instead of n-fold.
		:param FinalFeatureMode:         How to pick final features: ``'best_fold'``,
		                                 ``'frequency'``, or ``'reselect'``.
		:param Embed:                    Embed the data before feature selection.
		:param EmbedDimensions:          Embedding dimension (E).
		:param PredictionHorizon:        Prediction time horizon (Tp).
		:param knn:                      Number of nearest neighbours (0 → E+1).
		:param Step:                     Time-delay step size (tau).
		:param ExclusionRadius:          Temporal exclusion radius for neighbours.
		:param Verbose:                  Print diagnostic messages.
		:param UseSMap:                  Use S-Map instead of Simplex.
		:param Theta:                    S-Map localisation parameter.
		:param stdThreshold:             Minimum std dev to include a variable.
		:param CCMLibraryPercentiles:    Library-size percentiles for CCM pre-check.
		:param CCMNumSamples:            Random samples per library size for CCM.
		:param CCMConvergenceThreshold:  Minimum slope for CCM convergence.
		:param MinPredictionThreshold:   Minimum correlation to add a feature.
		:param EmbedDimCorrelationMin:   Minimum in-sample correlation for embedding.
		:param FirstEMax:                Use highest-E model in first step.
		:param TimeDelay:                Extra time delay (0 = none).
		:param progressBar:              Show progress bar during CV.
		"""
		super().__init__(progressBar)

		self.MaxD                    = MaxD
		self.IncludeTarget           = IncludeTarget
		self.Convergent              = Convergent
		self.Metric                  = Metric
		self.BatchSize               = BatchSize
		self.HalfPrecision           = HalfPrecision
		self.Folds                   = Folds
		self.LeaveOneRunOut          = LeaveOneRunOut
		self.FinalFeatureMode        = FinalFeatureMode
		self.EmbedDimensions         = EmbedDimensions
		self.PredictionHorizon       = PredictionHorizon
		self.KNN                     = knn
		self.Step                    = Step
		self.ExclusionRadius         = ExclusionRadius
		self.Verbose                 = Verbose
		self.UseSMap                 = UseSMap
		self.Theta                   = Theta
		self.embed                   = Embed
		self.stdThreshold            = stdThreshold
		self.CCMLibraryPercentiles   = CCMLibraryPercentiles
		self.CCMNumSamples           = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.MinPredictionThreshold  = MinPredictionThreshold
		self.EmbedDimCorrelationMin  = EmbedDimCorrelationMin
		self.FirstEMax               = FirstEMax
		self.TimeDelay               = TimeDelay

		self.trainDataAdapter  = None
		self.cvSplitter        = None
		self.foldResults: List[MDEResult] = []
		self.foldAccuracies: List[float]  = []
		self.bestFold: Optional[int]      = None
		self.bestFoldFeatures: Optional[List[int]] = None
		self.bestFoldAccuracy: Optional[float]     = None

	# ------------------------------------------------------------------
	# sklearn-compatible public interface
	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None,
			initialVariables: Optional[List[int]] = None) -> 'MDEFitterCV':
		"""
		Run cross-validated MDE feature selection on training data.

		:param X_train:           Training features (array or list of arrays for
		                          multiple independent runs).
		:param y_train:           Training target (array or list of arrays).
		:param TrainStart:        Rows to skip at the start of each training run.
		:param TrainEnd:          Rows to drop at the end of each training run.
		:param trainTime:         Optional time-stamp column for training data.
		:param initialVariables:  Optional pre-selected column indices to start from.
		:return: self
		"""
		super().fit(X_train, y_train, TrainStart, TrainEnd, trainTime)

		self.trainDataAdapter = DataAdapter.MakeDataAdapter(
			X_train, y_train, None, None,
			TrainStart, TrainEnd, 0, 0,
			trainTime, None,
		)

		self.cvSplitter = EDMCVSplitter(
			dataAdapter    = self.trainDataAdapter,
			nFolds         = self.Folds,
			leaveOneRunOut = self.LeaveOneRunOut,
			edmStyleIndices = True,
		)

		trainData = self.trainDataAdapter.fullData
		target    = trainData.shape[1] - 1

		self.foldResults    = []
		self.foldAccuracies = []

		numSplits   = self.cvSplitter.GetNSplits()
		progressBar = ProgressBar(total=numSplits, desc='MDE CV Fold', leave=False)

		for trainIndices, testIndices in self.cvSplitter.Split():
			foldResult = self._fit_single_fold(
				trainData, trainIndices, testIndices, target, initialVariables
			)
			self.foldResults.append(foldResult)
			self.foldAccuracies.append(foldResult.compute_error())
			progressBar.update(1)

		self.bestFold         = int(numpy.argmax(self.foldAccuracies))
		self.bestFoldAccuracy = self.foldAccuracies[self.bestFold]
		self.bestFoldFeatures = self.foldResults[self.bestFold].selected_features

		return self

	def predict(self,
				X_test: numpy.ndarray,
				y_test: Optional[numpy.ndarray] = None,
				TestStart: int = 0,
				TestEnd: int = 0,
				testTime: Optional[numpy.ndarray] = None) -> numpy.ndarray:
		"""
		Predict on test data using the feature set selected during ``fit``.

		:param X_test:   Test feature data.
		:param y_test:   Test target data (optional; used for evaluation only).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: 1-D array of predicted values (NaN where no prediction).
		:raises RuntimeError: if ``fit`` has not been called yet.
		:raises ValueError:   if no test data is provided.
		"""
		self._check_is_fitted()

		if len(self.foldResults) == 0:
			raise RuntimeError("Model not fitted. Call fit() first.")

		if X_test is None:
			raise ValueError("X_test must be provided")

		if y_test is None:
			y_test = numpy.zeros((X_test.shape[0], 1))

		# Build combined data adapter
		self.DataAdapter = DataAdapter.MakeDataAdapter(
			self.trainDataAdapter.XTrain,
			self.trainDataAdapter.YTrain,
			X_test, y_test,
			self.trainDataAdapter.TrainStart,
			self.trainDataAdapter.TrainEnd,
			TestStart, TestEnd,
			self.trainDataAdapter.trainTime,
			testTime,
		)

		# Select final feature set
		if self.FinalFeatureMode == "frequency":
			features = self._get_frequency_features()
		elif self.FinalFeatureMode == 'reselect':
			allSelected: List[int] = []
			for fold in self.foldResults:
				allSelected += fold.selected_features
			uniqueSortedFeatures = sorted(set(allSelected))
			last             = self.DataAdapter.YIndex
			reselect_columns = uniqueSortedFeatures + [last]
			res = self._fit_single_fold(
				self.DataAdapter.fullData[:, reselect_columns],
				self.DataAdapter.TrainIndices,
				self.DataAdapter.TestIndices,
				len(uniqueSortedFeatures),
				convergent=False,
			)
			features = res.selected_features
		else:  # "best_fold"
			features = self.bestFoldFeatures

		simplex = Simplex(
			data              = self.DataAdapter.fullData,
			columns           = features,
			target            = self.DataAdapter.YIndex,
			train             = self.DataAdapter.TrainIndices,
			test              = self.DataAdapter.TestIndices,
			embedDimensions   = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn               = self.KNN,
			step              = self.Step,
			exclusionRadius   = self.ExclusionRadius,
			noTime            = self.DataAdapter.hasTime,
			verbose           = self.Verbose,
			embedded          = True,
		)

		simplex_result = simplex.Run()
		self.result_   = MDECVResult(
			final_forecast    = simplex_result,
			selected_features = features,
			fold_results      = self.foldResults,
			accuracy          = self.foldAccuracies,
			best_fold         = self.bestFold,
		)
		return self.result_.predictions

	def get_params(self, deep: bool = True) -> dict:
		return {
			'MaxD':                    self.MaxD,
			'IncludeTarget':           self.IncludeTarget,
			'Convergent':              self.Convergent,
			'Metric':                  self.Metric,
			'BatchSize':               self.BatchSize,
			'HalfPrecision':           self.HalfPrecision,
			'Folds':                   self.Folds,
			'LeaveOneRunOut':          self.LeaveOneRunOut,
			'FinalFeatureMode':        self.FinalFeatureMode,
			'Embed':                   self.embed,
			'EmbedDimensions':         self.EmbedDimensions,
			'PredictionHorizon':       self.PredictionHorizon,
			'knn':                     self.KNN,
			'Step':                    self.Step,
			'ExclusionRadius':         self.ExclusionRadius,
			'Verbose':                 self.Verbose,
			'UseSMap':                 self.UseSMap,
			'Theta':                   self.Theta,
			'stdThreshold':            self.stdThreshold,
			'CCMLibraryPercentiles':   self.CCMLibraryPercentiles,
			'CCMNumSamples':           self.CCMNumSamples,
			'CCMConvergenceThreshold': self.CCMConvergenceThreshold,
			'MinPredictionThreshold':  self.MinPredictionThreshold,
			'EmbedDimCorrelationMin':  self.EmbedDimCorrelationMin,
			'FirstEMax':               self.FirstEMax,
			'TimeDelay':               self.TimeDelay,
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _fit_single_fold(self,
						 data: numpy.ndarray,
						 trainIndices: List[int],
						 testIndices: List[int],
						 target: int,
						 initialVariables: Optional[List[int]] = None,
						 convergent: Optional[bool] = None) -> MDEResult:
		"""
		Run MDE on a single CV fold.

		:param data:             Full data array for this fold.
		:param trainIndices:     EDM-style train indices.
		:param testIndices:      EDM-style test indices.
		:param target:           Target column index.
		:param initialVariables: Optional pre-selected column indices.
		:param convergent:       Override convergence setting (None → use self.Convergent).
		:return: MDEResult for this fold.
		"""
		mde = MDE(
			data                    = data,
			target                  = target,
			maxD                    = self.MaxD,
			include_target          = self.IncludeTarget,
			convergent              = convergent if convergent is not None else self.Convergent,
			metric                  = self.Metric,
			batch_size              = self.BatchSize,
			use_half_precision      = self.HalfPrecision,
			columns                 = initialVariables,
			train                   = trainIndices,
			test                    = testIndices,
			embedDimensions         = self.EmbedDimensions,
			predictionHorizon       = self.PredictionHorizon,
			knn                     = self.KNN,
			step                    = self.Step,
			exclusionRadius         = self.ExclusionRadius,
			embedded                = not self.embed,
			noTime                  = not self.trainDataAdapter.HasTime,
			verbose                 = self.Verbose,
			useSMap                 = self.UseSMap,
			theta                   = self.Theta,
			stdThreshold            = self.stdThreshold,
			CCMLibraryPercentiles   = self.CCMLibraryPercentiles,
			CCMNumSamples           = self.CCMNumSamples,
			CCMConvergenceThreshold = self.CCMConvergenceThreshold,
			MinPredictionThreshold  = self.MinPredictionThreshold,
			EmbedDimCorrelationMin  = self.EmbedDimCorrelationMin,
			FirstEMax               = self.FirstEMax,
			TimeDelay               = self.TimeDelay,
		)
		return mde.Run()

	def _get_frequency_features(self) -> List[int]:
		"""Return the MaxD most-frequent features across all CV folds."""
		featureCounts: dict = {}
		for result in self.foldResults:
			for feature in result.selected_features:
				featureCounts[feature] = featureCounts.get(feature, 0) + 1
		sortedFeatures = sorted(featureCounts.items(), key=lambda x: x[1], reverse=True)
		return [feature for feature, _ in sortedFeatures[: self.MaxD]]
