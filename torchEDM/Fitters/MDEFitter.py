
from typing import Optional, List, Union

import numpy

from torchEDM.EDM.MDE import MDE
from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter


class MDEFitter(EDMFitter):
	"""
	sklearn-compatible wrapper for Manifold Dimensional Expansion (MDE).

	MDE iteratively selects the best features for predicting a target variable
	using EDM (Simplex or S-Map) and optionally CCM convergence checking.

	Usage::

	    fitter = MDEFitter(MaxD=5, PredictionHorizon=1)
	    fitter.fit(X_train, y_train)
	    y_pred = fitter.predict(X_test, y_test)
	    result = fitter.result_   # MDEResult with selected_features, accuracy, etc.
	"""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'pre',
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 HalfPrecision: bool = False,
				 Embed: bool = False,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
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
		:param IncludeTarget:            Start with the target variable in the feature list.
		:param Convergent:               CCM convergence mode: ``'pre'``, ``'post'``, or ``False``.
		:param Metric:                   Optimisation metric: ``'correlation'`` or ``'MAE'``.
		:param BatchSize:                Features processed per GPU batch.
		:param HalfPrecision:            Use float16 tensors on GPU.
		:param Embed:                    Embed the data before feature selection.
		:param EmbedDimensions:          Embedding dimension (E).
		:param PredictionHorizon:        Prediction time horizon (Tp).
		:param KNN:                      Number of nearest neighbours (0 → E+1).
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
		:param progressBar:              Show progress bar.
		"""
		super().__init__(progressBar)

		self.MaxD                    = MaxD
		self.IncludeTarget           = IncludeTarget
		self.Convergent              = Convergent
		self.Metric                  = Metric
		self.BatchSize               = BatchSize
		self.HalfPrecision           = HalfPrecision
		self.EmbedDimensions         = EmbedDimensions
		self.PredictionHorizon       = PredictionHorizon
		self.KNN                     = KNN
		self.Step                    = Step
		self.ExclusionRadius         = ExclusionRadius
		self.Verbose                 = Verbose
		self.UseSMap                 = UseSMap
		self.Theta                   = Theta
		self.Embed                   = Embed
		self.stdThreshold            = stdThreshold
		self.CCMLibraryPercentiles   = CCMLibraryPercentiles
		self.CCMNumSamples           = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.MinPredictionThreshold  = MinPredictionThreshold
		self.EmbedDimCorrelationMin  = EmbedDimCorrelationMin
		self.FirstEMax               = FirstEMax
		self.TimeDelay               = TimeDelay

		self.MDE = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'MDEFitter':
		"""
		Store training data for MDE feature selection.

		:param X_train:    Training feature data.
		:param y_train:    Training target data.
		:param TrainStart: Rows to skip at the start of the training set.
		:param TrainEnd:   Rows to drop at the end of the training set.
		:param trainTime:  Optional time-stamp column for training data.
		:return: self
		"""
		super().fit(X_train, y_train, TrainStart, TrainEnd, trainTime)
		return self

	def predict(self,
				X_test: numpy.ndarray,
				y_test: Optional[numpy.ndarray] = None,
				TestStart: int = 0,
				TestEnd: int = 0,
				testTime: Optional[numpy.ndarray] = None) -> numpy.ndarray:
		"""
		Run MDE feature selection and return final predictions.

		Feature selection is performed on the training data; the identified
		features are then used to produce a final forecast on the test set.

		:param X_test:   Test feature data.
		:param y_test:   Test target data (optional; used for evaluation only).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: 1-D array of predicted values (NaN where no prediction).
		"""
		self._check_is_fitted()
		self._build_adapter(X_test, y_test, TestStart, TestEnd, testTime)

		Data         = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices  = self.GetTestIndices()
		Target       = self.GetYIndex()
		XStart, XEnd = self.GetXIndices()
		Columns      = list(range(XStart, XEnd + 1))
		NoTime       = not self.HasTime()

		self.MDE = MDE(
			data                    = Data,
			target                  = Target,
			maxD                    = self.MaxD,
			include_target          = self.IncludeTarget,
			convergent              = self.Convergent,
			metric                  = self.Metric,
			batch_size              = self.BatchSize,
			use_half_precision      = self.HalfPrecision,
			columns                 = Columns,
			train                   = TrainIndices,
			test                    = TestIndices,
			embedded                = not self.Embed,
			embedDimensions         = self.EmbedDimensions,
			predictionHorizon       = self.PredictionHorizon,
			knn                     = self.KNN,
			step                    = self.Step,
			exclusionRadius         = self.ExclusionRadius,
			noTime                  = NoTime,
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

		self.result_ = self.MDE.Run()
		return self.result_.predictions

	def get_params(self, deep: bool = True) -> dict:
		return {
			'MaxD':                    self.MaxD,
			'IncludeTarget':           self.IncludeTarget,
			'Convergent':              self.Convergent,
			'Metric':                  self.Metric,
			'BatchSize':               self.BatchSize,
			'HalfPrecision':           self.HalfPrecision,
			'Embed':                   self.Embed,
			'EmbedDimensions':         self.EmbedDimensions,
			'PredictionHorizon':       self.PredictionHorizon,
			'KNN':                     self.KNN,
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
