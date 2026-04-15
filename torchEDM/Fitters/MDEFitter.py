
from typing import Optional, List, Union

import numpy

from torchEDM.EDM.MDE import MDE
from .EDMFitter import EDMFitter

class MDEFitter(EDMFitter):
	"""
	Wrapper class for MDE that provides sklearn-like API.
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
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0,
				 progressBar: bool = True):
		"""
		Initialize MDE wrapper with sklearn-style separate arrays.

		:param MaxD: 				Maximum number of features to select
		:param IncludeTarget: 		Whether to start with target in feature list
		:param Convergent: 			Whether to use convergence checking
		:param Metric: 				Metric to use: "correlation" or "MAE"
		:param BatchSize: 			Number of features to process in each batch
		:param HalfPrecision: 		Use float16 instead of float32 for GPU tensors
		:param Embed:				whether to embed the data or not
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Verbose: 			Print diagnostic messages
		:param UseSMap: 			Whether to use SMap instead of Simplex
		:param Theta: 				S-Map localization parameter
		"""

		super().__init__(progressBar)

		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.HalfPrecision = HalfPrecision
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.Embed = Embed
		self.stdThreshold = stdThreshold

		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.MinPredictionThreshold = MinPredictionThreshold
		self.EmbedDimCorrelationMin = EmbedDimCorrelationMin
		self.FirstEMax = FirstEMax
		self.TimeDelay = TimeDelay

		self.MDE = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray, YTest: numpy.ndarray,
			TrainStart = 1, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		XStart, XEnd = self.GetXIndices()
		Columns = list(range(XStart, XEnd + 1))
		Target = self.GetYIndex()
		NoTime = not self.HasTime()

		# Determine columns to use

		self.MDE = MDE(
			data = Data,
			target = Target,
			maxD = self.MaxD,
			include_target = self.IncludeTarget,
			convergent = self.Convergent,
			metric = self.Metric,
			batch_size = self.BatchSize,
			use_half_precision = self.HalfPrecision,
			columns = Columns,
			train = TrainIndices,
			test = TestIndices,
			embedded = not self.Embed,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose,
			useSMap = self.UseSMap,
			theta = self.Theta,
			stdThreshold = self.stdThreshold,
			CCMLibraryPercentiles = self.CCMLibraryPercentiles,
			CCMNumSamples = self.CCMNumSamples,
			CCMConvergenceThreshold = self.CCMConvergenceThreshold,
			MinPredictionThreshold = self.MinPredictionThreshold,
			EmbedDimCorrelationMin = self.EmbedDimCorrelationMin,
			FirstEMax = self.FirstEMax,
			TimeDelay = self.TimeDelay
		)

		self.Result = self.MDE.Run()
		return self.Result
