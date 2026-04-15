from typing import Optional

import numpy

from torchEDM.EDM.SMap import SMap
from .EDMFitter import EDMFitter

class SMapFitter(EDMFitter):
	"""
	Wrapper class for SMap that provides sklearn-like API.
	"""

	def __init__(self,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 Theta: float = 0.0,
				 ExclusionRadius: int = 0,
				 Embedded: bool = False,
				 Verbose: bool = False):
		"""
		Initialize SMap wrapper with sklearn-style separate arrays.

		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param Theta: 				S-Map localization parameter
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Embedded: 			Whether data is already embedded
		:param Verbose: 			Print diagnostic messages
		"""

		super().__init__()

		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.Theta = Theta
		self.ExclusionRadius = ExclusionRadius
		self.Embedded = Embedded
		self.Verbose = Verbose

		self.SMap = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray, YTest: numpy.ndarray,
			TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		YIndex = self.GetYIndex()
		NoTime = not self.HasTime()

		XStart, XEnd = self.GetXIndices()
		Columns = list(range(XStart, XEnd + 1))
		Target = YIndex

		self.SMap = SMap(
			data = Data,
			columns = Columns,
			target = Target,
			train = TrainIndices,
			test = TestIndices,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			theta = self.Theta,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose,
			embedded = self.Embedded
		)

		self.Result = self.SMap.Run()
		return self.Result
