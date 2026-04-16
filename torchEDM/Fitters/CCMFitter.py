from typing import Optional, List

import numpy

from torchEDM.EDM.CCM_batch import BatchedCCM
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""
	Wrapper class for CCM that provides sklearn-like API.
	"""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 progressBar: bool = True,
				 directions: str = 'both',
				 device: str = 'cuda',
				 batchSize: int = 10000,
				 useHalfPrecision: bool = False,
				 batchMode: str = 'variable',
				 sampleBatchSize: Optional[int] = None,
				 seed: Optional[int] = None
				 ):
		"""
		Init.

		:param TrainSizes: 			train sizes to explore, when none, it defaults to the 10th, 25th, 50th, 75th, and 90th percentiles of the data size
		:param numRepeats: 			Number of repeats at each training size
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param directions: 			Which directions to compute: forward|reverse|both
		:param device: 				Device for torch tensors ('cpu', 'cuda', or torch.device object)
		:param batchSize: 			Number of variables to process per batch in 'variable' mode
		:param useHalfPrecision: 	Use float16 instead of float32 to save VRAM
		:param batchMode: 			'variable' (batch over source variables) or 'sample' (batch over subsamples per library size)
		:param sampleBatchSize: 	Number of subsamples to process per batch in 'sample' mode
		:param seed: 				Random seed for reproducible sampling
		"""

		super().__init__(progressBar)

		self.TrainSizes = TrainSizes
		self.Sample = numRepeats
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.directions = directions
		self.device = device
		self.batchSize = batchSize
		self.useHalfPrecision = useHalfPrecision
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize
		self.seed = seed

		self.CCM = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: Optional[numpy.ndarray] = None, XTest: numpy.ndarray = None, YTest: numpy.ndarray = None,
			TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		TrainIndices = self.GetTrainIndices()

		self.CCM = BatchedCCM(
			X = self.DataAdapter.XTrain,
			Y = self.DataAdapter.YTrain,
			trainSizes = self.TrainSizes,
			sample = self.Sample,
			forwardEmbedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			seed = self.seed,
			directions = self.directions,
			trainIndices = list(TrainIndices),
			device = self.device,
			batchSize = self.batchSize,
			useHalfPrecision = self.useHalfPrecision,
			showProgress = not self.hideProgress,
			batchMode = self.batchMode,
			sampleBatchSize = self.sampleBatchSize
		)

		self.Result = self.CCM.Run()
		return self.Result
