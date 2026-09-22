from typing import List, Optional

import torch

from ..EDM.ConvergentCrossMap import ConvergentCrossMap
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""Parameter holder for ConvergentCrossMap."""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 10,
				 EmbedDimensions = None,
				 MaxEmbedDimensions: int = 20,
				 PredictionHorizon: int = 1,
				 KNN: Optional[int] = None,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 progressBar: bool = True,
				 device: str = 'cuda',
				 sourceBatchSize: int = 1000,
				 targetBatchSize: int = 2000,
				 targetVRAM: Optional[float] = None,
				 dtype: torch.dtype = torch.float16,
				 batchMode: str = 'variables',
				 sampleBatchSize: Optional[int] = None,
				 seed: Optional[int] = None):
		"""
		:param TrainSizes:	training-subset sizes; None uses 10, 25, 50, 75 and 90 percent of the training rows
		:param numRepeats:	random subsets per size
		:param EmbedDimensions:	embedding dimensions per source; None searches up to MaxEmbedDimensions
		:param KNN:			neighbors; None means embedding dimensions + 1
		Other parameters as in ConvergentCrossMap.
		"""
		super().__init__(progressBar)
		self.TrainSizes = TrainSizes
		self.Repeats = numRepeats
		self.EmbedDimensions = EmbedDimensions
		self.MaxEmbedDimensions = MaxEmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.device = device
		self.sourceBatchSize = sourceBatchSize
		self.targetBatchSize = targetBatchSize
		self.targetVRAM = targetVRAM
		self.dtype = dtype
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize
		self.seed = seed

	def Fit(self, X_train, Y_train = None, X_test = None, Y_test = None):
		"""
		:param X_train:	source columns; :param Y_train: targets, None cross-maps X onto itself
		:param X_test:	None scores the training rows in-sample
		"""
		self.Result = ConvergentCrossMap(
			X_train, Y_train, X_test, Y_test,
			trainSizes = self.TrainSizes,
			repeats = self.Repeats,
			embedDimensions = self.EmbedDimensions,
			maxEmbedDimensions = self.MaxEmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			# 0 means the default (embedding dimensions + 1), which ConvergentCrossMap spells None
			knn = self.KNN if self.KNN else None,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			seed = self.seed,
			device = self.device,
			sourceBatchSize = self.sourceBatchSize,
			targetBatchSize = self.targetBatchSize,
			targetVRAM = self.targetVRAM,
			dtype = self.dtype,
			hasProgressBar = not self.hideProgress,
			batchMode = self.batchMode,
			sampleBatchSize = self.sampleBatchSize)
		return self.Result
