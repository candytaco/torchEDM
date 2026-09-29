from typing import List, Optional

import torch

from ..EDM.ConvergentCrossMap import ConvergentCrossMap
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""Parameter holder for ConvergentCrossMap, the cross-map skill screen across training-subset sizes."""

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
		:param TrainSizes:	training-subset sizes at which the cross-map skill is measured; None uses 10, 25, 50, 75 and 90 percent of the training rows
		:param numRepeats:	random subsets drawn per size; the reported skill is their mean
		:param EmbedDimensions:	embedding dimensions per source column: an int, [nSources], or [nSources, nTargets]; None searches each source's up to MaxEmbedDimensions
		:param MaxEmbedDimensions:	largest embedding dimension tried in that search
		:param PredictionHorizon:	rows between a source state and the target value it predicts
		:param KNN:	neighbors per source; None means the source's embedding dimensions + 1
		:param Step:	row offset between the stacked copies; negative reaches into the past
		:param ExclusionRadius:	in-sample only; training states within this many rows of a test state may not be its neighbors (the state itself never is)
		:param progressBar:	show progress bars over source batches, subset sizes and repeats
		:param device:	torch device; cuda falls back to cpu when unavailable
		:param sourceBatchSize:	source columns per batch in 'variables' mode, and columns per batch in the embedding-dimension search
		:param targetBatchSize:	target columns per batch within a source batch
		:param targetVRAM:	GB budget that sizes both batches when given; None uses the sizes as given
		:param dtype:	torch dtype of the distances and predictions
		:param batchMode:	'variables' batches over source columns and predicts the test rows; 'sample' batches over subsets and scores the training rows predicting themselves
		:param sampleBatchSize:	subsets per batch in 'sample' mode; None takes all at once
		:param seed:	seed of the random subset draws; None draws fresh subsets every call
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
		Run ConvergentCrossMap and keep the BatchedCCMResult in Result.

		:param X_train:	[nTrain, nSources] or a list of runs: the columns whose stacked histories predict the targets
		:param Y_train:	[nTrain, nTargets] or a list of runs; None cross-maps every X column onto every X column
		:param X_test:	[nTest, nSources] or a list of runs; None scores the training rows in-sample
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; required with X_test unless Y_train is None
		:return: BatchedCCMResult with forward_performance [nSizes, nSources, nTargets], singleton axes squeezed
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
