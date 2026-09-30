from typing import List, Optional

import torch

from ..EDM.ConvergentCrossMap import ConvergentCrossMap
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""Hold the parameters of ConvergentCrossMap, the cross-map performance screen across training-subset sizes."""

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
		:param TrainSizes:	training-subset sizes at which the cross-map performance is measured; None uses 10, 25, 50, 75 and 90 percent of the training samples
		:param numRepeats:	number of random subsets drawn per size; the reported performance is their mean
		:param EmbedDimensions:	embedding dimensions per source variable: an int, [nSources], or [nSources, nTargets]; None searches each source's up to MaxEmbedDimensions
		:param MaxEmbedDimensions:	largest embedding dimension tried in that search
		:param PredictionHorizon:	number of samples between a source state and the target value it predicts
		:param KNN:	number of nearest neighbors per source; None means the source's embedding dimensions + 1
		:param Step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
		:param ExclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
		:param progressBar:	True shows progress bars over source batches, subset sizes and repeats
		:param device:	torch device; cuda falls back to cpu when unavailable
		:param sourceBatchSize:	number of source variables per batch in 'variables' mode, and of variables per batch in the embedding-dimension search
		:param targetBatchSize:	number of target variables per batch within a source batch
		:param targetVRAM:	memory budget in GB that sizes both batches when given; None uses the sizes as given
		:param dtype:	torch dtype of the distances and predictions
		:param batchMode:	'variables' batches over source variables and predicts the test samples; 'sample' batches over subsets and scores the training samples predicting themselves
		:param sampleBatchSize:	number of subsets per batch in 'sample' mode; None takes all at once
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

		:param X_train:	source data, [nTrain, nSources] or a list of runs; the lagged history of each source variable predicts the targets
		:param Y_train:	target data, [nTrain, nTargets] or a list of runs; None cross-maps every source variable onto every source variable
		:param X_test:	test source data, [nTest, nSources] or a list of runs; None scores the training samples in-sample
		:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test unless Y_train is None
		:return: BatchedCCMResult holding forward_performance [nSizes, nSources, nTargets], singleton axes squeezed
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
