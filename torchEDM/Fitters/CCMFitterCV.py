from typing import List, Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .CVSplitter import RunSplitter, SliceRuns
from .EDMFitter import EDMFitter
from ..EDM.ConvergentCrossMap import ConvergentCrossMap
from ..EDM.Results import CCMCVResult
from ..EDM.Setup import AsRuns


class CCMFitterCV(EDMFitter):
	"""
	ConvergentCrossMap across leave-one-run-out or n-fold splits of the training runs: does
	the convergence signal reproduce across temporal subsets of the data?
	"""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 10,
				 EmbedDimensions = None,
				 MaxEmbedDimensions: int = 20,
				 PredictionHorizon: int = 1,
				 KNN: Optional[int] = None,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 device: str = 'cuda',
				 sourceBatchSize: int = 1000,
				 targetBatchSize: Optional[int] = None,
				 targetVRAM: Optional[float] = None,
				 dtype: torch.dtype = torch.float32,
				 batchMode: str = 'variables',
				 sampleBatchSize: Optional[int] = None,
				 seed: Optional[int] = None,
				 Folds: int = 5,
				 LeaveOneRunOut: bool = True,
				 progressBar: bool = True):
		"""
		:param TrainSizes:	training-subset sizes at which the cross-map skill is measured; None uses 10, 25, 50, 75 and 90 percent of the training rows
		:param numRepeats:	random subsets drawn per size; the reported skill is their mean
		:param EmbedDimensions:	embedding dimensions per source column: an int, [nSources], or [nSources, nTargets]; None searches each source's up to MaxEmbedDimensions
		:param MaxEmbedDimensions:	largest embedding dimension tried in that search
		:param PredictionHorizon:	rows between a source state and the target value it predicts
		:param KNN:	neighbors per source; None means the source's embedding dimensions + 1
		:param Step:	row offset between the stacked copies; negative reaches into the past
		:param ExclusionRadius:	in-sample only; training states within this many rows of a test state may not be its neighbors (the state itself never is)
		:param device:	torch device; cuda falls back to cpu when unavailable
		:param sourceBatchSize:	source columns per batch in 'variables' mode, and columns per batch in the embedding-dimension search
		:param targetBatchSize:	target columns per batch within a source batch
		:param targetVRAM:	GB budget that sizes both batches when given; None uses the sizes as given
		:param dtype:	torch dtype of the distances and predictions
		:param batchMode:	'variables' batches over source columns and predicts the test rows; 'sample' batches over subsets and scores the training rows predicting themselves
		:param sampleBatchSize:	subsets per batch in 'sample' mode; None takes all at once
		:param seed:	seed of the random subset draws; None draws fresh subsets every call
		:param Folds:	contiguous blocks each run is cut into when LeaveOneRunOut is False; fold k holds out block k of every run
		:param LeaveOneRunOut:	True holds out one whole run per split; False uses the n-fold blocks
		:param progressBar:	show a progress bar over the folds
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
		self.targetBatchSize = targetBatchSize if targetBatchSize is not None else 2000
		self.targetVRAM = targetVRAM
		self.dtype = dtype
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize
		self.seed = seed
		self.Folds = Folds
		self.LeaveOneRunOut = LeaveOneRunOut
		self.splitter = None
		self.foldResults = []

	def Fit(self, X_train, Y_train = None, X_test = None, Y_test = None) -> CCMCVResult:
		"""
		Run ConvergentCrossMap once per split, each split's training slices predicting its held-out
		slices, and keep the CCMCVResult in Result.

		:param X_train:	[nTrain, nSources] or a list of runs: the columns whose stacked histories predict the targets
		:param Y_train:	[nTrain, nTargets] or a list of runs; None cross-maps every X column onto every X column
		:param X_test:	unused; each fold predicts its own held-out slices
		:param Y_test:	unused
		:return: CCMCVResult with every fold's BatchedCCMResult and the mean and standard deviation of the skill over folds
		"""
		xRuns = AsRuns(X_train)
		yRuns = None if Y_train is None else AsRuns(Y_train)
		self.splitter = RunSplitter([run.shape[0] for run in xRuns], self.Folds, self.LeaveOneRunOut)

		self.foldResults = []
		progressBar = ProgressBar(total = self.splitter.GetNSplits(), desc = 'CCM CV Fold', leave = False, disable = self.hideProgress)
		for trainSlices, testSlices in self.splitter.Split():
			foldResult = ConvergentCrossMap(
				SliceRuns(xRuns, trainSlices),
				None if yRuns is None else SliceRuns(yRuns, trainSlices),
				SliceRuns(xRuns, testSlices),
				None if yRuns is None else SliceRuns(yRuns, testSlices),
				trainSizes = self.TrainSizes,
				repeats = self.Repeats,
				embedDimensions = self.EmbedDimensions,
				maxEmbedDimensions = self.MaxEmbedDimensions,
				predictionHorizon = self.PredictionHorizon,
				knn = self.KNN,
				step = self.Step,
				exclusionRadius = self.ExclusionRadius,
				seed = self.seed,
				device = self.device,
				sourceBatchSize = self.sourceBatchSize,
				targetBatchSize = self.targetBatchSize,
				targetVRAM = self.targetVRAM,
				dtype = self.dtype,
				hasProgressBar = False,
				batchMode = self.batchMode,
				sampleBatchSize = self.sampleBatchSize)
			self.foldResults.append(foldResult)
			progressBar.update(1)
		progressBar.close()

		foldForwardCorrelations = numpy.stack([r.forward_performance for r in self.foldResults], axis = 0)
		self.Result = CCMCVResult(
			fold_results = self.foldResults,
			fold_performances = foldForwardCorrelations,
			mean_performance = numpy.mean(foldForwardCorrelations, axis = 0),
			std_performance = numpy.std(foldForwardCorrelations, axis = 0),
			predictionHorizon = self.PredictionHorizon,
			fold_forward_embed_dimensions = [r.forward_embed_dimensions for r in self.foldResults])
		return self.Result
