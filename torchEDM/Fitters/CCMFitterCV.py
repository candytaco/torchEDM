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
		:param Folds:		folds per run when LeaveOneRunOut is False
		:param LeaveOneRunOut:	hold out one whole run per split
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
		:param X_train:	source columns, an array or a list of runs
		:param Y_train:	targets matching X_train; None cross-maps X onto itself
		X_test and Y_test are unused; each fold predicts its own held-out slices.
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
