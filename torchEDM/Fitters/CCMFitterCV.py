from typing import Optional, List, Union

import numpy
from tqdm import tqdm as ProgressBar

from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter
from .CVSplitter import EDMCVSplitter
from torchEDM.EDM.ConvergentCrossMap import ConvergentCrossMap
from torchEDM.EDM.Results import CCMCVResult


class CCMFitterCV(EDMFitter):
	"""
	CCM with cross-validation that supports both n-fold and leave-one-run-out CV.

	Each fold runs CCM using the user-specified library size convergence curve, restricted
	to only the training data for that fold. This tests whether the CCM convergence signal
	is reproducible across different temporal subsets of the data.
	"""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 10,
				 EmbedDimensions: int = None,
				 PredictionHorizon: int = 1,
				 KNN: Optional[int] = None,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 directions: str = 'both',
				 device: str = 'cuda',
				 batchSize: int = 10000,
				 y_batch: Optional[int] = None,
				 HalfPrecision: bool = False,
				 batchMode: str = 'variable',
				 sampleBatchSize: Optional[int] = None,
				 seed: Optional[int] = None,
				 Folds: int = 5,
				 LeaveOneRunOut: bool = True,
				 progressBar: bool = True):
		"""
		Initialize CCM cross-validation fitter.

		:param TrainSizes: 			Library sizes to evaluate for the convergence curve
		:param numRepeats: 			Number of random subsamples at each library size
		:param EmbedDimensions: 	Embedding dimension (E). None for auto-selection per fold.
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors, if none will be set to embed dims + 1
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param directions: 			Which directions to compute: forward|reverse|both
		:param device: 				Device for torch tensors ('cpu', 'cuda', or torch.device object)
		:param batchSize: 			Number of distance matrices to process per batch in 'variable' mode
		:param y_batch:				number of target variable to predict in batches, independent of batchSize, which is X
		:param HalfPrecision: 		Use float16 instead of float32 to save VRAM
		:param batchMode: 			'variable' (batch over source variables) or 'sample' (batch over subsamples)
		:param sampleBatchSize: 	Number of subsamples per batch in 'sample' mode
		:param seed: 				Random seed for reproducible sampling
		:param Folds: 				Number of cross-validation folds (ignored if LeaveOneRunOut is True)
		:param LeaveOneRunOut: 		If True, use leave-one-run-out CV instead of n-fold
		"""
		super().__init__(progressBar)

		self.TrainSizes = TrainSizes
		self.Repeats = numRepeats
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.directions = directions
		self.device = device
		self.batchSize = batchSize
		self.y_batch = y_batch
		self.useHalfPrecision = HalfPrecision
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize
		self.seed = seed
		self.Folds = Folds
		self.LeaveOneRunOut = LeaveOneRunOut

		self.trainDataAdapter = None
		self.cvSplitter = None
		self.foldResults = []

	def Fit(self,
			XTrain: Union[numpy.ndarray, List[numpy.ndarray]],
			YTrain: Union[numpy.ndarray, List[numpy.ndarray], None] = None,
			XTest: Optional[numpy.ndarray] = None,
			YTest: Optional[numpy.ndarray] = None,
			TrainStart: int = 0,
			TrainEnd: int = 0,
			TestStart: int = 0,
			TestEnd: int = 0,
			TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		"""
		Fit CCM using cross-validation.

		:param XTrain:		Source variables (single array or list of arrays for multiple runs)
		:param YTrain:		Target variable (single array or list of arrays for multiple runs)
		:param XTest:		Unused; included for API compatibility with EDMFitter
		:param YTest:		Unused; included for API compatibility with EDMFitter
		:param TrainStart:	Samples to exclude at the start of each run
		:param TrainEnd:	Samples to exclude at the end of each run
		:param TestStart:	Unused; included for API compatibility with EDMFitter
		:param TestEnd:		Unused; included for API compatibility with EDMFitter
		:param TrainTime:	Time labels for train data
		:param TestTime:	Unused; included for API compatibility with EDMFitter
		"""
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		if YTrain is None:
			self.directions = 'forward'

		self.trainDataAdapter = DataAdapter.MakeDataAdapter(
			XTrain, YTrain, None, None, TrainStart, TrainEnd, 0, 0, TrainTime, None
		)

		self.cvSplitter = EDMCVSplitter(
			dataAdapter = self.trainDataAdapter,
			nFolds = self.Folds,
			leaveOneRunOut = self.LeaveOneRunOut,
			edmStyleIndices = True
		)

		fullData = self.trainDataAdapter.fullData
		xStart, xEnd = self.trainDataAdapter.XIndices
		yColumnIndices = self.trainDataAdapter.YIndex

		XArray = fullData[:, xStart:xEnd + 1]
		YArray = fullData[:, yColumnIndices] if yColumnIndices else None

		self.foldResults = []
		numSplits = self.cvSplitter.GetNSplits()
		progressBarIterator = ProgressBar(total = numSplits, desc = 'CCM CV Fold', leave = False, disable = self.hideProgress)

		for foldTrainIndices, foldTestIndices in self.cvSplitter.Split():
			ccm = ConvergentCrossMap(X = XArray,
									 Y = YArray,
									 trainSizes = self.TrainSizes,
									 repeats = self.Repeats,
									 embedDimensions = self.EmbedDimensions,
									 predictionHorizon = self.PredictionHorizon,
									 knn = self.KNN,
									 step = self.Step,
									 exclusionRadius = self.ExclusionRadius,
									 seed = self.seed,
									 trainIndices = foldTrainIndices,
									 testIndices = foldTestIndices,
									 device = self.device,
									 batchSize = self.batchSize,
									 y_batch = self.y_batch,
									 HalfPrecision = self.useHalfPrecision,
									 showProgress = False,
									 batchMode = self.batchMode,
									 sampleBatchSize = self.sampleBatchSize)

			foldResult = ccm.Run()
			self.foldResults.append(foldResult)
			progressBarIterator.update(1)

		progressBarIterator.close()

		foldForwardCorrelations = None
		foldForwardEmbedDimensions = None

		foldForwardCorrelations = numpy.stack([r.forward_performance for r in self.foldResults], axis = 0)
		foldForwardEmbedDimensions = [r.forward_embed_dimensions for r in self.foldResults]

		meanForwardCorrelation = numpy.mean(foldForwardCorrelations, axis = 0) if foldForwardCorrelations is not None else None
		stdForwardCorrelation = numpy.std(foldForwardCorrelations, axis = 0) if foldForwardCorrelations is not None else None

		self.Result = CCMCVResult(
			fold_results = self.foldResults,
			fold_performances = foldForwardCorrelations,
			mean_performance = meanForwardCorrelation,
			std_performance = stdForwardCorrelation,
			predictionHorizon = self.PredictionHorizon,
			fold_forward_embed_dimensions = foldForwardEmbedDimensions,
		)
		return self.Result
