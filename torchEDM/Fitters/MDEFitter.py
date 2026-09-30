from typing import Optional, Union

import numpy
import torch

from ..EDM.MDE import MDE
from .EDMFitter import EDMFitter


class MDEFitter(EDMFitter):
	"""Hold the parameters of MDE, the greedy variable selection with an optional convergence gate."""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'pre',
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 dtype: torch.dtype = torch.float32,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 MinSelectedZScore: Optional[float] = None,
				 ExtraStepsBelowZScore: int = 0,
				 IterativeDimensionSearch: bool = False,
				 progressBar: bool = True,
				 device = None):
		"""
		:param MaxD:	number of variables to select per target; the target's own variable counts when IncludeTarget
		:param IncludeTarget:	True starts with the target series itself selected, so its own value is part of every state; it then appears in selected_variables as index nFeatures + targetIndex
		:param Convergent:	'pre' screens every candidate for cross-map convergence before selection, 'post' checks candidates in performance order at each step, False skips the check
		:param Metric:	'correlation' or 'r2': the performance metric that ranks candidates at each step
		:param BatchSize:	number of candidate variables whose distance matrices are held on the device at once
		:param dtype:	torch dtype of the selection tensors
		:param EmbedDimensions:	fixed embedding dimensions of the target's lagged history in the convergence check; 0 searches each candidate's own
		:param PredictionHorizon:	number of samples between a state and the target value it predicts
		:param KNN:	number of nearest neighbors in the convergence check and in a final SMapPredict, 0 meaning each one's default (the target's embedding dimensions + 1, every training state); the final SimplexPredict always uses the number of selected variables plus one
		:param Step:	sample offset between consecutive lagged copies in the convergence check and the embedding-dimension search; a negative offset reaches into the past
		:param ExclusionRadius:	training states within this many samples of a test state are excluded from its neighbors; always applied in the convergence check, which runs on the training samples, and to selection and prediction only in-sample
		:param Verbose:	True prints when a target stops expanding and other progress details
		:param UseSMap:	True makes the final prediction with SMapPredict instead of SimplexPredict
		:param Theta:	localization strength of that final SMapPredict; 0 fits one global linear map
		:param stdThreshold:	candidates whose standard deviation over the training samples is below this are dropped from the pool
		:param CCMLibraryPercentiles:	training-subset sizes of the convergence check, as percentages of the number of training states
		:param CCMNumSamples:	number of random subsets drawn per size in the convergence check
		:param CCMConvergenceThreshold:	minimum slope of cross-map performance against subset fraction for a candidate to count as convergent
		:param CCMSeed:	seed of the convergence check's subset draws; None draws fresh subsets
		:param CCMMaxEmbeddingDimensions:	largest embedding dimension tried in the per-candidate search
		:param MinPredictionThreshold:	minimum performance a candidate must reach at a step to be selectable
		:param MinCandidatePerformance:	minimum peak performance a candidate alone (at its best embedding dimension) must reach predicting the target to stay in the pool; 0 disables
		:param MinSelectedZScore:	stop rule on how far the selected candidate stands out from the other candidates: at each step the performance of every evaluated candidate is z-scored, and a target stops expanding once its selected candidate's z-score falls below this value; None disables the rule
		:param ExtraStepsBelowZScore:	number of further selection steps run after a selected candidate first falls below MinSelectedZScore, to confirm the drop; the target stops once this many further steps have also fallen below it, and a step back above it restarts the count. The candidates selected during these steps stay selected
		:param IterativeDimensionSearch:	True evaluates each embedding dimension of the per-candidate search on its own complete samples (slower, reproduces the reference); False shares the samples complete at the largest one in one pass
		:param progressBar:	True shows a progress bar over the selection steps
		:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
		"""
		super().__init__(progressBar)
		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.dtype = dtype
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.stdThreshold = stdThreshold
		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxEmbeddingDimensions = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.MinCandidatePerformance = MinCandidatePerformance
		self.MinSelectedZScore = MinSelectedZScore
		self.ExtraStepsBelowZScore = ExtraStepsBelowZScore
		self.IterativeDimensionSearch = IterativeDimensionSearch
		self.device = device

	def MDEKeywords(self) -> dict:
		"""
		Map these settings onto the keyword arguments of MDE.

		:return: dict ready to unpack into MDE(X_train, Y_train, X_test, Y_test, **keywords)
		"""
		return dict(maxVariables = self.MaxD, isTargetIncluded = self.IncludeTarget, convergenceCheck = self.Convergent,
					candidateMetric = self.Metric, batchSize = self.BatchSize, dtype = self.dtype,
					embedDimensions = self.EmbedDimensions, predictionHorizon = self.PredictionHorizon,
					knn = self.KNN, step = self.Step, exclusionRadius = self.ExclusionRadius, isVerbose = self.Verbose,
					isUsingSMap = self.UseSMap, theta = self.Theta, stdThreshold = self.stdThreshold,
					convergenceSubsetPercentiles = self.CCMLibraryPercentiles, convergenceRepeats = self.CCMNumSamples,
					convergenceSlopeThreshold = self.CCMConvergenceThreshold, convergenceSeed = self.CCMSeed,
					convergenceMaxEmbedDimensions = self.CCMMaxEmbeddingDimensions,
					minPredictionScore = self.MinPredictionThreshold,
					minCandidateScore = self.MinCandidatePerformance,
					minSelectedZScore = self.MinSelectedZScore, extraStepsBelowZScore = self.ExtraStepsBelowZScore,
					isIterativeDimensionSearch = self.IterativeDimensionSearch,
					hasProgressBar = not self.hideProgress,
					device = self.device)

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		"""
		Select variables with MDE, predict the test samples with them, and keep the MDEResult in Result.

		:param X_train:	candidate input data, [nTrain, nFeatures] or a list of runs
		:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
		:param X_test:	test input data, [nTest, nFeatures] or a list of runs; candidates are scored and the final prediction made on these samples. None scores the training samples in-sample
		:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test, and a NaN sample is predicted but not scored
		:return: MDEResult holding the selected variables, their performance, Y_pred, and the per-candidate diagnostics
		"""
		self.Result = MDE(X_train, Y_train, X_test, Y_test, **self.MDEKeywords())
		return self.Result
