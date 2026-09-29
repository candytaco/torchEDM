from typing import Union

import numpy
import torch

from ..EDM.MDE import MDE
from .EDMFitter import EDMFitter


class MDEFitter(EDMFitter):
	"""Parameter holder for MDE, the greedy variable selection with an optional convergence gate."""

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
				 IterativeDimensionSearch: bool = False,
				 progressBar: bool = True,
				 device = None):
		"""
		:param MaxD:	columns to select per target; the target's own column counts when IncludeTarget
		:param IncludeTarget:	start with the target series itself selected, so its own value is part of every state; it then appears in selected_variables as column nFeatures + targetIndex
		:param Convergent:	'pre' screens every candidate for cross-map convergence before selection, 'post' checks candidates in score order at each step, False skips the check
		:param Metric:	'correlation' or 'r2': the score that ranks candidates at each step
		:param BatchSize:	candidate columns whose distance matrices are held on the device at once
		:param dtype:	torch dtype of the selection tensors
		:param EmbedDimensions:	fixed embedding dimensions of the target's history in the convergence check; 0 searches each candidate's own
		:param PredictionHorizon:	rows between a state and the target value it predicts
		:param KNN:	neighbors in the convergence check and in a final SMapPredict, 0 meaning each one's default (the target's embedding dimensions + 1, every training state); the final SimplexPredict always uses the number of selected columns plus one
		:param Step:	row offset between the stacked copies in the convergence check and the embedding-dimension search; negative reaches into the past
		:param ExclusionRadius:	training states within this many rows of a test state may not be its neighbors; always applied in the convergence check, which runs on the training rows, and to selection and prediction only in-sample
		:param Verbose:	print when a target stops expanding and other progress details
		:param UseSMap:	final prediction with SMapPredict instead of SimplexPredict
		:param Theta:	localization of that final SMapPredict; 0 fits one global linear map
		:param stdThreshold:	candidates whose standard deviation over the training rows is below this are dropped from the pool
		:param CCMLibraryPercentiles:	training-subset sizes of the convergence check, as percentages of the training states
		:param CCMNumSamples:	random subsets drawn per size in the convergence check
		:param CCMConvergenceThreshold:	minimum slope of cross-map skill against subset fraction for a candidate to count as convergent
		:param CCMSeed:	seed of the convergence check's subset draws; None draws fresh subsets
		:param CCMMaxEmbeddingDimensions:	largest embedding dimension tried in the per-candidate search
		:param MinPredictionThreshold:	minimum score a candidate must reach at a step to be selectable
		:param MinCandidatePerformance:	minimum peak score a candidate alone (at its best embedding dimension) must reach predicting the target to stay in the pool; 0 disables
		:param IterativeDimensionSearch:	True evaluates each embedding dimension of the per-candidate search on its own complete rows (slower, reproduces the reference); False shares the rows complete at the largest one in one pass
		:param progressBar:	show a progress bar over the selection steps
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
		self.IterativeDimensionSearch = IterativeDimensionSearch
		self.device = device

	def MDEKeywords(self) -> dict:
		"""
		The keyword arguments of MDE that these settings map onto.

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
					isIterativeDimensionSearch = self.IterativeDimensionSearch,
					hasProgressBar = not self.hideProgress,
					device = self.device)

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		"""
		Select variables with MDE, predict the test rows with them, and keep the MDEResult in Result.

		:param X_train:	[nTrain, nFeatures] or a list of runs: the candidate columns
		:param Y_train:	[nTrain, nTargets] (1-D for one target) or a list of runs aligned row by row with X_train
		:param X_test:	[nTest, nFeatures] or a list of runs on which candidates are scored and the final prediction made; None scores the training rows in-sample
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; required with X_test, and a NaN row is predicted but never scored
		:return: MDEResult with the selected columns, their scores, Y_pred, and the per-candidate diagnostics
		"""
		self.Result = MDE(X_train, Y_train, X_test, Y_test, **self.MDEKeywords())
		return self.Result
