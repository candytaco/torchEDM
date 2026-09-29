import torch

from ..EDM.Multiview import MultiviewPredict
from .EDMFitter import EDMFitter


class MultiviewFitter(EDMFitter):
	"""Hold the parameters of MultiviewPredict, the ensemble over top-ranked combinations of state dimensions."""

	def __init__(self, ColumnsPerView: int = 0, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 NumMultiview: int = 0, ExclusionRadius: int = 0, IsRankedInSample: bool = True,
				 IsTieBreakDeterministic: bool = False, device = None, dtype: torch.dtype = torch.float64):
		"""
		:param ColumnsPerView:	number of state dimensions in each combination that predicts on its own; 0 means the number of input variables
		:param EmbedDimensions:	number of lagged copies of each input variable; the combinations are drawn from the nFeatures * EmbedDimensions state dimensions
		:param PredictionHorizon:	number of samples between a state and the target value it predicts
		:param KNN:	number of nearest neighbors per combination; 0 means ColumnsPerView + 1
		:param Step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
		:param NumMultiview:	number of top-ranked combinations whose predictions are averaged; 0 means the square root of the number of combinations
		:param ExclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
		:param IsRankedInSample:	True ranks the combinations on the training samples predicting themselves (faster, optimistic); False ranks them on X_test and Y_test
		:param IsTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position so repeated runs pick the same neighbors; False leaves the order to torch.topk
		:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
		:param dtype:	torch dtype of the distances, weights and predictions
		"""
		super().__init__()
		self.ColumnsPerView = ColumnsPerView
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.NumMultiview = NumMultiview
		self.ExclusionRadius = ExclusionRadius
		self.IsRankedInSample = IsRankedInSample
		self.IsTieBreakDeterministic = IsTieBreakDeterministic
		self.device = device
		self.dtype = dtype

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		"""
		Predict with MultiviewPredict and keep the MultiviewResult in Result.

		:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
		:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
		:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
		:param Y_test:	true data for the test samples, same layout as Y_train; when given, the result carries a performance score per target
		:return: MultiviewResult holding the ensemble Y_pred and each top combination's own predictions and statistics
		"""
		self.Result = MultiviewPredict(X_train, Y_train, X_test, Y_test,
									   embedDimensions = self.EmbedDimensions, step = self.Step,
									   predictionHorizon = self.PredictionHorizon, knn = self.KNN,
									   exclusionRadius = self.ExclusionRadius, columnsPerView = self.ColumnsPerView,
									   numTopViews = self.NumMultiview, isRankedInSample = self.IsRankedInSample,
									   isTieBreakDeterministic = self.IsTieBreakDeterministic,
									   device = self.device, dtype = self.dtype)
		return self.Result
