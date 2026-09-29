import torch

from ..EDM.Multiview import MultiviewPredict
from .EDMFitter import EDMFitter


class MultiviewFitter(EDMFitter):
	"""Parameter holder for MultiviewPredict, the ensemble over top-ranked column combinations."""

	def __init__(self, ColumnsPerView: int = 0, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 NumMultiview: int = 0, ExclusionRadius: int = 0, IsRankedInSample: bool = True,
				 IsTieBreakDeterministic: bool = False, device = None, dtype: torch.dtype = torch.float64):
		"""
		:param ColumnsPerView:	stacked columns in each combination that predicts on its own; 0 means the number of feature columns
		:param EmbedDimensions:	copies of each feature column, each shifted Step rows from the last; the combinations are drawn from the nFeatures * EmbedDimensions stacked columns
		:param PredictionHorizon:	rows between a state and the target value it predicts
		:param KNN:	neighbors per combination; 0 means ColumnsPerView + 1
		:param Step:	row offset between the stacked copies; negative reaches into the past
		:param NumMultiview:	top-ranked combinations whose predictions are averaged; 0 means the square root of the number of combinations
		:param ExclusionRadius:	in-sample only; training states within this many rows of a test state may not be its neighbors (the state itself never is)
		:param IsRankedInSample:	True ranks the combinations on the training rows predicting themselves (faster, optimistic); False ranks them on X_test and Y_test
		:param IsTieBreakDeterministic:	order exactly tied neighbor distances by row so repeated runs pick the same neighbors; False leaves the order to torch.topk
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

		:param X_train:	[nTrain, nFeatures] or a list of runs: the columns that form the states
		:param Y_train:	[nTrain, nTargets] (1-D for one target) or a list of runs aligned row by row with X_train
		:param X_test:	[nTest, nFeatures] or a list of runs whose rows are predicted; None predicts the training rows in-sample
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; when given, the result carries a score per target
		:return: MultiviewResult with the ensemble Y_pred and each top combination's own predictions and statistics
		"""
		self.Result = MultiviewPredict(X_train, Y_train, X_test, Y_test,
									   embedDimensions = self.EmbedDimensions, step = self.Step,
									   predictionHorizon = self.PredictionHorizon, knn = self.KNN,
									   exclusionRadius = self.ExclusionRadius, columnsPerView = self.ColumnsPerView,
									   numTopViews = self.NumMultiview, isRankedInSample = self.IsRankedInSample,
									   isTieBreakDeterministic = self.IsTieBreakDeterministic,
									   device = self.device, dtype = self.dtype)
		return self.Result
