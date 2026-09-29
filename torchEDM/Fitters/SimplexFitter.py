import torch

from ..EDM.Predictors import SimplexPredict
from .EDMFitter import EDMFitter


class SimplexFitter(EDMFitter):
	"""Parameter holder for SimplexPredict, the nearest-neighbor weighted-average predictor."""

	def __init__(self, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 ExclusionRadius: int = 0, IsTieBreakDeterministic: bool = False, device = None,
				 dtype: torch.dtype = torch.float64):
		"""
		:param EmbedDimensions:	copies of each feature column in the state, each shifted Step rows from the last; 1 uses the columns as given
		:param PredictionHorizon:	rows between a state and the target value it predicts
		:param KNN:	neighbors that vote on each prediction; 0 means the state size plus one
		:param Step:	row offset between the stacked copies; negative reaches into the past
		:param ExclusionRadius:	in-sample only; training states within this many rows of a test state may not be its neighbors (the state itself never is)
		:param IsTieBreakDeterministic:	order exactly tied neighbor distances by row so repeated runs pick the same neighbors; False leaves the order to torch.topk
		:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
		:param dtype:	torch dtype of the distances, weights and predictions
		"""
		super().__init__()
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.IsTieBreakDeterministic = IsTieBreakDeterministic
		self.device = device
		self.dtype = dtype

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		"""
		Predict with SimplexPredict and keep the SimplexResult in Result.

		:param X_train:	[nTrain, nFeatures] or a list of runs: the columns that form the states
		:param Y_train:	[nTrain, nTargets] (1-D for one target) or a list of runs aligned row by row with X_train
		:param X_test:	[nTest, nFeatures] or a list of runs whose rows are predicted; None predicts the training rows in-sample
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; when given, the result carries a score per target
		:return: SimplexResult with Y_pred shaped like Y_test (or Y_train in-sample)
		"""
		self.Result = SimplexPredict(X_train, Y_train, X_test, Y_test,
									 embedDimensions = self.EmbedDimensions, step = self.Step,
									 predictionHorizon = self.PredictionHorizon, knn = self.KNN,
									 exclusionRadius = self.ExclusionRadius,
									 isTieBreakDeterministic = self.IsTieBreakDeterministic,
									 device = self.device, dtype = self.dtype)
		return self.Result
