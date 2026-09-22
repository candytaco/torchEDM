import torch

from ..EDM.Predictors import SimplexPredict
from .EDMFitter import EDMFitter


class SimplexFitter(EDMFitter):
	"""Parameter holder for SimplexPredict."""

	def __init__(self, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 ExclusionRadius: int = 0, IsTieBreakDeterministic: bool = False, device = None,
				 dtype: torch.dtype = torch.float64):
		"""
		:param EmbedDimensions:	copies of each feature column in the state; 1 uses the columns as given
		:param PredictionHorizon:	rows between a state and the target it predicts
		:param KNN:			neighbors; 0 means state size plus one
		:param Step:		row offset between copies; negative reaches into the past
		:param ExclusionRadius:	in-sample only; training states this close in rows are not neighbors
		:param IsTieBreakDeterministic:	order exactly tied neighbor distances reproducibly
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
		self.Result = SimplexPredict(X_train, Y_train, X_test, Y_test,
									 embedDimensions = self.EmbedDimensions, step = self.Step,
									 predictionHorizon = self.PredictionHorizon, knn = self.KNN,
									 exclusionRadius = self.ExclusionRadius,
									 isTieBreakDeterministic = self.IsTieBreakDeterministic,
									 device = self.device, dtype = self.dtype)
		return self.Result
