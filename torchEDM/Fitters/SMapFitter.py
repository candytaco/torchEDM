import torch

from ..EDM.Predictors import SMapPredict
from .EDMFitter import EDMFitter


class SMapFitter(EDMFitter):
	"""Parameter holder for SMapPredict."""

	def __init__(self, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 Theta: float = 0.0, ExclusionRadius: int = 0, IsTieBreakDeterministic: bool = False,
				 device = None, dtype: torch.dtype = torch.float64):
		"""
		:param KNN:		neighbors; 0 means every available training state
		:param Theta:	localization; 0 fits one global linear map
		Other parameters as in SimplexFitter.
		"""
		super().__init__()
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.Theta = Theta
		self.ExclusionRadius = ExclusionRadius
		self.IsTieBreakDeterministic = IsTieBreakDeterministic
		self.device = device
		self.dtype = dtype

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		self.Result = SMapPredict(X_train, Y_train, X_test, Y_test,
								  embedDimensions = self.EmbedDimensions, step = self.Step,
								  predictionHorizon = self.PredictionHorizon, knn = self.KNN, theta = self.Theta,
								  exclusionRadius = self.ExclusionRadius,
								  isTieBreakDeterministic = self.IsTieBreakDeterministic,
								  device = self.device, dtype = self.dtype)
		return self.Result
