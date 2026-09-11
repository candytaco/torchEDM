import torch

from ..EDM.Multiview import MultiviewPredict
from .EDMFitter import EDMFitter


class MultiviewFitter(EDMFitter):
	"""Parameter holder for MultiviewPredict."""

	def __init__(self, ColumnsPerView: int = 0, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 NumMultiview: int = 0, ExclusionRadius: int = 0, IsRankedInSample: bool = True,
				 IsTieBreakDeterministic: bool = False, device = None, dtype: torch.dtype = torch.float64):
		"""
		:param ColumnsPerView:	stacked columns per combination; 0 means the number of feature columns
		:param NumMultiview:	combinations averaged; 0 means the square root of the number of combinations
		:param IsRankedInSample:	rank combinations on the training rows predicting themselves
		Other parameters as in SimplexFitter.
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
		self.Result = MultiviewPredict(X_train, Y_train, X_test, Y_test,
									   embedDimensions = self.EmbedDimensions, step = self.Step,
									   predictionHorizon = self.PredictionHorizon, knn = self.KNN,
									   exclusionRadius = self.ExclusionRadius, columnsPerView = self.ColumnsPerView,
									   numTopViews = self.NumMultiview, isRankedInSample = self.IsRankedInSample,
									   isTieBreakDeterministic = self.IsTieBreakDeterministic,
									   device = self.device, dtype = self.dtype)
		return self.Result
