import torch

from ..EDM.Predictors import SMapPredict
from .EDMFitter import EDMFitter


class SMapFitter(EDMFitter):
	"""Hold the parameters of SMapPredict, the locally weighted linear predictor."""

	def __init__(self, EmbedDimensions: int = 1, PredictionHorizon: int = 1, KNN: int = 0, Step: int = -1,
				 Theta: float = 0.0, ExclusionRadius: int = 0, IsTieBreakDeterministic: bool = False,
				 device = None, dtype: torch.dtype = torch.float64):
		"""
		:param EmbedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
		:param PredictionHorizon:	number of samples between a state and the target value it predicts
		:param KNN:	number of nearest neighbors whose equations enter each local linear fit; 0 means every available training state
		:param Step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
		:param Theta:	localization strength: neighbor weights are exp(-Theta * d / mean(d)); 0 fits one global linear map, larger values weight near neighbors more
		:param ExclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
		:param IsTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position so repeated runs pick the same neighbors; False leaves the order to torch.topk
		:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
		:param dtype:	torch dtype of the distances, weights and predictions
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
		"""
		Predict with SMapPredict and keep the SMapResult in Result.

		:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
		:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
		:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
		:param Y_test:	true data for the test samples, same layout as Y_train; when given, the result carries a performance score per target
		:return: SMapResult holding Y_pred with the same shape as Y_test and the coefficients of each sample's local fit
		"""
		self.Result = SMapPredict(X_train, Y_train, X_test, Y_test,
								  embedDimensions = self.EmbedDimensions, step = self.Step,
								  predictionHorizon = self.PredictionHorizon, knn = self.KNN, theta = self.Theta,
								  exclusionRadius = self.ExclusionRadius,
								  isTieBreakDeterministic = self.IsTieBreakDeterministic,
								  device = self.device, dtype = self.dtype)
		return self.Result
