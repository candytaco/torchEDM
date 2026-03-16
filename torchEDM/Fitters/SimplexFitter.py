
from typing import Optional, List, Union

import numpy

from torchEDM.EDM.Simplex import Simplex
from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter


class SimplexFitter(EDMFitter):
	"""
	sklearn-compatible wrapper for Simplex projection.

	Usage::

	    fitter = SimplexFitter(EmbedDimensions=3, PredictionHorizon=1)
	    fitter.fit(X_train, y_train)
	    y_pred = fitter.predict(X_test)
	    score  = fitter.score(X_test, y_test)
	    result = fitter.result_  # full SimplexResult
	"""

	def __init__(self,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Embedded: bool = False,
				 Verbose: bool = False):
		"""
		:param EmbedDimensions:   Embedding dimension (E).
		:param PredictionHorizon: Prediction time horizon (Tp).
		:param KNN:               Number of nearest neighbours (0 → E+1).
		:param Step:              Time-delay step size (tau).
		:param ExclusionRadius:   Temporal exclusion radius for neighbours.
		:param Embedded:          Whether data are already embedded.
		:param Verbose:           Print diagnostic messages.
		"""
		super().__init__()

		self.EmbedDimensions  = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN              = KNN
		self.Step             = Step
		self.ExclusionRadius  = ExclusionRadius
		self.Embedded         = Embedded
		self.Verbose          = Verbose

		self.Simplex = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'SimplexFitter':
		"""
		Store training data as the Simplex library.

		:param X_train:    Training feature data.
		:param y_train:    Training target data.
		:param TrainStart: Rows to skip at the start of the training set.
		:param TrainEnd:   Rows to drop at the end of the training set.
		:param trainTime:  Optional time-stamp column for training data.
		:return: self
		"""
		super().fit(X_train, y_train, TrainStart, TrainEnd, trainTime)
		return self

	def predict(self,
				X_test: numpy.ndarray,
				y_test: Optional[numpy.ndarray] = None,
				TestStart: int = 0,
				TestEnd: int = 0,
				testTime: Optional[numpy.ndarray] = None) -> numpy.ndarray:
		"""
		Predict using Simplex projection on the supplied test points.

		:param X_test:   Test feature data.
		:param y_test:   Test target data (optional; used for evaluation only).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: 1-D array of predicted values (NaN where no prediction).
		"""
		self._check_is_fitted()
		self._build_adapter(X_test, y_test, TestStart, TestEnd, testTime)

		Data         = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices  = self.GetTestIndices()
		YIndex       = self.GetYIndex()
		NoTime       = not self.HasTime()
		XStart, XEnd = self.GetXIndices()
		Columns      = list(range(XStart, XEnd + 1))

		self.Simplex = Simplex(
			data             = Data,
			columns          = Columns,
			target           = YIndex,
			train            = TrainIndices,
			test             = TestIndices,
			embedDimensions  = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn              = self.KNN,
			step             = self.Step,
			exclusionRadius  = self.ExclusionRadius,
			noTime           = NoTime,
			verbose          = self.Verbose,
			embedded         = self.Embedded,
		)

		self.result_ = self.Simplex.Run()
		return self.result_.predictions

	def get_params(self, deep: bool = True) -> dict:
		return {
			'EmbedDimensions':   self.EmbedDimensions,
			'PredictionHorizon': self.PredictionHorizon,
			'KNN':               self.KNN,
			'Step':              self.Step,
			'ExclusionRadius':   self.ExclusionRadius,
			'Embedded':          self.Embedded,
			'Verbose':           self.Verbose,
		}
