
from typing import Optional, List, Union

import numpy

from ..EDM.Multiview import Multiview
from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter


class MultiviewFitter(EDMFitter):
	"""
	sklearn-compatible wrapper for Multiview embedding.

	Multiview builds an ensemble of Simplex projections over many possible
	time-delay embeddings, ranks them by in-sample skill, and averages the
	top-ranked projections for the final forecast.

	Usage::

	    fitter = MultiviewFitter(EmbedDimensions=3, PredictionHorizon=1)
	    fitter.fit(X_train, y_train)
	    y_pred = fitter.predict(X_test, y_test)
	    result = fitter.result_  # full MultiviewResult (includes view rankings)
	"""

	def __init__(self,
				 dimensions: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 NumMultiview: int = 0,
				 ExclusionRadius: int = 0,
				 TrainLib: bool = True,
				 ExcludeTarget: bool = False,
				 Verbose: bool = False):
		"""
		:param dimensions:        State-space dimension (D).
		:param EmbedDimensions:   Embedding dimension per variable (E).
		:param PredictionHorizon: Prediction time horizon (Tp).
		:param KNN:               Number of nearest neighbours (0 → E+1).
		:param Step:              Time-delay step size (tau).
		:param NumMultiview:      Number of top-ranked views to average (0 → sqrt(C(N,E))).
		:param ExclusionRadius:   Temporal exclusion radius for neighbours.
		:param TrainLib:          If True, evaluate views on training data.
		:param ExcludeTarget:     If True, exclude target column from candidate columns.
		:param Verbose:           Print diagnostic messages.
		"""
		super().__init__()

		self.dimensions      = dimensions
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN             = KNN
		self.Step            = Step
		self.NumMultiview    = NumMultiview
		self.ExclusionRadius = ExclusionRadius
		self.TrainLib        = TrainLib
		self.ExcludeTarget   = ExcludeTarget
		self.Verbose         = Verbose

		self.Multiview = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'MultiviewFitter':
		"""
		Store training data as the Multiview library.

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
		Predict using Multiview ensemble on the supplied test points.

		:param X_test:   Test feature data.
		:param y_test:   Test target data (optional; used for evaluation only).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: 1-D array of ensemble-averaged predicted values.
		"""
		self._check_is_fitted()
		self._build_adapter(X_test, y_test, TestStart, TestEnd, testTime)

		Data         = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices  = self.GetTestIndices()
		YIndex       = self.GetYIndex()
		XStart, XEnd = self.GetXIndices()
		Columns      = list(range(XStart, XEnd + 1))

		self.Multiview = Multiview(
			data             = Data,
			columns          = Columns,
			target           = YIndex,
			train            = TrainIndices,
			test             = TestIndices,
			D                = self.dimensions,
			embedDimensions  = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn              = self.KNN,
			step             = self.Step,
			multiview        = self.NumMultiview,
			exclusionRadius  = self.ExclusionRadius,
			trainLib         = self.TrainLib,
			excludeTarget    = self.ExcludeTarget,
			verbose          = self.Verbose,
		)

		self.result_ = self.Multiview.Run()
		return self.result_.predictions

	def get_params(self, deep: bool = True) -> dict:
		return {
			'dimensions':        self.dimensions,
			'EmbedDimensions':   self.EmbedDimensions,
			'PredictionHorizon': self.PredictionHorizon,
			'KNN':               self.KNN,
			'Step':              self.Step,
			'NumMultiview':      self.NumMultiview,
			'ExclusionRadius':   self.ExclusionRadius,
			'TrainLib':          self.TrainLib,
			'ExcludeTarget':     self.ExcludeTarget,
			'Verbose':           self.Verbose,
		}
