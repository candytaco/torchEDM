from typing import Any, Optional, List, Union

import numpy

from ..Hyperparameters import FindSMapNeighborhood
from .EDMFitter import EDMFitter


class SMapNeighborhoodFitter(EDMFitter):
	"""
	sklearn-compatible selector for the S-Map localisation parameter theta.

	Sweeps over the supplied ``theta`` values and evaluates Pearson
	correlation on the test set for each value.  After :meth:`predict` the
	optimal theta is stored in :attr:`best_param_` and the full sweep table
	in :attr:`result_`.

	Usage::

	    selector = SMapNeighborhoodFitter(EmbedDimensions=4, PredictionHorizon=1)
	    selector.fit(X_train, y_train)
	    sweep = selector.predict(X_test, y_test)  # shape (n_theta, 2): [[theta, corr], ...]
	    best_theta = selector.best_param_
	    result     = selector.result_             # same as sweep
	"""

	def __init__(self,
				 theta: Any = None,
				 EmbedDimensions: int = 1,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: float = 0,
				 Embedded: bool = False):
		"""
		:param theta:             Theta values to sweep.  ``None`` uses the default
		                          sequence ``[0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2,
		                          3, 4, 5, 6, 7, 8, 9]``.  May also be a
		                          space-separated string.
		:param EmbedDimensions:   Embedding dimension (E).
		:param PredictionHorizon: Prediction time horizon (Tp).
		:param KNN:               Number of nearest neighbours (0 → all library points).
		:param Step:              Time-delay step size (tau).
		:param ExclusionRadius:   Temporal exclusion radius for neighbours.
		:param Embedded:          Whether data are already embedded.
		"""
		super().__init__()

		self.theta            = theta
		self.EmbedDimensions  = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN              = KNN
		self.Step             = Step
		self.ExclusionRadius  = ExclusionRadius
		self.Embedded         = Embedded

		self.best_param_: Optional[float] = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'SMapNeighborhoodFitter':
		"""
		Store training data as the S-Map library.

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
		Sweep over theta values and return the correlation at each theta.

		:param X_test:    Test feature data.
		:param y_test:    Test target data (optional; observations needed for correlation).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:   Rows to drop at the end of the test set.
		:param testTime:  Optional time-stamp column for test data.
		:return: 2-D array of shape ``(n_theta, 2)`` with columns ``[theta, correlation]``.
		         Also sets :attr:`best_param_` and :attr:`result_`.
		"""
		self._check_is_fitted()
		self._build_adapter(X_test, y_test, TestStart, TestEnd, testTime)

		Data         = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices  = self.GetTestIndices()
		XStart, XEnd = self.GetXIndices()
		Columns      = list(range(XStart, XEnd + 1))
		Target       = self.GetYIndex()
		NoTime       = not self.HasTime()

		sweep = FindSMapNeighborhood(
			data             = Data,
			columns          = Columns,
			target           = Target,
			theta            = self.theta,
			train            = list(TrainIndices),
			test             = list(TestIndices),
			embedDimensions  = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn              = self.KNN,
			step             = self.Step,
			exclusionRadius  = self.ExclusionRadius,
			embedded         = self.Embedded,
			validLib         = [],
			noTime           = NoTime,
			ignoreNan        = True,
		)

		self.result_     = sweep
		self.best_param_ = float(sweep[numpy.argmax(sweep[:, 1]), 0])
		return self.result_

	def score(self,
			  X_test: numpy.ndarray,
			  y_test: numpy.ndarray,
			  TestStart: int = 0,
			  TestEnd: int = 0,
			  testTime: Optional[numpy.ndarray] = None) -> float:
		"""
		Return the maximum Pearson correlation across all tested theta values.

		:return: Best (max) correlation found in the theta sweep.
		"""
		sweep = self.predict(X_test, y_test, TestStart, TestEnd, testTime)
		return float(numpy.nanmax(sweep[:, 1]))

	def get_params(self, deep: bool = True) -> dict:
		return {
			'theta':           self.theta,
			'EmbedDimensions': self.EmbedDimensions,
			'PredictionHorizon': self.PredictionHorizon,
			'KNN':             self.KNN,
			'Step':            self.Step,
			'ExclusionRadius': self.ExclusionRadius,
			'Embedded':        self.Embedded,
		}
