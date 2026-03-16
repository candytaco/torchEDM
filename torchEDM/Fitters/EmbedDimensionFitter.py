from typing import Optional, List, Union

import numpy

from ..Hyperparameters import FindOptimalEmbeddingDimensionality
from .EDMFitter import EDMFitter


class EmbedDimensionFitter(EDMFitter):
	"""
	sklearn-compatible selector for the Simplex embedding dimension (E).

	Sweeps E from 1 to ``maxE`` and evaluates Pearson correlation on the
	test set for each value.  After :meth:`predict` the optimal E is
	stored in :attr:`best_param_` and the full sweep table in
	:attr:`result_`.

	Usage::

	    selector = EmbedDimensionFitter(maxE=10, PredictionHorizon=1, Step=-1)
	    selector.fit(X_train, y_train)
	    sweep = selector.predict(X_test, y_test)  # shape (maxE, 2): [[E, corr], ...]
	    best_E = selector.best_param_
	    result = selector.result_                 # same as sweep
	"""

	def __init__(self,
				 maxE: int = 10,
				 PredictionHorizon: int = 1,
				 Step: int = -1,
				 ExclusionRadius: float = 0,
				 Embedded: bool = False,
				 batched: bool = False):
		"""
		:param maxE:              Maximum embedding dimension to test (tests 1 … maxE).
		:param PredictionHorizon: Prediction time horizon (Tp) used during sweep.
		:param Step:              Time-delay step size (tau).
		:param ExclusionRadius:   Temporal exclusion radius for neighbours.
		:param Embedded:          Whether data are already embedded.
		:param batched:           Use shared maxE indices for all E values (faster,
		                          slightly less accurate for low E).
		"""
		super().__init__()

		self.maxE             = maxE
		self.PredictionHorizon = PredictionHorizon
		self.Step             = Step
		self.ExclusionRadius  = ExclusionRadius
		self.Embedded         = Embedded
		self.batched          = batched

		self.best_param_: Optional[int] = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'EmbedDimensionFitter':
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
		Sweep over E values 1 … maxE and return the correlation at each E.

		:param X_test:    Test feature data.
		:param y_test:    Test target data (optional; observations needed for correlation).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:   Rows to drop at the end of the test set.
		:param testTime:  Optional time-stamp column for test data.
		:return: 2-D array of shape ``(maxE, 2)`` with columns ``[E, correlation]``.
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

		E_values, correlations = FindOptimalEmbeddingDimensionality(
			data             = Data,
			columns          = Columns,
			target           = Target,
			maxE             = self.maxE,
			train            = list(TrainIndices),
			test             = list(TestIndices),
			predictionHorizon = self.PredictionHorizon,
			step             = self.Step,
			exclusionRadius  = self.ExclusionRadius,
			embedded         = self.Embedded,
			validLib         = [],
			noTime           = NoTime,
			ignoreNan        = True,
			batched          = self.batched,
		)

		self.result_     = numpy.column_stack([E_values, correlations])
		self.best_param_ = int(E_values[numpy.argmax(correlations)])
		return self.result_

	def score(self,
			  X_test: numpy.ndarray,
			  y_test: numpy.ndarray,
			  TestStart: int = 0,
			  TestEnd: int = 0,
			  testTime: Optional[numpy.ndarray] = None) -> float:
		"""
		Return the maximum Pearson correlation across all tested E values.

		:return: Best (max) correlation found in the E sweep.
		"""
		sweep = self.predict(X_test, y_test, TestStart, TestEnd, testTime)
		return float(numpy.nanmax(sweep[:, 1]))

	def get_params(self, deep: bool = True) -> dict:
		return {
			'maxE':             self.maxE,
			'PredictionHorizon': self.PredictionHorizon,
			'Step':             self.Step,
			'ExclusionRadius':  self.ExclusionRadius,
			'Embedded':         self.Embedded,
			'batched':          self.batched,
		}
