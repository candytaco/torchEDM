from typing import Optional, List, Union

import numpy

from ..Hyperparameters import FindOptimalPredictionHorizon
from .EDMFitter import EDMFitter


class PredictionHorizonFitter(EDMFitter):
	"""
	sklearn-compatible selector for the Simplex prediction horizon (Tp).

	Sweeps Tp from 1 to ``maxTp`` and evaluates Pearson correlation on the
	test set for each value.  After :meth:`predict` the optimal Tp is
	stored in :attr:`best_param_` and the full sweep table in
	:attr:`result_`.

	Usage::

	    selector = PredictionHorizonFitter(maxTp=15, EmbedDimensions=3, Step=-1)
	    selector.fit(X_train, y_train)
	    sweep = selector.predict(X_test, y_test)  # shape (maxTp, 2): [[Tp, corr], ...]
	    best_Tp = selector.best_param_
	    result  = selector.result_                # same as sweep
	"""

	def __init__(self,
				 maxTp: int = 10,
				 EmbedDimensions: int = 1,
				 Step: int = -1,
				 ExclusionRadius: float = 0,
				 Embedded: bool = False,
				 batched: bool = False):
		"""
		:param maxTp:           Maximum prediction horizon to test (tests 1 … maxTp).
		:param EmbedDimensions: Embedding dimension (E) used during sweep.
		:param Step:            Time-delay step size (tau).
		:param ExclusionRadius: Temporal exclusion radius for neighbours.
		:param Embedded:        Whether data are already embedded.
		:param batched:         Use shared maxTp library for all Tp values (faster,
		                        slightly less accurate for low Tp).
		"""
		super().__init__()

		self.maxTp           = maxTp
		self.EmbedDimensions = EmbedDimensions
		self.Step            = Step
		self.ExclusionRadius = ExclusionRadius
		self.Embedded        = Embedded
		self.batched         = batched

		self.best_param_: Optional[int] = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'PredictionHorizonFitter':
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
		Sweep over Tp values 1 … maxTp and return the correlation at each Tp.

		:param X_test:    Test feature data.
		:param y_test:    Test target data (optional; observations needed for correlation).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:   Rows to drop at the end of the test set.
		:param testTime:  Optional time-stamp column for test data.
		:return: 2-D array of shape ``(maxTp, 2)`` with columns ``[Tp, correlation]``.
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

		sweep = FindOptimalPredictionHorizon(
			data             = Data,
			columns          = Columns,
			target           = Target,
			train            = list(TrainIndices),
			test             = list(TestIndices),
			maxTp            = self.maxTp,
			embedDimensions  = self.EmbedDimensions,
			step             = self.Step,
			exclusionRadius  = self.ExclusionRadius,
			embedded         = self.Embedded,
			validLib         = [],
			noTime           = NoTime,
			ignoreNan        = True,
			batched          = self.batched,
		)

		self.result_     = sweep
		self.best_param_ = int(sweep[numpy.argmax(sweep[:, 1]), 0])
		return self.result_

	def score(self,
			  X_test: numpy.ndarray,
			  y_test: numpy.ndarray,
			  TestStart: int = 0,
			  TestEnd: int = 0,
			  testTime: Optional[numpy.ndarray] = None) -> float:
		"""
		Return the maximum Pearson correlation across all tested Tp values.

		:return: Best (max) correlation found in the Tp sweep.
		"""
		sweep = self.predict(X_test, y_test, TestStart, TestEnd, testTime)
		return float(numpy.nanmax(sweep[:, 1]))

	def get_params(self, deep: bool = True) -> dict:
		return {
			'maxTp':          self.maxTp,
			'EmbedDimensions': self.EmbedDimensions,
			'Step':           self.Step,
			'ExclusionRadius': self.ExclusionRadius,
			'Embedded':       self.Embedded,
			'batched':        self.batched,
		}
