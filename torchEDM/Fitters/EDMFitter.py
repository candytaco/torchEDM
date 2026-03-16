from typing import Optional, Tuple, List, Union

import numpy

from .DataAdapter import DataAdapter


class EDMFitter:
	"""
	Base wrapper class for EDM methods that provides a sklearn-compatible API.

	This class handles the conversion from separate X/Y train/test arrays to the
	EDM single-array format using DataAdapter.

	The sklearn-compatible pattern is:
	  1. Instantiate with hyperparameters: ``estimator = SomeFitter(param=value)``
	  2. Fit on training data: ``estimator.fit(X_train, y_train)``
	  3. Predict on test data: ``y_pred = estimator.predict(X_test)``
	  4. Score: ``score = estimator.score(X_test, y_test)``

	After ``predict()`` is called the full result object is available as
	``estimator.result_``.
	"""

	def __init__(self, progressBar: bool = True):
		"""
		Init. Subclasses set their algorithm hyperparameters here.

		:param progressBar: Show progress bar during long operations.
		"""
		self.DataAdapter = None
		self.result_ = None
		self.hideProgress = not progressBar

		# Attributes set by fit()
		self.X_train_: Optional[Union[numpy.ndarray, List[numpy.ndarray]]] = None
		self.y_train_: Optional[Union[numpy.ndarray, List[numpy.ndarray]]] = None
		self.TrainStart_: int = 0
		self.TrainEnd_: int = 0
		self.trainTime_: Optional[numpy.ndarray] = None

	# ------------------------------------------------------------------
	# sklearn-compatible public interface
	# ------------------------------------------------------------------

	def fit(self,
			X_train: Union[numpy.ndarray, List[numpy.ndarray]],
			y_train: Union[numpy.ndarray, List[numpy.ndarray]],
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'EDMFitter':
		"""
		Store training data as the EDM library.

		In EDM, training data forms the "library" of state-space points used
		to find nearest-neighbour analogues during prediction.  No expensive
		computation is performed here; the algorithm runs on the first call to
		:meth:`predict`.

		:param X_train:    Training feature data (array or list of arrays for
		                   multiple independent runs).
		:param y_train:    Training target data.
		:param TrainStart: Rows to skip at the *start* of each training run
		                   (to provide embedding history for the first usable
		                   sample).
		:param TrainEnd:   Rows to drop at the *end* of each training run.
		:param trainTime:  Optional time-stamp column for training data.
		:return: self
		"""
		self.X_train_ = X_train
		self.y_train_ = y_train
		self.TrainStart_ = TrainStart
		self.TrainEnd_ = TrainEnd
		self.trainTime_ = trainTime
		return self

	def predict(self,
				X_test: numpy.ndarray,
				y_test: Optional[numpy.ndarray] = None,
				TestStart: int = 0,
				TestEnd: int = 0,
				testTime: Optional[numpy.ndarray] = None) -> numpy.ndarray:
		"""
		Make predictions on test data using the fitted library.

		Must be called after :meth:`fit`.  The full result object (including
		time, observations and predictions) is stored in ``self.result_``.

		:param X_test:   Test feature data.
		:param y_test:   Test target data.  When provided the result object
		                 will contain observed values for error analysis; when
		                 omitted zeros are used as a placeholder.
		:param TestStart: Rows to skip at the *start* of the test set.
		:param TestEnd:  Rows to drop at the *end* of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: 1-D array of predicted values.
		"""
		raise NotImplementedError

	def score(self,
			  X_test: numpy.ndarray,
			  y_test: numpy.ndarray,
			  TestStart: int = 0,
			  TestEnd: int = 0,
			  testTime: Optional[numpy.ndarray] = None) -> float:
		"""
		Return the Pearson correlation between predictions and observations.

		Calls :meth:`predict` internally and uses the aligned observation /
		prediction arrays stored in the result object.

		:param X_test:   Test feature data.
		:param y_test:   True target values.
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		:return: Pearson correlation coefficient (float), or NaN if not enough
		         valid (non-NaN) points exist.
		"""
		self.predict(X_test, y_test, TestStart, TestEnd, testTime)

		# Use the result's built-in aligned observation/prediction arrays
		# (these handle NaN padding and embedding lag correctly)
		if hasattr(self.result_, 'compute_error'):
			return float(self.result_.compute_error())  # None → Pearson correlation
		return float('nan')

	def get_params(self, deep: bool = True) -> dict:
		"""
		Get the hyperparameters of this estimator.

		Subclasses should override this to return their ``__init__`` parameters.

		:param deep: Ignored; provided for sklearn API compatibility.
		:return: Parameter name → value mapping.
		"""
		return {}

	def set_params(self, **params) -> 'EDMFitter':
		"""
		Set estimator hyperparameters.

		:param params: Keyword arguments matching the names returned by
		              :meth:`get_params`.
		:return: self
		"""
		for key, value in params.items():
			setattr(self, key, value)
		return self

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _check_is_fitted(self) -> None:
		"""Raise RuntimeError if fit() has not been called yet."""
		if self.X_train_ is None:
			raise RuntimeError(
				"This estimator is not fitted yet.  Call 'fit' with appropriate "
				"arguments before calling 'predict'."
			)

	def _build_adapter(self,
					   X_test: numpy.ndarray,
					   y_test: Optional[numpy.ndarray],
					   TestStart: int,
					   TestEnd: int,
					   testTime: Optional[numpy.ndarray]) -> None:
		"""
		Construct the DataAdapter from stored training data and the supplied
		test data, storing it in ``self.DataAdapter``.

		:param X_test:   Test feature data.
		:param y_test:   Test target data (may be None → zeros placeholder).
		:param TestStart: Rows to skip at the start of the test set.
		:param TestEnd:  Rows to drop at the end of the test set.
		:param testTime: Optional time-stamp column for test data.
		"""
		if y_test is None:
			if X_test.ndim == 1:
				X_test = X_test[:, None].copy()
			y_test = numpy.zeros((X_test.shape[0], 1))

		self.DataAdapter = DataAdapter.MakeDataAdapter(
			self.X_train_, self.y_train_,
			X_test, y_test,
			self.TrainStart_, self.TrainEnd_,
			TestStart, TestEnd,
			self.trainTime_, testTime,
		)

	def GetEDMData(self) -> numpy.ndarray:
		"""
		Return the combined EDM data array built by the last :meth:`_build_adapter` call.
		"""
		return self.DataAdapter.fullData

	def GetTrainIndices(self) -> Tuple[int, int]:
		"""Return train indices ``[start, end]`` (stop-inclusive)."""
		return self.DataAdapter.TrainIndices

	def GetTestIndices(self) -> Tuple[int, int]:
		"""Return test indices ``[start, end]`` (stop-inclusive)."""
		return self.DataAdapter.TestIndices

	def GetXIndices(self) -> Tuple[int, int]:
		"""Return feature column indices ``[start, end]`` (stop-inclusive)."""
		return self.DataAdapter.XIndices

	def GetYIndex(self) -> int:
		"""Return the column index of the target variable."""
		return self.DataAdapter.YIndex

	def HasTime(self) -> bool:
		"""Return ``True`` if the combined data array contains a time column."""
		return self.DataAdapter.HasTime
