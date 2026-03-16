from typing import Optional, List, Union

import numpy

from torchEDM.EDM.CCM import CCM
from torchEDM.EDM.Results import CCMResult
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""
	sklearn-compatible wrapper for Convergent Cross Mapping (CCM).

	CCM tests for causality between two time-series variables by measuring
	cross-map skill across increasing library sizes.  Because CCM is a
	whole-dataset analysis (not a train-then-predict workflow), the fit/predict
	split works as follows:

	* ``fit(X, y)`` – stores the two time series that will be cross-mapped.
	* ``predict()`` – runs the CCM analysis and returns the library-size vs
	  cross-map-skill array (``libMeans``).  The full :class:`CCMResult` is
	  stored in ``self.result_``.

	Usage::

	    fitter = CCMFitter(TrainSizes=[10, 30, 50, 70], numRepeats=100)
	    fitter.fit(X, y)
	    lib_means = fitter.predict()
	    result    = fitter.result_
	"""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 progressBar: bool = True):
		"""
		:param TrainSizes:        Library sizes to evaluate.
		:param numRepeats:        Number of random samples per library size.
		:param EmbedDimensions:   Embedding dimension (E).
		:param PredictionHorizon: Prediction time horizon (Tp).
		:param KNN:               Number of nearest neighbours (0 → E+1).
		:param Step:              Time-delay step size (tau).
		:param ExclusionRadius:   Temporal exclusion radius for neighbours.
		:param Verbose:           Print diagnostic messages.
		:param progressBar:       Show progress bar.
		"""
		super().__init__(progressBar)

		self.TrainSizes      = TrainSizes
		self.Sample          = numRepeats
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN             = KNN
		self.Step            = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose         = Verbose

		self.CCM = None

	# ------------------------------------------------------------------

	def fit(self,
			X_train: numpy.ndarray,
			y_train: numpy.ndarray,
			TrainStart: int = 0,
			TrainEnd: int = 0,
			trainTime: Optional[numpy.ndarray] = None) -> 'CCMFitter':
		"""
		Store the two time series to be cross-mapped.

		For CCM, *X_train* is the "effect" variable and *y_train* is the
		"cause" variable (or vice-versa; both directions are always evaluated).

		:param X_train:    First time-series variable (2-D column vector).
		:param y_train:    Second time-series variable (2-D column vector).
		:param TrainStart: Rows to skip at the start of the data.
		:param TrainEnd:   Rows to drop at the end of the data.
		:param trainTime:  Optional time-stamp column.
		:return: self
		"""
		super().fit(X_train, y_train, TrainStart, TrainEnd, trainTime)
		return self

	def predict(self,
				X_test: Optional[numpy.ndarray] = None,
				y_test: Optional[numpy.ndarray] = None,
				TestStart: int = 0,
				TestEnd: int = 0,
				testTime: Optional[numpy.ndarray] = None) -> numpy.ndarray:
		"""
		Run the CCM analysis on the stored time series.

		CCM does not use a separate test set; the *X_test* / *y_test*
		arguments are accepted for API consistency but are ignored.  The
		cross-mapping is performed on the data supplied to :meth:`fit`.

		:return: ``libMeans`` array of shape ``(n_lib_sizes, 3)`` with columns
		         ``[library_size, fwd_correlation, rev_correlation]``.
		"""
		self._check_is_fitted()

		# CCM uses only the training data; build adapter without test data
		from .DataAdapter import DataAdapter
		self.DataAdapter = DataAdapter.MakeDataAdapter(
			self.X_train_, self.y_train_,
			None, None,
			self.TrainStart_, self.TrainEnd_,
			0, 0,
			self.trainTime_, None,
		)

		Data   = self.GetEDMData()
		NoTime = not self.HasTime()

		self.CCM = CCM(
			data             = Data,
			columns          = [0],
			target           = [1],
			trainSizes       = self.TrainSizes,
			sample           = self.Sample,
			embedDimensions  = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn              = self.KNN,
			step             = self.Step,
			exclusionRadius  = self.ExclusionRadius,
			noTime           = NoTime,
			verbose          = self.Verbose,
		)

		self.CCM.FwdMap.EmbedData()
		self.CCM.FwdMap.RemoveNan()
		self.CCM.RevMap.EmbedData()
		self.CCM.RevMap.RemoveNan()

		self.result_ = self.CCM.Run()
		return self.result_.libMeans

	def get_params(self, deep: bool = True) -> dict:
		return {
			'TrainSizes':        self.TrainSizes,
			'numRepeats':        self.Sample,
			'EmbedDimensions':   self.EmbedDimensions,
			'PredictionHorizon': self.PredictionHorizon,
			'KNN':               self.KNN,
			'Step':              self.Step,
			'ExclusionRadius':   self.ExclusionRadius,
			'Verbose':           self.Verbose,
		}
