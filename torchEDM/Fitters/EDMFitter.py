from typing import Optional, Tuple, List

import numpy

from .DataAdapter import DataAdapter

class EDMFitter:
	"""
	Base wrapper class for EDM methods that provides sklearn-like API.

	This class handles the conversion from separate X/Y train/test arrays to the EDM single-array format using DataAdapter.

	It's difficult to provide an exact sklearn-compatible API, but, we *do* provide a
	sklearn-like API in that object creation sets the algorithm parameters,
	and calling Fit(XTrain, YTrain, ...) gives you a result object that is also stored in the fitter
	"""

	def __init__(self, progressBar = True):
		"""
		Init. Classes should set their algorithm parameters with this function
		:param progressBar:
		"""

		self.DataAdapter = None
		self.Result = None
		self.hideProgress = not progressBar

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray, YTest: numpy.ndarray,
			TrainStart = 1, TrainEnd = 0, TestStart = 0, TestEnd = 0,
			TrainTime: Optional[numpy.ndarray] = None, TestTime: Optional[numpy.ndarray] = None):
		"""
		Does the fitting and predicting

		:param XTrain: 		Training feature data
		:param YTrain: 		Training target data
		:param XTest: 		Test feature data
		:param YTest: 		Test target data
		:param TrainStart: 	Start index for train data
		:param TrainEnd: 	number of additional samples included beyond end of train data
		:param TestStart: 	Start index for test data
		:param TestEnd: 	number of additional samples included beyond end of test data
		:param TrainTime: 	Time labels for train data
		:param TestTime: 	Time labels for test data
		"""
		self.DataAdapter = DataAdapter.MakeDataAdapter(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd,
													   TestStart, TestEnd, TrainTime, TestTime)


	def GetEDMData(self) -> numpy.ndarray:
		"""
		Get the combined EDM data array.

		:return: Combined data array in EDM format
		"""

		return self.DataAdapter.fullData

	def GetTrainIndices(self) -> Tuple[int, int]:
		"""
		Get the train indices for EDM.

		:return: Train indices [start, end]
		"""

		return self.DataAdapter.TrainIndices

	def GetTestIndices(self) -> Tuple[int, int]:
		"""
		Get the test indices for EDM.

		:return: Test indices [start, end]
		"""

		return self.DataAdapter.TestIndices

	def GetXIndices(self) -> Tuple[int, int]:
		"""
		Get the X feature indices.

		:return: X indices [start, end]
		"""

		return self.DataAdapter.XIndices

	def GetYIndex(self) -> List[int]:
		"""
		Get the Y target column indices.

		:return: Y indices
		"""

		return self.DataAdapter.YIndex

	def HasTime(self) -> bool:
		"""
		Check if data has time column.

		:return: True if data has time column
		"""

		return self.DataAdapter.HasTime
