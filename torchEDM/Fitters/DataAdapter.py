"""
Data adapter for handling separate X/Y and train/test arrays.

Provides bridge from SKLearn style to EDM single-array style.

All index ranges are 0-based, half-open [start, stop) pairs.
"""
from typing import Optional, Tuple, List, Union

import numpy

class DataAdapter:
	"""
	Base abstract data adapter class
	"""

	@staticmethod
	def MakeDataAdapter(XTrain: [numpy.ndarray, List[numpy.ndarray]],
						YTrain: [numpy.ndarray, List[numpy.ndarray]],
						XTest: Optional[numpy.ndarray] = None,
						YTest: Optional[numpy.ndarray] = None,
						TrainStart = 0, TrainEnd = 0,
						TestStart = 0, TestEnd = 0,
						trainTime: Optional[numpy.ndarray] = None,
						testTime: Optional[numpy.ndarray] = None) -> 'DataAdapter':
		"""
		Make a data adapter depending on whether we get a single or multiple train runs
		:param XTrain: 		training features
		:param XTest: 		testing features
		:param YTrain: 		training value to predict, should be just a single column
		:param YTest: 		testing value to predict, should be just a single column
		:param TrainStart:	index at which to start the train data; used to provide history for the first train sample
		:param TrainEnd:	number of additional data samples at end of train data to ignore
		:param TestStart:	index at which to start the test data; used to provide history for the first test sample
		:param TestEnd:		number of additional data samples at end of test data to ignore
		:param trainTime: 	time labels for train data
		:param testTime: 	time labels for test data
		:return:
		"""
		if (type(XTrain) == numpy.ndarray):
			return DataAdapterSingleRun(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd,
										trainTime, testTime)
		elif (type(XTrain) == list):
			return DataAdapterMultipleRuns(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd,
										trainTime, testTime)
		raise ValueError

	def __init__(self,
				 XTrain: [numpy.ndarray, List[numpy.ndarray]],
				 YTrain: [numpy.ndarray, List[numpy.ndarray]],
				 XTest: Optional[numpy.ndarray] = None,
				 YTest: Optional[numpy.ndarray] = None,
				 TrainStart = 0, TrainEnd = 0,
				 TestStart = 0, TestEnd = 0,
				 trainTime: Optional[numpy.ndarray] = None, testTime: Optional[numpy.ndarray] = None):
		"""
		Data adapter init
		:param XTrain: 		training features
		:param XTest: 		testing features
		:param YTrain: 		training value to predict, should be just a single column
		:param YTest: 		testing value to predict, should be just a single column
		:param TrainStart:	index at which to start the train data; used to provide history for the first train sample
		:param TrainEnd:	number of additional data samples at end of train data to ignore
		:param TestStart:	index at which to start the test data; used to provide history for the first test sample
		:param TestEnd:		number of additional data samples at end of test data to ignore
		:param trainTime: 	time labels for train data
		:param testTime: 	time labels for test data
		"""
		self.XTrain = XTrain
		self.XTest = XTest
		self.YTrain = YTrain
		self.YTest = YTest
		self.TrainStart = TrainStart
		self.TrainEnd = TrainEnd
		self.TestStart = TestStart
		self.TestEnd = TestEnd
		self.trainTime = trainTime
		self.testTime = testTime
		self.hasTime = False

		self.trainOffset = None
		self.testOffset = None

		self.fullData = None

		self.trainTestSplitIndex = None

		self.StackData()

	@property
	def TrainData(self) -> numpy.ndarray:
		"""
		Get matrix for only the stacked train data
		:return:
		"""
		return self.fullData[:self.trainTestSplitIndex, :]

	@property
	def TestData(self) -> numpy.ndarray:
		"""
		Get matrix only for the stacked test data
		:return:
		"""
		if self.XTest is not None:
			return self.fullData[self.trainTestSplitIndex, :]
		else:
			raise ValueError

	def StackData(self):
		"""
		Function called to format the data into EDM style format
		:return:
		"""
		raise NotImplementedError

	@property
	def HasTime(self) -> bool:
		"""
		Check if data has time column.

		:return: True if data has time column
		"""
		return self.hasTime

	@property
	def TrainIndices(self) -> List[Tuple[int, int]]:
		"""
		Get train indices for EDM.

		:return: Train indices as list of (start, end) pairs
		"""
		raise NotImplementedError

	@property
	def TestIndices(self) -> List[Tuple[int, int]]:
		"""
		Get test indices for EDM.

		:return: Test indices as list of (start, end) pairs
		:raises ValueError: if no test data
		"""
		raise NotImplementedError

	@property
	def XIndices(self) -> Tuple[int, int]:
		"""
		Indices for X variables, half-open [start, stop).

		:return: X indices (start, stop)
		"""
		raise NotImplementedError

	@property
	def YIndex(self) -> List[int]:
		"""
		Indices for Y variable columns.

		:return: Y column indices
		"""
		raise NotImplementedError


class DataAdapterSingleRun(DataAdapter):

	def __init__(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: Optional[numpy.ndarray] = None,
				 YTest: Optional[numpy.ndarray] = None, TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0,
				 trainTime: Optional[numpy.ndarray] = None, testTime: Optional[numpy.ndarray] = None):
		"""
		Data adapter init
		:param XTrain: 		training features
		:param XTest: 		testing features
		:param YTrain: 		training value to predict, should be just a single column
		:param YTest: 		testing value to predict, should be just a single column
		:param TrainStart:	index at which to start the train data; used to provide history for the first train sample
		:param TrainEnd:	number of additional data samples at end of train data to ignore
		:param TestStart:	index at which to start the test data; used to provide history for the first test sample
		:param TestEnd:		number of additional data samples at end of test data to ignore
		:param trainTime: 	time labels for train data
		:param testTime: 	time labels for test data
		"""

		super().__init__(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, trainTime, testTime)

	def StackData(self):
		if self.YTrain is not None:
			if self.YTrain.ndim == 1:
				self.YTrain = self.YTrain[:, None]
			if self.YTest is not None and self.YTest.ndim == 1:
				self.YTest = self.YTest[:, None]
			train = numpy.hstack([self.XTrain, self.YTrain])
		else:
			train = self.XTrain
		self.trainOffset = self.TrainStart
		self.testOffset = self.TestStart + train.shape[0]

		self.trainTestSplitIndex = train.shape[0]

		if self.YTest is not None:
			test = numpy.hstack([self.XTest, self.YTest])
			data = numpy.vstack([train, test])
		else:
			data = train

		# add time if not none
		if self.trainTime is not None:
			self.trainTime = self.trainTime.squeeze()
			if self.testTime is not None:
				self.testTime = self.testTime.squeeze()
				time = numpy.concatenate([self.trainTime, self.testTime])
			else:
				time = self.trainTime
			data = numpy.hstack([time[:, None], data])
			self.hasTime = True

		self.fullData = data

	@property
	def TrainIndices(self) -> List[Tuple[int, int]]:
		return [(self.trainOffset, self.XTrain.shape[0] - self.TrainEnd)]

	@property
	def TestIndices(self) -> List[Tuple[int, int]]:
		if self.YTest is not None:
			return [(self.testOffset, self.fullData.shape[0] - self.TestEnd)]
		else:
			raise ValueError('No test data')

	@property
	def XIndices(self) -> Tuple[int, int]:
		return (0 + int(self.hasTime), self.XTrain.shape[1] + int(self.hasTime))

	@property
	def YIndex(self) -> List[int]:
		if self.YTrain is None:
			return []
		total = self.fullData.shape[1]
		nY = self.YTrain.shape[1]
		return list(range(total - nY, total))


class DataAdapterMultipleRuns(DataAdapter):
	"""
	A class that can take multiple train runs and adapt them to MDE code. Will specify separate
	'lib' indices for each run such that there's no bleedover between the stacked runs.

	Still takes only a single test run
	"""
	def __init__(self, XTrain: List[numpy.ndarray], YTrain: List[numpy.ndarray], XTest: Optional[numpy.ndarray] = None,
				 YTest: Optional[numpy.ndarray] = None, TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0,
				 trainTime: Optional[List[numpy.ndarray]] = None, testTime: Optional[numpy.ndarray] = None):
		self.numRuns = len(XTrain)
		self.trainIndices = []
		self.testIndices = None
		super().__init__(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, trainTime, testTime)

	def StackData(self):

		if type(self.TrainStart) == int:	# one trainStart for all runs
			self.TrainStart = [self.TrainStart] * self.numRuns
		if type(self.TrainEnd) == int:
			self.TrainEnd = [self.TrainEnd] * self.numRuns

		trainRuns = []
		if self.YTrain is not None:
			for X, Y in zip(self.XTrain, self.YTrain):
				if Y.ndim == 1:
					Y = Y[:, None]
				trainRuns.append(numpy.hstack([X, Y]))
		else:
			for X in self.XTrain:
				trainRuns.append(X)

		# calculate indices for each run in the stacked data
		n = 0
		for i, run in enumerate(trainRuns):
			start = n + self.TrainStart[i]
			end = n + run.shape[0] - self.TrainEnd[i]
			self.trainIndices.append((start, end))
			n += run.shape[0]

		data = numpy.vstack(trainRuns)
		self.trainTestSplitIndex = data.shape[0]

		# add test data if we have it
		if self.YTest is not None:
			if self.YTest.ndim == 1:
				self.YTest = self.YTest[:, None]
			test = numpy.hstack([self.XTest, self.YTest])
			data = numpy.vstack([data, test])

			start = n + self.TestStart
			end = n + self.XTest.shape[0] - self.TestEnd
			self.testIndices = (start, end)

		# add time if needed
		if self.trainTime is not None:
			time = numpy.concatenate([run.squeeze() for run in self.trainTime])
			if self.testTime is not None:
				time = numpy.concatenate([time, self.testTime.squeeze()])
			data = numpy.hstack([time[:, None], time])
			self.hasTime = True

		self.fullData = data

	@property
	def TrainIndices(self) -> List[Tuple[int, int]]:
		return self.trainIndices

	@property
	def TestIndices(self) -> List[Tuple[int, int]]:
		if self.YTest is not None:
			return [self.testIndices]
		else:
			raise ValueError('No test data')

	@property
	def XIndices(self) -> Tuple[int, int]:
		return (0 + int(self.hasTime), self.XTrain[0].shape[1] + int(self.hasTime) - 1)

	@property
	def YIndex(self) -> List[int]:
		if self.YTrain is None:
			return []
		total = self.fullData.shape[1]
		first_Y = self.YTrain[0]
		nY = first_Y.shape[1] if first_Y.ndim == 2 else 1
		return list(range(total - nY, total))