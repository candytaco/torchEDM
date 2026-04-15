
from typing import List, Tuple, Generator, Union
import numpy

from .DataAdapter import DataAdapter, DataAdapterMultipleRuns, DataAdapterSingleRun


class EDMCVSplitter:
	"""
	Cross-validation splitter that handles multiple data runs
	with exclusion indices at the start and end of each run.
	"""

	def __init__(self,
				 dataAdapter: DataAdapter,
				 nFolds: int = 5,
				 leaveOneRunOut: bool = True,
				 edmStyleIndices: bool = False):
		"""
		Initialize the cross-validation splitter.

		:param dataAdapter: 		DataAdapter object containing run data and exclusion indices
		:param nFolds: 				number of folds for n-fold CV (ignored if leaveOneRunOut is True)
		:param leaveOneRunOut: 		if True, use leave-one-run-out CV instead of n-fold
		:param edmStyleIndices: 	if True, return flat [start, stop, ...] lists; if False, return numpy index arrays
		"""
		self.dataAdapter = dataAdapter
		self.nFolds = nFolds
		self.leaveOneRunOut = leaveOneRunOut
		self.edmStyleIndices = edmStyleIndices

		self.runs = []
		self.trainStart = []
		self.trainEnd = []
		self.ExtractRunInfo()

		self.numRuns = len(self.runs)
		self.runBoundaries = self.ComputeRunBoundaries()
		self.validRangesPerRun = self.ComputeValidRangesPerRun()

	def ExtractRunInfo(self):
		"""
		Extract run lengths and exclusion indices from the data adapter.
		TrainStart and TrainEnd are lists with one value per run.
		"""
		if isinstance(self.dataAdapter, DataAdapterSingleRun):
			self.runs = [self.dataAdapter.XTrain.shape[0]]
			self.trainStart = [self.dataAdapter.TrainStart]
			self.trainEnd = [self.dataAdapter.TrainEnd]
		elif isinstance(self.dataAdapter, DataAdapterMultipleRuns):
			self.runs = [X.shape[0] for X in self.dataAdapter.XTrain]
			self.trainStart = self.dataAdapter.TrainStart
			self.trainEnd = self.dataAdapter.TrainEnd

	def ComputeRunBoundaries(self) -> List[Tuple[int, int]]:
		"""
		Compute the start and end indices for each run in the concatenated data.

		:return: list of (start, end) tuples for each run (end is exclusive)
		"""
		boundaries = []
		currentIndex = 0
		for runLength in self.runs:
			boundaries.append((currentIndex, currentIndex + runLength))
			currentIndex += runLength
		return boundaries

	def ComputeValidRangesPerRun(self) -> List[Tuple[int, int]]:
		"""
		Compute valid sample ranges for each run after applying exclusions.
		Uses per-run trainStart and trainEnd values.

		:return: list of (start, end) tuples with valid range for each run (end is inclusive for EDM)
		"""
		validRanges = []
		for i, (start, end) in enumerate(self.runBoundaries):
			runStart = start + self.trainStart[i]
			runEnd = end - self.trainEnd[i] - 1
			if runStart <= runEnd:
				validRanges.append((runStart, runEnd))
			else:
				validRanges.append(None)
		return validRanges

	def GetNSplits(self) -> int:
		"""
		Return the number of splits.

		:return: number of cross-validation splits
		"""
		if self.leaveOneRunOut:
			return self.numRuns
		else:
			return self.nFolds

	def RangesToSklearn(self, ranges: List[Tuple[int, int]]) -> numpy.ndarray:
		"""
		Convert list of (start, end) ranges to sklearn-style flat index array.

		:param ranges: 	list of (start, end) tuples (end is inclusive)
		:return: 		numpy array of all indices
		"""
		indices = []
		for start, end in ranges:
			indices.append(numpy.arange(start, end + 1))
		return numpy.concatenate(indices) if indices else numpy.array([], dtype = int)

	def RangesToEDM(self, ranges: List[Tuple[int, int]]) -> List[int]:
		"""
		Convert list of (start, end) ranges to EDM-style flat list.

		:param ranges: 	list of (start, end) tuples (end is inclusive)
		:return: 		flat list [start1, end1, start2, end2, ...]
		"""
		result = []
		for start, end in ranges:
			result.extend([start, end])
		return result

	def FormatIndices(self, ranges: List[Tuple[int, int]]) -> Union[numpy.ndarray, List[int]]:
		"""
		Format ranges according to the configured index style.

		:param ranges: 	list of (start, end) tuples
		:return: 		indices in the configured format
		"""
		if self.edmStyleIndices:
			return self.RangesToEDM(ranges)
		else:
			return self.RangesToSklearn(ranges)

	def Split(self) -> Generator[Tuple[Union[numpy.ndarray, List[int]], Union[numpy.ndarray, List[int]]], None, None]:
		"""
		Generate train/test index splits.

		:return: generator yielding (trainIndices, testIndices) tuples in configured format
		"""
		if self.leaveOneRunOut:
			yield from self.SplitLeaveOneRunOut()
		else:
			yield from self.SplitNFold()

	def SplitLeaveOneRunOut(self) -> Generator[Tuple[Union[numpy.ndarray, List[int]], Union[numpy.ndarray, List[int]]], None, None]:
		"""
		Generate leave-one-run-out splits. Each run is used as test once,
		with all other runs as training.

		:return: generator yielding (trainIndices, testIndices) tuples
		"""
		for testRunIndex in range(self.numRuns):
			testRanges = []
			trainRanges = []

			for i in range(self.numRuns):
				if self.validRangesPerRun[i] is None:
					continue
				if i == testRunIndex:
					testRanges.append(self.validRangesPerRun[i])
				else:
					trainRanges.append(self.validRangesPerRun[i])

			yield self.FormatIndices(trainRanges), self.FormatIndices(testRanges)

	def SplitNFold(self) -> Generator[Tuple[Union[numpy.ndarray, List[int]], Union[numpy.ndarray, List[int]]], None, None]:
		"""
		Generate n-fold cross-validation splits. Each run is split into n contiguous
		segments, preserving temporal structure within runs.

		:return: generator yielding (trainIndices, testIndices) tuples
		"""
		foldRangesPerRun = []
		for runIndex in range(self.numRuns):
			validRange = self.validRangesPerRun[runIndex]
			if validRange is None:
				foldRangesPerRun.append([None] * self.nFolds)
				continue

			start, end = validRange
			numSamples = end - start + 1
			foldSize = numSamples // self.nFolds
			remainder = numSamples % self.nFolds

			runFolds = []
			currentPosition = start
			for foldIndex in range(self.nFolds):
				size = foldSize + (1 if foldIndex < remainder else 0)
				if size > 0:
					foldEnd = currentPosition + size - 1
					runFolds.append((currentPosition, foldEnd))
					currentPosition = foldEnd + 1
				else:
					runFolds.append(None)
			foldRangesPerRun.append(runFolds)

		for testFoldIndex in range(self.nFolds):
			testRanges = []
			trainRanges = []

			for runIndex in range(self.numRuns):
				for foldIndex in range(self.nFolds):
					foldRange = foldRangesPerRun[runIndex][foldIndex]
					if foldRange is None:
						continue
					if foldIndex == testFoldIndex:
						testRanges.append(foldRange)
					else:
						trainRanges.append(foldRange)

			yield self.FormatIndices(trainRanges), self.FormatIndices(testRanges)
