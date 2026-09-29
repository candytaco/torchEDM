"""
Cross-validation splits over runs. A split is a list of (runIndex, start, stop) slices;
SliceRuns turns it into the list of arrays the predictors take.
"""
from typing import Generator, List, Tuple

import numpy

Slice = Tuple[int, int, int]


def SliceRuns(runs: List[numpy.ndarray], slices: List[Slice]) -> List[numpy.ndarray]:
	"""
	The rows of each named run, one array per slice.

	:param runs:	list of [nRows, nColumns] arrays
	:param slices:	list of (runIndex, start, stop): rows [start, stop) of runs[runIndex]
	:return: list of arrays, one per slice, in the order given
	"""
	return [runs[runIndex][start:stop] for runIndex, start, stop in slices]


class RunSplitter:
	"""
	Leave-one-run-out or n-fold splits that never cut a run's history across a fold:
	every held-out block is a contiguous slice of one run.
	"""

	def __init__(self, runLengths: List[int], nFolds: int = 5, leaveOneRunOut: bool = True):
		"""
		:param runLengths:	rows per run, in run order
		:param nFolds:	contiguous blocks each run is cut into in n-fold mode
		:param leaveOneRunOut:	True holds out one whole run per split; False holds out block k of every run in split k
		"""
		self.runLengths = list(runLengths)
		self.nFolds = nFolds
		self.leaveOneRunOut = leaveOneRunOut

	def GetNSplits(self) -> int:
		""":return: splits Split will yield: the number of runs in leave-one-run-out mode, else nFolds"""
		return len(self.runLengths) if self.leaveOneRunOut else self.nFolds

	def Split(self) -> Generator[Tuple[List[Slice], List[Slice]], None, None]:
		"""
		Yield every split.

		:return: generator of (trainSlices, testSlices), each a list of (runIndex, start, stop) half-open row ranges
		"""
		if self.leaveOneRunOut:
			yield from self.SplitLeaveOneRunOut()
		else:
			yield from self.SplitNFold()

	def SplitLeaveOneRunOut(self):
		"""Yield one split per run: that run held out whole, every other run training."""
		for testRun in range(len(self.runLengths)):
			train = [(run, 0, length) for run, length in enumerate(self.runLengths) if run != testRun]
			yield train, [(testRun, 0, self.runLengths[testRun])]

	def SplitNFold(self):
		"""
		Cut every run into nFolds contiguous blocks of near-equal length and yield one split per
		block index: that block of every run held out, the other blocks training. Blocks that
		come out empty (a run shorter than nFolds) are skipped.
		"""
		blocksPerRun = []
		for length in self.runLengths:
			edges = numpy.linspace(0, length, self.nFolds + 1).astype(int)
			blocksPerRun.append([(int(edges[i]), int(edges[i + 1])) for i in range(self.nFolds)])
		for testFold in range(self.nFolds):
			train, test = [], []
			for run, blocks in enumerate(blocksPerRun):
				for fold, (start, stop) in enumerate(blocks):
					if stop <= start:
						continue
					(test if fold == testFold else train).append((run, start, stop))
			yield train, test
