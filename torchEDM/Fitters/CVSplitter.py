"""
Cross-validation splits over runs. A split is a list of (runIndex, start, stop) slices;
SliceRuns turns it into the list of arrays the predictors take.
"""
from typing import Generator, List, Tuple

import numpy

Slice = Tuple[int, int, int]


def SliceRuns(runs: List[numpy.ndarray], slices: List[Slice]) -> List[numpy.ndarray]:
	"""The rows [start, stop) of each named run, one array per slice."""
	return [runs[runIndex][start:stop] for runIndex, start, stop in slices]


class RunSplitter:
	"""
	Leave-one-run-out or n-fold splits that never cut a run's history across a fold:
	every held-out block is a contiguous slice of one run.
	"""

	def __init__(self, runLengths: List[int], nFolds: int = 5, leaveOneRunOut: bool = True):
		"""
		:param runLengths:	rows per run
		:param nFolds:		folds per run in n-fold mode
		:param leaveOneRunOut:	True holds out one whole run per split; False splits every run into nFolds contiguous blocks
		"""
		self.runLengths = list(runLengths)
		self.nFolds = nFolds
		self.leaveOneRunOut = leaveOneRunOut

	def GetNSplits(self) -> int:
		return len(self.runLengths) if self.leaveOneRunOut else self.nFolds

	def Split(self) -> Generator[Tuple[List[Slice], List[Slice]], None, None]:
		"""Yield (trainSlices, testSlices)."""
		if self.leaveOneRunOut:
			yield from self.SplitLeaveOneRunOut()
		else:
			yield from self.SplitNFold()

	def SplitLeaveOneRunOut(self):
		for testRun in range(len(self.runLengths)):
			train = [(run, 0, length) for run, length in enumerate(self.runLengths) if run != testRun]
			yield train, [(testRun, 0, self.runLengths[testRun])]

	def SplitNFold(self):
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
