"""
Turning X and Y arrays into the training pairs and test states that the predictors consume.

Every array is [nSamples, nColumns]. A list of arrays is a list of runs; each run is
processed alone, so stacked history and horizon-shifted targets never cross a run
boundary. Samples are assumed evenly spaced. The state at row r stacks
X[r], X[r + step], ..., X[r + (embedDimensions - 1) * step].

Row semantics shared by every predictor:
- a training pair is the state at row r paired with Y_train[r + predictionHorizon],
  kept whenever the state is complete (no NaN) and the target row lies inside the run;
- output row i of Y_pred is predicted from the state at X_test[i - predictionHorizon],
  and stays NaN when that state is incomplete or its row lies outside the run;
- omitting X_test is in-sample: the training rows predict themselves, and a training
  state within exclusionRadius rows of a test state (the state itself at radius 0)
  may not serve as its neighbor.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy

from .utils import MakeDelays

ArrayOrRuns = Union[numpy.ndarray, Sequence[numpy.ndarray]]


def AsRuns(arrays: ArrayOrRuns) -> List[numpy.ndarray]:
	"""
	One 2-D float array per run. A single array is one run; a 1-D array is one column.
	"""
	runs = list(arrays) if isinstance(arrays, (list, tuple)) else [arrays]
	out = []
	for run in runs:
		run = numpy.asarray(run, dtype = numpy.float64)
		if run.ndim == 1:
			run = run[:, None]
		if run.ndim != 2:
			raise ValueError(f'Each run must be a 1-D or 2-D array, got shape {run.shape}')
		out.append(run)
	return out


def IsListOfRuns(arrays) -> bool:
	return isinstance(arrays, (list, tuple))


def StackHistory(X: numpy.ndarray, embedDimensions: int, step: int) -> numpy.ndarray:
	"""
	State vectors for every row of one run: each column of X followed by its shifted
	copies, column-major (all lags of column 0, then all lags of column 1, ...). Rows whose
	history falls outside the run hold NaN.

	:param X:				[nSamples, nColumns]
	:param embedDimensions:	copies of each column in the state; 1 uses the columns as given
	:param step:			row offset between copies; negative reaches into the past
	:return: [nSamples, nColumns * embedDimensions]
	"""
	if embedDimensions < 1:
		raise ValueError('embedDimensions must be at least 1')
	if step == 0:
		raise ValueError('step must be non-zero')
	if embedDimensions == 1:
		return X
	return MakeDelays(X, embedDimensions, step)


def BuildTrainingPairs(X_train: numpy.ndarray, Y_train: numpy.ndarray, embedDimensions: int, step: int,
					   predictionHorizon: int, rowMask: Optional[numpy.ndarray] = None):
	"""
	Every row of one run whose state is complete and whose horizon-shifted target lies
	inside the run. The target itself may be NaN; the predictors decide what that means.

	:param rowMask:	optional bool [nSamples]; False bars the row from serving as a training state
	:return: (states [nPairs, stateSize], targets [nPairs, nTargets], stateRows [nPairs])
	"""
	states = StackHistory(X_train, embedDimensions, step)
	nRows = X_train.shape[0]
	stateRows = numpy.arange(nRows)
	targetRows = stateRows + predictionHorizon
	isUsable = (targetRows >= 0) & (targetRows < nRows) & numpy.isfinite(states).all(axis = 1)
	if rowMask is not None:
		rowMask = numpy.asarray(rowMask, dtype = bool)
		if rowMask.shape[0] != nRows:
			raise ValueError(f'trainRowMask has {rowMask.shape[0]} entries for a run of {nRows} rows')
		isUsable &= rowMask
	stateRows = stateRows[isUsable]
	return states[stateRows], Y_train[stateRows + predictionHorizon], stateRows


def BuildTestStates(X_test: numpy.ndarray, embedDimensions: int, step: int, predictionHorizon: int):
	"""
	The complete states of one run that predict one of its rows: output row i is predicted
	from the state at row i - predictionHorizon.

	:return: (states [nStates, stateSize], stateRows [nStates], outputRows [nStates])
	"""
	states = StackHistory(X_test, embedDimensions, step)
	nRows = X_test.shape[0]
	outputRows = numpy.arange(nRows)
	stateRows = outputRows - predictionHorizon
	isInside = (stateRows >= 0) & (stateRows < nRows)
	outputRows, stateRows = outputRows[isInside], stateRows[isInside]
	isComplete = numpy.isfinite(states[stateRows]).all(axis = 1)
	return states[stateRows[isComplete]], stateRows[isComplete], outputRows[isComplete]


def BuildExclusionMask(trainRows: numpy.ndarray, trainRuns: numpy.ndarray,
					   testRows: numpy.ndarray, testRuns: numpy.ndarray,
					   exclusionRadius: int) -> Optional[numpy.ndarray]:
	"""
	In-sample neighbor exclusion: a training state within exclusionRadius rows of a test
	state in the same run (the test state itself at radius 0) may not be its neighbor.

	:return: bool [nTrain, nTest], True = excluded; None when nothing is excluded
	"""
	isSameRun = trainRuns[:, None] == testRuns[None, :]
	isNear = numpy.abs(trainRows[:, None] - testRows[None, :]) <= exclusionRadius
	mask = isSameRun & isNear
	return mask if mask.any() else None


@dataclass(frozen = True)
class PredictionInputs:
	"""
	Training pairs and test states gathered across runs.

	Rows are positions within their own run; runs are indices into the run lists.
	outputRows[j] is the row of Y_pred (within run testRuns[j]) that test state j predicts.
	"""
	trainStates: numpy.ndarray		# [nTrain, stateSize]
	trainTargets: numpy.ndarray		# [nTrain, nTargets]
	trainRows: numpy.ndarray		# [nTrain]
	trainRuns: numpy.ndarray		# [nTrain]
	testStates: numpy.ndarray		# [nTest, stateSize]
	testRows: numpy.ndarray			# [nTest]
	testRuns: numpy.ndarray			# [nTest]
	outputRows: numpy.ndarray		# [nTest]
	outputLengths: Tuple[int, ...]	# rows of Y_pred per test run
	numTargets: int
	isInSample: bool
	isSingleTestRun: bool
	exclusionMask: Optional[numpy.ndarray]	# bool [nTrain, nTest] or None

	@property
	def stateSize(self) -> int:
		return self.trainStates.shape[1]

	@property
	def numTrainingPairs(self) -> int:
		return self.trainStates.shape[0]

	@property
	def numAvailableNeighbors(self) -> int:
		"""Training states that every test state may use once exclusions are applied."""
		if self.exclusionMask is None:
			return self.numTrainingPairs
		return self.numTrainingPairs - int(self.exclusionMask.sum(axis = 0).max())

	def SharedAxisRows(self):
		"""
		Row positions of training and test states on one axis (in-sample only), for
		ordering tied distances by temporal proximity. Runs are spaced apart by more than
		any run length so cross-run pairs never look close.
		"""
		span = int(max(self.outputLengths)) + 1
		return self.trainRows + self.trainRuns * span, self.testRows + self.testRuns * span


def PreparePrediction(X_train: ArrayOrRuns, Y_train: ArrayOrRuns, X_test: Optional[ArrayOrRuns] = None,
					  embedDimensions: int = 1, step: int = -1, predictionHorizon: int = 1,
					  exclusionRadius: int = 0, trainRowMask: Optional[ArrayOrRuns] = None) -> PredictionInputs:
	"""
	Gather training pairs and test states for the predictors.

	:param X_train:		[nTrain, nFeatures] or a list of runs
	:param Y_train:		[nTrain, nTargets] (1-D is one target) or a list matching X_train
	:param X_test:		[nTest, nFeatures] or a list of runs; None predicts the training rows in-sample
	:param embedDimensions:	copies of each feature column in the state
	:param step:		row offset between copies; negative reaches into the past
	:param predictionHorizon:	rows between a state and the target it predicts
	:param exclusionRadius:	in-sample only; training states within this many rows of a test state are not its neighbors
	:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
	"""
	xRuns = AsRuns(X_train)
	yRuns = AsRuns(Y_train)
	if len(xRuns) != len(yRuns):
		raise ValueError(f'X_train has {len(xRuns)} runs but Y_train has {len(yRuns)}')
	for runIndex, (x, y) in enumerate(zip(xRuns, yRuns)):
		if x.shape[0] != y.shape[0]:
			raise ValueError(f'Run {runIndex}: X_train has {x.shape[0]} rows but Y_train has {y.shape[0]}')
	numTargets = yRuns[0].shape[1]
	if any(y.shape[1] != numTargets for y in yRuns):
		raise ValueError('Every Y_train run must have the same number of columns')

	if trainRowMask is None:
		masks = [None] * len(xRuns)
	elif IsListOfRuns(trainRowMask):
		masks = list(trainRowMask)
	else:
		masks = [trainRowMask]
	if len(masks) != len(xRuns):
		raise ValueError(f'trainRowMask has {len(masks)} runs but X_train has {len(xRuns)}')

	isInSample = X_test is None
	if isInSample:
		testRunsX = xRuns
		isSingleTestRun = not IsListOfRuns(X_train)
	else:
		if exclusionRadius > 0:
			raise ValueError('exclusionRadius applies only in-sample (X_test omitted): a separate test '
							 'array shares no sample axis with the training arrays')
		testRunsX = AsRuns(X_test)
		isSingleTestRun = not IsListOfRuns(X_test)
	numFeatures = xRuns[0].shape[1]
	if any(x.shape[1] != numFeatures for x in xRuns + testRunsX):
		raise ValueError('Every X run must have the same number of columns')

	trainStates, trainTargets, trainRows, trainRuns = [], [], [], []
	for runIndex, (x, y, mask) in enumerate(zip(xRuns, yRuns, masks)):
		states, targets, rows = BuildTrainingPairs(x, y, embedDimensions, step, predictionHorizon, mask)
		trainStates.append(states)
		trainTargets.append(targets)
		trainRows.append(rows)
		trainRuns.append(numpy.full(len(rows), runIndex, dtype = int))
	trainStates = numpy.concatenate(trainStates)
	if trainStates.shape[0] == 0:
		raise ValueError('No usable training states: every row has incomplete history, an out-of-range target, or is masked')

	testStates, testRows, testRuns, outputRows, outputLengths = [], [], [], [], []
	for runIndex, x in enumerate(testRunsX):
		states, rows, outRows = BuildTestStates(x, embedDimensions, step, predictionHorizon)
		testStates.append(states)
		testRows.append(rows)
		testRuns.append(numpy.full(len(rows), runIndex, dtype = int))
		outputRows.append(outRows)
		outputLengths.append(x.shape[0])
	testStates = numpy.concatenate(testStates)
	if testStates.shape[0] == 0:
		raise ValueError('No usable test states: every row has incomplete history or falls outside the run')

	inputs = PredictionInputs(
		trainStates = trainStates,
		trainTargets = numpy.concatenate(trainTargets),
		trainRows = numpy.concatenate(trainRows),
		trainRuns = numpy.concatenate(trainRuns),
		testStates = testStates,
		testRows = numpy.concatenate(testRows),
		testRuns = numpy.concatenate(testRuns),
		outputRows = numpy.concatenate(outputRows),
		outputLengths = tuple(outputLengths),
		numTargets = numTargets,
		isInSample = isInSample,
		isSingleTestRun = isSingleTestRun,
		exclusionMask = None)
	if isInSample:
		mask = BuildExclusionMask(inputs.trainRows, inputs.trainRuns, inputs.testRows, inputs.testRuns, exclusionRadius)
		inputs = PredictionInputs(**{**inputs.__dict__, 'exclusionMask': mask})
	return inputs


def TestTargets(Y_true: ArrayOrRuns, inputs: PredictionInputs) -> numpy.ndarray:
	"""
	The target rows that the test states predict, gathered from Y_true (Y_test, or Y_train
	in-sample) in the order of inputs.testStates.
	:return: [nTest, nTargets]
	"""
	runs = AsRuns(Y_true)
	if len(runs) != len(inputs.outputLengths):
		raise ValueError(f'Y_true has {len(runs)} runs but the test data has {len(inputs.outputLengths)}')
	targets = numpy.empty((inputs.testStates.shape[0], inputs.numTargets))
	for runIndex, y in enumerate(runs):
		if y.shape[0] != inputs.outputLengths[runIndex] or y.shape[1] != inputs.numTargets:
			raise ValueError(f'Run {runIndex}: Y_true has shape {y.shape}, expected ({inputs.outputLengths[runIndex]}, {inputs.numTargets})')
		inRun = inputs.testRuns == runIndex
		targets[inRun] = y[inputs.outputRows[inRun]]
	return targets


def ResolveNeighborCount(knn: Optional[int], inputs: PredictionInputs, isEveryNeighborDefault: bool = False) -> int:
	"""
	The neighbor count a predictor will use. Zero or None means the default: the state size
	plus one, or every available training state when isEveryNeighborDefault.
	"""
	available = inputs.numAvailableNeighbors
	if knn is None or knn <= 0:
		knn = available if isEveryNeighborDefault else inputs.stateSize + 1
	if knn > available:
		raise ValueError(f'knn = {knn} but only {available} training states are available to every test state')
	return int(knn)


def ScatterPredictions(values: numpy.ndarray, inputs: PredictionInputs) -> Union[numpy.ndarray, List[numpy.ndarray]]:
	"""
	Place per-test-state values [nTest, ...] at the rows they predict in NaN-filled arrays
	of [nRows, ...] per test run. A single test run comes back as one array.
	"""
	values = numpy.asarray(values)
	outputs = []
	for runIndex, length in enumerate(inputs.outputLengths):
		out = numpy.full((length,) + values.shape[1:], numpy.nan, dtype = values.dtype)
		inRun = inputs.testRuns == runIndex
		out[inputs.outputRows[inRun]] = values[inRun]
		outputs.append(out)
	return outputs[0] if inputs.isSingleTestRun else outputs


def MatchTargetLayout(arrays, isTargetOneDimensional: bool):
	"""Drop the trailing target axis when Y was given as a 1-D array, per run."""
	if not isTargetOneDimensional:
		return arrays
	if isinstance(arrays, list):
		return [a[..., 0] for a in arrays]
	return arrays[..., 0]


def ScorePredictions(scoringFunction: Callable, Y_true: ArrayOrRuns, Y_pred: ArrayOrRuns) -> numpy.ndarray:
	"""
	scoringFunction(actual, predicted) per target column over all runs; the scoring
	functions drop non-finite pairs themselves. A score the function declines to compute
	is NaN.
	:return: [nTargets]
	"""
	actual = numpy.concatenate(AsRuns(Y_true))
	predicted = numpy.concatenate(AsRuns(Y_pred))
	if actual.shape != predicted.shape:
		raise ValueError(f'Y_true has shape {actual.shape} but Y_pred has shape {predicted.shape}')
	scores = []
	for column in range(actual.shape[1]):
		score = scoringFunction(actual[:, column], predicted[:, column])
		scores.append(numpy.nan if score is None else float(score))
	return numpy.array(scores)
