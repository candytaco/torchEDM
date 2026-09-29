"""
Turn X and Y arrays into the training pairs and test states that the predictors consume.

Every array is [nSamples, nVariables]. A list of arrays is a list of runs; each run is
processed alone, so stacked lags and horizon-shifted targets never cross a run boundary.
Samples are assumed evenly spaced. The state at sample s stacks
X[s], X[s + step], ..., X[s + (embedDimensions - 1) * step].

Every predictor shares these sample semantics:
- a training pair is the state at sample s paired with Y_train[s + predictionHorizon],
  kept whenever the state is complete (no NaN) and the target sample lies inside the run;
- Y_pred[i] is predicted from the state at X_test[i - predictionHorizon], and stays NaN
  when that state is incomplete or its sample lies outside the run;
- omitting X_test is in-sample: the training samples predict themselves, and a training
  state within exclusionRadius samples of a test state (the state itself at radius 0)
  is excluded from its neighbors.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy

from .utils import MakeDelays

ArrayOrRuns = Union[numpy.ndarray, Sequence[numpy.ndarray]]


def AsRuns(arrays: ArrayOrRuns) -> List[numpy.ndarray]:
	"""
	Return one 2-D float array per run.

	:param arrays:	one array, or a list or tuple of arrays with one per run; a 1-D array holds one variable
	:return: list of [nSamples, nVariables] float64 arrays, one per run
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
	"""
	Return whether an X, Y or mask argument is a list or tuple of runs rather than a single array.

	:param arrays:	the argument as the caller passed it
	"""
	return isinstance(arrays, (list, tuple))


def StackHistory(X: numpy.ndarray, embedDimensions: int, step: int) -> numpy.ndarray:
	"""
	Build the state vector of every sample of one run: each variable of X followed by its
	lagged copies, grouped by variable (all lags of variable 0, then all lags of variable 1,
	and so on). Samples whose lags fall outside the run hold NaN.

	:param X:	one run, [nSamples, nVariables]
	:param embedDimensions:	number of lagged copies of each variable in the state; 1 returns X as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:return: states, [nSamples, nVariables * embedDimensions]
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
	Collect every sample of one run whose state is complete and whose horizon-shifted target
	lies inside the run. The target itself may be NaN; the predictors decide what that means.

	:param X_train:	one run of training input data, [nSamples, nFeatures]
	:param Y_train:	the same run's target data, [nSamples, nTargets]
	:param embedDimensions:	number of lagged copies of each variable in the state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target it is paired with
	:param rowMask:	optional boolean mask over the samples, [nSamples]; False excludes that sample from serving as a training state
	:return: (states [nPairs, stateSize], targets [nPairs, nTargets], stateSamples [nPairs]): the pairs, and the position of each state within the run
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
	Collect the complete states of one run that predict one of its samples: sample i is
	predicted from the state at sample i - predictionHorizon.

	:param X_test:	one run of test input data, [nSamples, nFeatures]
	:param embedDimensions:	number of lagged copies of each variable in the state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the sample it predicts
	:return: (states [nStates, stateSize], stateSamples [nStates], outputSamples [nStates]): the position of each state and the position of the sample it predicts
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
	Mark the in-sample neighbor exclusions: a training state within exclusionRadius samples of
	a test state in the same run (the test state itself at radius 0) is excluded from its
	neighbors.

	:param trainRows:	position of each training state within its run, [nTrain]
	:param trainRuns:	run index of each training state, [nTrain]
	:param testRows:	position of each test state within its run, [nTest]
	:param testRuns:	run index of each test state, [nTest]
	:param exclusionRadius:	largest sample distance that is still excluded
	:return: boolean [nTrain, nTest] with True for an excluded pair, or None when nothing is excluded
	"""
	isSameRun = trainRuns[:, None] == testRuns[None, :]
	isNear = numpy.abs(trainRows[:, None] - testRows[None, :]) <= exclusionRadius
	mask = isSameRun & isNear
	return mask if mask.any() else None


@dataclass(frozen = True)
class PredictionInputs:
	"""
	Hold the training pairs and test states gathered across runs. Positions count samples
	within their own run; runs index the run lists.

	:param trainStates:	complete training states, [nTrain, stateSize]
	:param trainTargets:	horizon-shifted target of each training state, [nTrain, nTargets]; may hold NaN
	:param trainRows:	position of each training state within its run, [nTrain]
	:param trainRuns:	run index of each training state, [nTrain]
	:param testStates:	complete test states, [nTest, stateSize]
	:param testRows:	position of each test state within its run, [nTest]
	:param testRuns:	run index of each test state, [nTest]
	:param outputRows:	position in Y_pred (within run testRuns[j]) that test state j predicts, [nTest]
	:param outputLengths:	number of samples in Y_pred per test run
	:param numTargets:	number of target variables
	:param isInSample:	True when the training samples predict themselves (X_test omitted)
	:param isSingleTestRun:	True when the test data came as one array, so Y_pred is one array rather than a list
	:param exclusionMask:	boolean [nTrain, nTest] marking the pairs excluded from neighbor search, or None when nothing is excluded
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
		"""Return the number of dimensions of a state: nFeatures * embedDimensions."""
		return self.trainStates.shape[1]

	@property
	def numTrainingPairs(self) -> int:
		"""Return the number of training states before exclusions."""
		return self.trainStates.shape[0]

	@property
	def numAvailableNeighbors(self) -> int:
		"""Return the number of training states that every test state may use once exclusions are applied."""
		if self.exclusionMask is None:
			return self.numTrainingPairs
		return self.numTrainingPairs - int(self.exclusionMask.sum(axis = 0).max())

	def SharedAxisRows(self):
		"""
		Return the positions of the training and test states on one shared sample axis (in-sample
		only), used to order tied distances by temporal proximity. Runs are spaced apart by more
		than any run length so cross-run pairs never look close.

		:return: (trainPositions [nTrain], testPositions [nTest])
		"""
		span = int(max(self.outputLengths)) + 1
		return self.trainRows + self.trainRuns * span, self.testRows + self.testRuns * span


def PreparePrediction(X_train: ArrayOrRuns, Y_train: ArrayOrRuns, X_test: Optional[ArrayOrRuns] = None,
					  embedDimensions: int = 1, step: int = -1, predictionHorizon: int = 1,
					  exclusionRadius: int = 0, trainRowMask: Optional[ArrayOrRuns] = None) -> PredictionInputs:
	"""
	Gather the training pairs and test states for the predictors.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train; NaN targets are allowed
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
	:param embedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:return: PredictionInputs; raises ValueError when the arrays disagree in runs, samples or variables, when exclusionRadius is given with a separate X_test, or when no usable training or test state remains
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
	Gather the true values of the samples that the test states predict, in the order of
	inputs.testStates.

	:param Y_true:	true data for the test samples (Y_test, or Y_train in-sample), [nSamples, nTargets] per test run
	:param inputs:	the PredictionInputs the test states came from
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
	Return the neighbor count a predictor will use.

	:param knn:	the requested number of neighbors; 0 or None asks for the default
	:param inputs:	the PredictionInputs, for the state size and the neighbors available after exclusions
	:param isEveryNeighborDefault:	False makes the default the state size plus one; True makes it every available training state
	:return: the neighbor count; raises ValueError when it exceeds the training states available to every test state
	"""
	available = inputs.numAvailableNeighbors
	if knn is None or knn <= 0:
		knn = available if isEveryNeighborDefault else inputs.stateSize + 1
	if knn > available:
		raise ValueError(f'knn = {knn} but only {available} training states are available to every test state')
	return int(knn)


def ScatterPredictions(values: numpy.ndarray, inputs: PredictionInputs) -> Union[numpy.ndarray, List[numpy.ndarray]]:
	"""
	Place per-test-state values at the samples they predict, in NaN-filled arrays per test run.

	:param values:	one entry per test state in the order of inputs.testStates, [nTest, ...]
	:param inputs:	the PredictionInputs the test states came from
	:return: [nSamples, ...] per test run; one array for a single test run, a list for several
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
	"""
	Drop the trailing target axis when Y was given as a 1-D array.

	:param arrays:	an array [nSamples, ..., nTargets] or a list of them, one per run
	:param isTargetOneDimensional:	True when the caller's Y had no target axis
	:return: the arrays with their last axis removed when isTargetOneDimensional, else unchanged
	"""
	if not isTargetOneDimensional:
		return arrays
	if isinstance(arrays, list):
		return [a[..., 0] for a in arrays]
	return arrays[..., 0]


def ScorePredictions(scoringFunction: Callable, Y_true: ArrayOrRuns, Y_pred: ArrayOrRuns) -> numpy.ndarray:
	"""
	Score the predictions per target over all runs. The scoring functions drop non-finite pairs
	themselves; a score the function declines to compute is NaN.

	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) on two 1-D arrays
	:param Y_true:	true data, an array or a list of runs
	:param Y_pred:	predicted data with the same layout as Y_true
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
