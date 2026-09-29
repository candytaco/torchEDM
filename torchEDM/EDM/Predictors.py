"""
Predict from arrays and return arrays.

Inputs are X_train, Y_train and optionally X_test and Y_test (see Setup.PreparePrediction
for the sample semantics). Y_pred always has the shape of Y_test (or of Y_train in-sample),
with NaN where no complete state predicts a sample. Nothing here knows about time; samples
are assumed evenly spaced.
"""
from typing import Callable, List, Optional, Union

import numpy
import torch

from ._core import (ComputePairwiseDistances, SelectNearestNeighbors, ComputeSimplexWeights, ProjectSimplex,
					ComputeSMapWeights, SolveWeightedLinearMap)
from .Setup import (ArrayOrRuns, AsRuns, IsListOfRuns, PreparePrediction, PredictionInputs, ResolveNeighborCount,
					ScatterPredictions, MatchTargetLayout, ScorePredictions, BuildTrainingPairs)
from .Results import SimplexResult, SMapResult
from ..Scoring import Correlation

ArrayOrList = Union[numpy.ndarray, List[numpy.ndarray]]


def ResolveDevice(device) -> torch.device:
	"""
	Return the torch device a caller asked for.

	:param device:	a torch.device, a device string, or None; None picks cuda when available, and a cuda request without cuda falls back to cpu
	:return: torch.device
	"""
	if device is None:
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device(device) if isinstance(device, str) else device
	if device.type == 'cuda' and not torch.cuda.is_available():
		return torch.device('cpu')
	return device


def _IsOneDimensionalTarget(Y) -> bool:
	"""
	Return whether Y was given without a target axis: a 1-D array, or a list of 1-D runs.

	:param Y:	Y_train or Y_test as the caller passed it
	"""
	first = Y[0] if IsListOfRuns(Y) else Y
	return numpy.ndim(first) == 1


def _CheckTestTargets(Y_test, inputs: PredictionInputs) -> None:
	"""
	Raise ValueError unless Y_test has one run per test run, each with that run's number of
	samples and the training targets' number of variables.

	:param Y_test:	the caller's Y_test
	:param inputs:	the PredictionInputs built from X_test
	"""
	runs = AsRuns(Y_test)
	if len(runs) != len(inputs.outputLengths):
		raise ValueError(f'Y_test has {len(runs)} runs but X_test has {len(inputs.outputLengths)}')
	for runIndex, (y, length) in enumerate(zip(runs, inputs.outputLengths)):
		if y.shape[0] != length or y.shape[1] != inputs.numTargets:
			raise ValueError(f'Run {runIndex}: Y_test has shape {y.shape}, expected ({length}, {inputs.numTargets})')


def _FindNeighbors(inputs: PredictionInputs, knn: int, isTieBreakDeterministic: bool, device, dtype):
	"""
	Compute the distances from every training state to every test state, apply the exclusions,
	and select the knn nearest neighbors per test state.

	:param inputs:	the PredictionInputs holding the states and the exclusion mask
	:param knn:	number of neighbors kept per test state
	:param isTieBreakDeterministic:	True orders exactly tied distances by sample position (in-sample, by proximity first) instead of leaving it to torch.topk
	:param device:	torch device the tensors are moved to
	:param dtype:	torch dtype of the state tensors and distances
	:return: neighborDistances [knn, nTest], neighborIndices [knn, nTest] into the training states, and trainStates, testStates as tensors on the device
	"""
	trainStates = torch.as_tensor(inputs.trainStates, device = device, dtype = dtype)
	testStates = torch.as_tensor(inputs.testStates, device = device, dtype = dtype)
	distances = ComputePairwiseDistances(trainStates, testStates)
	if inputs.exclusionMask is not None:
		distances[torch.as_tensor(inputs.exclusionMask, device = device)] = float('inf')
	trainRows = testRows = None
	if isTieBreakDeterministic and inputs.isInSample:
		trainRows, testRows = inputs.SharedAxisRows()
	neighborDistances, neighborIndices = SelectNearestNeighbors(distances, knn, isTieBreakDeterministic, trainRows, testRows)
	return neighborDistances, neighborIndices, trainStates, testStates


def SimplexPredict(X_train: ArrayOrRuns, Y_train: ArrayOrRuns,
				   X_test: Optional[ArrayOrRuns] = None, Y_test: Optional[ArrayOrRuns] = None,
				   embedDimensions: int = 1, step: int = -1, predictionHorizon: int = 1, knn: int = 0,
				   exclusionRadius: int = 0, trainRowMask: Optional[ArrayOrRuns] = None,
				   isTieBreakDeterministic: bool = False, scoringFunction: Callable = Correlation,
				   device = None, dtype: torch.dtype = torch.float64) -> SimplexResult:
	"""
	Predict each test sample as the weighted average of the targets of its nearest training
	states (Sugihara & May 1990). The knn nearest training states of each test state vote
	with weights exp(-d / dNearest); a NaN target among the neighbors makes the prediction NaN.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train; NaN targets are allowed
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; when given, the result carries a performance score per target, and a NaN sample is predicted but not scored
	:param embedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param knn:	number of nearest neighbors that vote on each prediction; 0 means the state size plus one
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) on two 1-D arrays, applied per target to the finite pairs
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: SimplexResult holding Y_pred (same shape as Y_test, or Y_train in-sample), the weighted variance of the neighbor targets around each prediction, and the performance per target when Y_test was given
	"""
	inputs = PreparePrediction(X_train, Y_train, X_test, embedDimensions, step, predictionHorizon,
							   exclusionRadius, trainRowMask)
	if Y_test is not None:
		_CheckTestTargets(Y_test, inputs)
	knn = ResolveNeighborCount(knn, inputs)
	device = ResolveDevice(device)

	neighborDistances, neighborIndices, _, _ = _FindNeighbors(inputs, knn, isTieBreakDeterministic, device, dtype)
	trainTargets = torch.as_tensor(inputs.trainTargets, device = device, dtype = dtype)
	weights = ComputeSimplexWeights(neighborDistances)
	predictions, variance = ProjectSimplex(weights, trainTargets[neighborIndices])

	isOneDimensional = _IsOneDimensionalTarget(Y_test if Y_test is not None else Y_train)
	Y_pred = MatchTargetLayout(ScatterPredictions(predictions.cpu().numpy(), inputs), isOneDimensional)
	variance = MatchTargetLayout(ScatterPredictions(variance.cpu().numpy(), inputs), isOneDimensional)
	score = ScorePredictions(scoringFunction, Y_test, Y_pred) if Y_test is not None else None
	return SimplexResult(Y_pred = Y_pred, variance = variance, score = score,
						 embedDimensions = embedDimensions, predictionHorizon = predictionHorizon, knn = knn)


def SMapPredict(X_train: ArrayOrRuns, Y_train: ArrayOrRuns,
				X_test: Optional[ArrayOrRuns] = None, Y_test: Optional[ArrayOrRuns] = None,
				embedDimensions: int = 1, step: int = -1, predictionHorizon: int = 1, knn: int = 0,
				theta: float = 0.0, exclusionRadius: int = 0, trainRowMask: Optional[ArrayOrRuns] = None,
				isTieBreakDeterministic: bool = False, scoringFunction: Callable = Correlation,
				device = None, dtype: torch.dtype = torch.float64) -> SMapResult:
	"""
	Predict each test sample with a linear map fitted locally to its nearest training states
	(Sugihara 1994). Per test state, a linear map with intercept is fitted from the knn nearest
	training states to their targets with weights exp(-theta * d / mean(d)) and applied to
	the test state. A NaN neighbor target drops that neighbor's equation for that target; a
	target with no finite neighbor target is predicted as NaN.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train; NaN targets are allowed
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; when given, the result carries a performance score per target, and a NaN sample is predicted but not scored
	:param embedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param knn:	number of nearest neighbors whose equations enter each local linear fit; 0 means every available training state
	:param theta:	localization strength; 0 fits one global linear map, larger values weight near neighbors more
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) on two 1-D arrays, applied per target to the finite pairs
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: SMapResult holding Y_pred (same shape as Y_test), the coefficients of each sample's fit [nSamples, stateSize + 1, nTargets] with the intercept first, the singular values of each weighted design matrix, the weighted residual variance, and the performance per target when Y_test was given
	"""
	inputs = PreparePrediction(X_train, Y_train, X_test, embedDimensions, step, predictionHorizon,
							   exclusionRadius, trainRowMask)
	if Y_test is not None:
		_CheckTestTargets(Y_test, inputs)
	knn = ResolveNeighborCount(knn, inputs, isEveryNeighborDefault = True)
	device = ResolveDevice(device)

	neighborDistances, neighborIndices, trainStates, testStates = _FindNeighbors(
		inputs, knn, isTieBreakDeterministic, device, dtype)
	trainTargets = torch.as_tensor(inputs.trainTargets, device = device, dtype = dtype)
	neighborsByTest = neighborIndices.t()	# [nTest, knn]
	weights = ComputeSMapWeights(neighborDistances, theta).t()
	coefficients, predictions, variance, singularValues = SolveWeightedLinearMap(
		weights, trainStates[neighborsByTest], trainTargets[neighborsByTest], testStates)

	isOneDimensional = _IsOneDimensionalTarget(Y_test if Y_test is not None else Y_train)
	Y_pred = MatchTargetLayout(ScatterPredictions(predictions.cpu().numpy(), inputs), isOneDimensional)
	variance = MatchTargetLayout(ScatterPredictions(variance.cpu().numpy(), inputs), isOneDimensional)
	coefficients = MatchTargetLayout(ScatterPredictions(coefficients.cpu().numpy(), inputs), isOneDimensional)
	singularValues = MatchTargetLayout(ScatterPredictions(singularValues.cpu().numpy(), inputs), isOneDimensional)
	score = ScorePredictions(scoringFunction, Y_test, Y_pred) if Y_test is not None else None
	return SMapResult(Y_pred = Y_pred, variance = variance, coefficients = coefficients,
					  singularValues = singularValues, score = score,
					  embedDimensions = embedDimensions, predictionHorizon = predictionHorizon, knn = knn, theta = theta)


def _StateAtRow(series: numpy.ndarray, row: int, embedDimensions: int, step: int) -> numpy.ndarray:
	"""
	Build the stacked state of one sample, grouped by variable like StackHistory.

	:param series:	the series, [nSamples, nVariables]
	:param row:	position of the sample whose state is built
	:param embedDimensions:	number of lagged copies of each variable in the state
	:param step:	sample offset between consecutive lagged copies
	:return: [1, nVariables * embedDimensions]; raises ValueError when a lag leaves the series
	"""
	lagRows = row + step * numpy.arange(embedDimensions)
	if lagRows.min() < 0 or lagRows.max() >= series.shape[0]:
		raise ValueError(f'Row {row} has no complete history of {embedDimensions} samples at step {step}')
	return series[lagRows, :].T.reshape(1, -1)


def _PrepareGeneration(X_train, embedDimensions, step, knn, isEveryNeighborDefault, device, dtype):
	"""
	Build the fixed training set for feeding a series forward: every complete state of the
	series paired with the sample after it.

	:param X_train:	one series, [nSamples, nVariables]; a list of one run is accepted
	:param embedDimensions:	number of lagged copies of each variable in the state
	:param step:	sample offset between consecutive lagged copies
	:param knn:	the requested number of neighbors; 0 or None asks for the default
	:param isEveryNeighborDefault:	False makes the default the state size plus one; True every training state
	:param device:	torch device; None picks cuda when available
	:param dtype:	torch dtype of the state and target tensors
	:return: (series as a 2-D array, positions of the training states, knn, device, trainStates [nStates, stateSize], trainTargets [nStates, nVariables]), the last two as tensors on the device
	"""
	runs = AsRuns(X_train)
	if len(runs) != 1:
		raise ValueError('Generation continues a single series')
	X = runs[0]
	states, targets, rows = BuildTrainingPairs(X, X, embedDimensions, step, 1)
	if len(rows) == 0:
		raise ValueError('No usable training states: the series is shorter than its history')
	if knn is None or knn <= 0:
		knn = len(rows) if isEveryNeighborDefault else states.shape[1] + 1
	if knn > len(rows):
		raise ValueError(f'knn = {knn} but only {len(rows)} training states are available')
	device = ResolveDevice(device)
	return (X, rows, int(knn), device,
			torch.as_tensor(states, device = device, dtype = dtype),
			torch.as_tensor(targets, device = device, dtype = dtype))


def SimplexGenerate(X_train: ArrayOrRuns, numSteps: int, embedDimensions: int = 1, step: int = -1, knn: int = 0,
					isTieBreakDeterministic: bool = False, device = None,
					dtype: torch.dtype = torch.float64) -> numpy.ndarray:
	"""
	Feed the series forward with the nearest-neighbor predictor: the state at the last sample
	predicts the next value of every variable, the new sample is appended, and the loop repeats
	numSteps times. Training states come from the given series and stay fixed.

	:param X_train:	one series, [nSamples, nVariables]
	:param numSteps:	number of samples to generate
	:param embedDimensions:	number of lagged copies of each variable in the state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param knn:	number of nearest neighbors that vote on each generated sample; 0 means the state size plus one
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: generated samples, [numSteps, nVariables]
	"""
	X, rows, knn, device, trainStates, trainTargets = _PrepareGeneration(
		X_train, embedDimensions, step, knn, False, device, dtype)
	nRows = X.shape[0]
	series = numpy.concatenate([X, numpy.full((numSteps, X.shape[1]), numpy.nan)])
	for generated in range(numSteps):
		lastRow = nRows - 1 + generated
		state = torch.as_tensor(_StateAtRow(series, lastRow, embedDimensions, step), device = device, dtype = dtype)
		distances = ComputePairwiseDistances(trainStates, state)
		neighborDistances, neighborIndices = SelectNearestNeighbors(
			distances, knn, isTieBreakDeterministic,
			rows if isTieBreakDeterministic else None, [lastRow] if isTieBreakDeterministic else None)
		predictions, _ = ProjectSimplex(ComputeSimplexWeights(neighborDistances), trainTargets[neighborIndices])
		series[nRows + generated] = predictions[0].cpu().numpy()
	return series[nRows:]


def SMapGenerate(X_train: ArrayOrRuns, numSteps: int, embedDimensions: int = 1, step: int = -1, knn: int = 0,
				 theta: float = 0.0, isTieBreakDeterministic: bool = False, device = None,
				 dtype: torch.dtype = torch.float64) -> numpy.ndarray:
	"""
	Feed the series forward with the locally linear predictor: the state at the last sample
	predicts the next value of every variable, the new sample is appended, and the loop repeats
	numSteps times. Training states come from the given series and stay fixed.

	:param X_train:	one series, [nSamples, nVariables]
	:param numSteps:	number of samples to generate
	:param embedDimensions:	number of lagged copies of each variable in the state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param knn:	number of nearest neighbors whose equations enter each local linear fit; 0 means every training state
	:param theta:	localization strength; 0 fits one global linear map, larger values weight near neighbors more
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: generated samples, [numSteps, nVariables]
	"""
	X, rows, knn, device, trainStates, trainTargets = _PrepareGeneration(
		X_train, embedDimensions, step, knn, True, device, dtype)
	nRows = X.shape[0]
	series = numpy.concatenate([X, numpy.full((numSteps, X.shape[1]), numpy.nan)])
	for generated in range(numSteps):
		lastRow = nRows - 1 + generated
		state = torch.as_tensor(_StateAtRow(series, lastRow, embedDimensions, step), device = device, dtype = dtype)
		distances = ComputePairwiseDistances(trainStates, state)
		neighborDistances, neighborIndices = SelectNearestNeighbors(
			distances, knn, isTieBreakDeterministic,
			rows if isTieBreakDeterministic else None, [lastRow] if isTieBreakDeterministic else None)
		neighborsByTest = neighborIndices.t()
		_, predictions, _, _ = SolveWeightedLinearMap(
			ComputeSMapWeights(neighborDistances, theta).t(),
			trainStates[neighborsByTest], trainTargets[neighborsByTest], state)
		series[nRows + generated] = predictions[0].cpu().numpy()
	return series[nRows:]
