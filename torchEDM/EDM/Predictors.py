"""
Array-in, array-out predictors.

Inputs are X_train, Y_train and optionally X_test and Y_test (see Setup.PreparePrediction
for the row semantics). Y_pred always has the shape of Y_test (or of Y_train in-sample),
with NaN where no complete state predicts a row. Nothing here knows about time; samples
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
	"""None picks cuda when available; a cuda request without cuda falls back to cpu."""
	if device is None:
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device(device) if isinstance(device, str) else device
	if device.type == 'cuda' and not torch.cuda.is_available():
		return torch.device('cpu')
	return device


def _IsOneDimensionalTarget(Y) -> bool:
	first = Y[0] if IsListOfRuns(Y) else Y
	return numpy.ndim(first) == 1


def _CheckTestTargets(Y_test, inputs: PredictionInputs) -> None:
	runs = AsRuns(Y_test)
	if len(runs) != len(inputs.outputLengths):
		raise ValueError(f'Y_test has {len(runs)} runs but X_test has {len(inputs.outputLengths)}')
	for runIndex, (y, length) in enumerate(zip(runs, inputs.outputLengths)):
		if y.shape[0] != length or y.shape[1] != inputs.numTargets:
			raise ValueError(f'Run {runIndex}: Y_test has shape {y.shape}, expected ({length}, {inputs.numTargets})')


def _FindNeighbors(inputs: PredictionInputs, knn: int, isTieBreakDeterministic: bool, device, dtype):
	"""
	Distances from every training state to every test state, exclusions applied, and the
	knn nearest per test state.
	:return: neighborDistances [knn, nTest], neighborIndices [knn, nTest], trainStates, testStates (tensors)
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
	Nearest-neighbor weighted-average prediction (Sugihara & May 1990).

	Each test state's knn nearest training states vote on the target with weights
	exp(-d / dNearest); a NaN target among the neighbors makes the prediction NaN.

	:param X_train:		[nTrain, nFeatures] or a list of runs
	:param Y_train:		[nTrain, nTargets] (1-D for one target) or a list of runs
	:param X_test:		[nTest, nFeatures] or a list of runs; None predicts the training rows in-sample
	:param Y_test:		matching Y_train's layout; when given, the result carries a score per target
	:param embedDimensions:	copies of each feature column in the state; 1 uses the columns as given
	:param step:		row offset between copies; negative reaches into the past
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors; 0 means state size plus one
	:param exclusionRadius:	in-sample only; training states this close in rows are not neighbors
	:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
	:param isTieBreakDeterministic:	order exactly tied distances reproducibly (slower than topk)
	:param scoringFunction:	scoringFunction(actual, predicted) -> float, applied per target
	:param device:		torch device; None picks cuda when available
	:param dtype:		torch dtype for the computation
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
	Locally weighted linear prediction (Sugihara 1994).

	Per test state, a linear map with intercept is fitted from the knn nearest training
	states to their targets with weights exp(-theta * d / mean(d)) and applied to the test
	state. A NaN neighbor target drops that neighbor's equation for that target.

	:param knn:		neighbors; 0 means every available training state
	:param theta:	localization; 0 fits one global linear map
	Other parameters as in SimplexPredict.
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
	"""The stacked state of one row, column-major like StackHistory, as [1, stateSize]."""
	lagRows = row + step * numpy.arange(embedDimensions)
	if lagRows.min() < 0 or lagRows.max() >= series.shape[0]:
		raise ValueError(f'Row {row} has no complete history of {embedDimensions} samples at step {step}')
	return series[lagRows, :].T.reshape(1, -1)


def _PrepareGeneration(X_train, embedDimensions, step, knn, isEveryNeighborDefault, device, dtype):
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
	Feed the series forward: the state at its last row predicts the next row of every
	column, the row is appended, and the loop repeats numSteps times. Training states
	come from the given series and stay fixed.

	:param X_train:	[nSamples, nColumns] one series
	:param numSteps:	rows to generate
	:return: [numSteps, nColumns] generated rows
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
	Feed the series forward with the locally weighted linear predictor; see SimplexGenerate.
	knn 0 means every training state.
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
