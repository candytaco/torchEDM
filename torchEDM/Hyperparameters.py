"""
Parameter sweeps scored on X_train/Y_train -> X_test/Y_test with the predictors' row
semantics (see EDM.Setup): embedding dimensions, prediction horizon, and localization.
"""
from typing import Callable, List, Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .EDM._core import (Correlation as TorchCorrelation, batch_simplex_predict, batch_get_simplex_weights,
						ComputeSimplexWeights, ProjectSimplex, ComputeSMapWeights,
						SolveWeightedLinearMap)
from .EDM.Setup import ArrayOrRuns, AsRuns, PreparePrediction, TestTargets, ResolveNeighborCount
from .EDM.Predictors import SimplexPredict, ResolveDevice, _FindNeighbors
from .Scoring import Correlation, _FilterNonFinite


def _ScoreFinitePairs(scoringFunction, Y_true, Y_pred):
	"""Score only the pairs where both values are finite."""
	Y_true, Y_pred = _FilterNonFinite(Y_true, Y_pred)
	return scoringFunction(Y_true, Y_pred)


def _ScoreColumns(scoringFunction, isScoringFinitePairsOnly, Y_true, Y_pred) -> numpy.ndarray:
	"""scoringFunction per target column; a declined score is NaN. :return: [nTargets]"""
	scores = []
	for column in range(Y_true.shape[1]):
		if isScoringFinitePairsOnly:
			score = _ScoreFinitePairs(scoringFunction, Y_true[:, column], Y_pred[:, column])
		else:
			score = scoringFunction(Y_true[:, column], Y_pred[:, column])
		scores.append(numpy.nan if score is None else float(score))
	return numpy.array(scores)


def _GatherAtOffset(runs: List[numpy.ndarray], rows: numpy.ndarray, runIds: numpy.ndarray, offset: int) -> numpy.ndarray:
	"""Y[row + offset] per state, NaN where the shifted row leaves its run. :return: [nStates, nTargets]"""
	out = numpy.full((len(rows), runs[0].shape[1]), numpy.nan)
	for runIndex, y in enumerate(runs):
		inRun = runIds == runIndex
		shifted = rows[inRun] + offset
		isInside = (shifted >= 0) & (shifted < y.shape[0])
		values = numpy.full((inRun.sum(), y.shape[1]), numpy.nan)
		values[isInside] = y[shifted[isInside]]
		out[inRun] = values
	return out


# ---------------------------------------------------------------------------------------
# embedding dimensions
# ---------------------------------------------------------------------------------------

def FindOptimalEmbeddingDimensionality(X_train: ArrayOrRuns,
									   Y_train: Optional[ArrayOrRuns] = None,
									   X_test: Optional[ArrayOrRuns] = None,
									   Y_test: Optional[ArrayOrRuns] = None,
									   maxDims: int = 10,
									   step: int = -1,
									   predictionHorizon: int = 1,
									   exclusionRadius: int = 0,
									   trainRowMask: Optional[ArrayOrRuns] = None,
									   isBatched: bool = True,
									   isJoint: bool = True,
									   batchSize: Optional[int] = None,
									   device = None,
									   dtype: torch.dtype = torch.float32) -> numpy.ndarray:
	"""
	Pearson correlation of the neighbor-averaging prediction at every embedding dimension 1..maxDims.

	:param X_train:		[nTrain, nFeatures] or a list of runs
	:param Y_train:		[nTrain, nTargets] or a list of runs; None makes every X column predict every X column
	:param X_test:		[nTest, nFeatures] or a list of runs; None scores the training rows in-sample
	:param Y_test:		targets for X_test (required with X_test unless Y_train is None)
	:param maxDims:		largest embedding dimension to test
	:param predictionHorizon:	rows between a state and the target it predicts
	:param step:		row offset between stacked copies; negative reaches into the past
	:param exclusionRadius:	in-sample only; training states this close in rows are not neighbors
	:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
	:param isBatched:		True: every embedding dimension shares the test states complete at maxDims and the training rows
						usable there, in one pass. False: each embedding dimension is scored on its own complete rows
						(more rows at smaller embedding dimensions, slower).
	:param isJoint:		True: all X columns stacked together predict each Y column, [nTargets, maxDims].
						False: each X column alone predicts each Y column, [nTargets, nVars, maxDims].
						Self-prediction (Y_train None) is always per column, [nVars, nVars, maxDims].
	:param dtype:		torch dtype for the computation
	:param batchSize:	X columns per pass in the per-column sweeps; None takes all at once
	:param device:		torch device; None picks cuda when available
	:return: scores with the leading target axis dropped when there is one target
	"""
	isSelfPrediction = Y_train is None
	if isSelfPrediction:
		Y_train, Y_test, isJoint = X_train, X_test, False
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score predictions on X_test')

	if isBatched:
		scores = _FindOptimalEmbeddingDimensionalityBatched(
			X_train, Y_train, X_test, Y_test, maxDims, predictionHorizon, step, exclusionRadius,
			trainRowMask, isJoint, dtype, batchSize, device)
	else:
		perEmbedDimension = []
		for embedDimension in range(1, maxDims + 1):
			scores = _FindOptimalEmbeddingDimensionalityBatched(
				X_train, Y_train, X_test, Y_test, embedDimension, predictionHorizon, step, exclusionRadius,
				trainRowMask, isJoint, dtype, batchSize, device)
			perEmbedDimension.append(scores[..., -1])
		scores = numpy.stack(perEmbedDimension, axis = -1)

	scores = numpy.asarray(scores)
	if scores.ndim >= 2 and scores.shape[0] == 1:
		scores = scores[0]
	return scores


def _FindOptimalEmbeddingDimensionalityBatched(X_train, Y_train, X_test, Y_test, maxDims, predictionHorizon, step,
											   exclusionRadius, trainRowMask, joint, dtype, batchSize, device):
	"""
	One pass over the rows complete at maxDims; per-column squared distances are accumulated
	over the stacked history so every embedding dimension comes from one cumulative sum.
	"""
	inputs = PreparePrediction(X_train, Y_train, X_test, maxDims, step, predictionHorizon, exclusionRadius, trainRowMask)
	testTargets = TestTargets(Y_test if X_test is not None else Y_train, inputs)
	isScored = numpy.isfinite(testTargets).all(axis = 1)
	device = ResolveDevice(device)

	trainStates = inputs.trainStates
	testStates = inputs.testStates[isScored]
	trainTargetTensor = torch.as_tensor(inputs.trainTargets, device = device, dtype = dtype).T		# [nTargets, nTrain]
	testTargetTensor = torch.as_tensor(testTargets[isScored], device = device, dtype = dtype).T		# [nTargets, nTest]
	maskTensor = None
	if inputs.exclusionMask is not None:
		maskTensor = torch.as_tensor(inputs.exclusionMask[:, isScored], device = device)
	nVars = trainStates.shape[1] // maxDims

	if joint:
		return _BatchedJointPrediction(trainStates, trainTargetTensor, testStates, testTargetTensor, maxDims, nVars, device, dtype, maskTensor)
	return _BatchedSeparatePrediction(trainStates, trainTargetTensor, testStates, testTargetTensor, maxDims, nVars, device, dtype,
									  maskTensor, batchSize)


def _ComputeJointEmbeddingDistances(X_train, X_test, maxDims, nVars, device, dtype):
	"""
	Cumulative squared distances of all columns stacked together, one matrix per embedding dimension.
	Stacked columns arrive variable-major (all lags of column 0, then column 1, ...); they
	are reordered lag-major so the cumulative sum at position embedDimension * nVars - 1 holds every
	column through that embedding dimension. :return: [maxDims, nTrain, nTest]
	"""
	trainTensor = torch.as_tensor(X_train, device = device, dtype = dtype)
	testTensor = torch.as_tensor(X_test, device = device, dtype = dtype)
	nStacked = trainTensor.shape[1]
	nTrain, nTest = trainTensor.shape[0], testTensor.shape[0]

	distances = torch.zeros(nStacked, nTrain, nTest, device = device, dtype = dtype)
	for column in range(nStacked):
		diff = trainTensor[:, column].unsqueeze(1) - testTensor[:, column].unsqueeze(0)
		distances[column] = diff * diff
	del trainTensor, testTensor

	if nVars > 1:
		lagMajor = [column * maxDims + lag for lag in range(maxDims) for column in range(nVars)]
		distances = distances[lagMajor]
	cumulative = torch.cumsum(distances, dim = 0)
	del distances
	perEmbedDimension = cumulative[[embedDimension * nVars - 1 for embedDimension in range(1, maxDims + 1)]]
	del cumulative
	return perEmbedDimension


def _ComputePerVariableEmbeddingDistances(X_train, X_test, maxDims, batchNumVars, colStart, colEnd, device, dtype):
	"""
	Cumulative squared distances per column for a batch of columns.
	:return: [batchNumVars * maxDims, nTrain, nTest], row v * maxDims + d is column v through embedding dimension d + 1
	"""
	numBatch = batchNumVars * maxDims
	trainTensor = torch.as_tensor(X_train[:, colStart:colEnd], device = device, dtype = dtype)
	testTensor = torch.as_tensor(X_test[:, colStart:colEnd], device = device, dtype = dtype)
	nTrain, nTest = trainTensor.shape[0], testTensor.shape[0]

	distances = torch.zeros(numBatch, nTrain, nTest, device = device, dtype = dtype)
	for column in range(numBatch):
		diff = trainTensor[:, column].unsqueeze(1) - testTensor[:, column].unsqueeze(0)
		distances[column] = diff * diff
	del trainTensor, testTensor

	cumulative = torch.cumsum(distances.view(batchNumVars, maxDims, nTrain, nTest), dim = 1)
	del distances
	return cumulative.view(numBatch, nTrain, nTest)


def _BatchedJointPrediction(X_train, Y_train, X_test, Y_test, maxDims, nVars, device, dtype, maskTensor):
	"""
	All X columns jointly predict each target; embedding dimension d uses d * nVars + 1 neighbors.

	:param X_train:	stacked training states [nTrain, nVars * maxDims], source-major
	:param Y_train:	[nTargets, nTrain] tensor
	:param X_test:	stacked test states [nTest, nVars * maxDims]
	:param Y_test:	[nTargets, nTest] tensor
	:return: [nTargets, maxDims]
	"""
	embeddingDistances = _ComputeJointEmbeddingDistances(X_train, X_test, maxDims, nVars, device, dtype)
	if maskTensor is not None:
		embeddingDistances[:, maskTensor] = float('inf')
	neighborCounts = torch.arange(1, maxDims + 1, device = device, dtype = torch.long) * nVars + 1

	out = torch.zeros(Y_train.shape[0], maxDims, device = device, dtype = dtype)
	for targetIndex in range(Y_train.shape[0]):
		predictions = batch_simplex_predict(embeddingDistances, neighborCounts, Y_train[targetIndex])
		TorchCorrelation(Y_test[targetIndex], predictions, out[targetIndex])

	del embeddingDistances
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return out.cpu().numpy()


def _BatchedSeparatePrediction(X_train, Y_train, X_test, Y_test, maxDims, nVars, device, dtype, maskTensor, batchSize):
	"""
	Each X column alone predicts each target, columns in batches. Arrays as in
	_BatchedJointPrediction.
	:return: [nTargets, nVars, maxDims]
	"""
	nTargets = Y_train.shape[0]
	scores = numpy.zeros((nTargets, nVars, maxDims), dtype = numpy.float32)
	actualBatchSize = batchSize if batchSize is not None else nVars

	for varBatchStart in range(0, nVars, actualBatchSize):
		varBatchEnd = min(varBatchStart + actualBatchSize, nVars)
		batchNumVars = varBatchEnd - varBatchStart
		embeddingDistances = _ComputePerVariableEmbeddingDistances(
			X_train, X_test, maxDims, batchNumVars, varBatchStart * maxDims, varBatchEnd * maxDims, device, dtype)
		if maskTensor is not None:
			embeddingDistances[:, maskTensor] = float('inf')

		# matrix v * maxDims + d is column v at embedding dimension d + 1, predicted with its own d + 2 neighbors
		neighborCounts = torch.arange(2, maxDims + 2, dtype = torch.long, device = device).repeat(batchNumVars)
		out = torch.zeros(nTargets, batchNumVars * maxDims, device = device, dtype = dtype)
		for targetIndex in range(nTargets):
			predictions = batch_simplex_predict(embeddingDistances, neighborCounts, Y_train[targetIndex])
			TorchCorrelation(Y_test[targetIndex], predictions, out[targetIndex])
		scores[:, varBatchStart:varBatchEnd, :] = out.cpu().numpy().reshape(nTargets, batchNumVars, maxDims)

		del embeddingDistances, out
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return scores


def FindSelfPredictionEmbeddingDimension(X_train: ArrayOrRuns,
										  X_test: Optional[ArrayOrRuns] = None,
										  maxDims: int = 10,
										  step: int = -1,
										  predictionHorizon: int = 1,
										  exclusionRadius: int = 0,
										  trainRowMask: Optional[ArrayOrRuns] = None,
										  batchSize: int = 1000,
										  targetVRAM: Optional[float] = None,
										  hasProgressBar: bool = True,
										  device = 'cuda',
										  dtype: torch.dtype = torch.float16) -> numpy.ndarray:
	"""
	For every column, the embedding dimension in 1..maxDims at which its own stacked history best
	predicts its future value. This is the embedding dimension to give a source column when it cross-maps
	a target.

	Columns are processed in batches whose dominant tensor is [sourceBatch, nTrain, nTest].
	Exclusions are pre-applied as inf so they survive the incremental lag accumulation.

	:param X_train:		[nTrain, nColumns] or a list of runs
	:param X_test:		[nTest, nColumns] or a list of runs; None scores the training rows in-sample
	:param batchSize:	columns per batch (auto-raised to fill targetVRAM when given)
	:param targetVRAM:	GB budget; None uses batchSize as is
	:param hasProgressBar:	show a progress bar
	:return: [nColumns] best embedding dimension per column, 1-based
	"""
	inputs = PreparePrediction(X_train, X_train, X_test, maxDims, step, predictionHorizon, exclusionRadius, trainRowMask)
	targets = TestTargets(X_test if X_test is not None else X_train, inputs)
	isScored = numpy.isfinite(targets).all(axis = 1)
	torchDevice = ResolveDevice(device)

	numSources = inputs.numTargets
	numTrain = inputs.numTrainingPairs
	numTest = int(isScored.sum())
	exclusionMask = None if inputs.exclusionMask is None else inputs.exclusionMask[:, isScored]

	scores = numpy.zeros([numSources, maxDims], dtype = numpy.float32)

	# batch size from the two coexisting [sourceBatch, numTrain, numTest] tensors
	elementSize = torch.zeros(1, dtype = dtype).element_size()
	if targetVRAM is None:
		sourceBatchSize = batchSize
	else:
		vramBatchSize = max(1, int(targetVRAM * 1e9 / (2 * numTrain * numTest * elementSize)))
		sourceBatchSize = max(batchSize, vramBatchSize)

	yTrain = torch.as_tensor(inputs.trainTargets, dtype = dtype, device = torchDevice)		# [numTrain, numSources]
	yTest = torch.as_tensor(targets[isScored], dtype = dtype, device = torchDevice)		# [numTest, numSources]
	trainStates = inputs.trainStates
	testStates = inputs.testStates[isScored]

	for sourceBatchStart in ProgressBar(range(0, numSources, sourceBatchSize), desc = 'Embedding dim search',
										leave = False, disable = not hasProgressBar):
		sourceBatchEnd = min(sourceBatchStart + sourceBatchSize, numSources)
		actualSourceBatchSize = sourceBatchEnd - sourceBatchStart

		# stacked histories per source: [sourceBatch, numTrain/numTest, maxDims]
		trainEmbeddings = torch.as_tensor(
			trainStates[:, sourceBatchStart * maxDims:sourceBatchEnd * maxDims].reshape(numTrain, actualSourceBatchSize, maxDims),
			dtype = dtype, device = torchDevice).permute(1, 0, 2).contiguous()
		testEmbeddings = torch.as_tensor(
			testStates[:, sourceBatchStart * maxDims:sourceBatchEnd * maxDims].reshape(numTest, actualSourceBatchSize, maxDims),
			dtype = dtype, device = torchDevice).permute(1, 0, 2).contiguous()

		cumulativeDistances = torch.zeros([actualSourceBatchSize, numTrain, numTest], dtype = dtype, device = torchDevice)
		if exclusionMask is not None:
			cumulativeDistances[:, torch.as_tensor(exclusionMask, device = torchDevice)] = float('inf')

		yTrainBatch = yTrain[:, sourceBatchStart:sourceBatchEnd].T.contiguous()	# [sourceBatch, numTrain]
		yTestBatch = yTest[:, sourceBatchStart:sourceBatchEnd].T.contiguous()		# [sourceBatch, numTest]

		for embedDimIndex in range(maxDims):
			numKnn = embedDimIndex + 2

			lagDiffs = trainEmbeddings[:, :, embedDimIndex].unsqueeze(2) - testEmbeddings[:, :, embedDimIndex].unsqueeze(1)
			lagDiffs.square_()
			cumulativeDistances.add_(lagDiffs)
			del lagDiffs

			neighborIndices, neighborWeights = batch_get_simplex_weights(cumulativeDistances, numKnn)
			# source i predicts itself: gather yTrainBatch[i] with neighborIndices[i]
			flatNeighborIndices = neighborIndices.reshape(actualSourceBatchSize, -1)
			selectedTargets = yTrainBatch.gather(1, flatNeighborIndices).reshape(actualSourceBatchSize, numKnn, numTest)
			del flatNeighborIndices
			predictions = (neighborWeights * selectedTargets).sum(dim = 1)	# [sourceBatch, numTest]
			del selectedTargets

			targetCentered = yTestBatch - yTestBatch.mean(dim = 1, keepdim = True)
			predCentered = predictions - predictions.mean(dim = 1, keepdim = True)
			del predictions
			targetStd = torch.sqrt((targetCentered ** 2).sum(dim = 1))
			predStd = torch.sqrt((predCentered ** 2).sum(dim = 1))

			isValid = (targetStd > 0) & (predStd > 0)
			batchScores = torch.zeros(actualSourceBatchSize, dtype = dtype, device = torchDevice)
			if isValid.any():
				batchScores[isValid] = ((targetCentered[isValid] * predCentered[isValid]).sum(dim = 1) /
										(targetStd[isValid] * predStd[isValid]))
			scores[sourceBatchStart:sourceBatchEnd, embedDimIndex] = batchScores.cpu().float().numpy()

		del trainEmbeddings, testEmbeddings, cumulativeDistances, yTrainBatch, yTestBatch

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return numpy.argmax(scores, axis = 1) + 1


# ---------------------------------------------------------------------------------------
# prediction horizon
# ---------------------------------------------------------------------------------------

def FindOptimalPredictionHorizon(X_train: ArrayOrRuns,
								 Y_train: ArrayOrRuns,
								 X_test: Optional[ArrayOrRuns] = None,
								 Y_test: Optional[ArrayOrRuns] = None,
								 maxHorizon: int = 10,
								 embedDimensions: int = 1,
								 step: int = -1,
								 knn: int = 0,
								 exclusionRadius: int = 0,
								 trainRowMask: Optional[ArrayOrRuns] = None,
								 isBatched: bool = False,
								 isScoringFinitePairsOnly: bool = False,
								 isTieBreakDeterministic: bool = False,
								 scoringFunction: Callable = Correlation,
								 device = None,
								 dtype: torch.dtype = torch.float64) -> numpy.ndarray:
	"""
	Score of the neighbor-averaging prediction at every horizon 1..maxHorizon.

	:param isBatched:		False (default): each horizon is refitted on its own training rows with
						SimplexPredict. True: neighbors are found once on the training states usable at
						maxHorizon and reused at every horizon (faster; the shared training set loses
						maxHorizon - horizon usable rows at each shorter horizon).
	:param scoringFunction:	scoringFunction(actual, predicted) -> float, applied per target
	:param isScoringFinitePairsOnly:	drop pairs with a non-finite value before scoring
	Other parameters as in SimplexPredict.
	:return: [maxHorizon, 1 + nTargets], the horizon in column 0
	"""
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score predictions on X_test')
	horizons = numpy.arange(1, maxHorizon + 1)
	Y_true = Y_test if X_test is not None else Y_train

	if isBatched:
		scores = _FindOptimalPredictionHorizonBatched(
			X_train, Y_train, X_test, Y_true, horizons, embedDimensions, step, knn, exclusionRadius,
			trainRowMask, scoringFunction, isScoringFinitePairsOnly, isTieBreakDeterministic, device, dtype)
	else:
		scorer = scoringFunction
		if isScoringFinitePairsOnly:
			scorer = lambda actual, predicted: _ScoreFinitePairs(scoringFunction, actual, predicted)
		scores = [SimplexPredict(X_train, Y_train, X_test, Y_true, embedDimensions, step, int(horizon), knn,
								 exclusionRadius, trainRowMask, isTieBreakDeterministic, scorer, device, dtype).score
				  for horizon in horizons]
	return numpy.column_stack([horizons, numpy.array(scores)])


def _FindOptimalPredictionHorizonBatched(X_train, Y_train, X_test, Y_true, horizons, embedDimensions, step, knn,
										 exclusionRadius, trainRowMask, scoringFunction, isScoringFinitePairsOnly,
										 isTieBreakDeterministic, device, dtype):
	"""
	Training rows usable at the largest horizon are usable at every smaller one, so their
	neighbors and weights are computed once; only the gathered targets change per horizon.
	"""
	maxHorizon = int(numpy.max(horizons))
	inputs = PreparePrediction(X_train, Y_train, X_test, embedDimensions, step, maxHorizon, exclusionRadius, trainRowMask)
	knn = ResolveNeighborCount(knn, inputs)
	device = ResolveDevice(device)
	neighborDistances, neighborIndices, _, _ = _FindNeighbors(inputs, knn, isTieBreakDeterministic, device, dtype)
	weights = ComputeSimplexWeights(neighborDistances)

	yTrainRuns = AsRuns(Y_train)
	yTrueRuns = AsRuns(Y_true)
	scores = []
	for horizon in horizons:
		trainTargets = torch.as_tensor(_GatherAtOffset(yTrainRuns, inputs.trainRows, inputs.trainRuns, int(horizon)),
									   device = device, dtype = dtype)
		testTargets = _GatherAtOffset(yTrueRuns, inputs.testRows, inputs.testRuns, int(horizon))
		predictions, _ = ProjectSimplex(weights, trainTargets[neighborIndices])
		scores.append(_ScoreColumns(scoringFunction, isScoringFinitePairsOnly, testTargets, predictions.cpu().numpy()))
	return numpy.array(scores)


# ---------------------------------------------------------------------------------------
# localization
# ---------------------------------------------------------------------------------------

def FindSMapNeighborhood(X_train: ArrayOrRuns,
						 Y_train: ArrayOrRuns,
						 X_test: Optional[ArrayOrRuns] = None,
						 Y_test: Optional[ArrayOrRuns] = None,
						 theta: Optional[List[float]] = None,
						 embedDimensions: int = 1,
						 step: int = -1,
						 predictionHorizon: int = 1,
						 knn: int = 0,
						 exclusionRadius: int = 0,
						 trainRowMask: Optional[ArrayOrRuns] = None,
						 isScoringFinitePairsOnly: bool = False,
						 isTieBreakDeterministic: bool = False,
						 scoringFunction: Callable = Correlation,
						 device = None,
						 dtype: torch.dtype = torch.float64) -> numpy.ndarray:
	"""
	Score of the locally weighted linear prediction at every localization value. Neighbors
	are found once; only the weights change per value.

	:param theta:	localization values; None uses 0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9
	:param knn:		neighbors; 0 means every available training state
	Other parameters as in SMapPredict and FindOptimalPredictionHorizon.
	:return: [nTheta, 1 + nTargets], theta in column 0
	"""
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score predictions on X_test')
	if theta is None:
		theta = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
	thetaValues = numpy.asarray(theta, dtype = float)
	Y_true = Y_test if X_test is not None else Y_train

	inputs = PreparePrediction(X_train, Y_train, X_test, embedDimensions, step, predictionHorizon, exclusionRadius, trainRowMask)
	knn = ResolveNeighborCount(knn, inputs, isEveryNeighborDefault = True)
	device = ResolveDevice(device)
	neighborDistances, neighborIndices, trainStates, testStates = _FindNeighbors(
		inputs, knn, isTieBreakDeterministic, device, dtype)
	trainTargets = torch.as_tensor(inputs.trainTargets, device = device, dtype = dtype)
	neighborsByTest = neighborIndices.t()
	neighborStates = trainStates[neighborsByTest]
	neighborTargets = trainTargets[neighborsByTest]
	testTargets = TestTargets(Y_true, inputs)

	scores = []
	for value in thetaValues:
		weights = ComputeSMapWeights(neighborDistances, float(value)).t()
		_, predictions, _, _ = SolveWeightedLinearMap(weights, neighborStates, neighborTargets, testStates)
		scores.append(_ScoreColumns(scoringFunction, isScoringFinitePairsOnly, testTargets, predictions.cpu().numpy()))

	del neighborDistances, neighborIndices, trainStates, testStates, trainTargets, neighborStates, neighborTargets
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return numpy.column_stack([thetaValues, numpy.array(scores)])
