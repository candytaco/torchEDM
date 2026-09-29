"""
Sweep parameters and score each value on X_train, Y_train and X_test, Y_test with the
predictors' sample semantics (see EDM.Setup): embedding dimensions, prediction horizon, and
localization strength.
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
	"""
	Score only the pairs where both values are finite.

	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData)
	:param Y_true:	true data, [n]
	:param Y_pred:	predicted data, [n]
	:return: the performance, or None when the metric declines
	"""
	Y_true, Y_pred = _FilterNonFinite(Y_true, Y_pred)
	return scoringFunction(Y_true, Y_pred)


def _ScoreColumns(scoringFunction, isScoringFinitePairsOnly, Y_true, Y_pred) -> numpy.ndarray:
	"""
	Apply the performance metric per target; a declined score is NaN.

	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData)
	:param isScoringFinitePairsOnly:	True drops non-finite pairs before scoring
	:param Y_true:	true data, [n, nTargets]
	:param Y_pred:	predicted data, [n, nTargets]
	:return: [nTargets]
	"""
	scores = []
	for column in range(Y_true.shape[1]):
		if isScoringFinitePairsOnly:
			score = _ScoreFinitePairs(scoringFunction, Y_true[:, column], Y_pred[:, column])
		else:
			score = scoringFunction(Y_true[:, column], Y_pred[:, column])
		scores.append(numpy.nan if score is None else float(score))
	return numpy.array(scores)


def _GatherAtOffset(runs: List[numpy.ndarray], rows: numpy.ndarray, runIds: numpy.ndarray, offset: int) -> numpy.ndarray:
	"""
	Gather Y[position + offset] per state, NaN where the shifted position leaves its run.

	:param runs:	list of [nSamples, nTargets] arrays, one per run
	:param rows:	position of each state within its run, [nStates]
	:param runIds:	run index of each state, [nStates]
	:param offset:	number of samples added to each state's position
	:return: [nStates, nTargets]
	"""
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
	Score the nearest-neighbor prediction at every embedding dimension 1..maxDims by Pearson
	correlation.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] or a list of runs; None makes every input variable predict every input variable
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are scored. None scores the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test unless Y_train is None, and a NaN sample is never scored
	:param maxDims:	largest embedding dimension tested
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param isBatched:	True scores every embedding dimension in one pass on the test states complete at maxDims and the training samples usable there. False scores each embedding dimension on its own complete samples (more samples at smaller embedding dimensions, slower)
	:param isJoint:	True stacks all input variables together to predict each target, giving [nTargets, maxDims]. False makes each input variable alone predict each target, giving [nTargets, nVariables, maxDims]. Self-prediction (Y_train None) is always per variable, [nVariables, nVariables, maxDims]
	:param batchSize:	number of input variables per pass in the per-variable sweeps; None takes all at once
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances and predictions
	:return: the performance, with the leading target axis dropped when there is one target
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
											   exclusionRadius, trainRowMask, isJoint, dtype, batchSize, device):
	"""
	Score every embedding dimension in one pass over the samples complete at maxDims: squared
	distances per variable are accumulated over the lagged copies, so every embedding dimension
	comes from one cumulative sum.

	:param X_train:	the caller's training input data
	:param Y_train:	the caller's training target data
	:param X_test:	the caller's test input data, or None in-sample
	:param Y_test:	the caller's true test data, or None in-sample
	:param maxDims:	largest embedding dimension tested; the samples must be complete at this depth
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param step:	sample offset between consecutive lagged copies
	:param exclusionRadius:	in-sample neighbor exclusion radius in samples
	:param trainRowMask:	optional mask over the usable training samples
	:param isJoint:	True scores all input variables together, False each variable alone
	:param dtype:	torch dtype of the distances and predictions
	:param batchSize:	number of variables per pass in the per-variable sweep; None takes all
	:param device:	torch device; None picks cuda when available
	:return: [nTargets, maxDims] jointly, or [nTargets, nVariables, maxDims] per variable
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

	if isJoint:
		return _BatchedJointPrediction(trainStates, trainTargetTensor, testStates, testTargetTensor, maxDims, nVars, device, dtype, maskTensor)
	return _BatchedSeparatePrediction(trainStates, trainTargetTensor, testStates, testTargetTensor, maxDims, nVars, device, dtype,
									  maskTensor, batchSize)


def _ComputeJointEmbeddingDistances(X_train, X_test, maxDims, nVars, device, dtype):
	"""
	Accumulate the squared distances of all input variables stacked together, one matrix per
	embedding dimension. Stacked dimensions arrive grouped by variable (all lags of variable 0,
	then variable 1, and so on); they are reordered by lag so the cumulative sum at position
	embedDimension * nVars - 1 holds every variable through that embedding dimension.

	:param X_train:	training states grouped by variable, [nTrain, nVars * maxDims]
	:param X_test:	test states grouped the same way, [nTest, nVars * maxDims]
	:param maxDims:	number of lagged copies the states were stacked to
	:param nVars:	number of input variables
	:param device:	torch device
	:param dtype:	torch dtype
	:return: [maxDims, nTrain, nTest]
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
	Accumulate the squared distances per input variable for a batch of variables.

	:param X_train:	training states grouped by variable, [nTrain, nVars * maxDims]
	:param X_test:	test states grouped the same way, [nTest, nVars * maxDims]
	:param maxDims:	number of lagged copies the states were stacked to
	:param batchNumVars:	number of variables in this batch
	:param colStart:	first state dimension of the batch
	:param colEnd:	one past the last state dimension of the batch
	:param device:	torch device
	:param dtype:	torch dtype
	:return: [batchNumVars * maxDims, nTrain, nTest]; matrix v * maxDims + d is variable v through embedding dimension d + 1
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
	Predict each target from all input variables jointly; embedding dimension d uses
	d * nVars + 1 neighbors.

	:param X_train:	training states grouped by variable, [nTrain, nVars * maxDims]
	:param Y_train:	targets of the training states, [nTargets, nTrain] tensor
	:param X_test:	test states grouped the same way, [nTest, nVars * maxDims]
	:param Y_test:	true data for the test states, [nTargets, nTest] tensor
	:param maxDims:	number of lagged copies the states were stacked to
	:param nVars:	number of input variables
	:param device:	torch device
	:param dtype:	torch dtype
	:param maskTensor:	optional boolean [nTrain, nTest] marking the pairs excluded from neighbor search
	:return: correlations, [nTargets, maxDims]
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
	Predict each target from each input variable alone, variables in batches; embedding
	dimension d uses d + 1 neighbors.

	:param X_train:	training states grouped by variable, [nTrain, nVars * maxDims]
	:param Y_train:	targets of the training states, [nTargets, nTrain] tensor
	:param X_test:	test states grouped the same way, [nTest, nVars * maxDims]
	:param Y_test:	true data for the test states, [nTargets, nTest] tensor
	:param maxDims:	number of lagged copies the states were stacked to
	:param nVars:	number of input variables
	:param device:	torch device
	:param dtype:	torch dtype
	:param maskTensor:	optional boolean [nTrain, nTest] marking the pairs excluded from neighbor search
	:param batchSize:	number of variables per batch; None takes all at once
	:return: correlations, [nTargets, nVars, maxDims]
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
	Find, for every variable, the embedding dimension in 1..maxDims at which its own lagged
	history best predicts its future value. This is the embedding dimension to give a source
	variable when it cross-maps a target.

	Variables are processed in batches whose dominant tensor is [sourceBatch, nTrain, nTest].
	Exclusions are pre-applied as inf so they survive the incremental lag accumulation.

	:param X_train:	training data, [nTrain, nVariables] or a list of runs; every variable is both a source of states and its own target
	:param X_test:	test data, [nTest, nVariables] or a list of runs; these samples are scored. None scores the training samples in-sample
	:param maxDims:	largest embedding dimension tested
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param batchSize:	number of variables per batch; raised to fill targetVRAM when that is given
	:param targetVRAM:	memory budget in GB that sizes the batch when given; None uses batchSize as is
	:param hasProgressBar:	True shows a progress bar over the variable batches
	:param device:	torch device; cuda falls back to cpu when unavailable
	:param dtype:	torch dtype of the distances and predictions
	:return: best embedding dimension per variable, [nVariables], counted from 1
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
	Score the nearest-neighbor prediction at every prediction horizon 1..maxHorizon.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted and scored. None scores the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test, and a NaN sample is predicted but not scored
	:param maxHorizon:	largest prediction horizon tested
	:param embedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param knn:	number of nearest neighbors that vote on each prediction; 0 means the state size plus one
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param isBatched:	False refits each horizon on its own training samples with SimplexPredict. True finds the neighbors once on the training states usable at maxHorizon and reuses them at every horizon (faster; the shared training set loses maxHorizon - horizon usable samples at each shorter horizon)
	:param isScoringFinitePairsOnly:	True drops (true, predicted) pairs with a non-finite value before calling scoringFunction; the built-in metrics already do this
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) on two 1-D arrays, applied per target
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: sweep table, [maxHorizon, 1 + nTargets]: per horizon, the horizon itself and then the performance per target
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
	Score every horizon from one neighbor search: training samples usable at the largest
	horizon are usable at every smaller one, so their neighbors and weights are computed once
	and only the gathered targets change per horizon.

	:param X_train:	the caller's training input data
	:param Y_train:	the caller's training target data
	:param X_test:	the caller's test input data, or None in-sample
	:param Y_true:	true data for the test samples (Y_test, or Y_train in-sample)
	:param horizons:	the horizons scored, [nHorizons]
	:param embedDimensions:	number of lagged copies of each input variable in the state
	:param step:	sample offset between consecutive lagged copies
	:param knn:	number of nearest neighbors per prediction; 0 means the state size plus one
	:param exclusionRadius:	in-sample neighbor exclusion radius in samples
	:param trainRowMask:	optional mask over the usable training samples
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData)
	:param isScoringFinitePairsOnly:	True drops non-finite pairs before scoring
	:param isTieBreakDeterministic:	True orders tied neighbor distances by sample position
	:param device:	torch device; None picks cuda when available
	:param dtype:	torch dtype
	:return: performance, [nHorizons, nTargets]
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
	Score the locally weighted linear prediction at every localization strength. Neighbors
	are found once; only the weights change per value.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted and scored. None scores the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test, and a NaN sample is predicted but not scored
	:param theta:	localization strengths tested; None uses 0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9
	:param embedDimensions:	number of lagged copies of each input variable that form a state; 1 uses the variables as given
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param knn:	number of nearest neighbors whose equations enter each local linear fit; 0 means every available training state
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param trainRowMask:	optional boolean mask over the training samples, [nTrain] or a list with one per run; False excludes that sample from serving as a training state
	:param isScoringFinitePairsOnly:	True drops (true, predicted) pairs with a non-finite value before calling scoringFunction; the built-in metrics already do this
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData) on two 1-D arrays, applied per target
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: sweep table, [nTheta, 1 + nTargets]: per localization strength, theta itself and then the performance per target
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
