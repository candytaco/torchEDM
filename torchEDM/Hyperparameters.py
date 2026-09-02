from typing import List, Tuple, Any, Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .EDM._core import Correlation, batch_simplex_predict, batch_get_simplex_weights
from .EDM.utils import BuildEmbeddingIndices, build_exclusion_mask, MakeDelays
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex
from .Utils import IsNonStringIterable

# TODO: these should all be cross-validated


def FindOptimalEmbeddingDimensionality(X: numpy.ndarray,
									   Y: Optional[numpy.ndarray] = None,
									   maxDims: int = 10,
									   train: List[Tuple[int, int]] = None,
									   test: List[Tuple[int, int]] = None,
									   predictionHorizon: int = 1,
									   step: int = -1,
									   exclusionRadius: float = 0,
									   embedded: bool = False,
									   validLib: List = [],
									   ignoreNan: bool = True,
									   batched: bool = True,
									   scoring_function = Correlation,
									   joint: bool = True,
									   dtype: torch.dtype = torch.float32,
									   BatchSize: Optional[int] = None):
	"""
	Estimate optimal embedding dimension for simplex. When Y is not provided, each X variable
	separately predicts every X variable, returning a [nVars, nVars, maxDims] score array.
	When Y is provided and joint is True, all X variables are used jointly to predict each Y variable.
	When joint is False, each X variable separately predicts each Y variable.

	When batched = False, each train and test indices are computed per embedding dimensionality. When batched = True,
	the indices are computed from the maximum, which is the most restrictive, which enables
	shared distance precomputation but slightly penalizes lower dimensions values
	by excluding a few extra rows.

	:param X: 					2D numpy array of predictor columns, shape (N, numFeatures)
	:param Y: 					1D or 2D numpy array of target values, shape (N,) or (N, 1)
	:param maxDims: 			maximum number of embedding dimensions to test
	:param train: 				Train indices as list of (start, end) pairs
	:param test: 				Test indices as list of (start, end) pairs
	:param predictionHorizon: 	Prediction horizon
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param ignoreNan: 			Whether to ignore NaN values
	:param batched: 			True (default): all dimensions share the most restrictive (maxDims) row set in one
								batched pass. False: each dimension uses its own valid rows (exact, slower; will be
								deprecated)
	:param scoring_function: 	Scoring function taking (actual, predicted) and returning a scalar
	:param joint:				when X is 2D, use all vars together to predict Y? If False, each X is used separately to predict Y
	:param dtype:				Torch dtype for tensors (e.g. torch.float32 or torch.float16)
	:param BatchSize:			Number of variables to process per batch (non-joint/self-prediction only); None processes all at once
	:return: score for each embedding dimension
	"""

	# force column vectors
	if len(X.shape) < 2:
		X = X[:, None]
	if Y is not None and len(Y.shape) < 2:
		Y = Y[:, None]

	# TODO: this needs to be refactored for some things because the sub-calls are growing too long
	# we should be able to accomodate these options:
	# parallel process all embedding dimensions vs sequentially process all embedding dimensions
	# each X separately predict Y vs all Xs jointly predict Y
	#	when Xs are separate, we can sequentially process embedding dimensions, but for each dimension, parallel batch across X
	#	when Xs are joint, they only produce one matrix
	# when X and Y are provided vs only when X are provided
	#	when X and Y are provided, X vars are used to predict Y vars
	#	when only X is provided, each X is used to predict every other X (this collapses the previous choice to no joint)
	# these options can be handled by separate functions that branch, but there should only be one level of branching in
	# each function (currently the Batched variant has two levels of nested branching).
	# these options should all make use of the batch_simplex* methods in the _core.py file

	if batched:
		scores = _FindOptimalEmbeddingDimensionalityBatched(
			X, Y, maxDims, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, ignoreNan, scoring_function, joint,
			dtype = dtype, batchSize = BatchSize)
	else:
		scores = _FindOptimalEmbeddingDimensionalityIterative(
			X, Y, maxDims, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, ignoreNan, scoring_function, joint,
			dtype = dtype, batchSize = BatchSize)

	# Squeeze the target axis for single-target calls so the original 1D return
	# shape is preserved.
	scores = numpy.asarray(scores)
	if scores.ndim >= 2 and scores.shape[0] == 1:
		scores = scores[0]
	return scores


def FindSelfPredictionEmbeddingDimension(X,
										  maxDims: int = 10,
										  train: List[Tuple[int, int]] = None,
										  test: List[Tuple[int, int]] = None,
										  predictionHorizon: int = 1,
										  step: int = -1,
										  exclusionRadius: float = 0,
										  embedded: bool = False,
										  validLib: List = [],
										  dtype: torch.dtype = torch.float16,
										  device = 'cuda',
										  batchSize: int = 1000,
										  targetVRAM: Optional[float] = None,
										  showProgress: bool = True) -> numpy.ndarray:
	"""
	Find the optimal embedding dimension for each source variable via self-prediction.

	For each variable, finds the embedding dimension E in [1, maxDims] that maximises the
	correlation between simplex predictions of that variable from its own shadow manifold
	and the actual observed values. This is the correct embedding dimension to use for CCM,
	where the quality of source s's shadow manifold governs how well it cross-maps target t.

	Processes variables in batches. The dominant memory cost is [sourceBatch, numTrain, numTest],
	auto-tuned to stay near 2 GB. Exclusion mask positions are pre-applied as inf so they
	propagate correctly through incremental lag accumulation.

	:param X:               2D numpy array (N_timepoints, M_variables)
	:param maxDims:         Maximum embedding dimension to test
	:param train:           Train block index pairs [(start, end), ...]
	:param test:            Test block index pairs [(start, end), ...]
	:param predictionHorizon: Prediction time horizon
	:param step:            Time delay step size
	:param exclusionRadius: Temporal exclusion radius for neighbors
	:param embedded:        Whether data is already embedded
	:param validLib:        Boolean mask for valid library points
	:param dtype:           Torch dtype for tensors
	:param device:          Device string or torch.device
	:param batchSize:       Maximum number of source variables per batch (auto-reduced to fit VRAM)
	:param targetVRAM:      Target VRAM in GB. If None, sourceBatchSize = batchSize. If given, scales sourceBatchSize up from batchSize to fill the budget.
	:param showProgress:    Show tqdm progress bar
	:return:                1D int array [M_variables] of optimal embedding dimensions, 1-indexed
	"""
	if X.ndim == 1:
		X = X[:, None]
	numSources = X.shape[1]
	torchDevice = torch.device(device) if isinstance(device, str) else device

	if train is None:
		train = [(0, X.shape[0])]
	if test is None:
		test = [(0, X.shape[0])]

	train_indices, test_indices = BuildEmbeddingIndices(
		X.shape[0], X.shape[1],
		train, test,
		maxDims, predictionHorizon, step,
		embedded, validLib
	)
	exclusionMask = build_exclusion_mask(train_indices, test_indices, exclusionRadius)

	numTrain = train_indices.shape[0]
	numTest = test_indices.shape[0]

	scores = numpy.zeros([numSources, maxDims], dtype = numpy.float32)

	# Auto-tune source batch size so peak VRAM stays within targetVRAM.
	# Peak tensors that scale with sourceBatch: cumulativeDistances and lagDiffs coexist at [sourceBatch, numTrain, numTest].
	elementSize = torch.zeros(1, dtype = dtype).element_size()
	if targetVRAM is None:
		sourceBatchSize = batchSize
	else:
		targetVRAMBytes = targetVRAM * 1e9
		vramBatchSize = max(1, int(targetVRAMBytes / (2 * numTrain * numTest * elementSize)))
		sourceBatchSize = max(batchSize, vramBatchSize)

	yTrain = torch.tensor(X[train_indices + predictionHorizon, :], dtype = dtype, device = torchDevice)
	yTest = torch.tensor(X[test_indices + predictionHorizon, :], dtype = dtype, device = torchDevice)

	for sourceBatchStart in ProgressBar(range(0, numSources, sourceBatchSize), desc = 'Embedding dim search',
										leave = False, disable = not showProgress):
		sourceBatchEnd = min(sourceBatchStart + sourceBatchSize, numSources)
		actualSourceBatchSize = sourceBatchEnd - sourceBatchStart

		# Build time-delay embeddings for all sources in this batch: [sourceBatch, numTrain/numTest, maxDims]
		trainEmbeddings = torch.zeros([actualSourceBatchSize, numTrain, maxDims], dtype = dtype, device = torchDevice)
		testEmbeddings = torch.zeros([actualSourceBatchSize, numTest, maxDims], dtype = dtype, device = torchDevice)
		for localSourceIndex in range(actualSourceBatchSize):
			globalSourceIndex = sourceBatchStart + localSourceIndex
			if embedded:
				delayed = X[:, globalSourceIndex][:, None]
			else:
				delayed = MakeDelays(data = X[:, globalSourceIndex], num_delays = maxDims, stepSize = step, fill = 0.0)
			trainEmbeddings[localSourceIndex] = torch.tensor(delayed[train_indices, :], dtype = dtype, device = torchDevice)
			testEmbeddings[localSourceIndex] = torch.tensor(delayed[test_indices, :], dtype = dtype, device = torchDevice)

		# Cumulative squared Euclidean distances across lags: [sourceBatch, numTrain, numTest].
		# Pre-apply exclusion mask as inf so that inf + finite = inf keeps those positions excluded
		# through all subsequent lag additions.
		cumulativeDistances = torch.zeros([actualSourceBatchSize, numTrain, numTest], dtype = dtype, device = torchDevice)
		if exclusionMask is not None:
			cumulativeDistances[:, exclusionMask] = float('inf')

		# yTrain/yTest rows for this source batch, transposed for efficient diagonal gather
		yTrainBatch = yTrain[:, sourceBatchStart:sourceBatchEnd].T.contiguous()  # [sourceBatch, numTrain]
		yTestBatch = yTest[:, sourceBatchStart:sourceBatchEnd].T.contiguous()    # [sourceBatch, numTest]

		for embedDimIndex in range(maxDims):
			numKnn = embedDimIndex + 2

			# Accumulate this lag's squared differences in-place
			lagDiffs = trainEmbeddings[:, :, embedDimIndex].unsqueeze(2) - testEmbeddings[:, :, embedDimIndex].unsqueeze(1)
			lagDiffs.square_()
			cumulativeDistances.add_(lagDiffs)
			del lagDiffs

			neighborIndices, neighborWeights = batch_get_simplex_weights(cumulativeDistances, numKnn)
			# neighborIndices: [sourceBatch, numKnn, numTest]
			# neighborWeights: [sourceBatch, numKnn, numTest]

			# Diagonal gather: source i predicts itself, so index yTrainBatch[i] with neighborIndices[i]
			flatNeighborIndices = neighborIndices.reshape(actualSourceBatchSize, -1)  # [sourceBatch, numKnn * numTest]
			selectedTargets = yTrainBatch.gather(1, flatNeighborIndices).reshape(actualSourceBatchSize, numKnn, numTest)
			del flatNeighborIndices

			predictions = (neighborWeights * selectedTargets).sum(dim = 1)  # [sourceBatch, numTest]
			del selectedTargets

			targetCentered = yTestBatch - yTestBatch.mean(dim = 1, keepdim = True)
			predCentered = predictions - predictions.mean(dim = 1, keepdim = True)
			del predictions
			targetStd = torch.sqrt((targetCentered ** 2).sum(dim = 1))
			predStd = torch.sqrt((predCentered ** 2).sum(dim = 1))

			validMask = (targetStd > 0) & (predStd > 0)
			batchScores = torch.zeros(actualSourceBatchSize, dtype = dtype, device = torchDevice)
			if validMask.any():
				batchScores[validMask] = (
					(targetCentered[validMask] * predCentered[validMask]).sum(dim = 1) /
					(targetStd[validMask] * predStd[validMask])
				)
			scores[sourceBatchStart:sourceBatchEnd, embedDimIndex] = batchScores.cpu().float().numpy()

		del trainEmbeddings, testEmbeddings, cumulativeDistances, yTrainBatch, yTestBatch

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return numpy.argmax(scores, axis = 1) + 1


def _FindOptimalEmbeddingDimensionalityIterative(X, Y, maxDims,
												 train, test, predictionHorizon,
												 step, exclusionRadius, embedded,
												 validLib, ignoreNan,
												 scoring_function,
												 joint,
												 dtype = torch.float32,
												 batchSize = None):
	"""
	Evaluate each dimension with its own proper train/test indices: rows are
	trimmed only by the lags that dimension actually uses, so low dimensions keep
	the rows the shared maxDims trim would discard. Exact but ~(maxDims/2)x the
	work of the shared-index variant: each dimension reruns the batched engine at
	that depth and keeps its last score.
	"""
	print('Iterative search will be deprecated soon')
	perDimensionScores = []
	for E in range(1, maxDims + 1):
		scores = _FindOptimalEmbeddingDimensionalityBatched(
			X, Y, E, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, ignoreNan, scoring_function, joint,
			dtype = dtype, batchSize = batchSize)
		perDimensionScores.append(scores[..., -1])

	return numpy.stack(perDimensionScores, axis = -1)


def _FindOptimalEmbeddingDimensionalityBatched(X, Y, maxDims,
											   train, test, predictionHorizon,
											   step, exclusionRadius, embedded,
											   validLib, ignoreNan,
											   scoring_function,
											   joint,
											   dtype = torch.float32,
											   batchSize = None):
	"""
	Evaluate all embedding dimensions using shared maxDims indices and precomputed
	cumulative per-column distances on GPU. Uses the most restrictive NaN filtering
	(from maxDims) for all embedding dimension values.

	When joint is True, all X variables are used together to predict each Y variable,
	returning a [nTargets, maxDims] array of scores.

	When joint is False, each X variable is used separately to predict each Y variable,
	returning a [nTargets, nVars, maxDims] array of scores.

	When Y is None, each X variable separately predicts every X variable as target,
	returning a [nVars, nVars, maxDims] array of scores.
	"""
	nVars = X.shape[1]
	selfPrediction = Y is None

	combinedData = X if selfPrediction else numpy.column_stack([X, Y])
	target = 0 if selfPrediction else nVars
	columns = list(range(nVars))

	train_indices, _ = BuildEmbeddingIndices(combinedData.shape[0], combinedData.shape[1],
											 train, test,
											 maxDims, predictionHorizon, step,
											 embedded, validLib)

	dummy = Simplex(data = combinedData, columns = columns, target = target,
				train = train, test = test, embedDimensions = maxDims,
				predictionHorizon = predictionHorizon, knn = 0,
				step = step, exclusionRadius = exclusionRadius,
				embedded = embedded, validLib = validLib,
				noTime = True, ignoreNan = ignoreNan)

	dummy.EmbedData()
	dummy.RemoveNan()

	device = dummy.device

	trainEmbedding = dummy.Embedding[dummy.trainIndices, :]
	testEmbedding = dummy.Embedding[dummy.testIndices, :]
	nTrain = len(dummy.trainIndices)
	nTest = len(dummy.testIndices)

	exclusionMask = dummy._BuildExclusionMask()
	hasMask = exclusionMask.any()
	maskTensor = torch.tensor(exclusionMask, device = device, dtype = torch.bool) if hasMask else None

	if selfPrediction:
		scores = _BatchedSeparatePrediction(
			X, dummy, trainEmbedding, testEmbedding,
			nTrain, nTest, maxDims, nVars, device, dtype,
			hasMask, maskTensor, predictionHorizon, embedded, batchSize)
	elif joint:
		scores = _BatchedJointPrediction(
			Y, dummy, trainEmbedding, testEmbedding,
			nTrain, nTest, maxDims, nVars, device, dtype,
			hasMask, maskTensor, predictionHorizon, embedded)
	else:
		scores = _BatchedSeparatePrediction(
			Y, dummy, trainEmbedding, testEmbedding,
			nTrain, nTest, maxDims, nVars, device, dtype,
			hasMask, maskTensor, predictionHorizon, embedded, batchSize)

	if hasMask:
		del maskTensor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return scores


def _ComputeJointEmbeddingDistances(trainEmbedding, testEmbedding, nTrain, nTest,
									maxDims, nVars, device, dtype, embedded):
	"""
	Compute cumulative pairwise squared-distance matrices for the joint-prediction case.
	Columns are reordered lag-first so that cumsum position dim*nVars - 1 accumulates
	all variables through lag depth dim. Returns [maxDims, nTrain, nTest].
	"""
	trainTensor = torch.tensor(trainEmbedding, device = device, dtype = dtype)
	testTensor = torch.tensor(testEmbedding, device = device, dtype = dtype)
	nEmbedded = trainTensor.shape[1]

	distances = torch.zeros(nEmbedded, nTrain, nTest, device = device, dtype = dtype)
	for c in range(nEmbedded):
		diff = trainTensor[:, c].unsqueeze(1) - testTensor[:, c].unsqueeze(0)
		distances[c] = diff * diff
	del trainTensor, testTensor

	# Embed() orders columns variable-first: [var0_lag0, var0_lag1, ..., var1_lag0, ...]
	# Reorder to lag-first: [var0_lag0, var1_lag0, var0_lag1, var1_lag1, ...]
	# so cumsum at position dim*nVars - 1 correctly accumulates all variables through lag depth dim.
	if nVars > 1:
		perm = [c * maxDims + l for l in range(maxDims) for c in range(nVars)]
		distances = distances[perm]

	cumulativeDistances = torch.cumsum(distances, dim = 0)
	del distances

	indices = [dim * nVars - 1 if not embedded else dim - 1 for dim in range(1, maxDims + 1)]
	embeddingDistances = cumulativeDistances[indices]
	del cumulativeDistances
	return embeddingDistances


def _ComputePerVariableEmbeddingDistances(trainEmbedding, testEmbedding, nTrain, nTest,
										  maxDims, batchNumVars, colStart, colEnd, device, dtype):
	"""
	Compute cumulative pairwise squared-distance matrices for a batch of variables.
	Returns [batchNumVars * maxDims, nTrain, nTest] in variable-first ordering where
	row v*maxDims + e holds the cumulative distance for variable v through lag depth e+1.
	"""
	numBatch = batchNumVars * maxDims
	trainTensor = torch.tensor(trainEmbedding[:, colStart:colEnd], device = device, dtype = dtype)
	testTensor = torch.tensor(testEmbedding[:, colStart:colEnd], device = device, dtype = dtype)

	distances = torch.zeros(numBatch, nTrain, nTest, device = device, dtype = dtype)
	for c in range(numBatch):
		diff = trainTensor[:, c].unsqueeze(1) - testTensor[:, c].unsqueeze(0)
		distances[c] = diff * diff
	del trainTensor, testTensor

	cumulativeDistances = torch.cumsum(distances.view(batchNumVars, maxDims, nTrain, nTest), dim = 1)
	del distances
	return cumulativeDistances.view(numBatch, nTrain, nTest)


def _BatchedJointPrediction(Y, dummy, trainEmbedding, testEmbedding,
							 nTrain, nTest, maxDims, nVars, device, dtype,
							 hasMask, maskTensor, predictionHorizon, embedded):
	"""
	All X variables jointly predict Y. Returns [nTargets, maxDims].
	"""
	embeddingDistances = _ComputeJointEmbeddingDistances(
		trainEmbedding, testEmbedding, nTrain, nTest, maxDims, nVars, device, dtype, embedded)

	if hasMask:
		embeddingDistances[:, maskTensor] = float('inf')

	testIndices = dummy.testIndices + predictionHorizon
	testIndices = testIndices[testIndices < len(dummy.targetVec)]
	nTestValid = len(testIndices)

	nTargets = Y.shape[1]
	y_train = torch.tensor(Y[dummy.trainIndices + predictionHorizon, :], device = device, dtype = dtype).T
	y_test_all = torch.tensor(Y[testIndices, :], device = device, dtype = dtype).T

	out = torch.zeros(nTargets, maxDims, device = device, dtype = dtype)
	for targetIndex in range(nTargets):
		predictions = batch_simplex_predict(embeddingDistances, maxDims + 1, y_train[targetIndex])
		Correlation(y_test_all[targetIndex, :nTestValid], predictions[:, :nTestValid], out[targetIndex])

	del embeddingDistances, y_train, y_test_all
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return out.cpu().numpy()


def _BatchedSeparatePrediction(Y, dummy, trainEmbedding, testEmbedding,
								nTrain, nTest, maxDims, nVars, device, dtype,
								hasMask, maskTensor, predictionHorizon, embedded, batchSize):
	"""
	Each X variable separately predicts Y. Variables are processed in batches.
	Returns [nTargets, nVars, maxDims].
	"""
	nTargets = Y.shape[1]
	scores = numpy.zeros((nTargets, nVars, maxDims), dtype = numpy.float32)

	testIndices = dummy.testIndices + predictionHorizon
	testIndices = testIndices[testIndices < len(dummy.targetVec)]
	nTestValid = len(testIndices)

	y_train = torch.tensor(Y[dummy.trainIndices + predictionHorizon, :], device = device, dtype = dtype).T
	y_test_all = torch.tensor(Y[testIndices, :], device = device, dtype = dtype).T

	actualBatchSize = batchSize if batchSize is not None else nVars

	for varBatchStart in range(0, nVars, actualBatchSize):
		varBatchEnd = min(varBatchStart + actualBatchSize, nVars)
		batchNumVars = varBatchEnd - varBatchStart
		colStart = varBatchStart * maxDims
		colEnd = varBatchEnd * maxDims

		embeddingDistances = _ComputePerVariableEmbeddingDistances(
			trainEmbedding, testEmbedding, nTrain, nTest, maxDims, batchNumVars, colStart, colEnd, device, dtype)

		if hasMask:
			embeddingDistances[:, maskTensor] = float('inf')

		# matrix v*maxDims + d is variable v at history depth d+1, predicted with its own d+2 neighbors
		neighborCountsPerMatrix = torch.arange(2, maxDims + 2, dtype = torch.long, device = device).repeat(batchNumVars)

		out = torch.zeros(nTargets, batchNumVars * maxDims, device = device, dtype = dtype)
		for targetIndex in range(nTargets):
			predictions = batch_simplex_predict(embeddingDistances, neighborCountsPerMatrix, y_train[targetIndex])
			Correlation(y_test_all[targetIndex, :nTestValid], predictions[:, :nTestValid], out[targetIndex])
		scores[:, varBatchStart:varBatchEnd, :] = out.cpu().numpy().reshape(nTargets, batchNumVars, maxDims)

		del embeddingDistances, out
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	return scores


def FindOptimalPredictionHorizon(data: numpy.ndarray,
								 columns: List[int] = None,
								 target: int = None,
								 train: List[Tuple[int, int]] = None,
								 test: List[Tuple[int, int]] = None,
								 maxTp: int = 10,
								 embedDimensions: int = 1,
								 step: int = -1,
								 exclusionRadius: float = 0,
								 embedded: bool = False,
								 validLib: List = [],
								 noTime: bool = False,
								 ignoreNan: bool = True,
								 batched: bool = False,
								 scoring_function = Correlation) -> numpy.ndarray:
	"""
	Estimate optimal prediction interval [1:maxTp] using GPU-accelerated Simplex.

	When batched=False, each Tp gets its own proper train library (CreateIndices
	adjusts the library endpoint by predictionHorizon). When batched=True, the
	maxTp library (most restrictive) is used for all Tp values, and distances
	and neighbors are computed once with predictions batched across all Tp.

	:param data: 			2D numpy array where column 0 is time
	:param columns: 		Column indices to use (defaults to all except time)
	:param target: 			Target column index (defaults to column 1)
	:param maxTp: 			Maximum prediction horizon to test
	:param train: 			Train indices as list of (start, end) pairs
	:param test: 			Test indices as list of (start, end) pairs
	:param embedDimensions: Embedding dimension
	:param step: 			Step size for embedding
	:param exclusionRadius: Exclusion radius
	:param embedded: 		Whether data is already embedded
	:param validLib: 		Valid library indices
	:param noTime: 			Whether to exclude time column
	:param ignoreNan: 		Whether to ignore NaN values
	:param batched: 		Use shared maxTp library for all Tp (faster, slightly less accurate for low Tp)
	:param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
	:return: Array with columns [predictionHorizon, correlation]
	"""

	TpVals = list(range(1, maxTp + 1))

	if batched:
		correlations = _FindOptimalPredictionHorizonBatched(
			data, columns, target, TpVals, train, test,
			embedDimensions, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan, scoring_function)
	else:
		correlations = _FindOptimalPredictionHorizonIterative(
			data, columns, target, TpVals, train, test,
			embedDimensions, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan, scoring_function)

	return numpy.column_stack([TpVals, correlations])


def _FindOptimalPredictionHorizonIterative(data, columns, target, TpVals,
										   train, test, embedDimensions,
										   step, exclusionRadius, embedded,
										   validLib, noTime, ignoreNan,
										   scoring_function):
	"""
	Evaluate each Tp with its own proper train library.
	"""
	correlations = []

	for Tp in TpVals:
		S = Simplex(data=data, columns=columns, target=target,
					train=train, test=test, embedDimensions=embedDimensions,
					predictionHorizon=Tp, knn=0,
					step=step, exclusionRadius=exclusionRadius,
					embedded=embedded, validLib=validLib,
					noTime=noTime, ignoreNan=ignoreNan)

		result = S.Run()
		correlation = scoring_function(result.projection[:, 1], result.projection[:, 2])
		correlations.append(correlation)

	return correlations


def _FindOptimalPredictionHorizonBatched(data, columns, target, TpVals,
										 train, test, embedDimensions,
										 step, exclusionRadius, embedded,
										 validLib, noTime, ignoreNan,
										 scoring_function):
	"""
	Evaluate all Tp values using shared maxTp library. Distances and neighbors
	are computed once (using the most restrictive library from maxTp), then
	predictions for all Tp values are batched in a single tensor operation.
	"""
	maxTp = numpy.max(TpVals)

	# Create Simplex with maxTp to get the most restrictive library
	S = Simplex(data=data, columns=columns, target=target,
				train=train, test=test, embedDimensions=embedDimensions,
				predictionHorizon=maxTp, knn=0,
				step=step, exclusionRadius=exclusionRadius,
				embedded=embedded, validLib=validLib,
				noTime=noTime, ignoreNan=ignoreNan)

	S.EmbedData()
	S.RemoveNan()
	S.FindNeighborsTorch()

	device = S.device
	dtype = S.dtype

	distances = torch.tensor(S.knn_distances, device=device, dtype=dtype)
	neighbors = torch.tensor(S.knn_neighbors, device=device, dtype=torch.long)
	targetVector = torch.tensor(S.targetVec.squeeze(), device=device, dtype=dtype)

	# Weights depend only on distances, shared across all Tp
	minDist = distances[:, 0].clone()
	torch.clamp_min(minDist, 1e-6, out=minDist)
	scaledDistances = distances / minDist.unsqueeze(1)
	weights = torch.exp(-scaledDistances)
	weightRowSum = torch.sum(weights, dim=1)

	# Batch predictions: neighborsPlusTp is [maxTp, nTest, knn]
	TpTensor = torch.tensor(TpVals, device=device, dtype=torch.long)
	neighborsPlusTp = neighbors.unsqueeze(0) + TpTensor.unsqueeze(1).unsqueeze(2)

	# libTargetValues: [maxTp, nTest, knn]
	libTargetValues = targetVector[neighborsPlusTp]

	# predictions: [maxTp, nTest]
	predictions = torch.sum(weights.unsqueeze(0) * libTargetValues, dim=2) / weightRowSum.unsqueeze(0)

	predictionsNumpy = predictions.cpu().numpy()

	# Compute correlations for each Tp
	correlations = []
	for i, Tp in enumerate(TpVals):
		observationIndices = S.testIndices + Tp
		validObsIndices = observationIndices[observationIndices < len(S.targetVec)]
		observations = S.targetVec[validObsIndices, 0]
		nValid = len(validObsIndices)
		correlation = scoring_function(observations[:nValid], predictionsNumpy[i, :nValid])
		correlations.append(correlation)

	del distances, neighbors, targetVector, weights, weightRowSum
	del TpTensor, neighborsPlusTp, libTargetValues, predictions
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return correlations


def FindSMapNeighborhood(data: numpy.ndarray,
						 columns: List[int] = None,
						 target: int = None,
						 theta: Any = None,
						 train: List[Tuple[int, int]] = None,
						 test: List[Tuple[int, int]] = None,
						 embedDimensions: int = 1,
						 predictionHorizon: int = 1,
						 knn: int = 0,
						 step: int = -1,
						 exclusionRadius: float = 0,
						 solver: Any = None,
						 embedded: bool = False,
						 validLib: List = [],
						 noTime: bool = False,
						 ignoreNan: bool = True,
						 numProcess: int = 4,
						 mpMethod: Any = None,
						 chunksize: int = 1,
						 scoring_function = Correlation) -> numpy.ndarray:
	"""
	Estimate the best neighborhood size for SMap, i.e. the
	exponential decay factor for weighing neighbors by distance.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param theta: 				Theta values to test
	:param train: 				Train indices as list of (start, end) pairs
	:param test: 				Test indices as list of (start, end) pairs
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param solver: 				SMap solver (unused, kept for API compatibility)
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param numProcess: 			Unused, kept for API compatibility
	:param mpMethod: 			Unused, kept for API compatibility
	:param chunksize: 			Unused, kept for API compatibility
	:param scoring_function: 	Scoring function taking (actual, predicted) and returning a scalar
	:return: Array with columns [theta, correlation]
	"""

	if theta is None:
		theta = [0.01, 0.1, 0.3, 0.5, 0.75, 1,
				 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
	elif not IsNonStringIterable(theta):
		theta = [float(t) for t in theta.split()]

	correlations = _FindSMapNeighborhoodBatched(
		data, columns, target, theta, train, test,
		embedDimensions, predictionHorizon, knn, step,
		exclusionRadius, embedded, validLib, noTime, ignoreNan, scoring_function)

	return numpy.column_stack([theta, correlations])


def _FindSMapNeighborhoodBatched(data, columns, target, thetaValues, train, test,
								 embedDimensions, predictionHorizon, knn, step,
								 exclusionRadius, embedded, validLib, noTime, ignoreNan,
								 scoring_function):
	"""
	Evaluate all theta values using shared neighbor computation.
	Neighbors are found once, then projections for all theta values
	are computed by varying only the distance weighting.
	"""
	# Create SMap with theta=0 (won't affect neighbor finding)
	S = SMap(data = data,
			 columns = columns,
			 target = target,
			 train = train,
			 test = test,
			 embedDimensions = embedDimensions,
			 predictionHorizon = predictionHorizon,
			 knn = knn,
			 step = step,
			 theta = 0,
			 exclusionRadius = exclusionRadius,
			 embedded = embedded,
			 validLib = validLib,
			 noTime = noTime,
			 ignoreNan = ignoreNan)

	S.EmbedData()
	S.RemoveNan()
	S.FindNeighborsTorch()

	device = S.device
	dtype = S.dtype

	numberOfPredictions = len(S.testIndices)
	numberOfDimensions = S.embedDimensions + 1

	# Convert data to tensors once
	distances = torch.tensor(S.knn_distances, device = device, dtype = dtype)
	neighbors = torch.tensor(S.knn_neighbors, device = device, dtype = torch.long)
	embedding = torch.tensor(S.Embedding, device = device, dtype = dtype)
	targetVector = torch.tensor(S.targetVec.squeeze(), device = device, dtype = dtype)
	testIndices = torch.tensor(S.testIndices, device = device, dtype = torch.long)

	# Precompute values shared across all theta
	distanceRowMean = torch.mean(distances, dim = 1, keepdim = True)
	torch.clamp_min_(distanceRowMean, 1e-10)

	neighborsPlusTp = neighbors + predictionHorizon
	targetValues = targetVector[neighborsPlusTp]

	validMask = torch.isfinite(targetValues)
	maskedTargetValues = torch.where(validMask, targetValues, torch.zeros_like(targetValues))

	neighborEmbeddings = embedding[neighbors]
	testEmbeddings = embedding[testIndices]

	# Observation values for correlation computation
	observationIndices = S.testIndices + predictionHorizon
	validObsIndices = observationIndices[observationIndices < len(S.targetVec)]
	observations = S.targetVec[validObsIndices, 0]
	nValid = len(validObsIndices)

	correlations = []

	for theta in thetaValues:
		# Compute weights for this theta
		if theta == 0:
			weights = torch.ones_like(distances)
		else:
			distanceRowScale = theta / distanceRowMean
			weights = torch.exp(-distanceRowScale * distances)

		maskedWeights = torch.where(validMask, weights, torch.zeros_like(weights))
		weightedTargets = maskedWeights * maskedTargetValues

		# Build design matrix
		designMatrix = torch.zeros(numberOfPredictions, S.knn, numberOfDimensions,
								   device = device, dtype = dtype)
		designMatrix[:, :, 0] = maskedWeights
		designMatrix[:, :, 1:] = maskedWeights.unsqueeze(2) * neighborEmbeddings

		# Solve least squares
		lstsqResult = torch.linalg.lstsq(designMatrix, weightedTargets)
		coefficients = lstsqResult.solution

		# Compute predictions
		predictions = coefficients[:, 0] + torch.sum(coefficients[:, 1:] * testEmbeddings, dim = 1)
		predictionsNumpy = predictions.cpu().numpy()

		correlation = scoring_function(observations[:nValid], predictionsNumpy[:nValid])
		correlations.append(correlation)

	# Clean up
	del distances, neighbors, embedding, targetVector, testIndices
	del distanceRowMean, neighborsPlusTp, targetValues, validMask
	del maskedTargetValues, neighborEmbeddings, testEmbeddings
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return correlations
