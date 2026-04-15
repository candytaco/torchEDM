from typing import List, Tuple, Any, Optional

import numpy
import torch

from .EDM._MDE import FloorArray, RowwiseCorrelation
from .Scoring import Correlation
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex
from .Utils import IsNonStringIterable

# TODO: these should all be cross-validated


def FindOptimalEmbeddingDimensionality(X: numpy.ndarray,
									   Y: Optional[numpy.ndarray] = None,
									   maxDims: int = 10,
									   train: Tuple[int, int] = None,
									   test: Tuple[int, int] = None,
									   predictionHorizon: int = 1,
									   step: int = -1,
									   exclusionRadius: float = 0,
									   embedded: bool = False,
									   validLib: List = [],
									   ignoreNan: bool = True,
									   batched: bool = True,
									   scoring_function = Correlation,
									   joint: bool = True):
	"""
	Estimate optimal embedding dimension for simplex. When Y is not provided, each X is used to predict itself and find
	the optimal number of dimensions for it self. When Y is provided and joint is True, all Xs are used to jointly
	predict Y. When joint is False, each X is used to separately predict Y and we have a separate dimensionality for
	each X when paired to Y.

	When batched = False, each train and test indices are computed per embedding dimensionality. When batched = True,
	the indices are computed from the maximum, which is the most restrictive, which enables
	shared distance precomputation but slightly penalizes lower dimensions values
	by excluding a few extra rows.

	:param X: 					2D numpy array of predictor columns, shape (N, numFeatures)
	:param Y: 					1D or 2D numpy array of target values, shape (N,) or (N, 1)
	:param maxDims: 			maximum number of embedding dimensions to test
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param predictionHorizon: 	Prediction horizon
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param ignoreNan: 			Whether to ignore NaN values
	:param batched: 			Use shared maxE indices for all E (faster, slightly less accurate for low E)
	:param scoring_function: 	Scoring function taking (actual, predicted) and returning a scalar
	:param joint:				when X is 2D, use all vars together to predict Y? If False, each X is used separately to predict Y
	:return: score for each embedding dimension
	"""

	# force column vectors
	if len(X.shape) < 2:
		X = X[:, None]
	if Y is not None and len(Y.shape) < 2:
		Y = Y[:, None]

	if batched:
		scores = _FindOptimalEmbeddingDimensionalityBatched(
			X, Y, maxDims, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, ignoreNan, scoring_function, joint)
	else:
		scores = _FindOptimalEmbeddingDimensionalityIterative(
			X, Y, maxDims, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, ignoreNan, scoring_function, joint)

	return scores.squeeze()


def _FindOptimalEmbeddingDimensionalityIterative(X, Y, maxDims,
												 train, test, predictionHorizon,
												 step, exclusionRadius, embedded,
												 validLib, ignoreNan,
												 scoring_function,
												 joint):
	"""
	Evaluate each E with its own proper train/test indices.
	Each E creates a Simplex and runs GPU-accelerated FindNeighbors/Project.
	"""
	print('Iterative search will be deprecated soon and this is not updated')
	if Y is not None:
		combinedData = numpy.column_stack([X, Y])
	else:
		combinedData = X
	columns = list(range(X.shape[1]))
	target = X.shape[1]

	scores = []

	for E in range(1, maxDims + 1):
		S = Simplex(data = combinedData, columns = columns, target = target,
					train = train, test = test, embedDimensions = E,
					predictionHorizon = predictionHorizon, knn = 0,
					step = step, exclusionRadius = exclusionRadius,
					embedded = embedded, validLib = validLib,
					noTime = True, ignoreNan = ignoreNan)

		result = S.Run()
		correlation = scoring_function(result.projection[:, 1], result.projection[:, 2])
		scores.append(correlation)

	return scores


def _FindOptimalEmbeddingDimensionalityBatched(X, Y, maxDims,
											   train, test, predictionHorizon,
											   step, exclusionRadius, embedded,
											   validLib, ignoreNan,
											   scoring_function,
											   joint):
	"""
	Evaluate all embedding dimensions using shared maxE indices and precomputed
	cumulative per-column distances on GPU. Uses the most restrictive
	NaN filtering (from maxE) for all E values.

	When joint is True, all variables in X are used together to predict Y,
	returning a [maxDims] array of scores.

	When joint is False, each variable in X is used separately to predict Y.
	Per-variable distances are concatenated along the first dimension of the
	stacked distance matrices, and scores are returned as a [nVars x maxDims] array.

	When Y is None, each column of X is used to predict itself (self-prediction),
	returning a [nVars x maxDims] array of scores.
	"""
	nVars = X.shape[1]
	selfPrediction = Y is None

	if selfPrediction:
		combinedData = X
		target = 0  # dummy target; actual targets come from X columns below
	else:
		combinedData = numpy.column_stack([X, Y])
		target = nVars
	columns = list(range(nVars))

	# Create a Simplex at maxE to get proper indices, embedding, and target
	dummy = Simplex(data = combinedData, columns = columns, target = target,
				train = train, test = test, embedDimensions = maxDims,
				predictionHorizon = predictionHorizon, knn = 0,
				step = step, exclusionRadius = exclusionRadius,
				embedded = embedded, validLib = validLib,
				noTime = True, ignoreNan = ignoreNan)

	dummy.EmbedData()
	dummy.RemoveNan()

	device = dummy.device
	dtype = dummy.dtype

	trainEmbedding = dummy.Embedding[dummy.trainIndices, :]
	testEmbedding = dummy.Embedding[dummy.testIndices, :]
	nTrain = len(dummy.trainIndices)
	nTest = len(dummy.testIndices)

	trainTensor = torch.tensor(trainEmbedding, device = device, dtype = dtype)
	testTensor = torch.tensor(testEmbedding, device = device, dtype = dtype)
	nEmbedded = trainTensor.shape[1]

	# Compute per-column squared pairwise distances: [nEmbedded, nTrain, nTest]
	distances = torch.zeros(nEmbedded, nTrain, nTest, device = device, dtype = dtype)
	for c in range(nEmbedded):
		diff = trainTensor[:, c].unsqueeze(1) - testTensor[:, c].unsqueeze(0)
		distances[c, :, :] = diff * diff

	del trainTensor, testTensor

	if joint and not selfPrediction:
		# Embed() orders columns variable-first: [var0_lag0, var0_lag1, ..., var1_lag0, ...]
		# Reorder to lag-first: [var0_lag0, var1_lag0, var0_lag1, var1_lag1, ...]
		# so that totalDistance[dim*nVars - 1] correctly sums the first dim lags of all variables.
		if nVars > 1:
			perm = [c * maxDims + l for l in range(maxDims) for c in range(nVars)]
			distances = distances[perm]

		# cumulative sum across all embeddings
		totalDistance = torch.cumsum(distances, dim = 0)

		indices = []
		for dim in range(1, maxDims + 1):
			# For multi-variable embeddings, E dimensions use E * len(columns) actual columns
			# so we index into the cumulative sum at the position that sums to the embedding dims
			index = dim * nVars if not embedded else dim
			indices.append(index - 1)
		# select the indices that correspond to sums across all vars at each N dimensions
		embeddingDistances = totalDistance[indices, :, :]
		numBatch = maxDims

	else:
		# For non-joint and self-prediction, compute cumulative sums per variable.
		# Embed() orders columns variable-first: [var0_lag0, var0_lag1, ..., var1_lag0, ...]
		# Reshape to [nVars, maxDims, nTrain, nTest] so cumsum runs along the lag dimension
		# independently for each variable. Index v*maxDims + E holds the cumulative distance
		# for variable v over its first E+1 lags. Per-variable distances are thus concatenated
		# along the first dimension of the stacked distance matrices.
		totalDistance = torch.cumsum(distances.view(nVars, maxDims, nTrain, nTest), dim = 1)
		embeddingDistances = totalDistance.view(nVars * maxDims, nTrain, nTest)
		numBatch = nVars * maxDims

	del distances, totalDistance

	# Build exclusion mask once (same for all E since indices are shared)
	exclusionMask = dummy._BuildExclusionMask()
	hasMask = exclusionMask.any()
	if hasMask:
		maskTensor = torch.tensor(exclusionMask, device = device, dtype = torch.bool)
		embeddingDistances[:, maskTensor] = float('inf')

	# batched over distances
	neighborDistances, neighborIndices = torch.topk(embeddingDistances, maxDims + 1, dim = 1, largest = False)

	neighborDistances.sqrt_()
	FloorArray(neighborDistances, 1e-6)

	# Compute weighted predictions
	minDistances = torch.amin(neighborDistances, dim = 1)
	weights = neighborDistances / minDistances.unsqueeze(1)
	weights.neg_().exp_()

	# Zero out extra neighbors: for embedding dimension E (0-indexed), keep E+2 neighbors.
	# For non-joint and self-prediction, the [0..maxDims-1] pattern repeats once per variable.
	dimIndices = torch.arange(maxDims, device = device).repeat(numBatch // maxDims).view(numBatch, 1, 1)
	kIndices = torch.arange(maxDims + 1, device = device).view(1, maxDims + 1, 1)
	weights.masked_fill_(kIndices > dimIndices + 1, 0)

	weightSum = torch.sum(weights, dim = 1)

	if selfPrediction:
		# Each variable predicts itself: for row i = v*maxDims + E_0, look up y_train for variable v.
		# y_train[v] = X[trainIndices + predictionHorizon, v], shape [nVars, nTrain]
		y_train = torch.tensor(
			X[dummy.trainIndices + predictionHorizon, :], device = device, dtype = dtype
		).T

		# Reshape neighborIndices to [nVars, maxDims, maxDims+1, nTest] so variable axis is explicit,
		# then index into y_train per-variable using the variable index as the first dimension.
		neighborIndices4d = neighborIndices.view(nVars, maxDims, maxDims + 1, nTest)
		variableIndex = torch.arange(nVars, device = device).view(nVars, 1, 1, 1).expand_as(neighborIndices4d)
		select = y_train[variableIndex, neighborIndices4d].view(nVars * maxDims, maxDims + 1, nTest)

		predictions = torch.sum(weights * select, dim = 1) / weightSum

		testIndices = dummy.testIndices + predictionHorizon
		testIndices = testIndices[testIndices < X.shape[0]]
		nTestValid = len(testIndices)

		# y_test[v] = X[testIndices, v], shape [nVars, nTestValid]
		# Expand to [nVars * maxDims, nTestValid] by repeating each variable's test values maxDims times.
		y_test = torch.tensor(X[testIndices, :], device = device, dtype = dtype).T
		y_test_expanded = y_test.unsqueeze(1).expand(nVars, maxDims, nTestValid).reshape(nVars * maxDims, nTestValid)

		y_pred = predictions[:, :nTestValid]

		# Vectorized per-row Pearson correlation between each (variable, dim) pair and its self-target.
		y_test_centered = y_test_expanded - y_test_expanded.mean(dim = 1, keepdim = True)
		y_pred_centered = y_pred - y_pred.mean(dim = 1, keepdim = True)
		numerator = (y_test_centered * y_pred_centered).sum(dim = 1)
		denominator = y_test_centered.pow(2).sum(dim = 1).sqrt() * y_pred_centered.pow(2).sum(dim = 1).sqrt()
		out = numerator / denominator

		scores = out.cpu().numpy().reshape(nVars, maxDims)

	else:
		nTargets = Y.shape[1]
		# y_train[t] = Y[trainIndices + Tp, t], shape [nTargets, nTrain]
		y_train = torch.tensor(Y[dummy.trainIndices + predictionHorizon, :], device = device, dtype = dtype).T

		testIndices = dummy.testIndices + predictionHorizon
		testIndices = testIndices[testIndices < len(dummy.targetVec)]
		nTestValid = len(testIndices)
		# y_test_all[t] = Y[testIndices, t], shape [nTargets, nTestValid]
		y_test_all = torch.tensor(Y[testIndices, :], device = device, dtype = dtype).T

		# y_train[:, neighborIndices] gathers training target values for every
		# target simultaneously. neighborIndices is [numBatch, maxDims+1, nTest],
		# so the result is [nTargets, numBatch, maxDims+1, nTest].
		select = y_train[:, neighborIndices]
		predictions = (weights.unsqueeze(0) * select).sum(dim = 2) / weightSum.unsqueeze(0)
		# predictions: [nTargets, numBatch, nTest]

		y_pred = predictions[:, :, :nTestValid]  # [nTargets, numBatch, nTestValid]

		out = torch.zeros(nTargets, numBatch, device = device)
		for targetIndex in range(nTargets):
			RowwiseCorrelation(y_test_all[targetIndex], y_pred[targetIndex], out[targetIndex])

		scores = out.cpu().numpy()  # [nTargets, numBatch]
		if not joint:
			scores = scores.reshape(nTargets, nVars, maxDims)

	if hasMask:
		del maskTensor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return scores


def FindOptimalPredictionHorizon(data: numpy.ndarray,
								 columns: List[int] = None,
								 target: int = None,
								 train: Tuple[int, int] = None,
								 test: Tuple[int, int] = None,
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
	:param train: 			Train indices [start, end]
	:param test: 			Test indices [start, end]
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
						 train: Tuple[int, int] = None,
						 test: Tuple[int, int] = None,
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
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
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
