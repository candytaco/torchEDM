from typing import List, Tuple, Any

import numpy
import torch

from .Utils import ComputeError
from .EDM.Embed import Embed
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex
from .Utils import IsNonStringIterable

# TODO: these should all be cross-validated


def FindOptimalEmbeddingDimensionality(data: numpy.ndarray,
									   columns: List[int] = None,
									   target: int = None,
									   maxE: int = 10,
									   train: Tuple[int, int] = None,
									   test: Tuple[int, int] = None,
									   predictionHorizon: int = 1,
									   step: int = -1,
									   exclusionRadius: float = 0,
									   embedded: bool = False,
									   validLib: List = [],
									   noTime: bool = False,
									   ignoreNan: bool = True,
									   batched: bool = False):
	"""
	Estimate optimal embedding dimension for simplex

	When batched=False, each E gets its own proper train/test indices derived
	from that E's embedding. When batched=True, the maxE indices (most
	restrictive NaN filtering) are used for all E values, which enables
	shared distance precomputation but slightly penalizes lower E values
	by excluding a few extra rows.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param maxE: 				Maximum embedding dimension to test
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param predictionHorizon: 	Prediction horizon
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param batched: 			Use shared maxE indices for all E (faster, slightly less accurate for low E)
	:return: best embedding dimensions
	"""

	dimensions = list(range(1, maxE + 1))

	if batched:
		correlations = _FindOptimalEmbeddingDimensionalityBatched(
			data, columns, target, maxE, dimensions, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)
	else:
		correlations = _FindOptimalEmbeddingDimensionalityIterative(
			data, columns, target, dimensions, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)

	return dimensions, correlations


def _FindOptimalEmbeddingDimensionalityIterative(data, columns, target, Evals,
												  train, test, predictionHorizon,
												  step, exclusionRadius, embedded,
												  validLib, noTime, ignoreNan):
	"""
	Evaluate each E with its own proper train/test indices.
	Each E creates a Simplex and runs GPU-accelerated FindNeighbors/Project.
	"""
	correlations = []

	for E in Evals:
		S = Simplex(data=data, columns=columns, target=target,
					train=train, test=test, embedDimensions=E,
					predictionHorizon=predictionHorizon, knn=0,
					step=step, exclusionRadius=exclusionRadius,
					embedded=embedded, validLib=validLib,
					noTime=noTime, ignoreNan=ignoreNan)

		result = S.Run()
		correlation = ComputeError(result.projection[:, 1], result.projection[:, 2], None)
		correlations.append(correlation)

	return correlations


def _FindOptimalEmbeddingDimensionalityBatched(data, columns, target, maxE, Evals,
												train, test, predictionHorizon,
												step, exclusionRadius, embedded,
												validLib, noTime, ignoreNan):
	"""
	Evaluate all E values using shared maxE indices and precomputed
	cumulative per-column distances on GPU. Uses the most restrictive
	NaN filtering (from maxE) for all E values.
	"""
	# Create a Simplex at maxE to get proper indices, embedding, and target
	S = Simplex(data=data, columns=columns, target=target,
				train=train, test=test, embedDimensions=maxE,
				predictionHorizon=predictionHorizon, knn=0,
				step=step, exclusionRadius=exclusionRadius,
				embedded=embedded, validLib=validLib,
				noTime=noTime, ignoreNan=ignoreNan)

	S.EmbedData()
	S.RemoveNan()

	device = S.device
	dtype = S.dtype

	trainEmbedding = S.Embedding[S.trainIndices, :]
	testEmbedding = S.Embedding[S.testIndices, :]
	nTrain = len(S.trainIndices)
	nTest = len(S.testIndices)
	numEmbeddingColumns = trainEmbedding.shape[1]

	trainTensor = torch.tensor(trainEmbedding, device=device, dtype=dtype)
	testTensor = torch.tensor(testEmbedding, device=device, dtype=dtype)
	targetVector = torch.tensor(S.targetVec.squeeze(), device=device, dtype=dtype)

	# Compute per-column squared pairwise distances: [numCols, nTrain, nTest]
	perColumnDistancesSq = torch.zeros(numEmbeddingColumns, nTrain, nTest, device=device, dtype=dtype)
	for c in range(numEmbeddingColumns):
		diff = trainTensor[:, c].unsqueeze(1) - testTensor[:, c].unsqueeze(0)
		perColumnDistancesSq[c] = diff * diff

	# Cumulative sum gives squared distances for each E
	cumulativeDistancesSq = torch.cumsum(perColumnDistancesSq, dim=0)

	del perColumnDistancesSq, trainTensor, testTensor

	# Build exclusion mask once (same for all E since indices are shared)
	exclusionMask = S._BuildExclusionMask()
	hasMask = exclusionMask.any()
	if hasMask:
		maskTensor = torch.tensor(exclusionMask, device=device, dtype=torch.bool)

	correlations = []

	for E in Evals:
		knn = E + 1

		# For multi-column embeddings, E dimensions use E * len(columns) actual columns
		embeddingColumnsForE = E * len(S.columns) if not embedded else E
		if embeddingColumnsForE > numEmbeddingColumns:
			embeddingColumnsForE = numEmbeddingColumns
		distancesSq = cumulativeDistancesSq[embeddingColumnsForE - 1]

		distances = torch.sqrt(distancesSq)

		if hasMask:
			distances[maskTensor] = float('inf')

		topkDistances, topkIndices = torch.topk(distances, knn, dim=0, largest=False)

		neighborDistances = topkDistances.t()
		neighborIndices = topkIndices.t()

		# Compute weighted predictions
		minDist = neighborDistances[:, 0].clone()
		torch.clamp_min(minDist, 1e-6, out=minDist)
		scaledDistances = neighborDistances / minDist.unsqueeze(1)
		weights = torch.exp(-scaledDistances)
		weightRowSum = torch.sum(weights, dim=1)

		neighborIndicesData = neighborIndices.cpu().numpy()
		neighborIndicesData = S._MapKNNIndicesToLibraryIndices(neighborIndicesData)
		neighborIndicesDataTp = torch.tensor(neighborIndicesData + predictionHorizon, device=device, dtype=torch.long)

		libTargetValues = targetVector[neighborIndicesDataTp]
		predictions = torch.sum(weights * libTargetValues, dim=1) / weightRowSum

		observationIndices = S.testIndices + predictionHorizon
		validObsIndices = observationIndices[observationIndices < len(S.targetVec)]
		observations = S.targetVec[validObsIndices, 0]

		predictionsNumpy = predictions.cpu().numpy()
		nValid = len(validObsIndices)
		correlation = ComputeError(observations[:nValid], predictionsNumpy[:nValid], None)
		correlations.append(correlation)

	del cumulativeDistancesSq
	if hasMask:
		del maskTensor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return correlations


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
								 batched: bool = False) -> numpy.ndarray:
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
	:return: Array with columns [predictionHorizon, correlation]
	"""

	TpVals = list(range(1, maxTp + 1))

	if batched:
		correlations = _FindOptimalPredictionHorizonBatched(
			data, columns, target, TpVals, train, test,
			embedDimensions, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)
	else:
		correlations = _FindOptimalPredictionHorizonIterative(
			data, columns, target, TpVals, train, test,
			embedDimensions, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)

	return numpy.column_stack([TpVals, correlations])


def _FindOptimalPredictionHorizonIterative(data, columns, target, TpVals,
											train, test, embedDimensions,
											step, exclusionRadius, embedded,
											validLib, noTime, ignoreNan):
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
		correlation = ComputeError(result.projection[:, 1], result.projection[:, 2], None)
		correlations.append(correlation)

	return correlations


def _FindOptimalPredictionHorizonBatched(data, columns, target, TpVals,
										  train, test, embedDimensions,
										  step, exclusionRadius, embedded,
										  validLib, noTime, ignoreNan):
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
		correlation = ComputeError(observations[:nValid], predictionsNumpy[i, :nValid], None)
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
						 chunksize: int = 1) -> numpy.ndarray:
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
		exclusionRadius, embedded, validLib, noTime, ignoreNan)

	return numpy.column_stack([theta, correlations])


def _FindSMapNeighborhoodBatched(data, columns, target, thetaValues, train, test,
								 embedDimensions, predictionHorizon, knn, step,
								 exclusionRadius, embedded, validLib, noTime, ignoreNan):
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

		correlation = ComputeError(observations[:nValid], predictionsNumpy[:nValid], None)
		correlations.append(correlation)

	# Clean up
	del distances, neighbors, embedding, targetVector, testIndices
	del distanceRowMean, neighborsPlusTp, targetValues, validMask
	del maskedTargetValues, neighborEmbeddings, testEmbeddings
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return correlations


def FindOptimalDelay(data: numpy.ndarray,
					 columns: List[int] = None,
					 target: int = None,
					 stepValues: List[int] = None,
					 maxTau: int = 10,
					 train: Tuple[int, int] = None,
					 test: Tuple[int, int] = None,
					 embedDimensions: int = 1,
					 predictionHorizon: int = 1,
					 exclusionRadius: float = 0,
					 embedded: bool = False,
					 validLib: List = [],
					 noTime: bool = False,
					 ignoreNan: bool = True) -> numpy.ndarray:
	"""
	Estimate optimal time-delay step size (tau) using Simplex projection.

	:param data: 			2D numpy array where column 0 is time
	:param columns: 		Column indices to use (defaults to all except time)
	:param target: 			Target column index (defaults to column 1)
	:param stepValues: 		Step (tau) values to test; if None, uses [-1, -2, ..., -maxTau]
	:param maxTau: 			Maximum absolute tau value when stepValues is None
	:param train: 			Train indices [start, end]
	:param test: 			Test indices [start, end]
	:param embedDimensions: Embedding dimension
	:param predictionHorizon: Prediction horizon
	:param exclusionRadius: Exclusion radius
	:param embedded: 		Whether data is already embedded
	:param validLib: 		Valid library indices
	:param noTime: 			Whether to exclude time column
	:param ignoreNan: 		Whether to ignore NaN values
	:return: Array with columns [step, correlation]
	"""

	if stepValues is None:
		stepValues = [-delayStep for delayStep in range(1, maxTau + 1)]

	correlations = []

	for step in stepValues:
		simplexModel = Simplex(data = data, columns = columns, target = target,
							   train = train, test = test, embedDimensions = embedDimensions,
							   predictionHorizon = predictionHorizon, knn = 0,
							   step = step, exclusionRadius = exclusionRadius,
							   embedded = embedded, validLib = validLib,
							   noTime = noTime, ignoreNan = ignoreNan)

		result = simplexModel.Run()
		correlation = ComputeError(result.projection[:, 1], result.projection[:, 2], None)
		correlations.append(correlation)

	return numpy.column_stack([stepValues, correlations])


def _BatchedSimplexPredict(X, Y, embeddingDimension, predictionHorizon, step, exclusionRadius,
						   train, test, ignoreNan, device, dtype, batchSize):
	"""
	Core batched Simplex prediction helper. For each of M source variables
	in X (N, M), embed with given embeddingDimension and step, find kNN,
	predict Y, and return per-variable Pearson correlations.

	Uses a dummy Simplex to obtain shared train/test indices (based on the
	first column of X), then processes all M variables in batches on GPU.

	:param X: 					(N, M) array without time column
	:param Y: 					(N,) target array without time column
	:param embeddingDimension: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param step: 				Time delay step size (tau)
	:param exclusionRadius: 	Temporal exclusion radius
	:param train: 				[start, end] train indices (1-indexed)
	:param test: 				[start, end] test indices (1-indexed)
	:param ignoreNan: 			Whether to ignore NaN values
	:param device: 				torch.device
	:param dtype: 				torch dtype
	:param batchSize: 			Number of variables per GPU batch
	:return: 					(M,) array of correlations
	"""
	numTimepoints, numVariables = X.shape
	knn = embeddingDimension + 1

	# Use a dummy Simplex on the first column to determine shared indices
	dummy = Simplex(data = X, columns = [0], target = 0,
					train = train, test = test, embedDimensions = embeddingDimension,
					predictionHorizon = predictionHorizon, knn = knn,
					step = step, exclusionRadius = exclusionRadius,
					embedded = False, validLib = [], noTime = True, ignoreNan = ignoreNan)
	dummy.EmbedData()
	dummy.RemoveNan()

	trainIndices = dummy.trainIndices
	testIndices  = dummy.testIndices
	numTrain     = len(trainIndices)
	numTest      = len(testIndices)

	targetArray = Y  # (N,)

	# Observations at testIndices + predictionHorizon
	observationIndices   = testIndices + predictionHorizon
	validObservationMask = (observationIndices >= 0) & (observationIndices < numTimepoints)
	validObservations    = observationIndices[validObservationMask]
	numValid             = int(validObservationMask.sum())
	observations         = targetArray[validObservations]

	# Build exclusion mask once (shared for all variables)
	exclusionMask = dummy._BuildExclusionMask()  # (numTrain, numTest)
	hasMask = exclusionMask.any()
	if hasMask:
		maskTensor = torch.tensor(exclusionMask, device = device, dtype = torch.bool)

	# Pre-embed all M variables for this parameter value
	allTrainEmbeddings = []
	allTestEmbeddings  = []
	for variableIndex in range(numVariables):
		embedding = Embed(data = X, columns = [variableIndex], embeddingDimensions = embeddingDimension,
						  stepSize = step, includeTime = False)
		allTrainEmbeddings.append(embedding[trainIndices, :])
		allTestEmbeddings.append(embedding[testIndices, :])

	targetTensor     = torch.tensor(targetArray, device = device, dtype = dtype)
	trainIndexTensor = torch.tensor(trainIndices, device = device, dtype = torch.long)

	correlations = numpy.zeros(numVariables)

	for batchStart in range(0, numVariables, batchSize):
		batchEnd         = min(batchStart + batchSize, numVariables)
		batchNumVariables = batchEnd - batchStart

		trainStack = numpy.stack(allTrainEmbeddings[batchStart:batchEnd])  # (batchNumVariables, numTrain, embeddingDimension)
		testStack  = numpy.stack(allTestEmbeddings[batchStart:batchEnd])   # (batchNumVariables, numTest, embeddingDimension)

		trainTensor = torch.tensor(trainStack, device = device, dtype = dtype)
		testTensor  = torch.tensor(testStack,  device = device, dtype = dtype)

		# Pairwise distances: (batchNumVariables, numTrain, numTest)
		diff            = trainTensor.unsqueeze(2) - testTensor.unsqueeze(1)  # (batchNumVariables, numTrain, numTest, embeddingDimension)
		distanceMatrix  = torch.sqrt((diff * diff).sum(dim = -1))             # (batchNumVariables, numTrain, numTest)

		if hasMask:
			distanceMatrix[:, maskTensor] = float('inf')

		# topk along train dim → (batchNumVariables, knn, numTest), then transpose → (batchNumVariables, numTest, knn)
		topKDistances, topKIndices = torch.topk(distanceMatrix, knn, dim = 1, largest = False)
		topKDistances = topKDistances.permute(0, 2, 1)  # (batchNumVariables, numTest, knn)
		topKIndices   = topKIndices.permute(0, 2, 1)    # (batchNumVariables, numTest, knn) — local indices into trainIndices

		# Exponential weights
		minDistances = topKDistances[:, :, 0].clone()
		torch.clamp_min_(minDistances, 1e-6)
		weights    = torch.exp(-topKDistances / minDistances.unsqueeze(-1))  # (batchNumVariables, numTest, knn)
		weightSum  = weights.sum(dim = -1)                                   # (batchNumVariables, numTest)

		# Map local kNN indices → actual row indices → +predictionHorizon
		actualIndices  = trainIndexTensor[topKIndices] + predictionHorizon  # (batchNumVariables, numTest, knn)
		libraryTargets = targetTensor[actualIndices]                         # (batchNumVariables, numTest, knn)

		predictions      = (weights * libraryTargets).sum(dim = -1) / weightSum  # (batchNumVariables, numTest)
		predictionsNumpy = predictions.cpu().numpy()

		for variableIndex in range(batchNumVariables):
			correlation = ComputeError(observations[:numValid], predictionsNumpy[variableIndex, :numValid], None)
			correlations[batchStart + variableIndex] = correlation if correlation is not None else numpy.nan

		del trainTensor, testTensor, diff, distanceMatrix, topKDistances, topKIndices
		del weights, weightSum, actualIndices, libraryTargets, predictions
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	del targetTensor, trainIndexTensor
	if hasMask:
		del maskTensor

	return correlations


def BatchedFindOptimalEmbeddingDimensionality(X: numpy.ndarray,
											  Y: numpy.ndarray,
											  maxE: int = 10,
											  predictionHorizon: int = 1,
											  step: int = -1,
											  exclusionRadius: float = 0,
											  trainBlockIndices: List[int] = None,
											  testBlockIndices: List[int] = None,
											  ignoreNan: bool = True,
											  device = None,
											  batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal embedding dimension for multiple source variables in
	parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	:param X: 				(N, M) array of source variables (no time column)
	:param Y: 				(N,) or (N, 1) target variable (no time column)
	:param maxE: 			Maximum embedding dimension to test
	:param predictionHorizon: Prediction horizon
	:param step: 			Time delay step size (tau)
	:param exclusionRadius: Temporal exclusion radius
	:param trainBlockIndices: [start, end] train indices (1-indexed)
	:param testBlockIndices:  [start, end] test indices (1-indexed)
	:param ignoreNan: 		Whether to ignore NaN values
	:param device: 			torch device or device string ('cpu', 'cuda')
	:param batchSize: 		Number of variables per GPU batch
	:return: (dimensions, correlations) where dimensions is [1..maxE] and
	         correlations is (M, maxE) — row m holds correlations for variable m
	"""
	if X.ndim == 1:
		X = X[:, numpy.newaxis]
	if Y.ndim == 2:
		Y = Y.squeeze(axis = 1)

	numTimepoints, numVariables = X.shape

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	dtype = torch.float64

	train = trainBlockIndices if trainBlockIndices is not None else [1, numTimepoints]
	test  = testBlockIndices  if testBlockIndices  is not None else [1, numTimepoints]

	dimensions   = list(range(1, maxE + 1))
	correlations = numpy.zeros((numVariables, maxE))

	for embeddingDimension in dimensions:
		variableCorrelations = _BatchedSimplexPredict(
			X, Y, embeddingDimension, predictionHorizon, step, exclusionRadius,
			train, test, ignoreNan, device, dtype, batchSize)
		correlations[:, embeddingDimension - 1] = variableCorrelations

	return dimensions, correlations


def BatchedFindOptimalPredictionHorizon(X: numpy.ndarray,
										Y: numpy.ndarray,
										maxTp: int = 10,
										embedDimensions: int = 1,
										step: int = -1,
										exclusionRadius: float = 0,
										trainBlockIndices: List[int] = None,
										testBlockIndices: List[int] = None,
										ignoreNan: bool = True,
										device = None,
										batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal prediction horizon for multiple source variables in
	parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	:param X: 				(N, M) array of source variables (no time column)
	:param Y: 				(N,) or (N, 1) target variable (no time column)
	:param maxTp: 			Maximum prediction horizon to test
	:param embedDimensions: Embedding dimension
	:param step: 			Time delay step size (tau)
	:param exclusionRadius: Temporal exclusion radius
	:param trainBlockIndices: [start, end] train indices (1-indexed)
	:param testBlockIndices:  [start, end] test indices (1-indexed)
	:param ignoreNan: 		Whether to ignore NaN values
	:param device: 			torch device or device string ('cpu', 'cuda')
	:param batchSize: 		Number of variables per GPU batch
	:return: (predictionHorizonValues, correlations) where predictionHorizonValues is [1..maxTp] and
	         correlations is (M, maxTp) — row m holds correlations for variable m
	"""
	if X.ndim == 1:
		X = X[:, numpy.newaxis]
	if Y.ndim == 2:
		Y = Y.squeeze(axis = 1)

	numTimepoints, numVariables = X.shape

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	dtype = torch.float64

	train                    = trainBlockIndices if trainBlockIndices is not None else [1, numTimepoints]
	test                     = testBlockIndices  if testBlockIndices  is not None else [1, numTimepoints]
	predictionHorizonValues  = list(range(1, maxTp + 1))
	correlations             = numpy.zeros((numVariables, maxTp))

	for index, horizon in enumerate(predictionHorizonValues):
		variableCorrelations = _BatchedSimplexPredict(
			X, Y, embedDimensions, horizon, step, exclusionRadius,
			train, test, ignoreNan, device, dtype, batchSize)
		correlations[:, index] = variableCorrelations

	return predictionHorizonValues, correlations


def BatchedFindOptimalDelay(X: numpy.ndarray,
							Y: numpy.ndarray,
							stepValues: List[int] = None,
							maxTau: int = 10,
							embedDimensions: int = 1,
							predictionHorizon: int = 1,
							exclusionRadius: float = 0,
							trainBlockIndices: List[int] = None,
							testBlockIndices: List[int] = None,
							ignoreNan: bool = True,
							device = None,
							batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal time-delay step size (tau) for multiple source variables
	in parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	:param X: 				(N, M) array of source variables (no time column)
	:param Y: 				(N,) or (N, 1) target variable (no time column)
	:param stepValues: 		Step (tau) values to test; if None, uses [-1, -2, ..., -maxTau]
	:param maxTau: 			Maximum absolute tau when stepValues is None
	:param embedDimensions: Embedding dimension
	:param predictionHorizon: Prediction horizon
	:param exclusionRadius: Temporal exclusion radius
	:param trainBlockIndices: [start, end] train indices (1-indexed)
	:param testBlockIndices:  [start, end] test indices (1-indexed)
	:param ignoreNan: 		Whether to ignore NaN values
	:param device: 			torch device or device string ('cpu', 'cuda')
	:param batchSize: 		Number of variables per GPU batch
	:return: (stepValues, correlations) where correlations is (M, len(stepValues)) —
	         row m holds correlations for variable m across all step values
	"""
	if X.ndim == 1:
		X = X[:, numpy.newaxis]
	if Y.ndim == 2:
		Y = Y.squeeze(axis = 1)

	numTimepoints, numVariables = X.shape

	if stepValues is None:
		stepValues = [-delayStep for delayStep in range(1, maxTau + 1)]

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	dtype = torch.float64

	train        = trainBlockIndices if trainBlockIndices is not None else [1, numTimepoints]
	test         = testBlockIndices  if testBlockIndices  is not None else [1, numTimepoints]
	correlations = numpy.zeros((numVariables, len(stepValues)))

	for index, delayStep in enumerate(stepValues):
		variableCorrelations = _BatchedSimplexPredict(
			X, Y, embedDimensions, predictionHorizon, delayStep, exclusionRadius,
			train, test, ignoreNan, device, dtype, batchSize)
		correlations[:, index] = variableCorrelations

	return stepValues, correlations
