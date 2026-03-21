from typing import List, Tuple, Any

import numpy
import torch

from .Scoring import Correlation
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
									   batched: bool = False,
									   scoring_function = Correlation):
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
	:param scoring_function: 	Scoring function taking (actual, predicted) and returning a scalar
	:return: best embedding dimensions
	"""

	dimensions = list(range(1, maxE + 1))

	if batched:
		correlations = _FindOptimalEmbeddingDimensionalityBatched(
			data, columns, target, maxE, dimensions, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan, scoring_function)
	else:
		correlations = _FindOptimalEmbeddingDimensionalityIterative(
			data, columns, target, dimensions, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan, scoring_function)

	return dimensions, correlations


def _FindOptimalEmbeddingDimensionalityIterative(data, columns, target, Evals,
												  train, test, predictionHorizon,
												  step, exclusionRadius, embedded,
												  validLib, noTime, ignoreNan,
												  scoring_function):
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
		correlation = scoring_function(result.projection[:, 1], result.projection[:, 2])
		correlations.append(correlation)

	return correlations


def _FindOptimalEmbeddingDimensionalityBatched(data, columns, target, maxE, Evals,
												train, test, predictionHorizon,
												step, exclusionRadius, embedded,
												validLib, noTime, ignoreNan,
												scoring_function):
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
		correlation = scoring_function(observations[:nValid], predictionsNumpy[:nValid])
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
