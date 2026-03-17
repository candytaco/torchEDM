from typing import List, Optional, Tuple, Any

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




def _EvaluateCandidateDistanceMatrices(
	distanceMatrices: torch.Tensor,
	targetTensor: torch.Tensor,
	trainIndexTensor: torch.Tensor,
	testIndices: numpy.ndarray,
	predictionHorizon: int,
	knn: int,
	numTimepoints: int,
) -> numpy.ndarray:
	"""
	Evaluate prediction performance for a batch of candidates given their
	pre-computed (and pre-masked) pairwise distance matrices.

	Each candidate corresponds to one (source variable, hyperparameter value)
	combination. Distance matrices must already have the temporal exclusion mask
	applied (masked positions set to infinity) before being passed in.

	Pearson correlations are computed in a fully vectorized way across all
	candidates using tensor broadcast operations — no Python loop over candidates.

	:param distanceMatrices:   (numCandidates, numTrain, numTest) distance matrices on device
	:param targetTensor:       (N,) full target time-series on device
	:param trainIndexTensor:   (numTrain,) global train indices (long) on device
	:param testIndices:        (numTest,) global test indices (numpy int array)
	:param predictionHorizon:  Steps ahead to predict
	:param knn:                Number of nearest neighbors
	:param numTimepoints:      Total number of time points (N)
	:return:                   (numCandidates,) Pearson correlations as numpy array
	"""
	device = distanceMatrices.device
	dtype  = distanceMatrices.dtype

	# kNN along train dimension → (numCandidates, knn, numTest), permute → (numCandidates, numTest, knn)
	topKDistances, topKIndices = torch.topk(distanceMatrices, knn, dim = 1, largest = False)
	topKDistances = topKDistances.permute(0, 2, 1)  # (numCandidates, numTest, knn)
	topKIndices   = topKIndices.permute(0, 2, 1)    # (numCandidates, numTest, knn)

	# Exponential weights using nearest-neighbor distance as scale
	minDistances = topKDistances[:, :, 0].unsqueeze(-1).clamp(min = 1e-6)
	weights      = torch.exp(-topKDistances / minDistances)  # (numCandidates, numTest, knn)
	weightSum    = weights.sum(dim = -1)                     # (numCandidates, numTest)

	# Map local kNN indices → global time indices → shift by predictionHorizon
	actualIndices  = trainIndexTensor[topKIndices] + predictionHorizon  # (numCandidates, numTest, knn)
	libraryTargets = targetTensor[actualIndices]                         # (numCandidates, numTest, knn)

	# Weighted predictions: (numCandidates, numTest)
	predictions = (weights * libraryTargets).sum(dim = -1) / weightSum

	# Build valid observation window: testIndices + predictionHorizon must be within [0, numTimepoints)
	observationIndices = testIndices + predictionHorizon
	validObsMask       = (observationIndices >= 0) & (observationIndices < numTimepoints)
	numValid           = int(validObsMask.sum())

	validObsGlobal = torch.tensor(
		observationIndices[validObsMask], device = device, dtype = torch.long)
	observations   = targetTensor[validObsGlobal]  # (numValid,)

	# Slice predictions to the valid range (invalid obs are at the end when Tp > 0)
	validPredictions = predictions[:, :numValid]   # (numCandidates, numValid)

	# Vectorized Pearson correlation across all candidates
	# Joint finite mask: both observations and predictions must be finite
	obsFiniteMask    = observations.isfinite().unsqueeze(0)   # (1, numValid)
	predFiniteMask   = validPredictions.isfinite()            # (numCandidates, numValid)
	jointFiniteMask  = obsFiniteMask & predFiniteMask         # (numCandidates, numValid)
	jointFiniteFloat = jointFiniteMask.to(dtype)

	validCount = jointFiniteMask.sum(dim = -1).to(dtype)      # (numCandidates,)

	# Masked means
	maskedPredictions  = validPredictions * jointFiniteFloat
	maskedObservations = observations.unsqueeze(0) * jointFiniteFloat

	predictionMean  = maskedPredictions.sum(dim = -1) / validCount.clamp(min = 1)   # (numCandidates,)
	observationMean = maskedObservations.sum(dim = -1) / validCount.clamp(min = 1)  # (numCandidates,)

	# Centered and masked deviations
	predCentered = (validPredictions - predictionMean.unsqueeze(-1)) * jointFiniteFloat
	obsCentered  = (observations.unsqueeze(0) - observationMean.unsqueeze(-1)) * jointFiniteFloat

	covariance    = (predCentered * obsCentered).sum(dim = -1)
	predVariance  = (predCentered ** 2).sum(dim = -1)
	obsVariance   = (obsCentered ** 2).sum(dim = -1)

	denominator   = torch.sqrt(predVariance * obsVariance).clamp(min = 1e-10)
	correlations  = covariance / denominator

	# Set to NaN where fewer than 5 valid points (matching ComputeError behaviour)
	insufficientMask = validCount < 5
	if insufficientMask.any():
		correlations = correlations.clone()
		correlations[insufficientMask] = float('nan')

	return correlations.cpu().numpy()


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
	Estimate optimal time-delay step size (tau) using tensor-batched Simplex
	projection.  All step-value candidates are evaluated in a single pass using
	pre-computed distance matrices rather than naively iterating with Simplex.Run().

	A single reference Simplex (at the most restrictive step = most negative value
	in stepValues) provides shared train/test indices.  Distance matrices for every
	step value are then stacked into a (numStepValues, numTrain, numTest) tensor and
	evaluated simultaneously by _EvaluateCandidateDistanceMatrices.

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

	# Reference step: the most restrictive (most negative) provides the largest
	# NaN-exclusion window, so all other steps are valid subsets of those indices.
	referenceStep = min(stepValues)

	referenceModel = Simplex(data = data, columns = columns, target = target,
							 train = train, test = test, embedDimensions = embedDimensions,
							 predictionHorizon = predictionHorizon, knn = 0,
							 step = referenceStep, exclusionRadius = exclusionRadius,
							 embedded = embedded, validLib = validLib,
							 noTime = noTime, ignoreNan = ignoreNan)
	referenceModel.EmbedData()
	referenceModel.RemoveNan()

	device = referenceModel.device
	dtype  = referenceModel.dtype

	trainIndices  = referenceModel.trainIndices
	testIndices   = referenceModel.testIndices
	numTimepoints = len(referenceModel.targetVec)

	targetTensor     = torch.tensor(
		referenceModel.targetVec.squeeze(), device = device, dtype = dtype)
	trainIndexTensor = torch.tensor(trainIndices, device = device, dtype = torch.long)

	knn = embedDimensions + 1

	exclusionMask       = referenceModel._BuildExclusionMask()
	hasMask             = exclusionMask.any()
	exclusionMaskTensor = (
		torch.tensor(exclusionMask, device = device, dtype = torch.bool) if hasMask else None)

	# Compute distance matrices for all step values and stack into a single tensor
	allDistanceMatrices = []
	for step in stepValues:
		stepModel = Simplex(data = data, columns = columns, target = target,
							train = train, test = test, embedDimensions = embedDimensions,
							predictionHorizon = predictionHorizon, knn = 0,
							step = step, exclusionRadius = exclusionRadius,
							embedded = embedded, validLib = validLib,
							noTime = noTime, ignoreNan = ignoreNan)
		stepModel.EmbedData()

		trainEmbedding = torch.tensor(
			stepModel.Embedding[trainIndices, :], device = device, dtype = dtype)
		testEmbedding  = torch.tensor(
			stepModel.Embedding[testIndices, :],  device = device, dtype = dtype)

		# Pairwise distances: (numTrain, numTest)
		diff           = trainEmbedding.unsqueeze(1) - testEmbedding.unsqueeze(0)
		distanceMatrix = torch.sqrt((diff * diff).sum(dim = -1))
		allDistanceMatrices.append(distanceMatrix.unsqueeze(0))

	# Stack → (numStepValues, numTrain, numTest)
	distanceMatrices = torch.cat(allDistanceMatrices, dim = 0)

	if hasMask:
		distanceMatrices[:, exclusionMaskTensor] = float('inf')

	correlations = _EvaluateCandidateDistanceMatrices(
		distanceMatrices, targetTensor, trainIndexTensor, testIndices,
		predictionHorizon, knn, numTimepoints)

	del distanceMatrices, targetTensor, trainIndexTensor, allDistanceMatrices
	if hasMask:
		del exclusionMaskTensor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return numpy.column_stack([stepValues, correlations])


def BatchedFindOptimalEmbeddingDimensionality(X: numpy.ndarray,
											  Y: numpy.ndarray,
											  maxE: int = 10,
											  predictionHorizon: int = 1,
											  step: int = -1,
											  exclusionRadius: float = 0,
											  trainBlockIndices: List[int] = None,
											  testBlockIndices: List[int] = None,
											  ignoreNan: bool = True,
											  device: Optional[torch.device] = None,
											  batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal embedding dimension for multiple source variables in
	parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	For each embedding dimension E, distance matrices for up to batchSize source
	variables are computed and passed as a single (batchSize, numTrain, numTest)
	candidate tensor to _EvaluateCandidateDistanceMatrices.  A separate Simplex at
	each E provides the correct per-E train/test indices.

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
	:param batchSize: 		Number of candidate variables per GPU batch
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

	targetTensor = torch.tensor(Y, device = device, dtype = dtype)

	dimensions   = list(range(1, maxE + 1))
	correlations = numpy.zeros((numVariables, maxE))

	for embeddingDimension in dimensions:
		knn = embeddingDimension + 1

		# Per-E dummy Simplex provides correct NaN-filtered train/test indices for this E
		dummy = Simplex(data = X, columns = [0], target = 0,
						train = train, test = test, embedDimensions = embeddingDimension,
						predictionHorizon = predictionHorizon, knn = knn,
						step = step, exclusionRadius = exclusionRadius,
						embedded = False, validLib = [], noTime = True, ignoreNan = ignoreNan)
		dummy.EmbedData()
		dummy.RemoveNan()

		trainIndices     = dummy.trainIndices
		testIndices      = dummy.testIndices
		trainIndexTensor = torch.tensor(trainIndices, device = device, dtype = torch.long)

		exclusionMask       = dummy._BuildExclusionMask()
		hasMask             = exclusionMask.any()
		exclusionMaskTensor = (
			torch.tensor(exclusionMask, device = device, dtype = torch.bool) if hasMask else None)

		for batchStart in range(0, numVariables, batchSize):
			batchEnd           = min(batchStart + batchSize, numVariables)
			batchVariableCount = batchEnd - batchStart

			# Compute embeddings for this batch of variables at the current E
			trainEmbeddingsList = []
			testEmbeddingsList  = []
			for variableIndex in range(batchStart, batchEnd):
				embedding = Embed(data = X, columns = [variableIndex],
								  embeddingDimensions = embeddingDimension,
								  stepSize = step, includeTime = False)
				trainEmbeddingsList.append(embedding[trainIndices, :])
				testEmbeddingsList.append(embedding[testIndices, :])

			trainStack = torch.tensor(
				numpy.stack(trainEmbeddingsList), device = device, dtype = dtype)
			testStack  = torch.tensor(
				numpy.stack(testEmbeddingsList),  device = device, dtype = dtype)

			# Pairwise distances: (batchVariableCount, numTrain, numTest)
			diff             = trainStack.unsqueeze(2) - testStack.unsqueeze(1)
			distanceMatrices = torch.sqrt((diff * diff).sum(dim = -1))

			if hasMask:
				distanceMatrices[:, exclusionMaskTensor] = float('inf')

			batchCorrelations = _EvaluateCandidateDistanceMatrices(
				distanceMatrices, targetTensor, trainIndexTensor, testIndices,
				predictionHorizon, knn, numTimepoints)

			correlations[batchStart:batchEnd, embeddingDimension - 1] = batchCorrelations

			del trainStack, testStack, diff, distanceMatrices
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		del trainIndexTensor
		if hasMask:
			del exclusionMaskTensor

	del targetTensor

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
										device: Optional[torch.device] = None,
										batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal prediction horizon for multiple source variables in
	parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	Distance matrices are invariant to predictionHorizon, so they are computed
	once per variable batch and passed to _EvaluateCandidateDistanceMatrices
	separately for each Tp value.  A per-Tp dummy Simplex provides the correct
	NaN-filtered indices for each Tp.

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
	:param batchSize: 		Number of candidate variables per GPU batch
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

	train                   = trainBlockIndices if trainBlockIndices is not None else [1, numTimepoints]
	test                    = testBlockIndices  if testBlockIndices  is not None else [1, numTimepoints]
	predictionHorizonValues = list(range(1, maxTp + 1))
	correlations            = numpy.zeros((numVariables, maxTp))

	targetTensor = torch.tensor(Y, device = device, dtype = dtype)
	knn          = embedDimensions + 1

	for index, horizon in enumerate(predictionHorizonValues):
		# Per-Tp dummy Simplex: CreateIndices trims the library by horizon rows
		dummy = Simplex(data = X, columns = [0], target = 0,
						train = train, test = test, embedDimensions = embedDimensions,
						predictionHorizon = horizon, knn = knn,
						step = step, exclusionRadius = exclusionRadius,
						embedded = False, validLib = [], noTime = True, ignoreNan = ignoreNan)
		dummy.EmbedData()
		dummy.RemoveNan()

		trainIndices     = dummy.trainIndices
		testIndices      = dummy.testIndices
		trainIndexTensor = torch.tensor(trainIndices, device = device, dtype = torch.long)

		exclusionMask       = dummy._BuildExclusionMask()
		hasMask             = exclusionMask.any()
		exclusionMaskTensor = (
			torch.tensor(exclusionMask, device = device, dtype = torch.bool) if hasMask else None)

		for batchStart in range(0, numVariables, batchSize):
			batchEnd           = min(batchStart + batchSize, numVariables)

			trainEmbeddingsList = []
			testEmbeddingsList  = []
			for variableIndex in range(batchStart, batchEnd):
				embedding = Embed(data = X, columns = [variableIndex],
								  embeddingDimensions = embedDimensions,
								  stepSize = step, includeTime = False)
				trainEmbeddingsList.append(embedding[trainIndices, :])
				testEmbeddingsList.append(embedding[testIndices, :])

			trainStack = torch.tensor(
				numpy.stack(trainEmbeddingsList), device = device, dtype = dtype)
			testStack  = torch.tensor(
				numpy.stack(testEmbeddingsList),  device = device, dtype = dtype)

			# Pairwise distances: (batchVariableCount, numTrain, numTest)
			diff             = trainStack.unsqueeze(2) - testStack.unsqueeze(1)
			distanceMatrices = torch.sqrt((diff * diff).sum(dim = -1))

			if hasMask:
				distanceMatrices[:, exclusionMaskTensor] = float('inf')

			batchCorrelations = _EvaluateCandidateDistanceMatrices(
				distanceMatrices, targetTensor, trainIndexTensor, testIndices,
				horizon, knn, numTimepoints)

			correlations[batchStart:batchEnd, index] = batchCorrelations

			del trainStack, testStack, diff, distanceMatrices
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		del trainIndexTensor
		if hasMask:
			del exclusionMaskTensor

	del targetTensor

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
							device: Optional[torch.device] = None,
							batchSize: int = 1000) -> Tuple[List[int], numpy.ndarray]:
	"""
	Find the optimal time-delay step size (tau) for multiple source variables
	in parallel, analogous to how BatchedCCM evaluates multiple Xs against the
	same Y simultaneously.

	A single reference Simplex at the most restrictive step (most negative value in
	stepValues) provides shared train/test indices valid for all step candidates.
	For each step value, distance matrices for up to batchSize variables are
	computed and passed as a (batchSize, numTrain, numTest) candidate tensor to
	_EvaluateCandidateDistanceMatrices.

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
	:param batchSize: 		Number of candidate variables per GPU batch
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

	targetTensor = torch.tensor(Y, device = device, dtype = dtype)
	knn          = embedDimensions + 1

	# Use the most restrictive step (most negative) to build shared train/test indices;
	# this matches FindOptimalDelay's approach and keeps the two functions numerically
	# consistent so that results for a single variable agree to floating-point precision.
	referenceStep = min(stepValues)
	referenceDummy = Simplex(data = X, columns = [0], target = 0,
							 train = train, test = test, embedDimensions = embedDimensions,
							 predictionHorizon = predictionHorizon, knn = knn,
							 step = referenceStep, exclusionRadius = exclusionRadius,
							 embedded = False, validLib = [], noTime = True, ignoreNan = ignoreNan)
	referenceDummy.EmbedData()
	referenceDummy.RemoveNan()

	trainIndices     = referenceDummy.trainIndices
	testIndices      = referenceDummy.testIndices
	trainIndexTensor = torch.tensor(trainIndices, device = device, dtype = torch.long)

	exclusionMask       = referenceDummy._BuildExclusionMask()
	hasMask             = exclusionMask.any()
	exclusionMaskTensor = (
		torch.tensor(exclusionMask, device = device, dtype = torch.bool) if hasMask else None)

	for stepIndex, delayStep in enumerate(stepValues):
		for batchStart in range(0, numVariables, batchSize):
			batchEnd = min(batchStart + batchSize, numVariables)

			trainEmbeddingsList = []
			testEmbeddingsList  = []
			for variableIndex in range(batchStart, batchEnd):
				embedding = Embed(data = X, columns = [variableIndex],
								  embeddingDimensions = embedDimensions,
								  stepSize = delayStep, includeTime = False)
				trainEmbeddingsList.append(embedding[trainIndices, :])
				testEmbeddingsList.append(embedding[testIndices, :])

			trainStack = torch.tensor(
				numpy.stack(trainEmbeddingsList), device = device, dtype = dtype)
			testStack  = torch.tensor(
				numpy.stack(testEmbeddingsList),  device = device, dtype = dtype)

			# Pairwise distances: (batchVariableCount, numTrain, numTest)
			diff             = trainStack.unsqueeze(2) - testStack.unsqueeze(1)
			distanceMatrices = torch.sqrt((diff * diff).sum(dim = -1))

			if hasMask:
				distanceMatrices[:, exclusionMaskTensor] = float('inf')

			batchCorrelations = _EvaluateCandidateDistanceMatrices(
				distanceMatrices, targetTensor, trainIndexTensor, testIndices,
				predictionHorizon, knn, numTimepoints)

			correlations[batchStart:batchEnd, stepIndex] = batchCorrelations

			del trainStack, testStack, diff, distanceMatrices
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	del targetTensor, trainIndexTensor
	if hasMask:
		del exclusionMaskTensor

	return stepValues, correlations
