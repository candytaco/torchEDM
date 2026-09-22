"""
Cross-map skill of every source column against every target column, measured across
training-subset sizes: skill that keeps growing with the subset is the convergence signal.
One direction per call; the reverse direction is a second call with X and Y exchanged.
"""
from dataclasses import dataclass
from typing import Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .Results import BatchedCCMResult
from ._core import batch_get_simplex_weights, CorrelationInPlace, ComputeSimplexWeights
from .Setup import ArrayOrRuns, AsRuns, PreparePrediction, PredictionInputs, TestTargets
from .Predictors import ResolveDevice
from .utils import _get_embedding_dimension
from ..Hyperparameters import FindSelfPredictionEmbeddingDimension


@dataclass(frozen = True)
class _CrossMapSettings:
	"""The settings the batch helpers read."""
	trainSizes: list
	repeats: int
	knn: Optional[int]
	exclusionRadius: int
	sourceBatchSize: int
	targetBatchSize: int
	sampleBatchSize: Optional[int]
	targetVRAM: Optional[float]
	hasProgressBar: bool
	device: torch.device
	dtype: torch.dtype


def ConvergentCrossMap(X_train: ArrayOrRuns,
					   Y_train: Optional[ArrayOrRuns] = None,
					   X_test: Optional[ArrayOrRuns] = None,
					   Y_test: Optional[ArrayOrRuns] = None,
					   embedDimensions = None,
					   step: int = -1,
					   predictionHorizon: int = 1,
					   knn: Optional[int] = None,
					   exclusionRadius: int = 0,
					   trainRowMask: Optional[ArrayOrRuns] = None,
					   trainSizes = None,
					   repeats: int = 10,
					   maxEmbedDimensions: int = 20,
					   seed = None,
					   batchMode: str = 'variables',
					   sourceBatchSize: int = 1000,
					   targetBatchSize: int = 2000,
					   sampleBatchSize: Optional[int] = None,
					   targetVRAM: Optional[float] = None,
					   hasProgressBar: bool = True,
					   device = 'cuda',
					   dtype: torch.dtype = torch.float16) -> BatchedCCMResult:
	"""
	Every source column, stacked to its own embedding dimensions, predicts every target
	column from random training subsets of increasing size.

	:param X_train:		[nTrain, nSources] or a list of runs: the columns whose stacked histories predict the targets
	:param Y_train:		[nTrain, nTargets] or a list of runs; None cross-maps every X column onto every X column
	:param X_test:		[nTest, nSources] or a list of runs; None scores the training rows in-sample (the usual setting)
	:param Y_test:		targets for X_test; required with X_test unless Y_train is None
	:param embedDimensions:	embedding dimensions per source: an int, [nSources], or [nSources, nTargets]; None searches
		each source's embedding dimension up to maxEmbedDimensions with FindSelfPredictionEmbeddingDimension
	:param step:		row offset between stacked copies; negative reaches into the past
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors; None means embedding dimensions + 1 per source
	:param exclusionRadius:	in-sample only; training states this close in rows are not neighbors
	:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
	:param trainSizes:	training-subset sizes; None uses 10, 25, 50, 75 and 90 percent of the training rows
	:param repeats:		random subsets drawn per size
	:param maxEmbedDimensions:	largest embedding dimension tried when embedDimensions is None
	:param seed:		random seed for the subsets
	:param batchMode:	'variables' batches over source columns and predicts the test rows;
		'sample' batches over subsets and scores the training rows predicting themselves
	:param sourceBatchSize:	source columns per batch in 'variables' mode (also the search batch size)
	:param targetBatchSize:	target columns per batch within a source batch
	:param sampleBatchSize:	subsets per batch in 'sample' mode; None takes all at once
	:param targetVRAM:	GB budget that sizes the batches when given
	:param hasProgressBar:	show progress bars
	:param device:		torch device; cuda falls back to cpu when unavailable
	:param dtype:		torch dtype for the computation
	:return: BatchedCCMResult with forward_performance [nSizes, nSources, nTargets], singleton axes squeezed
	"""
	isSelfMap = Y_train is None
	Y_train = X_train if isSelfMap else Y_train
	Y_test = X_test if isSelfMap else Y_test
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score predictions on X_test')
	xRuns = AsRuns(X_train)
	numSources = xRuns[0].shape[1]
	numTargets = AsRuns(Y_train)[0].shape[1]
	device = ResolveDevice(device)
	if trainSizes is None:
		numTrainSamples = sum(x.shape[0] for x in xRuns)
		trainSizes = [int(p * numTrainSamples) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
	RNG = numpy.random.default_rng(seed)

	if embedDimensions is None:
		embedDimensions = FindSelfPredictionEmbeddingDimension(
			X_train, X_test,
			maxDims = maxEmbedDimensions,
			step = step,
			predictionHorizon = predictionHorizon,
			exclusionRadius = exclusionRadius,
			trainRowMask = trainRowMask,
			batchSize = sourceBatchSize,
			targetVRAM = targetVRAM,
			hasProgressBar = hasProgressBar,
			device = device,
			dtype = dtype)

	maxEmbeddingDims = int(numpy.max(embedDimensions))
	inputs = PreparePrediction(X_train, Y_train, X_test, maxEmbeddingDims, step, predictionHorizon, exclusionRadius, trainRowMask)
	testTargets = TestTargets(Y_test if X_test is not None else Y_train, inputs)
	isScored = numpy.isfinite(testTargets).all(axis = 1)

	trainTargetTensor = torch.as_tensor(inputs.trainTargets, dtype = dtype, device = device)
	testTargetTensor = torch.as_tensor(testTargets[isScored], dtype = dtype, device = device)

	settings = _CrossMapSettings(trainSizes = list(trainSizes), repeats = repeats, knn = knn, exclusionRadius = exclusionRadius,
								 sourceBatchSize = sourceBatchSize, targetBatchSize = targetBatchSize,
								 sampleBatchSize = sampleBatchSize, targetVRAM = targetVRAM,
								 hasProgressBar = hasProgressBar, device = device, dtype = dtype)

	# kept in CPU RAM: [nSizes, repeats, nSources, nTargets] can be large
	performance = numpy.zeros([len(trainSizes), repeats, numSources, numTargets])

	if batchMode == 'sample':
		_CrossMapSampleBatched(inputs, trainTargetTensor, performance, RNG, embedDimensions, settings)
	else:
		exclusion = None if inputs.exclusionMask is None else inputs.exclusionMask[:, isScored]
		_CrossMapVariableBatched(inputs.trainStates, trainTargetTensor, inputs.testStates[isScored], testTargetTensor,
								 performance, RNG, embedDimensions, settings, exclusion)

	return BatchedCCMResult(
		forward_performance = numpy.mean(performance, axis = 1).squeeze(),
		predictionHorizon = predictionHorizon,
		library_sizes = trainSizes,
		forward_embed_dimensions = embedDimensions)


def _CrossMapVariableBatched(X_train, Y_train, X_test, Y_test, performance, RNG, embedDims, settings: _CrossMapSettings,
							 exclusion = None):
	"""
	Batch over source columns: one distance matrix per source, reused across every
	(subset size, repeat, target). X_train/X_test hold every source stacked to
	the largest embedding dimension, source-major; Y_train/Y_test are [nTrain, nTargets] and
	[nTest, nTargets] tensors.
	"""
	numTrain = X_train.shape[0]
	numTest = X_test.shape[0]
	numTargets = Y_train.shape[1]
	maxEmbeddingDims = int(numpy.max(embedDims))
	numSources = X_train.shape[1] // maxEmbeddingDims
	embedDimsArray = numpy.asarray(embedDims)
	if embedDimsArray.ndim == 0:
		embedDimsArray = numpy.full(numSources, int(embedDimsArray))
	elif embedDimsArray.ndim == 2:
		# one embedding dimension per source in this mode: the largest over its targets
		embedDimsArray = embedDimsArray.max(axis = 1)

	# When targetVRAM is given, the budget sets both batch sizes; sourceBatchSize and
	# targetBatchSize are the defaults otherwise. Tensors scaling with the source batch:
	#   sourceDistanceMatrices [sourceBatch, numTrain, numTest]       (per source batch)
	#   subsampledDistances    [sourceBatch, maxTrainSize, numTest]   (transient during topk)
	#   neighborIndices/Weights [sourceBatch, maxKnn, numTest]         (per sample)
	#   flatIndices, weightsForBmm                                     (permuted copies per sample)
	#   valuesForBmm           [sourceBatch * numTest, maxKnn, targetBatchSize] (transient per target batch; dominant)
	conservativeMaxKnn = maxEmbeddingDims + 1
	elementSize = torch.zeros(1, dtype = settings.dtype).element_size()
	if settings.targetVRAM is None:
		sourceBatchSize = settings.sourceBatchSize
		targetBatchSize = settings.targetBatchSize
	else:
		targetVRAMBytes = settings.targetVRAM * 1e9
		maxTrainSize = min(max(settings.trainSizes), numTrain)
		sourceOnlyBytesPerSource = ((numTrain + maxTrainSize) * numTest * elementSize +
		                            2 * conservativeMaxKnn * numTest * (8 + elementSize))
		yBytesPerSourcePerTarget = conservativeMaxKnn * numTest * elementSize
		perSourceBytes = sourceOnlyBytesPerSource + yBytesPerSourcePerTarget * settings.targetBatchSize
		sourceBatchSize = max(1, int(targetVRAMBytes / perSourceBytes))
		actualBatchSizeEstimate = min(sourceBatchSize, numSources)
		remainingBytes = targetVRAMBytes - actualBatchSizeEstimate * sourceOnlyBytesPerSource
		vramTargetBatch = max(1, int(remainingBytes / (actualBatchSizeEstimate * yBytesPerSourcePerTarget)))
		targetBatchSize = max(settings.targetBatchSize, vramTargetBatch)

	perLagSquaredDistances = torch.zeros([maxEmbeddingDims, numTrain, numTest], dtype = settings.dtype, device = settings.device)

	for sourceBatchStart in ProgressBar(range(0, numSources, sourceBatchSize), desc = 'Source batch', leave = False,
										disable = not settings.hasProgressBar):
		sourceBatchEnd = min(sourceBatchStart + sourceBatchSize, numSources)
		actualSourceBatchSize = sourceBatchEnd - sourceBatchStart
		sourceEmbedDims = embedDimsArray[sourceBatchStart:sourceBatchEnd]

		if settings.knn is not None:
			numNeighbors = settings.knn
			maxKnn = settings.knn
		else:
			maxKnn = int(numpy.max(sourceEmbedDims)) + 1
			numNeighborsPerSource = torch.tensor(sourceEmbedDims + 1, dtype = torch.long, device = settings.device)
			if int(torch.unique(numNeighborsPerSource).shape[0]) == 1:
				numNeighbors = int(numNeighborsPerSource[0].item())
			else:
				numNeighbors = numNeighborsPerSource

		sourceDistanceMatrices = torch.zeros([actualSourceBatchSize, numTrain, numTest], dtype = settings.dtype, device = settings.device)

		for localSourceIndex in range(actualSourceBatchSize):
			globalSourceIndex = sourceBatchStart + localSourceIndex
			sourceColumns = slice(globalSourceIndex * maxEmbeddingDims, (globalSourceIndex + 1) * maxEmbeddingDims)
			trainSourceTensor = torch.as_tensor(X_train[:, sourceColumns], dtype = settings.dtype, device = settings.device)
			testSourceTensor = torch.as_tensor(X_test[:, sourceColumns], dtype = settings.dtype, device = settings.device)

			for lagIndex in range(maxEmbeddingDims):
				perLagSquaredDistances[lagIndex] = trainSourceTensor[:, lagIndex].unsqueeze(1) - testSourceTensor[:, lagIndex].unsqueeze(0)
			del trainSourceTensor, testSourceTensor

			perLagSquaredDistances.square_()
			torch.cumsum(perLagSquaredDistances, dim = 0, out = perLagSquaredDistances)

			sourceDim = int(sourceEmbedDims[localSourceIndex])
			sourceDistanceMatrices[localSourceIndex] = perLagSquaredDistances[sourceDim - 1]

		if exclusion is not None:
			sourceDistanceMatrices[:, torch.as_tensor(exclusion, device = settings.device)] = float('inf')

		performanceBuffer = torch.zeros([actualSourceBatchSize, targetBatchSize], dtype = settings.dtype, device = settings.device)

		for sizeIndex, trainSize in enumerate(ProgressBar(settings.trainSizes, desc = 'Training-subset sizes', leave = False,
														disable = not settings.hasProgressBar)):
			trainSize = min(trainSize, numTrain)
			for repeatIndex in ProgressBar(range(settings.repeats), desc = 'Repeats', leave = False, disable = not settings.hasProgressBar):
				sampledIndices = torch.as_tensor(RNG.choice(numTrain, size = trainSize, replace = False),
												 dtype = torch.long, device = settings.device)

				subsampledDistances = sourceDistanceMatrices[:, sampledIndices, :]
				neighborIndices, neighborWeights = batch_get_simplex_weights(subsampledDistances, numNeighbors, sampledIndices)
				del subsampledDistances

				flatIndices = neighborIndices.permute(0, 2, 1).contiguous().view(actualSourceBatchSize * numTest, maxKnn)
				weightsForBmm = neighborWeights.permute(0, 2, 1).contiguous().view(actualSourceBatchSize * numTest, 1, maxKnn)

				for targetBatchStart in range(0, numTargets, targetBatchSize):
					targetBatchEnd = min(targetBatchStart + targetBatchSize, numTargets)
					actualTargetBatchSize = targetBatchEnd - targetBatchStart

					yBatch = Y_train[:, targetBatchStart:targetBatchEnd]
					valuesForBmm = yBatch[flatIndices]
					predictions = torch.bmm(weightsForBmm, valuesForBmm).view(actualSourceBatchSize, numTest, actualTargetBatchSize)
					del valuesForBmm

					CorrelationInPlace(Y_test[:, targetBatchStart:targetBatchEnd], predictions,
					                   out = performanceBuffer[:, :actualTargetBatchSize])
					performance[sizeIndex, repeatIndex, sourceBatchStart:sourceBatchEnd,
								targetBatchStart:targetBatchEnd] = performanceBuffer[:, :actualTargetBatchSize].cpu().numpy()

				del flatIndices, weightsForBmm

		del sourceDistanceMatrices, performanceBuffer
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

def _CrossMapSampleBatched(inputs: PredictionInputs, Y_train, performance, RNG, embedDims, settings: _CrossMapSettings):
	"""
	Batch over subsets per size; the training rows predict themselves. Efficient when
	there are few source columns. Cumulative per-lag squared distances are built once per
	source so each (source, target) pair reads its own embedding dimension off the prefix sum. Without a
	user-set knn, each pair uses embedding dimensions + 1 neighbors, enforced by masking the extra
	neighbors to zero weight after one shared topk.
	"""
	numSamplesInBatch = settings.sampleBatchSize if settings.sampleBatchSize is not None else settings.repeats
	numTargets = Y_train.shape[1]
	dims = int(numpy.max(embedDims))
	numSources = inputs.trainStates.shape[1] // dims
	numTrainStates = inputs.numTrainingPairs

	# [numSources, N, dims] from the source-major stacked states
	sourceStates = torch.as_tensor(
		inputs.trainStates.reshape(numTrainStates, numSources, dims), dtype = settings.dtype,
		device = settings.device).permute(1, 0, 2).contiguous()

	maxNeighbors = settings.knn if settings.knn is not None else dims + 1

	d = torch.zeros([dims, numTrainStates, numTrainStates], dtype = settings.dtype, device = settings.device)
	cumulativeSqDist = torch.zeros([numSources, dims, numTrainStates, numTrainStates], dtype = settings.dtype, device = settings.device)
	for i in range(numSources):
		for lag in range(dims):
			d[lag] = sourceStates[i, :, lag].unsqueeze(1) - sourceStates[i, :, lag].unsqueeze(0)
		d.square_()
		cumulativeSqDist[i] = torch.cumsum(d, dim = 0)
	del sourceStates, d

	fullDistances = torch.zeros([numSources, numTargets, numTrainStates, numTrainStates], dtype = settings.dtype, device = settings.device)
	for i in range(numSources):
		for t in range(numTargets):
			e = _get_embedding_dimension(embedDims, i, t)
			fullDistances[i, t] = torch.sqrt(cumulativeSqDist[i, e - 1])
	del cumulativeSqDist

	# the self-match is always excluded; a positive exclusionRadius also excludes every
	# pair within that many rows of each other in the same run
	diagIndices = torch.arange(numTrainStates, device = settings.device)
	fullDistances[:, :, diagIndices, diagIndices] = float('inf')
	if settings.exclusionRadius > 0:
		rowNumbers = torch.as_tensor(inputs.SharedAxisRows()[0], dtype = torch.long, device = settings.device)
		excludedPairs = (rowNumbers.unsqueeze(0) - rowNumbers.unsqueeze(1)).abs() <= settings.exclusionRadius
		fullDistances[:, :, excludedPairs] = float('inf')

	kIndices = torch.arange(maxNeighbors, device = settings.device).view(1, 1, maxNeighbors, 1)

	for sizeIndex, trainSize in enumerate(ProgressBar(settings.trainSizes, desc = 'Training-subset sizes', leave = False,
												 disable = not settings.hasProgressBar)):
		actualTrainSize = min(trainSize, numTrainStates)

		for batchStart in ProgressBar(range(0, settings.repeats, numSamplesInBatch), desc = 'Sample batch',
									  leave = False, disable = not settings.hasProgressBar):
			batchEnd = min(batchStart + numSamplesInBatch, settings.repeats)
			numSamplesInThisBatch = batchEnd - batchStart

			subsampleIndices = numpy.stack([RNG.choice(numTrainStates, size = actualTrainSize, replace = False)
											for _ in range(numSamplesInThisBatch)])
			subsampleTorch = torch.as_tensor(subsampleIndices, dtype = torch.long, device = settings.device)

			for t in range(numTargets):
				subsampledDistances = fullDistances[:, t][:, subsampleTorch, :]
				distances, neighbors = torch.topk(subsampledDistances, maxNeighbors, dim = 2, largest = False)

				if settings.knn is None:
					knnPerSource = torch.tensor([_get_embedding_dimension(embedDims, i, t) + 1 for i in range(numSources)],
												dtype = torch.long, device = settings.device).view(numSources, 1, 1, 1)
					distances.masked_fill_(kIndices >= knnPerSource, float('inf'))

				subsampleExpanded = subsampleTorch.unsqueeze(0).unsqueeze(-1)
				globalNeighbors = subsampleExpanded.expand(numSources, -1, actualTrainSize, numTrainStates).gather(dim = 2, index = neighbors)

				weights = ComputeSimplexWeights(distances)
				weightSum = weights.sum(dim = 2)

				targetT = Y_train[:, t]
				selectedTargets = targetT[globalNeighbors]
				predictions = (weights * selectedTargets).sum(dim = 2) / weightSum

				targetCentered = targetT - targetT.mean()
				targetStd = torch.sqrt((targetCentered ** 2).sum())
				predCentered = predictions - predictions.mean(dim = 2, keepdim = True)
				predStd = torch.sqrt((predCentered ** 2).sum(dim = 2))
				correlations = (targetCentered * predCentered).sum(dim = 2) / (targetStd * predStd)

				performance[sizeIndex, batchStart:batchEnd, :, t] = correlations.permute(1, 0).cpu().numpy()

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
