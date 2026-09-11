"""
Cross-map skill of every source column against every target column, measured across
training-subset sizes: skill that keeps growing with the subset is the convergence signal.
One direction per call; the reverse direction is a second call with X and Y exchanged.
"""
from typing import Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .Results import BatchedCCMResult
from ._core import batch_get_simplex_weights, CorrelationInPlace, ComputeSimplexWeights
from .Setup import ArrayOrRuns, AsRuns, PreparePrediction, TestTargets
from .Predictors import ResolveDevice
from .utils import _get_embedding_dimension
from ..Hyperparameters import FindSelfPredictionEmbeddingDimension


class ConvergentCrossMap:
	"""
	Vectorized cross mapping: every source column, stacked to its own embedding dimensions,
	predicts every target column from random training subsets of increasing size.
	"""

	def __init__(self,
				 X_train: ArrayOrRuns,
				 Y_train: Optional[ArrayOrRuns] = None,
				 X_test: Optional[ArrayOrRuns] = None,
				 Y_test: Optional[ArrayOrRuns] = None,
				 trainSizes = None,
				 repeats: int = 10,
				 embedDimensions = None,
				 maxEmbedDimensions: int = 20,
				 predictionHorizon: int = 1,
				 knn: Optional[int] = None,
				 step: int = -1,
				 exclusionRadius: int = 0,
				 seed = None,
				 trainRowMask: Optional[ArrayOrRuns] = None,
				 device = 'cuda',
				 x_batch: int = 1000,
				 y_batch: int = 2000,
				 targetVRAM: Optional[float] = None,
				 dtype: torch.dtype = torch.float16,
				 showProgress: bool = True,
				 batchMode: str = 'variables',
				 sampleBatchSize: Optional[int] = None):
		"""
		:param X_train:		[nTrain, nSources] or a list of runs: the columns whose stacked histories predict the targets
		:param Y_train:		[nTrain, nTargets] or a list of runs; None cross-maps every X column onto every X column
		:param X_test:		[nTest, nSources] or a list of runs; None scores the training rows in-sample (the usual setting)
		:param Y_test:		targets for X_test; required with X_test unless Y_train is None
		:param trainSizes:	training-subset sizes; None uses 10, 25, 50, 75 and 90 percent of the training rows
		:param repeats:		random subsets drawn per size
		:param embedDimensions:	embedding dimensions per source: an int, [nSources], or [nSources, nTargets]; None searches
			each source's embedding dimension up to maxEmbedDimensions with FindSelfPredictionEmbeddingDimension
		:param maxEmbedDimensions:	largest embedding dimension tried when embedDimensions is None
		:param predictionHorizon:	rows between a state and the target it predicts
		:param knn:			neighbors; None means embedding dimensions + 1 per source
		:param step:		row offset between stacked copies; negative reaches into the past
		:param exclusionRadius:	in-sample only; training states this close in rows are not neighbors
		:param seed:		random seed for the subsets
		:param trainRowMask:	optional bool array (or list per run) barring rows from serving as training states
		:param device:		torch device; cuda falls back to cpu when unavailable
		:param x_batch:		source columns per batch in 'variables' mode (also the search batch size)
		:param y_batch:		target columns per batch within a source batch
		:param targetVRAM:	GB budget that sizes the batches when given
		:param dtype:		torch dtype for the computation
		:param showProgress:	show progress bars
		:param batchMode:	'variables' batches over source columns and predicts the test rows;
			'sample' batches over subsets and scores the training rows predicting themselves
		:param sampleBatchSize:	subsets per batch in 'sample' mode; None takes all at once
		"""
		self.isSelfMap = Y_train is None
		self.X_train = X_train
		self.Y_train = X_train if self.isSelfMap else Y_train
		self.X_test = X_test
		self.Y_test = X_test if self.isSelfMap else Y_test
		if X_test is not None and self.Y_test is None:
			raise ValueError('Y_test is needed to score predictions on X_test')
		xRuns = AsRuns(self.X_train)
		self.numSources = xRuns[0].shape[1]
		self.numTargets = AsRuns(self.Y_train)[0].shape[1]

		self.embedDimensions = embedDimensions
		self.maxEmbedDimensions = maxEmbedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.trainRowMask = trainRowMask
		self.x_batch = x_batch
		self.y_batch = y_batch
		self.targetVRAM = targetVRAM
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize
		self.sample = repeats
		self.seed = seed
		self.device = ResolveDevice(device)
		self.dtype = dtype
		self.showProgress = showProgress

		if trainSizes is not None:
			self.trainSizes = trainSizes
		else:
			numTrainSamples = sum(x.shape[0] for x in xRuns)
			self.trainSizes = [int(p * numTrainSamples) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]

		self.forward_performance_ = None
		self.selectedForwardEmbedDimensions = None

	def Run(self) -> BatchedCCMResult:
		embedDims = self.embedDimensions
		RNG = numpy.random.default_rng(self.seed)

		if embedDims is None:
			embedDims = FindSelfPredictionEmbeddingDimension(
				self.X_train, self.X_test,
				maxDims = self.maxEmbedDimensions,
				predictionHorizon = self.predictionHorizon,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				trainRowMask = self.trainRowMask,
				dtype = self.dtype,
				device = self.device,
				batchSize = self.x_batch,
				targetVRAM = self.targetVRAM,
				showProgress = self.showProgress)

		maxEmbeddingDims = int(numpy.max(embedDims))
		inputs = PreparePrediction(self.X_train, self.Y_train, self.X_test, maxEmbeddingDims, self.step,
								   self.predictionHorizon, self.exclusionRadius, self.trainRowMask)
		testTargets = TestTargets(self.Y_test if self.X_test is not None else self.Y_train, inputs)
		isScored = numpy.isfinite(testTargets).all(axis = 1)

		trainTargetTensor = torch.as_tensor(inputs.trainTargets, dtype = self.dtype, device = self.device)
		testTargetTensor = torch.as_tensor(testTargets[isScored], dtype = self.dtype, device = self.device)

		# kept in CPU RAM: [nSizes, repeats, nSources, nTargets] can be large
		performance = numpy.zeros([len(self.trainSizes), self.sample, self.numSources, self.numTargets])

		if self.batchMode == 'sample':
			self.CrossMapSampleBatched(inputs, trainTargetTensor, performance, RNG, embedDims)
		else:
			exclusion = None if inputs.exclusionMask is None else inputs.exclusionMask[:, isScored]
			self.CrossMapVariableBatched(inputs.trainStates, trainTargetTensor, inputs.testStates[isScored], testTargetTensor,
										 performance, RNG, embedDims, exclusion)

		self.forward_performance_ = numpy.mean(performance, axis = 1).squeeze()
		self.selectedForwardEmbedDimensions = embedDims

		return BatchedCCMResult(
			forward_performance = self.forward_performance_,
			predictionHorizon = self.predictionHorizon,
			library_sizes = self.trainSizes,
			forward_embed_dimensions = self.selectedForwardEmbedDimensions)

	def CrossMapVariableBatched(self, X_train, Y_train, X_test, Y_test, performance, RNG, embedDims, exclusion = None):
		"""
		Batch over source columns: one distance matrix per source, reused across every
		(subset size, repeat, target). X_train/X_test hold every source stacked to
		the largest embedding dimension, source-major; Y_train/Y_test are [nTrain, nTargets] and
		[nTest, nTargets] tensors.
		"""
		numTrain = X_train.shape[0]
		numTest = X_test.shape[0]
		numSources = self.numSources
		numTargets = Y_train.shape[1]
		maxEmbeddingDims = int(numpy.max(embedDims))
		embedDimsArray = numpy.asarray(embedDims)
		if embedDimsArray.ndim == 0:
			embedDimsArray = numpy.full(numSources, int(embedDimsArray))
		elif embedDimsArray.ndim == 2:
			# one embedding dimension per source in this mode: the largest over its targets
			embedDimsArray = embedDimsArray.max(axis = 1)

		# When targetVRAM is given, the budget sets both batch sizes; x_batch and y_batch are the
		# defaults otherwise. Tensors scaling with the source batch:
		#   sourceDistanceMatrices [sourceBatch, numTrain, numTest]       (per source batch)
		#   subsampledDistances    [sourceBatch, maxTrainSize, numTest]   (transient during topk)
		#   neighborIndices/Weights [sourceBatch, maxKnn, numTest]         (per sample)
		#   flatIndices, weightsForBmm                                     (permuted copies per sample)
		#   valuesForBmm           [sourceBatch * numTest, maxKnn, y_batch] (transient per target batch; dominant)
		conservativeMaxKnn = maxEmbeddingDims + 1
		elementSize = torch.zeros(1, dtype = self.dtype).element_size()
		if self.targetVRAM is None:
			sourceBatchSize = self.x_batch
			targetBatchSize = self.y_batch
		else:
			targetVRAMBytes = self.targetVRAM * 1e9
			maxTrainSize = min(max(self.trainSizes), numTrain)
			sourceOnlyBytesPerSource = ((numTrain + maxTrainSize) * numTest * elementSize +
			                            2 * conservativeMaxKnn * numTest * (8 + elementSize))
			yBytesPerSourcePerTarget = conservativeMaxKnn * numTest * elementSize
			perSourceBytes = sourceOnlyBytesPerSource + yBytesPerSourcePerTarget * self.y_batch
			sourceBatchSize = max(1, int(targetVRAMBytes / perSourceBytes))
			actualBatchSizeEstimate = min(sourceBatchSize, numSources)
			remainingBytes = targetVRAMBytes - actualBatchSizeEstimate * sourceOnlyBytesPerSource
			vramTargetBatch = max(1, int(remainingBytes / (actualBatchSizeEstimate * yBytesPerSourcePerTarget)))
			targetBatchSize = max(self.y_batch, vramTargetBatch)

		perLagSquaredDistances = torch.zeros([maxEmbeddingDims, numTrain, numTest], dtype = self.dtype, device = self.device)

		for sourceBatchStart in ProgressBar(range(0, numSources, sourceBatchSize), desc = 'Source batch', leave = False,
											disable = not self.showProgress):
			sourceBatchEnd = min(sourceBatchStart + sourceBatchSize, numSources)
			actualSourceBatchSize = sourceBatchEnd - sourceBatchStart
			sourceEmbedDims = embedDimsArray[sourceBatchStart:sourceBatchEnd]

			if self.knn is not None:
				numNeighbors = self.knn
				maxKnn = self.knn
			else:
				maxKnn = int(numpy.max(sourceEmbedDims)) + 1
				numNeighborsPerSource = torch.tensor(sourceEmbedDims + 1, dtype = torch.long, device = self.device)
				if int(torch.unique(numNeighborsPerSource).shape[0]) == 1:
					numNeighbors = int(numNeighborsPerSource[0].item())
				else:
					numNeighbors = numNeighborsPerSource

			sourceDistanceMatrices = torch.zeros([actualSourceBatchSize, numTrain, numTest], dtype = self.dtype, device = self.device)

			for localSourceIndex in range(actualSourceBatchSize):
				globalSourceIndex = sourceBatchStart + localSourceIndex
				sourceColumns = slice(globalSourceIndex * maxEmbeddingDims, (globalSourceIndex + 1) * maxEmbeddingDims)
				trainSourceTensor = torch.as_tensor(X_train[:, sourceColumns], dtype = self.dtype, device = self.device)
				testSourceTensor = torch.as_tensor(X_test[:, sourceColumns], dtype = self.dtype, device = self.device)

				for lagIndex in range(maxEmbeddingDims):
					perLagSquaredDistances[lagIndex] = trainSourceTensor[:, lagIndex].unsqueeze(1) - testSourceTensor[:, lagIndex].unsqueeze(0)
				del trainSourceTensor, testSourceTensor

				perLagSquaredDistances.square_()
				torch.cumsum(perLagSquaredDistances, dim = 0, out = perLagSquaredDistances)

				sourceDim = int(sourceEmbedDims[localSourceIndex])
				sourceDistanceMatrices[localSourceIndex] = perLagSquaredDistances[sourceDim - 1]

			if exclusion is not None:
				sourceDistanceMatrices[:, torch.as_tensor(exclusion, device = self.device)] = float('inf')

			performanceBuffer = torch.zeros([actualSourceBatchSize, targetBatchSize], dtype = self.dtype, device = self.device)

			for size_i, trainSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
															disable = not self.showProgress)):
				trainSize = min(trainSize, numTrain)
				for sample_i in ProgressBar(range(self.sample), desc = 'Repeats', leave = False, disable = not self.showProgress):
					sampledIndices = torch.as_tensor(RNG.choice(numTrain, size = trainSize, replace = False),
													 dtype = torch.long, device = self.device)

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
						performance[size_i, sample_i, sourceBatchStart:sourceBatchEnd,
									targetBatchStart:targetBatchEnd] = performanceBuffer[:, :actualTargetBatchSize].cpu().numpy()

					del flatIndices, weightsForBmm

			del sourceDistanceMatrices, performanceBuffer
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def CrossMapSampleBatched(self, inputs, Y_train, performance, RNG, embedDims):
		"""
		Batch over subsets per size; the training rows predict themselves. Efficient when
		there are few source columns. Cumulative per-lag squared distances are built once per
		source so each (source, target) pair reads its own embedding dimension off the prefix sum. Without a
		user-set knn, each pair uses embedding dimensions + 1 neighbors, enforced by masking the extra
		neighbors to zero weight after one shared topk.
		"""
		numSamplesInBatch = self.sampleBatchSize if self.sampleBatchSize is not None else self.sample
		numSources = self.numSources
		numTargets = Y_train.shape[1]
		dims = int(numpy.max(embedDims))
		N_libraryIndices = inputs.numTrainingPairs

		# [numSources, N, dims] from the source-major stacked states
		train_embeddings = torch.as_tensor(
			inputs.trainStates.reshape(N_libraryIndices, numSources, dims), dtype = self.dtype,
			device = self.device).permute(1, 0, 2).contiguous()

		max_knn = self.knn if self.knn is not None else dims + 1

		d = torch.zeros([dims, N_libraryIndices, N_libraryIndices], dtype = self.dtype, device = self.device)
		cumulativeSqDist = torch.zeros([numSources, dims, N_libraryIndices, N_libraryIndices], dtype = self.dtype, device = self.device)
		for i in range(numSources):
			for lag in range(dims):
				d[lag] = train_embeddings[i, :, lag].unsqueeze(1) - train_embeddings[i, :, lag].unsqueeze(0)
			d.square_()
			cumulativeSqDist[i] = torch.cumsum(d, dim = 0)
		del train_embeddings, d

		fullDistances = torch.zeros([numSources, numTargets, N_libraryIndices, N_libraryIndices], dtype = self.dtype, device = self.device)
		for i in range(numSources):
			for t in range(numTargets):
				e = _get_embedding_dimension(embedDims, i, t)
				fullDistances[i, t] = torch.sqrt(cumulativeSqDist[i, e - 1])
		del cumulativeSqDist

		# the self-match is always excluded; a positive exclusionRadius also excludes every
		# pair within that many rows of each other in the same run
		diagIndices = torch.arange(N_libraryIndices, device = self.device)
		fullDistances[:, :, diagIndices, diagIndices] = float('inf')
		if self.exclusionRadius > 0:
			rowNumbers = torch.as_tensor(inputs.SharedAxisRows()[0], dtype = torch.long, device = self.device)
			excludedPairs = (rowNumbers.unsqueeze(0) - rowNumbers.unsqueeze(1)).abs() <= self.exclusionRadius
			fullDistances[:, :, excludedPairs] = float('inf')

		kIndices = torch.arange(max_knn, device = self.device).view(1, 1, max_knn, 1)

		for size_i, libSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
													 disable = not self.showProgress)):
			libSizeActual = min(libSize, N_libraryIndices)

			for batchStart in ProgressBar(range(0, self.sample, numSamplesInBatch), desc = 'Sample batch',
										  leave = False, disable = not self.showProgress):
				batchEnd = min(batchStart + numSamplesInBatch, self.sample)
				numSamplesInThisBatch = batchEnd - batchStart

				subsampleIndices = numpy.stack([RNG.choice(N_libraryIndices, size = libSizeActual, replace = False)
												for _ in range(numSamplesInThisBatch)])
				subsampleTorch = torch.as_tensor(subsampleIndices, dtype = torch.long, device = self.device)

				for t in range(numTargets):
					subsampledDistances = fullDistances[:, t][:, subsampleTorch, :]
					distances, neighbors = torch.topk(subsampledDistances, max_knn, dim = 2, largest = False)

					if self.knn is None:
						knnPerSource = torch.tensor([_get_embedding_dimension(embedDims, i, t) + 1 for i in range(numSources)],
													dtype = torch.long, device = self.device).view(numSources, 1, 1, 1)
						distances.masked_fill_(kIndices >= knnPerSource, float('inf'))

					subsampleExpanded = subsampleTorch.unsqueeze(0).unsqueeze(-1)
					globalNeighbors = subsampleExpanded.expand(numSources, -1, libSizeActual, N_libraryIndices).gather(dim = 2, index = neighbors)

					weights = ComputeSimplexWeights(distances)
					weightSum = weights.sum(dim = 2)

					targetT = Y_train[:, t]
					selectedTargets = targetT[globalNeighbors]
					predictions = (weights * selectedTargets).sum(dim = 2) / weightSum

					targetCentered = targetT - targetT.mean()
					targetStd = torch.sqrt((targetCentered ** 2).sum())
					predCentered = predictions - predictions.mean(dim = 2, keepdim = True)
					predStd = torch.sqrt((predCentered ** 2).sum(dim = 2))
					perfs_ = (targetCentered * predCentered).sum(dim = 2) / (targetStd * predStd)

					performance[size_i, batchStart:batchEnd, :, t] = perfs_.permute(1, 0).cpu().numpy()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
