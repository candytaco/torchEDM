import numpy
import torch
from tqdm import tqdm as ProgressBar

from .Results import BatchedCCMResult
from ._core import batch_get_simplex_weights, Correlation
from torchEDM.EDM._core import ElementwisePairwiseDistance
from torchEDM.EDM.utils import BuildEmbeddingIndices, MakeDelays, _get_embedding_dimension, build_exclusion_mask
from torchEDM.Hyperparameters import FindSelfPredictionEmbeddingDimension

class ConvergentCrossMap:
	"""
	BatchedCCM class: Vectorized CCM where M predictor variables predict the same target simultaneously.
	If Y is none, all X are used to cross-map each other
	"""

	def __init__(self,
				 X,
				 Y = None,
				 trainSizes = None,
				 repeats = 10,
				 embedDimensions = None,
				 maxEmbedDimensions = 20,
				 predictionHorizon = 1,
				 knn = None,
				 step = -1,
				 exclusionRadius = 0,
				 seed = None,
				 embedded = False,
				 validLib = None,
				 ignoreNan = True,
				 trainIndices = None,
				 testIndices = None,
				 device = 'cuda',
				 batchSize = 1000,
				 y_batch = 2000,
				 dtype: torch.dtype = torch.float16,
				 showProgress = True,
				 batchMode = 'variables',
				 sampleBatchSize = None):
		"""
		Initialize BatchedCCM.

		:param X: 					2D numpy array of predictor variables (N_timepoints, M_variables)
		:param Y: 					1D or 2D numpy array of target variable (N_timepoints,) or (N_timepoints, 1)
		:param trainSizes: 			Library sizes to evaluate
		:param repeats: 			Number of repeat with random train samples at each library size
		:param embedDimensions: 	Embedding dimension, if None, explore up to max provided
		:param maxEmbedDimensions:	max embedding dimension to explore, only used if embedDimension is None
		:param predictionHorizon: 	Prediction time horizon
		:param knn: 				Number of nearest neighbors
		:param step: 				Time delay step size
		:param exclusionRadius: 	Temporal exclusion radius for neighbors
		:param seed: 				Random seed for reproducible sampling
		:param embedded: 			Whether data is already embedded
		:param validLib:			Boolean mask for valid library points
		:param ignoreNan: 			Remove NaN values from embedding
		:param trainIndices: 		Train block index range [start, end]. If None, uses all data.
		:param testIndices: 		Test block index range [start, end]. If None, uses all data.
		:param device: 				Device for torch tensors ('cpu', 'cuda', or torch.device object)
		:param batchSize: 			Max number of source variables to process per batch (auto-reduced to fit VRAM)
		:param y_batch:				Number of Y variables to predict per batch within each source batch
		:param dtype: 				Torch dtype for tensors (e.g. torch.float32 or torch.float16)
		:param batchMode:			'variables' to batch over variables, 'sample' to batch over samples per library size
		:param sampleBatchSize:		Number of subsamples to process per batch in 'sample' mode. Defaults to all samples at once.
		"""

		self.X = X[:, None] if X.ndim == 1 else X
		if Y is not None:
			self.Y = Y[:, None] if Y.ndim == 1 else Y
		else:
			self.Y = None
		self.numSources = self.X.shape[1]
		self.numTargets = self.X.shape[1] if self.Y is None else self.Y.shape[1]
		self.embedDimensions = embedDimensions
		self.maxEmbedDimensions = maxEmbedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.ignoreNan = ignoreNan
		self.batchSize = batchSize
		self.y_batch = y_batch
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize

		self.sample = repeats
		self.seed = seed

		self.device = torch.device(device) if isinstance(device, str) else device
		self.dtype = dtype
		self.showProgress = showProgress

		if trainIndices is not None:
			self.train = trainIndices
		else:
			self.train = [(1, self.X.shape[0])]

		if trainSizes is not None:
			self.trainSizes = trainSizes
		else:
			numTrainSamples = 0
			for start, stop in self.train:
				numTrainSamples += (stop - start)
			self.trainSizes = [int(p * numTrainSamples) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]

		if testIndices is not None:
			self.test = testIndices
		else:
			self.test = [(1, self.X.shape[0])]

		self.forward_performance_ = None
		self.selectedForwardEmbedDimensions = None

	def Run(self):
		"""
		Run CCM
		"""

		X = self.X
		Y = self.Y if self.Y is not None else self.X
		embedDims = self.embedDimensions

		if X.ndim == 1:
			X = X[:, None]
		if Y.ndim == 1:
			Y = Y[:, None]

		numSources = X.shape[1]
		numTargets = Y.shape[1]

		RNG = numpy.random.default_rng(self.seed)

		if embedDims is None:
			embedDims = FindSelfPredictionEmbeddingDimension(
				X,
				maxDims = self.maxEmbedDimensions,
				train = self.train,
				test = self.test,
				predictionHorizon = self.predictionHorizon,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				embedded = self.embedded,
				validLib = self.validLib,
				dtype = self.dtype,
				device = self.device,
				batchSize = self.batchSize,
				showProgress = self.showProgress
			)

		train_indices, test_indices = BuildEmbeddingIndices(X.shape[0], X.shape[1],
												 self.train, self.test,
												 int(numpy.max(embedDims)), self.predictionHorizon, self.step,
												 self.embedded, self.validLib)
		exclusion = build_exclusion_mask(train_indices, test_indices, self.exclusionRadius)

		y_train = torch.tensor(Y[train_indices + self.predictionHorizon, :], dtype = self.dtype, device = self.device)
		y_test = torch.tensor(Y[test_indices + self.predictionHorizon, :], dtype = self.dtype, device = self.device)

		# note performance is kept in numpy array in CPU RAM because it might get lorge
		performance = numpy.zeros([len(self.trainSizes), self.sample, numSources, numTargets])

		# the rate-limiting factor is the number of and size of the distance matrices because they are a 3D tensor
		# so sample mode is better suited for a few smallish distance matrices

		if self.batchMode == 'sample':
			self.CrossMapSampleBatched(X, train_indices,
									   y_train, numSources, performance, RNG, embedDims, exclusion)
		else:
			self.CrossMapVariableBatched(X, train_indices, test_indices,
										 y_train, y_test, performance, RNG, embedDims, exclusion)

		self.forward_performance_ = numpy.mean(performance, axis = 1).squeeze()
		self.selectedForwardEmbedDimensions = embedDims

		return BatchedCCMResult(
			forward_performance = self.forward_performance_,
			predictionHorizon = self.predictionHorizon,
			library_sizes = self.trainSizes,
			forward_embed_dimensions = self.selectedForwardEmbedDimensions,
		)

	def CrossMapVariableBatched(self, X, train_indices, test_indices,
								y_train, y_test, performance, RNG, embedDims, exclusion = None):
		"""
		Batch over source variables. Computes one distance matrix per source and reuses it across
		all (library size, repeat, target) iterations. Source batch size is auto-tuned to keep
		the distance matrix block near 2 GB.
		"""
		numTrain = train_indices.shape[0]
		numTest = test_indices.shape[0]
		numSources = X.shape[1]
		numTargets = y_train.shape[1]
		maxEmbeddingDims = int(numpy.max(embedDims))
		embedDimsArray = numpy.asarray(embedDims)

		# Auto-tune source batch size so [sourceBatchSize, numTrain, numTest] stays near 2 GB
		elementSize = torch.zeros(1, dtype = self.dtype).element_size()
		sourceBatchSize = min(self.batchSize, max(1, int(2e9 / (numTrain * numTest * elementSize))))

		# Reusable per-lag squared distance buffer for one source: [maxEmbeddingDims, numTrain, numTest]
		perLagSquaredDistances = torch.zeros([maxEmbeddingDims, numTrain, numTest],
											  dtype = self.dtype, device = self.device)

		for sourceBatchStart in ProgressBar(range(0, numSources, sourceBatchSize), desc = 'Source batch', leave = False,
											disable = not self.showProgress):
			sourceBatchEnd = min(sourceBatchStart + sourceBatchSize, numSources)
			actualSourceBatchSize = sourceBatchEnd - sourceBatchStart
			sourceEmbedDims = embedDimsArray[sourceBatchStart:sourceBatchEnd]  # [actualSourceBatchSize]

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

			# Build one distance matrix per source in this batch: [actualSourceBatchSize, numTrain, numTest]
			sourceDistanceMatrices = torch.zeros([actualSourceBatchSize, numTrain, numTest],
												  dtype = self.dtype, device = self.device)

			for localSourceIndex in range(actualSourceBatchSize):
				globalSourceIndex = sourceBatchStart + localSourceIndex
				if self.embedded:
					delayed = X[:, globalSourceIndex][:, None]
				else:
					delayed = MakeDelays(data = X[:, globalSourceIndex], num_delays = maxEmbeddingDims, stepSize = self.step, fill = 0.0)

				trainEmbedding = torch.tensor(delayed[train_indices, :], dtype = self.dtype, device = self.device)
				testEmbedding = torch.tensor(delayed[test_indices, :], dtype = self.dtype, device = self.device)

				for lagIndex in range(maxEmbeddingDims):
					perLagSquaredDistances[lagIndex] = trainEmbedding[:, lagIndex].unsqueeze(1) - testEmbedding[:, lagIndex].unsqueeze(0)
				del trainEmbedding, testEmbedding

				perLagSquaredDistances.square_()
				torch.cumsum(perLagSquaredDistances, dim = 0, out = perLagSquaredDistances)

				sourceDim = int(sourceEmbedDims[localSourceIndex])
				sourceDistanceMatrices[localSourceIndex] = perLagSquaredDistances[sourceDim - 1]

			if exclusion is not None:
				sourceDistanceMatrices[:, exclusion] = float('inf')

			# Pre-allocate score output buffer for this source batch: [actualSourceBatchSize, y_batch]
			performanceBuffer = torch.zeros([actualSourceBatchSize, self.y_batch], dtype = self.dtype, device = self.device)

			for size_i, trainSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
															disable = not self.showProgress)):
				trainSize = min(trainSize, numTrain)
				for sample_i in range(self.sample):
					sampledIndices = torch.as_tensor(
						RNG.choice(numTrain, size = trainSize, replace = False),
						dtype = torch.long, device = self.device
					)

					subsampledDistances = sourceDistanceMatrices[:, sampledIndices, :]
					neighborIndices, neighborWeights = batch_get_simplex_weights(subsampledDistances, numNeighbors, sampledIndices)
					# neighborIndices: [actualSourceBatchSize, maxKnn, numTest] — row positions into y_train
					# neighborWeights: [actualSourceBatchSize, maxKnn, numTest]

					for targetBatchStart in range(0, numTargets, self.y_batch):
						targetBatchEnd = min(targetBatchStart + self.y_batch, numTargets)
						actualTargetBatchSize = targetBatchEnd - targetBatchStart

						yBatch = y_train[:, targetBatchStart:targetBatchEnd]  # [numTrain, actualTargetBatchSize]
						neighborTargetValues = yBatch[neighborIndices]  # [sourceBatch, maxKnn, numTest, actualTargetBatchSize]

						# Use bmm to compute weighted sum without materializing the element-wise product intermediate.
						# Reshape weights to [sourceBatch * numTest, 1, maxKnn] and values to [sourceBatch * numTest, maxKnn, T].
						weightsForBmm = neighborWeights.permute(0, 2, 1).reshape(actualSourceBatchSize * numTest, 1, maxKnn)
						valuesForBmm = neighborTargetValues.permute(0, 2, 1, 3).reshape(actualSourceBatchSize * numTest, maxKnn, actualTargetBatchSize)
						predictions = torch.bmm(weightsForBmm, valuesForBmm).reshape(actualSourceBatchSize, numTest, actualTargetBatchSize)
						del neighborTargetValues, weightsForBmm, valuesForBmm

						Correlation(y_test[:, targetBatchStart:targetBatchEnd], predictions,
									out = performanceBuffer[:, :actualTargetBatchSize])
						performance[size_i, sample_i,
									sourceBatchStart:sourceBatchEnd,
									targetBatchStart:targetBatchEnd] = performanceBuffer[:, :actualTargetBatchSize].cpu().numpy()

			del sourceDistanceMatrices, performanceBuffer
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def CrossMapSampleBatched(self, X, train_indices,
							  target, numSources, performance, RNG, embedDims, exclusion = None):
		"""
		Batch over subsamples per library size. Efficient when the number of source variables is small.

		Per-lag cumulative squared distances are pre-computed for all source variables. For each
		(source, target) pair the appropriate lag prefix is selected so kNN uses exactly the
		optimal E lags.

		When knn was not user-specified, each (source, target) pair uses E[s,t]+1 neighbors.
		This is enforced by masking distances beyond the pair's knn to inf after the global
		topk, so excess neighbor weights become zero and do not affect the prediction.

		Note that this function still operates only over the train data, using the train data to predict itself
		"""
		numSamplesInBatch = self.sampleBatchSize if self.sampleBatchSize is not None else self.sample
		numTargets = target.shape[1]
		dims = int(numpy.max(embedDims))
		N_libraryIndices = train_indices.shape[0]

		embeddings = []
		for varIndex in range(numSources):
			if self.embedded:
				delayed = X[:, varIndex][:, None]
			else:
				delayed = MakeDelays(data = X[:, varIndex], num_delays = dims, stepSize = self.step)
			embeddings.append(delayed[train_indices, :])

		max_knn = self.knn if self.knn is not None else int(numpy.max(embedDims)) + 1

		# Build per-lag cumulative distance matrices for all source variables: [numSources, dims, N, N]
		train_embeddingeddings = torch.tensor(numpy.array(embeddings), dtype = self.dtype, device = self.device)
		d = torch.zeros([dims, N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)
		cumulativeSqDist = torch.zeros([numSources, dims, N_libraryIndices, N_libraryIndices],
									   dtype = self.dtype, device = self.device)
		for i in range(numSources):
			ElementwisePairwiseDistance(train_embeddingeddings[i, :, :], train_embeddingeddings[i, :, :], d)
			cumulativeSqDist[i] = torch.cumsum(d, dim = 0)

		del train_embeddingeddings
		del d

		# Select per-(source, target) sqrt distances: [numSources, numTargets, N, N]
		fullDistances = torch.zeros([numSources, numTargets, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)
		for i in range(numSources):
			for t in range(numTargets):
				e = _get_embedding_dimension(embedDims, i, t)
				fullDistances[i, t] = torch.sqrt(cumulativeSqDist[i, e - 1])

		del cumulativeSqDist

		if self.exclusionRadius == 0:
			diagIndices = torch.arange(N_libraryIndices, device = self.device)
			fullDistances[:, :, diagIndices, diagIndices] = float('inf')

		# kIndices reused per target loop: [1, 1, max_knn, 1]
		kIndices = torch.arange(max_knn, device = self.device).view(1, 1, max_knn, 1)

		for size_i, libSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
													 disable = not self.showProgress)):
			libSizeActual = min(libSize, N_libraryIndices)

			for batchStart in ProgressBar(range(0, self.sample, numSamplesInBatch), desc = 'Sample batch',
										  leave = False, disable = not self.showProgress):
				batchEnd = min(batchStart + numSamplesInBatch, self.sample)
				numSamplesInThisBatch = batchEnd - batchStart

				# Draw subsamples: [numSamplesInThisBatch, libSizeActual]
				subsampleIndices = numpy.stack([
					RNG.choice(N_libraryIndices, size = libSizeActual, replace = False)
					for _ in range(numSamplesInThisBatch)
					])
				subsampleTorch = torch.as_tensor(subsampleIndices, dtype = torch.long, device = self.device)

				for t in range(numTargets):
					# fullDistances[:, t]: [numSources, N, N]
					# Gather subsampled distances: [numSources, numSamplesInThisBatch, libSizeActual, N]
					subsampledDistances = fullDistances[:, t][:, subsampleTorch, :]

					distances, neighbors = torch.topk(subsampledDistances, max_knn, dim = 2, largest = False)
					# distances: [numSources, numSamplesInThisBatch, max_knn, N]

					# Mask out neighbors beyond dims+1 for each source->target pair.
					if self.knn is None:
						knnPerSource = torch.tensor(
								[_get_embedding_dimension(embedDims, i, t) + 1 for i in range(numSources)],
								dtype = torch.long, device = self.device
								).view(numSources, 1, 1, 1)
						distances.masked_fill_(kIndices >= knnPerSource, float('inf'))

					subsampleExpanded = subsampleTorch.unsqueeze(0).unsqueeze(-1)
					globalNeighbors = subsampleExpanded.expand(numSources, -1, libSizeActual, N_libraryIndices).gather(
							dim = 2, index = neighbors
							)

					torch.clamp_min(distances, 1e-6, out = distances)
					minDistances = distances.min(dim = 2)[0]
					weights = torch.exp(-distances / minDistances.unsqueeze(2))
					weightSum = weights.sum(dim = 2)

					targetT = target[:, t]  # [N]
					selectedTargets = targetT[globalNeighbors]  # [numSources, numSamplesInThisBatch, max_knn, N]
					predictions = (weights * selectedTargets).sum(
							dim = 2) / weightSum  # [numSources, numSamplesInThisBatch, N]

					targetCentered = targetT - targetT.mean()  # [N]
					targetStd = torch.sqrt((targetCentered ** 2).sum())
					predCentered = predictions - predictions.mean(dim = 2,
																  keepdim = True)  # [numSources, numSamplesInThisBatch, N]
					predStd = torch.sqrt((predCentered ** 2).sum(dim = 2))  # [numSources, numSamplesInThisBatch]
					perfs_ = (targetCentered * predCentered).sum(dim = 2) / (
							targetStd * predStd)  # [numSources, numSamplesInThisBatch]

					performance[size_i, batchStart:batchEnd, :, t] = perfs_.permute(1, 0).cpu().numpy()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
