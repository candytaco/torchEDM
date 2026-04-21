import numpy
import torch
from typing import List, Union
from tqdm import tqdm as ProgressBar

from .Results import BatchedCCMResult
from ._core import batch_simplex_predict_and_score, batch_get_simplex_weights, Correlation
from torchEDM.EDM._core import ElementwisePairwiseDistance
from torchEDM.EDM.utils import BuildEmbeddingIndices, MakeDelays, _get_embedding_dimension, build_exclusion_mask
from torchEDM.Hyperparameters import FindOptimalEmbeddingDimensionality

class ConvergentCrossMap:
	"""
	BatchedCCM class: Vectorized CCM where M predictor variables predict the same target simultaneously.
	If Y is none, all X are used to cross-map each other
	"""

	def __init__(self,
				 X,
				 Y = None,
				 trainSizes = None,
				 sample = 10,
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
				 batchSize = 10000,
				 y_batch = None,
				 HalfPrecision = False,
				 showProgress = True,
				 batchMode = 'variables',
				 sampleBatchSize = None):
		"""
		Initialize BatchedCCM.

		:param X: 					2D numpy array of predictor variables (N_timepoints, M_variables)
		:param Y: 					1D or 2D numpy array of target variable (N_timepoints,) or (N_timepoints, 1)
		:param trainSizes: 			Library sizes to evaluate
		:param sample: 				Number of random samples at each library size
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
		:param batchSize: 			Number of distance matrices / embeddings to process per batch
		:param y_batch:				Number of Y variables to predict per batch within each embedding batch
		:param HalfPrecision: 		Use float16 instead of float32 to save VRAM
		:param batchMode:			'variables' to batch over variables, 'sample' to batch over samples per library size
		:param sampleBatchSize:		Number of subsamples to process per batch in 'sample' mode. Defaults to all samples at once.
		"""

		self.name = 'BatchedCCM'
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
		if y_batch is not None:
			self.y_batch = y_batch
		elif Y is not None:
			self.y_batch = Y.shape[0]
		else:
			self.y_batch = X.shape[0]
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize

		self.sample = sample
		self.seed = seed

		self.device = torch.device(device) if isinstance(device, str) else device
		self.dtype = torch.float16 if HalfPrecision else torch.float32
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
			scores = FindOptimalEmbeddingDimensionality(X, Y, maxDims = self.maxEmbedDimensions, train = self.train, test = self.test,
														predictionHorizon = self.predictionHorizon, step = self.step,
														ignoreNan = self.ignoreNan,
														batched = True, joint = False,
														HalfPrecision = (self.dtype == torch.float16),
														BatchSize = self.batchSize)
			# scores: [nVars, maxDims] (single target, squeezed) or [nTargets, nVars, maxDims].
			# Per-(source, target) best num dims is (argmax over the maxDims axis) + 1  since it is 1-indexed internally.
			if scores.ndim == 2:
				embedDims = numpy.argmax(scores, axis = 1) + 1  # [nVars]
			else:
				embedDims = numpy.argmax(scores, axis = 2).T + 1  # [nVars, nTargets]

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
								y_train: Union[numpy.ndarray, torch.tensor], y_test: Union[numpy.ndarray, torch.tensor],
								performance, RNG, embedDims,
								exclusion = None):
		"""
		Batch over source->target pairs. Embeddings are computed per-batch to minimize VRAM usage.
		"""
		maxEmbeddingDims = int(numpy.max(embedDims))
		numTrain = train_indices.shape[0]
		numTest = test_indices.shape[0]

		uniqueSourceDelayPairs = []	# list of the parameters for all evaluated distance matrices
		sourceIndices = []			# source variable for each evaluated distance matrix
		uniqueSourceDims = {}
		delayIndices = []			# embedding dimension for each evaluated distance matrix
		for s in range(X.shape[1]):
			uniqueSourceDims[s] = numpy.sort(numpy.unique(embedDims[s, :])) if not isinstance(embedDims, int) else [embedDims]
			for d in uniqueSourceDims[s]:
				uniqueSourceDelayPairs.append((s, d))
				sourceIndices.append(s)
				delayIndices.append(d)
		numUniqueSourceDelayPairs = len(uniqueSourceDelayPairs)

		batchSize = numUniqueSourceDelayPairs if numUniqueSourceDelayPairs < self.batchSize else self.batchSize

		# Reusable per-lag squared distance buffer for one source: [lag number, train samples, test samples]
		# in the case we need to compute multiple embedding dimensions per source
		delay_dists = torch.zeros([maxEmbeddingDims, numTrain, numTest],
						dtype = self.dtype, device = self.device)

		# matrices for batch tensor processing
		distances = torch.zeros([batchSize, numTrain, numTest], dtype = self.dtype, device = self.device)

		for batchStart in ProgressBar(range(0, numUniqueSourceDelayPairs, batchSize), desc = 'Batch', leave = False,
									  disable = not self.showProgress):
			batchEnd = min(batchStart + self.batchSize, numUniqueSourceDelayPairs)
			batchPairs = uniqueSourceDelayPairs[batchStart:batchEnd]
			batchSources = sourceIndices[batchStart:batchEnd]
			batchDelays = delayIndices[batchStart:batchEnd]

			# collect unique source and embedding dimensions in this batch
			theseSources = []
			theseDims = {}
			for source, dim in batchPairs:
				if source not in theseSources:
					theseSources.append(source)
					theseDims[source] = []
				theseDims[source].append(dim) # it's guaranteed to be unique here

			numNeighbors = []

			i = 0	# ith distance matrix because it could be variable across sources
			# compute the distances needed per source in this batch
			for source in theseSources:
				if self.embedded:
					delayed = X[:, source][:, None]
				else:
					delayed = MakeDelays(data = X[:, source], num_delays = maxEmbeddingDims, stepSize = self.step, fill = 0.0)
				train_embedding = torch.tensor(delayed[train_indices, :], dtype = self.dtype, device = self.device)
				test_embedding = torch.tensor(delayed[test_indices, :], dtype = self.dtype, device = self.device)

				for d in range(maxEmbeddingDims):
					delay_dists[d, :, :] = train_embedding[:, d].unsqueeze(1) - test_embedding[:, d].unsqueeze(0)
				del train_embedding, test_embedding

				delay_dists **= 2
				torch.cumsum(delay_dists, dim = 0, out = delay_dists)
				# insert only needed ones into the batched tensor matrix
				for d in theseDims[source]:
					distances[i, :, :] = delay_dists[d - 1, :, :] # d-1 because they are the i+1 number of embedding dimensions
					i += 1
					numNeighbors.append(d + 1)

			if self.knn is not None:
				numNeighbors = self.knn
			else:
				numNeighbors = torch.tensor(numNeighbors, device = self.device)

			if exclusion is not None:
				distances[:, exclusion] = float('inf')

			# because we make extra predictions from the tensor operations, these get out the actual (source x dim)->target pairs
			batchSourcesArray = numpy.array(batchSources)
			batchDelaysArray = numpy.array(batchDelays)
			if isinstance(embedDims, int):
				fullMask = numpy.ones((len(batchPairs), y_train.shape[1]), dtype = bool)
			else:
				fullMask = (embedDims[batchSourcesArray[:, None], numpy.arange(y_train.shape[1])[None, :]] == batchDelaysArray[:, None])

			predictions = torch.zeros(len(batchPairs), numTest, self.y_batch, dtype = self.dtype, device = self.device)
			perf_out = torch.zeros(len(batchPairs), self.y_batch, dtype = self.dtype, device = self.device)

			for size_i, train_size in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
															disable = not self.showProgress)):
				train_size = min(train_size, distances.shape[1]) # not great when this happens
				for sample_i in ProgressBar(range(self.sample), desc = 'Repeats', leave = False, disable = not self.showProgress):
					indices = torch.as_tensor(RNG.choice(distances.shape[1], size = train_size, replace = False),
											  dtype = torch.long, device = self.device)

					# subsampledDistances: [distance matrices, subsampled train, test indices]
					subsampledDistances = distances[:len(batchPairs), indices, :]

					# 1. compute neighbors and weights
					neighbors, weights = batch_get_simplex_weights(subsampledDistances, numNeighbors, indices)
					# 2. batch compute predictions over Ys and score; 4. place valid (source, target) pairs into performance
					for start in range(0, y_train.shape[1], self.y_batch):
						end = min(start + self.y_batch, y_train.shape[1])
						thisTargetBatchSize = end - start
						yBatch = y_train[:, start:end]
						select = yBatch[neighbors]
						torch.sum(weights[:, :, :, None] * select, dim = 1, out = predictions[:, :, :thisTargetBatchSize])
						Correlation(y_test[:, start:end], predictions[:, :, :thisTargetBatchSize], out = perf_out[:, :thisTargetBatchSize])
						thisMask = fullMask[:, start:end]
						rowPositions, colPositions = numpy.where(thisMask)
						performance[size_i, sample_i, batchSourcesArray[rowPositions], start + colPositions] = perf_out.cpu().numpy()[rowPositions, colPositions]

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
