import numpy
import torch
from tqdm import tqdm as ProgressBar

from torchEDM.EDM.Simplex import Simplex
from torchEDM.EDM._core import ElementwisePairwiseDistance
from torchEDM.EDM.utils import BuildEmbeddingIndices, MakeDelays, _get_embedding_dimension
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
				 HalfPrecision = False,
				 showProgress = True,
				 batchMode = 'variables',
				 sampleBatchSize = None):
		"""
		Initialize BatchedCCM.

		:param X: 					2D numpy array of predictor variables (N_timepoints, M_variables)
		:param Y: 					1D or 2D numpy array of target variable (N_timepoints,) or (N_timepoints, 1)
		:param trainSizes: 			Library sizes to evaluate [start, stop, increment]
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
		Execute BatchedCCM and return BatchedCCMResult.
		"""
		self.forward_performance_, self.selectedForwardEmbedDimensions = self.CrossMap(self.X,
																					   self.Y if self.Y is not None else self.X,
																					   self.embedDimensions,
																					   self.maxEmbedDimensions)

		from .Results import BatchedCCMResult
		return BatchedCCMResult(
				forward_performance = self.forward_performance_,
				predictionHorizon = self.predictionHorizon,
				library_sizes = self.trainSizes,
				forward_embed_dimensions = self.selectedForwardEmbedDimensions,
				)

	def CrossMap(self, X, Y, embedDims, maxDims):
		from .Embed import Embed

		if X.ndim == 1:
			X = X[:, None]
		if Y.ndim == 1:
			Y = Y[:, None]

		numSources = X.shape[1]
		numTargets = Y.shape[1]

		RNG = numpy.random.default_rng(self.seed)

		if embedDims is None:
			scores = FindOptimalEmbeddingDimensionality(X, Y, maxDims = maxDims, train = self.train, test = self.test,
														predictionHorizon = self.predictionHorizon, step = self.step,
														ignoreNan = self.ignoreNan,
														batched = True, joint = False,
														HalfPrecision = (self.dtype == torch.float16),
														BatchSize = self.batchSize)
			# scores: [nVars, maxDims] (single target, squeezed) or [nTargets, nVars, maxDims].
			# Per-variable best E = argmax over the dims axis + 1 (1-indexed).
			# For multi-target, keep per-target E separately so distance computation can
			# use the right number of lags for each (source, target) pair.
			if scores.ndim == 2:
				embedDims = numpy.argmax(scores, axis = 1) + 1  # [nVars]
			else:
				embedDims = numpy.argmax(scores, axis = 2).T + 1  # [nVars, nTargets]

		dims = int(numpy.max(embedDims))

		train_indices, _ = BuildEmbeddingIndices(X.shape[0], X.shape[1],
												 self.train, self.test,
												 maxDims, self.predictionHorizon, self.step,
												 self.embedded, self.validLib)

		libraryIndices = numpy.array(train_indices)
		N_libraryIndices = len(libraryIndices)

		# Always embed with the maximum number of lags. The CrossMap functions select
		# the appropriate per-(source, target) lag prefix during distance computation.
		with_delays = []
		for varIndex in range(numSources):
			if self.embedded:
				delayed = X[:, varIndex][:, None]
			else:
				delayed = MakeDelays(data = X[:, varIndex], num_delays = dims, stepSize = self.step)
			with_delays.append(delayed[train_indices, :])

		target = torch.tensor(Y[libraryIndices + self.predictionHorizon, :], dtype = self.dtype, device = self.device)
		performance = numpy.zeros([len(self.trainSizes), self.sample, numSources, numTargets])

		if self.batchMode == 'sample':
			self.CrossMapSampleBatched(with_delays, test_indices.shape[0],
									   target, numSources, performance, RNG, embedDims)
		else:
			self.CrossMapVariableBatched(with_delays, test_indices.shape[0],
										 target, numSources, numTargets, performance, RNG, embedDims)

		return numpy.mean(performance, axis = 1).squeeze(), embedDims

	def _get_embedding_dimension(self, embedDims, sourceIndex, targetIndex):
		"""Return the optimal embedding dimension for source sourceIndex predicting target targetIndex."""
		if isinstance(embedDims, int):
			return embedDims
		arr = numpy.asarray(embedDims)
		if arr.ndim == 1:
			return int(arr[sourceIndex])
		return int(arr[sourceIndex, targetIndex])  # [nVars, nTargets]

	def CrossMapVariableBatched(self, embeddings, N_libraryIndices,
								target, numSources, numTargets, performance, RNG, embedDims):
		"""
		Batch over source->target pairs. Distances are computed on the fly for each batch,
		keeping memory proportional to batchSize * N^2 rather than numSources * numTargets * N^2.
		This function is kinda not great in that it allows for the case where there are different
		embedding dimensions for source per target and uses a separate distance matrix for each, even
		though there can be many redundancies.

		Pairs are enumerated in source-major order so consecutive pairs within a batch share
		the same source. The per-lag squared distance buffer is reused and only recomputed
		when the source changes, avoiding redundant computation while caching only one
		source's distances at a time.
		"""
		dims = embeddings[0].shape[1]

		pairs = [(s, t) for s in range(numSources) for t in range(numTargets)]
		numPairs = len(pairs)

		max_knn = self.knn if self.knn is not None else int(numpy.max(embedDims)) + 1

		# kIndices: [1, max_knn, 1] for knn masking via broadcasting
		kIndices = torch.arange(max_knn, device = self.device).view(1, max_knn, 1)

		# Reusable per-lag squared distance buffer for one source: [dims, N, N]
		d = torch.zeros([dims, N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)

		diagIndices = torch.arange(N_libraryIndices, device = self.device) if self.exclusionRadius == 0 else None

		# for each source variable:
			# calculate per-lag squared distance matrices
			# calculate cumulative squared distance matrices
			# across all target variables, identify the unique set of embedding dimensions from this source variable
				# select the slices in the cumulative squared distance matrices for these
			# for all unique embedding dimensionality,
			 	# for each library size
					# compute predictions across target variables n times with random samples of train indices


		for batchStart in ProgressBar(range(0, numPairs, self.batchSize), desc = 'Pair batch', leave = False,
									  disable = not self.showProgress):
			batchEnd = min(batchStart + self.batchSize, numPairs)
			batchPairs = pairs[batchStart:batchEnd]
			batchNumPairs = len(batchPairs)

			# Compute Euclidean distances on the fly for each pair.
			# d is reused: recomputed only when the source changes (pairs are source-major ordered).
			pairDistances = torch.zeros([batchNumPairs, N_libraryIndices, N_libraryIndices],
										dtype = self.dtype, device = self.device)

			lastSource = -1
			for p, (s, t) in enumerate(batchPairs):
				e = self._get_embedding_dimension(embedDims, s, t)
				if s != lastSource:
					emb = torch.tensor(embeddings[s], dtype = self.dtype, device = self.device)
					ElementwisePairwiseDistance(emb, emb, d)
					lastSource = s
				pairDistances[p] = torch.sqrt(d[:e].sum(dim = 0))

			if self.exclusionRadius == 0:
				pairDistances[:, diagIndices, diagIndices] = float('inf')

			if self.knn is None:
				knnPerPair = torch.tensor([self._get_embedding_dimension(embedDims, s, t) + 1 for s, t in batchPairs],
										  dtype = torch.long, device = self.device).view(batchNumPairs, 1, 1)

			# pairTargets: [batchNumPairs, N_libraryIndices]
			target_index = torch.tensor([t for s, t in batchPairs], dtype = torch.long, device = self.device)
			pairTargets = target[:, target_index].T.contiguous()

			source_indices = numpy.array([s for s, t in batchPairs])
			target_indices = numpy.array([t for s, t in batchPairs])

			for size_i, train_size in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False,
															disable = not self.showProgress)):
				num_train = min(train_size, N_libraryIndices)

				for sample_i in ProgressBar(range(self.sample), desc = 'Repeats', leave = False,
											disable = not self.showProgress):
					indices = torch.as_tensor(RNG.choice(N_libraryIndices, size = num_train, replace = False),
											dtype = torch.long, device = self.device)

					# subsampledDistances: [batchNumPairs, num_train, N_libraryIndices]
					subsampledDistances = pairDistances[:, indices, :]

					# theseDistances: [batchNumPairs, max_knn, N_libraryIndices]
					theseDistances, theseNeighbors = torch.topk(subsampledDistances, max_knn, dim = 1, largest = False)

					if self.knn is None:
						theseDistances.masked_fill_(kIndices >= knnPerPair, float('inf'))

					# Map subsampled neighbor indices to global indices
					globalNeighbors = indices[theseNeighbors]  # [batchNumPairs, max_knn, N_libraryIndices]

					torch.clamp_min(theseDistances, 1e-6, out = theseDistances)
					minDistances = theseDistances.min(dim = 1)[0]  # [batchNumPairs, N_libraryIndices]
					weights = torch.exp(
							-theseDistances / minDistances.unsqueeze(1))  # [batchNumPairs, max_knn, N_libraryIndices]
					weightSum = weights.sum(dim = 1)  # [batchNumPairs, N_libraryIndices]

					# Gather target values at each neighbor position.
					# pairTargets[p, globalNeighbors[p, k, n]] for all p, k, n simultaneously.
					selectedTargets = torch.gather(
							pairTargets.unsqueeze(1).expand(-1, max_knn, -1),
							dim = 2,
							index = globalNeighbors
							)  # [batchNumPairs, max_knn, N_libraryIndices]

					predictions = (weights * selectedTargets).sum(
							dim = 1) / weightSum  # [batchNumPairs, N_libraryIndices]

					pairTargetsCentered = pairTargets - pairTargets.mean(dim = 1, keepdim = True)
					pairTargetsStd = torch.sqrt((pairTargetsCentered ** 2).sum(dim = 1))
					predCentered = predictions - predictions.mean(dim = 1, keepdim = True)
					predStd = torch.sqrt((predCentered ** 2).sum(dim = 1))
					perfs = (pairTargetsCentered * predCentered).sum(dim = 1) / (pairTargetsStd * predStd)

					performance[size_i, sample_i, source_indices, target_indices] = perfs.cpu().numpy()

			del pairDistances, pairTargets
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def CrossMapSampleBatched(self, embeddings, N_libraryIndices,
							  target, numSources, performance, RNG, embedDims):
		"""
		Batch over subsamples per library size. Efficient when the number of source variables is small.

		Per-lag cumulative squared distances are pre-computed for all source variables. For each
		(source, target) pair the appropriate lag prefix is selected so kNN uses exactly the
		optimal E lags.

		When knn was not user-specified, each (source, target) pair uses E[s,t]+1 neighbors.
		This is enforced by masking distances beyond the pair's knn to inf after the global
		topk, so excess neighbor weights become zero and do not affect the prediction.
		"""
		numSamplesInBatch = self.sampleBatchSize if self.sampleBatchSize is not None else self.sample
		numTargets = target.shape[1]
		dims = embeddings[0].shape[1]

		max_knn = self.knn if self.knn is not None else int(numpy.max(embedDims)) + 1

		# Build per-lag cumulative distance matrices for all source variables: [numSources, dims, N, N]
		trainEmbeddings = torch.tensor(numpy.array(embeddings), dtype = self.dtype, device = self.device)
		d = torch.zeros([dims, N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)
		cumulativeSqDist = torch.zeros([numSources, dims, N_libraryIndices, N_libraryIndices],
									   dtype = self.dtype, device = self.device)
		for i in range(numSources):
			ElementwisePairwiseDistance(trainEmbeddings[i, :, :], trainEmbeddings[i, :, :], d)
			cumulativeSqDist[i] = torch.cumsum(d, dim = 0)

		del trainEmbeddings
		del d

		# Select per-(source, target) sqrt distances: [numSources, numTargets, N, N]
		fullDistances = torch.zeros([numSources, numTargets, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)
		for i in range(numSources):
			for t in range(numTargets):
				e = self._get_embedding_dimension(embedDims, i, t)
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
					if not self.knnUserSpecified:
						knnPerSource = torch.tensor(
								[self._get_embedding_dimension(embedDims, i, t) + 1 for i in range(numSources)],
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
