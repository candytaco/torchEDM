import numpy
import torch
from tqdm import tqdm as ProgressBar

from torchEDM.Hyperparameters import FindOptimalEmbeddingDimensionality
from torchEDM.EDM.Simplex import Simplex
from torchEDM.EDM._MDE import ElementwisePairwiseDistance, FloorArray, MinAxis1, ComputeWeights, SumAxis1


class BatchedCCM:
	"""
	BatchedCCM class: Vectorized CCM where M predictor variables predict the same target simultaneously.
	Only supports pairwise distance mode (no KDTree).
	"""

	def __init__(self,
				 X,
				 Y,
				 trainSizes = None,
				 sample = 0,
				 forwardEmbedDimensions = 0,
				 reverseEmbedDimensions = None,
				 maxForwardDimensions = 20,
				 maxReverseDimensions = None,
				 predictionHorizon = 1,
				 knn = 0,
				 step = -1,
				 exclusionRadius = 0,
				 seed = None,
				 embedded = False,
				 validLib = None,
				 includeData = False,
				 ignoreNan = True,
				 directions: str = 'both',
				 trainBlockIndices = None,
				 testBlockIndices = None,
				 device = 'cuda',
				 batchSize = 10000,
				 useHalfPrecision = False,
				 showProgress = True,
				 batchMode = 'variable',
				 sampleBatchSize = None):
		"""
		Initialize BatchedCCM.

		:param X: 					2D numpy array of predictor variables (N_timepoints, M_variables)
		:param Y: 					1D or 2D numpy array of target variable (N_timepoints,) or (N_timepoints, 1)
		:param trainSizes: 			Library sizes to evaluate [start, stop, increment]
		:param sample: 				Number of random samples at each library size
		:param forwardEmbedDimensions: 	Embedding dimension
		:param predictionHorizon: 	Prediction time horizon
		:param knn: 				Number of nearest neighbors
		:param step: 				Time delay step size
		:param exclusionRadius: 	Temporal exclusion radius for neighbors
		:param seed: 				Random seed for reproducible sampling
		:param embedded: 			Whether data is already embedded
		:param validLib:			Boolean mask for valid library points
		:param includeData: 		Whether to include detailed prediction statistics
		:param ignoreNan: 			Remove NaN values from embedding
		:param directions: 			Which directions to compute: forward|reverse|both
		:param trainBlockIndices: 	Train block index range [start, end]. If None, uses all data.
		:param testBlockIndices: 	Test block index range [start, end]. If None, uses all data.
		:param device: 				Device for torch tensors ('cpu', 'cuda', or torch.device object)
		:param batchSize: 			Number of variables to process per batch in 'variable' mode
		:param useHalfPrecision: 	Use float16 instead of float32 to save VRAM
		:param batchMode:			'variable' (batch over source variables) or 'sample' (batch over subsamples per library size)
		:param sampleBatchSize:		Number of subsamples to process per batch in 'sample' mode. Defaults to all samples at once.
		"""

		self.name = 'BatchedCCM'
		self.X = X[:, None] if X.ndim == 1 else X
		self.Y = Y[:, None] if Y.ndim == 1 else Y
		self.numSources = self.X.shape[1]
		self.numTargets = self.Y.shape[1]
		self.forwardEmbedDimensions = forwardEmbedDimensions
		self.reverseEmbedDimensions = reverseEmbedDimensions if reverseEmbedDimensions is not None else forwardEmbedDimensions
		self.maxForwardDims = maxForwardDimensions
		self.maxReverseDims = maxReverseDimensions if maxReverseDimensions is not None else maxForwardDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.knnUserSpecified = knn > 0
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.ignoreNan = ignoreNan
		self.directions = directions
		self.batchSize = batchSize
		self.batchMode = batchMode
		self.sampleBatchSize = sampleBatchSize

		self.trainSizes = trainSizes if trainSizes is not None else []
		self.sample = sample
		self.seed = seed
		self.includeData = includeData

		self.device = torch.device(device) if isinstance(device, str) else device
		self.dtype = torch.float16 if useHalfPrecision else torch.float32
		self.showProgress = showProgress

		if trainBlockIndices is not None:
			self.train = trainBlockIndices
		else:
			self.train = [1, self.X.shape[0]]

		if testBlockIndices is not None:
			self.test = testBlockIndices
		else:
			self.test = [1, self.X.shape[0]]

		self.forward_performance_ = None
		self.reverse_performance_ = None
		self.PredictStatsFwd = None
		self.PredictStatsRev = None

	def Run(self):
		"""
		Execute BatchedCCM and return BatchedCCMResult.
		"""
		self.Project()

		from .Results import BatchedCCMResult
		return BatchedCCMResult(
			forward_performance = self.forward_performance_,
			reverse_performance = self.reverse_performance_,
			embedDimensions = self.forwardEmbedDimensions,
			predictionHorizon = self.predictionHorizon,
			library_sizes = self.trainSizes
		)

	def Project(self):
		"""
		Execute batched cross-mapping for all predictor variables.
		"""
		if self.directions in ['forward', 'both']:
			self.forward_performance_ = self.CrossMap(self.X, self.Y, self.forwardEmbedDimensions, self.maxForwardDims)

		if self.directions in ['reverse', 'both']:
			self.reverse_performance_ = self.CrossMap(self.Y, self.X, self.reverseEmbedDimensions, self.maxReverseDims)

	def CrossMap(self, X, Y, embedDims, maxDims):
		from .Embed import Embed

		if X.ndim == 1:
			X = X[:, None]
		if Y.ndim == 1:
			Y = Y[:, None]

		numSources = X.shape[1]
		numTargets = Y.shape[1]

		RNG = numpy.random.default_rng(self.seed)

		if embedDims == 0:
			scores = FindOptimalEmbeddingDimensionality(X, Y, maxDims = maxDims, train = self.train, test = self.test,
				predictionHorizon = self.predictionHorizon, step = self.step, ignoreNan = self.ignoreNan,
				batched = True, joint = False)
			# scores: [nVars, maxDims] (single target, squeezed) or [nTargets, nVars, maxDims].
			# Per-variable best E = argmax over the dims axis + 1 (1-indexed).
			# For multi-target, keep per-target E separately so distance computation can
			# use the right number of lags for each (source, target) pair.
			if scores.ndim == 2:
				embedDims = numpy.argmax(scores, axis = 1) + 1        # [nVars]
			else:
				embedDims = numpy.argmax(scores, axis = 2).T + 1      # [nVars, nTargets]

		dims = int(numpy.max(embedDims))

		dummy = Simplex(
			data = X,
			columns = numpy.arange(numSources).tolist(),
			target = 0,
			train = self.train,
			test = self.test,
			embedDimensions = dims,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			embedded = self.embedded,
			validLib = self.validLib,
			noTime = True,
			ignoreNan = self.ignoreNan,
			verbose = False
		)
		dummy.EmbedData()
		dummy.RemoveNan()

		libraryIndices = numpy.array(dummy.trainIndices)
		N_libraryIndices = len(libraryIndices)

		# Always embed with the maximum number of lags. The CrossMap functions select
		# the appropriate per-(source, target) lag prefix during distance computation.
		embeddings = []
		for varIndex in range(numSources):
			if self.embedded:
				embedding = X[:, varIndex].reshape(-1, 1)
			else:
				embedding = Embed(data = X,
								  columns = [varIndex],
								  embeddingDimensions = dims,
								  stepSize = self.step,
								  includeTime = False)
			embeddings.append(embedding[libraryIndices, :])

		target = torch.tensor(Y[libraryIndices + self.predictionHorizon, :], dtype = self.dtype, device = self.device)
		performance = numpy.zeros([len(self.trainSizes), self.sample, numSources, numTargets])

		if self.batchMode == 'sample':
			self.CrossMapSampleBatched(embeddings, N_libraryIndices,
									   target, numSources, performance, RNG, embedDims)
		else:
			self.CrossMapVariableBatched(embeddings, N_libraryIndices,
										 target, numSources, numTargets, performance, RNG, embedDims)

		return numpy.mean(performance, axis = 1).squeeze()

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
		Batch over source variables. Efficient when the number of source variables is large.

		For each source variable the per-lag squared pairwise distances are computed and
		cumulatively summed. The (source, target)-specific optimal E selects the prefix of
		that cumulative sum, so the kNN search for source v predicting target t uses exactly
		the first bestE[v, t] lags.

		When knn was not user-specified, each (source, target) pair uses E[s,t]+1 neighbors.
		This is enforced by masking distances beyond the pair's knn to inf after the global
		topk, so excess neighbor weights become zero and do not affect the prediction.
		"""
		dims = embeddings[0].shape[1]

		# max_knn is the largest neighbor count needed across all (source, target) pairs.
		# If the user fixed knn, use that for all pairs (no masking needed).
		# If knn is auto, each pair uses E[s,t]+1; allocate for the maximum and mask the rest.
		max_knn = self.knn if self.knnUserSpecified else int(numpy.max(embedDims)) + 1

		# Reusable buffers for per-lag distances of a single source variable: [dims, N, N]
		d = torch.zeros([dims, N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)

		# Per-(source, target) sqrt cumulative distances for the current batch: [batchSize, numTargets, N, N]
		fullDistances = torch.zeros([self.batchSize, numTargets, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)

		distances = torch.zeros([self.batchSize, max_knn, N_libraryIndices], dtype = self.dtype, device = self.device)
		neighbors = torch.zeros([self.batchSize, max_knn, N_libraryIndices], dtype = torch.long, device = self.device)
		minDistances = torch.zeros([self.batchSize, N_libraryIndices], dtype = self.dtype, device = self.device)
		weights = torch.zeros([self.batchSize, max_knn, N_libraryIndices], dtype = self.dtype, device = self.device)
		weightSum = torch.zeros([self.batchSize, N_libraryIndices], dtype = self.dtype, device = self.device)
		select = torch.zeros([self.batchSize, max_knn, N_libraryIndices], dtype = self.dtype, device = self.device)
		predictions = torch.zeros([self.batchSize, N_libraryIndices], dtype = self.dtype, device = self.device)
		perfs_ = torch.zeros([self.batchSize], dtype = self.dtype, device = self.device)

		# kIndices is reused across targets for the per-pair masking: [1, max_knn, 1]
		kIndices = torch.arange(max_knn, device = self.device).view(1, max_knn, 1)

		for batchStart in ProgressBar(range(0, numSources, self.batchSize), desc = 'Variable batch', leave = False, disable = not self.showProgress):
			batchEnd = min(batchStart + self.batchSize, numSources)
			batchEmbeddings = embeddings[batchStart:batchEnd]
			batchNumSources = len(batchEmbeddings)

			trainEmbeddings = torch.tensor(numpy.array(batchEmbeddings), dtype = self.dtype, device = self.device)

			# For each source in the batch, compute cumulative per-lag squared distances
			# and select the right prefix for each target.
			for i in range(batchNumSources):
				ElementwisePairwiseDistance(trainEmbeddings[i, :, :], trainEmbeddings[i, :, :], d)
				# d[lag, :, :] = squared pairwise distance for lag 'lag' of source i
				cumulativeDistances = torch.cumsum(d, dim = 0)  # [dims, N, N]
				for t in range(numTargets):
					thisDim = self._get_embedding_dimension(embedDims, batchStart + i, t)
					fullDistances[i, t] = torch.sqrt(cumulativeDistances[thisDim - 1])

			if self.exclusionRadius == 0:
				diagIndices = torch.arange(N_libraryIndices, device = self.device)
				fullDistances[:batchNumSources, :, diagIndices, diagIndices] = float('inf')

			for size_i, libSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False, disable = not self.showProgress)):
				for sample_i in ProgressBar(range(self.sample), desc = 'Repeats', leave = False, disable = not self.showProgress):
					subsampleIndices = RNG.choice(N_libraryIndices,
												  size = min(libSize, N_libraryIndices),
												  replace = False)
					tensorIndices = torch.as_tensor(subsampleIndices, dtype = torch.long, device = self.device)

					for t in range(numTargets):
						subsampledDistances = fullDistances[:batchNumSources, t, tensorIndices, :]
						theseDistances, theseNeighbors = torch.topk(subsampledDistances, max_knn, dim = 1,
																	   largest = False)
						# Mask out neighbors beyond dims[s,t]+1 for each source->target in the batch.
						# Masked distances become inf so their weights become zero.
						if not self.knnUserSpecified:
							knnPerSource = torch.tensor(
								[self._get_embedding_dimension(embedDims, batchStart + i, t) + 1
								 for i in range(batchNumSources)],
								dtype = torch.long, device = self.device
							).view(batchNumSources, 1, 1)
							theseDistances.masked_fill_(kIndices[:batchNumSources] >= knnPerSource, float('inf'))

						distances[:batchNumSources] = theseDistances
						neighbors[:batchNumSources] = tensorIndices[theseNeighbors]
						FloorArray(distances[:batchNumSources], 1e-6)

						minDistances[:batchNumSources] = MinAxis1(distances[:batchNumSources])
						weights[:batchNumSources] = ComputeWeights(distances[:batchNumSources],
																   minDistances[:batchNumSources])
						weightSum[:batchNumSources] = SumAxis1(weights[:batchNumSources])
						select[:batchNumSources] = target[:, t][neighbors[:batchNumSources]]
						predictions[:batchNumSources] = (weights[:batchNumSources] * select[:batchNumSources]).sum(dim = 1) / weightSum[:batchNumSources]

						targetT = target[:, t]
						targetCentered = targetT - targetT.mean()
						predCentered = predictions[:batchNumSources] - predictions[:batchNumSources].mean(dim = 1,
																										  keepdim = True)
						targetStd = torch.sqrt((targetCentered ** 2).sum())
						predStd = torch.sqrt((predCentered ** 2).sum(dim = 1))
						perfs_[:batchNumSources] = (targetCentered * predCentered).sum(dim = 1) / (targetStd * predStd)

						performance[size_i, sample_i, batchStart:batchEnd, t] = perfs_[:batchNumSources].cpu().numpy()

			del trainEmbeddings
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

		max_knn = self.knn if self.knnUserSpecified else int(numpy.max(embedDims)) + 1

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

		for size_i, libSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False, disable = not self.showProgress)):
			libSizeActual = min(libSize, N_libraryIndices)

			for batchStart in ProgressBar(range(0, self.sample, numSamplesInBatch), desc = 'Sample batch', leave = False, disable = not self.showProgress):
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

					FloorArray(distances, 1e-6)
					minDistances = distances.min(dim = 2)[0]
					weights = torch.exp(-distances / minDistances.unsqueeze(2))
					weightSum = weights.sum(dim = 2)

					targetT = target[:, t]                                                          # [N]
					selectedTargets = targetT[globalNeighbors]                                      # [numSources, numSamplesInThisBatch, max_knn, N]
					predictions = (weights * selectedTargets).sum(dim = 2) / weightSum              # [numSources, numSamplesInThisBatch, N]

					targetCentered = targetT - targetT.mean()                                       # [N]
					targetStd = torch.sqrt((targetCentered ** 2).sum())
					predCentered = predictions - predictions.mean(dim = 2, keepdim = True)          # [numSources, numSamplesInThisBatch, N]
					predStd = torch.sqrt((predCentered ** 2).sum(dim = 2))                          # [numSources, numSamplesInThisBatch]
					perfs_ = (targetCentered * predCentered).sum(dim = 2) / (targetStd * predStd)   # [numSources, numSamplesInThisBatch]

					performance[size_i, batchStart:batchEnd, :, t] = perfs_.permute(1, 0).cpu().numpy()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
