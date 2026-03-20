import numpy
import torch
from tqdm import tqdm as ProgressBar

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
				 embedDimensions = 0,
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
		:param embedDimensions: 	Embedding dimension (E)
		:param predictionHorizon: 	Prediction time horizon (Tp)
		:param knn: 				Number of nearest neighbors
		:param step: 				Time delay step size (tau)
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
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn if knn > 0 else embedDimensions + 1
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
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			library_sizes = self.trainSizes
		)

	def Project(self):
		"""
		Execute batched cross-mapping for all predictor variables.
		"""
		if self.directions in ['forward', 'both']:
			self.forward_performance_ = self.CrossMap(self.X, self.Y)

		if self.directions in ['reverse', 'both']:
			self.reverse_performance_ = self.CrossMap(self.Y, self.X)

	def CrossMap(self, X, Y):
		from .Embed import Embed

		if X.ndim == 1:
			X = X[:, None]
		if Y.ndim == 1:
			Y = Y[:, None]

		numSources = X.shape[1]
		numTargets = Y.shape[1]

		RNG = numpy.random.default_rng(self.seed)

		dummy = Simplex(
			data = X,
			columns = numpy.arange(numSources).tolist(),
			target = 0,
			train = self.train,
			test = self.test,
			embedDimensions = self.embedDimensions,
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

		embeddings = []
		for varIndex in range(numSources):
			if self.embedded:
				embedding = X[:, varIndex].reshape(-1, 1)
			else:
				embedding = Embed(data = X,
								  columns = [varIndex],
								  embeddingDimensions = self.embedDimensions,
								  stepSize = self.step,
								  includeTime = False)
			embeddings.append(embedding[libraryIndices, :])

		target = torch.tensor(Y[libraryIndices + self.predictionHorizon, :], dtype = self.dtype, device = self.device)
		performance = numpy.zeros([len(self.trainSizes), self.sample, numSources, numTargets])

		if self.batchMode == 'sample':
			self.CrossMapSampleBatched(embeddings, N_libraryIndices,
									   target, numSources, performance, RNG)
		else:
			self.CrossMapVariableBatched(embeddings, N_libraryIndices,
										 target, numSources, numTargets, performance, RNG)

		return numpy.mean(performance, axis = 1).squeeze()

	def CrossMapVariableBatched(self, embeddings, N_libraryIndices,
								target, numSources, numTargets, performance, RNG):
		"""
		Batch over source variables. Efficient when the number of source variables is large.
		"""
		d = torch.zeros([embeddings[0].shape[1], N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)
		fullDistances = torch.zeros([self.batchSize, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)

		for batchStart in ProgressBar(range(0, numSources, self.batchSize), desc = 'Variable batch', leave = False, disable = not self.showProgress):
			batchEnd = min(batchStart + self.batchSize, numSources)
			batchEmbeddings = embeddings[batchStart:batchEnd]
			batchNumSources = len(batchEmbeddings)

			trainEmbeddings = torch.tensor(numpy.array(batchEmbeddings), dtype = self.dtype, device = self.device)
			for i in range(batchNumSources):
				ElementwisePairwiseDistance(trainEmbeddings[i, :, :], trainEmbeddings[i, :, :], d)
				fullDistances[i, :, :] = torch.sum(d, dim = 0)
			fullDistances[:batchNumSources, :, :] = torch.sqrt(fullDistances[:batchNumSources, :, :])
			distances = torch.zeros([batchNumSources, self.knn, N_libraryIndices], dtype = self.dtype, device = self.device)
			neighbors = torch.zeros([batchNumSources, self.knn, N_libraryIndices], dtype = torch.long, device = self.device)
			minDistances = torch.zeros([batchNumSources, N_libraryIndices], dtype = self.dtype, device = self.device)
			weights = torch.zeros([batchNumSources, self.knn, N_libraryIndices], dtype = self.dtype, device = self.device)
			weightSum = torch.zeros([batchNumSources, N_libraryIndices], dtype = self.dtype, device = self.device)
			select = torch.zeros([batchNumSources, self.knn, N_libraryIndices, numTargets], dtype = self.dtype, device = self.device)
			predictions = torch.zeros([batchNumSources, N_libraryIndices, numTargets], dtype = self.dtype, device = self.device)
			perfs_ = torch.zeros([batchNumSources, numTargets], dtype = self.dtype, device = self.device)

			if self.exclusionRadius == 0:
				diagIndices = torch.arange(fullDistances.shape[1], device = self.device)
				for i in range(batchNumSources):
					fullDistances[i, diagIndices, diagIndices] = float('inf')

			for size_i, libSize in enumerate(ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False, disable = not self.showProgress)):
				for sample_i in ProgressBar(range(self.sample), desc = 'Repeats', leave = False, disable = not self.showProgress):
					subsampleIndices = RNG.choice(N_libraryIndices,
												  size = min(libSize, N_libraryIndices),
												  replace = False)

					subsampleTorch = torch.as_tensor(subsampleIndices, dtype = torch.long, device = self.device)
					subsampledDistances = fullDistances[:batchNumSources, subsampleTorch, :]
					topkDistances, topkLocalNeighbors = torch.topk(subsampledDistances, self.knn, dim = 1, largest = False)
					distances[:] = topkDistances
					neighbors[:] = subsampleTorch[topkLocalNeighbors]
					FloorArray(distances, 1e-6)

					minDistances[:] = MinAxis1(distances)
					weights[:] = ComputeWeights(distances, minDistances)
					weightSum[:] = SumAxis1(weights)
					select[:] = target[neighbors]
					predictions[:] = (weights.unsqueeze(-1) * select).sum(dim = 1) / weightSum.unsqueeze(-1)

					targetCentered = target - target.mean(dim = 0, keepdim = True)
					targetStd = torch.sqrt((targetCentered ** 2).sum(dim = 0, keepdim = True))
					predCentered = predictions - predictions.mean(dim = 1, keepdim = True)
					predStd = torch.sqrt((predCentered ** 2).sum(dim = 1))
					perfs_[:] = (targetCentered.unsqueeze(0) * predCentered).sum(dim = 1) / (targetStd * predStd)

					performance[size_i, sample_i, batchStart:batchEnd, :] = perfs_.cpu().numpy()

			del trainEmbeddings
			del perfs_
			del distances
			del neighbors
			del minDistances
			del weights
			del weightSum
			del select
			del predictions
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def CrossMapSampleBatched(self, embeddings, N_libraryIndices,
							  target, numSources, performance, RNG):
		"""
		Batch over subsamples per library size. Efficient when the number of source variables is small.

		All M source variables are processed simultaneously. For each library size, subsamples are
		drawn and evaluated in batches of sampleBatchSize, so the batch dimension runs over samples
		rather than variables.
		"""
		numSamplesInBatch = self.sampleBatchSize if self.sampleBatchSize is not None else self.sample

		# Build full distance matrix for all variables at once: [numSources, N_lib, N_lib]
		trainEmbeddings = torch.tensor(numpy.array(embeddings), dtype = self.dtype, device = self.device)
		d = torch.zeros([embeddings[0].shape[1], N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)
		fullDistances = torch.zeros([numSources, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)
		for i in range(numSources):
			ElementwisePairwiseDistance(trainEmbeddings[i, :, :], trainEmbeddings[i, :, :], d)
			fullDistances[i, :, :] = torch.sum(d, dim = 0)
		fullDistances = torch.sqrt(fullDistances)

		if self.exclusionRadius == 0:
			diagIndices = torch.arange(N_libraryIndices, device = self.device)
			fullDistances[:, diagIndices, diagIndices] = float('inf')

		del trainEmbeddings
		del d

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

				# Gather subsampled distances: [numSources, numSamplesInThisBatch, libSizeActual, N_lib]
				# fullDistances[:, subsampleTorch, :] uses advanced indexing:
				#   subsampleTorch has shape [numSamplesInThisBatch, libSizeActual]
				#   result has shape [numSources, numSamplesInThisBatch, libSizeActual, N_lib]
				subsampledDistances = fullDistances[:, subsampleTorch, :]

				# topk over libSizeActual dimension (dim=2): results are [numSources, numSamplesInThisBatch, knn, N_lib]
				topkDistances, topkLocalNeighbors = torch.topk(subsampledDistances, self.knn, dim = 2, largest = False)

				# Remap local neighbor indices (into the subsample) back to global library indices
				# subsampleTorch[s, local] -> global index, for each sample s
				# topkLocalNeighbors: [numSources, numSamplesInThisBatch, knn, N_lib]
				# subsampleTorch:     [numSamplesInThisBatch, libSizeActual]
				# We need to gather along the libSizeActual dim of subsampleTorch using topkLocalNeighbors
				# Expand subsampleTorch to [1, numSamplesInThisBatch, libSizeActual, 1] for gathering
				subsampleExpanded = subsampleTorch.unsqueeze(0).unsqueeze(-1)
				# topkLocalNeighbors used as gather index along libSizeActual: [numSources, numSamplesInThisBatch, knn, N_lib]
				globalNeighbors = subsampleExpanded.expand(numSources, -1, libSizeActual, N_libraryIndices).gather(
					dim = 2, index = topkLocalNeighbors
				)

				FloorArray(topkDistances, 1e-6)
				minDistances = topkDistances.min(dim = 2)[0]                                           # [numSources, numSamplesInThisBatch, N_lib]
				weights = torch.exp(-topkDistances / minDistances.unsqueeze(2))                        # [numSources, numSamplesInThisBatch, knn, N_lib]
				weightSum = weights.sum(dim = 2)                                                       # [numSources, numSamplesInThisBatch, N_lib]

				# target[globalNeighbors]: [numSources, numSamplesInThisBatch, knn, N_lib, numTargets]
				selectedTargets = target[globalNeighbors]
				predictions = (weights.unsqueeze(-1) * selectedTargets).sum(dim = 2) / weightSum.unsqueeze(-1)
				# predictions: [numSources, numSamplesInThisBatch, N_lib, numTargets]

				# Correlation over the N_lib dimension (dim=2)
				targetCentered = target - target.mean(dim = 0, keepdim = True)                         # [N_lib, numTargets]
				targetStd = torch.sqrt((targetCentered ** 2).sum(dim = 0))                             # [numTargets]
				predCentered = predictions - predictions.mean(dim = 2, keepdim = True)                 # [numSources, numSamplesInThisBatch, N_lib, numTargets]
				predStd = torch.sqrt((predCentered ** 2).sum(dim = 2))                                 # [numSources, numSamplesInThisBatch, numTargets]
				# targetCentered broadcast: [1, 1, N_lib, numTargets]
				perfs_ = (targetCentered.unsqueeze(0).unsqueeze(0) * predCentered).sum(dim = 2) / (targetStd * predStd)
				# perfs_: [numSources, numSamplesInThisBatch, numTargets]

				performance[size_i, batchStart:batchEnd, :, :] = perfs_.permute(1, 0, 2).cpu().numpy()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
