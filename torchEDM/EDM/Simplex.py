import numpy
import torch

from numpy import array, full, nan, linspace, column_stack

from .EDM import EDM
from .Results import SimplexResult


#-----------------------------------------------------------
class Simplex(EDM):
	"""
	Simplex class : child of EDM
	CCM & Multiview are composed of Simplex instances
	"""

	def __init__(self,
				 data,
				 columns=None,
				 target=None,
				 train=None,
				 test=None,
				 embedDimensions=0,
				 predictionHorizon=1,
				 knn=0,
				 step=-1,
				 exclusionRadius=0,
				 embedded=False,
				 validLib=None,
				 noTime=False,
				 ignoreNan=True,
				 verbose=False,
				 generateSteps=0,
				 generateConcat=False,
				 device=None,
				 dtype=None):
		"""
		Initialize Simplex as child of EDM.

		:param data: 2D numpy array where column 0 is time (unless noTime=True)
		:param columns: Column indices to use for embedding (defaults to all except time)
		:param target: Target column index (defaults to column 1)
		:param train: Training set indices [start, end]
		:param test: Test set indices [start, end]
		:param embedDimensions: Embedding dimension (E). If 0, will be set by Validate()
		:param predictionHorizon: Prediction time horizon (Tp)
		:param knn: Number of nearest neighbors. If 0, will be set to E+1 by Validate()
		:param step: Time delay step size (tau). Negative values indicate lag
		:param exclusionRadius: Temporal exclusion radius for neighbors
		:param embedded: Whether data is already embedded
		:param validLib: Boolean mask for valid library points
		:param noTime: Whether first column is time or data
		:param ignoreNan: Remove NaN values from embedding
		:param verbose: Print diagnostic messages
		:param generateSteps: Number of iterative generation steps. If 0, uses standard prediction.
		:param generateConcat: Whether to concatenate generated predictions
		:param device: torch device to use (None for auto-detect)
		:param dtype: torch dtype to use (None for float32)
		"""

		super(Simplex, self).__init__(data, isEmbedded=False, name='Simplex')

		self.columns         = columns
		self.target          = target
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn             = knn
		self.step            = step
		self.exclusionRadius = exclusionRadius
		self.embedded        = embedded
		self.validLib        = validLib if validLib is not None else []
		self.noTime          = noTime
		self.ignoreNan       = ignoreNan
		self.verbose         = verbose

		self.train = train if train is not None else []
		self.test = test if test is not None else []

		self.generateSteps = generateSteps
		self.generateConcat = generateConcat

		self.embedStep         = self.step
		self.isEmbedded        = self.embedded

		# GPU setup
		if device is not None:
			self.device = device
		elif torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.dtype = dtype if dtype is not None else torch.float64

		# Setup
		self.Validate()
		self.CreateIndices()

		self.targetVec = self.Data[:, [self.target[0]]]

		if self.noTime:
			timeIndex = [i for i in range(1, self.Data.shape[0] + 1)]
			self.time = array(timeIndex, dtype=int)
		else:
			self.time = self.Data[:, 0]

	#-------------------------------------------------------------------
	# Methods
	#-------------------------------------------------------------------
	def Run(self, return_predictions: bool = True):
	#-------------------------------------------------------------------
		"""
		Execute standard prediction and return SimplexResult.

		:param return_predictions: If False, the predictions field of the result will not be populated
		"""
		self.EmbedData()
		self.RemoveNan()
		self.FindNeighborsTorch()
		self.Project()
		self.FormatProjection()

		return SimplexResult(
			time=self.Projection[:, 0],
			projection=self.Projection if return_predictions else None,
			embedDimensions=self.embedDimensions,
			predictionHorizon=self.predictionHorizon
		)

	#-------------------------------------------------------------------
	def FindNeighborsTorch(self):
	#-------------------------------------------------------------------
		"""
		Find k nearest neighbors using torch on GPU.
		Computes pairwise squared Euclidean distances, applies exclusion
		mask, and selects k nearest via torch.topk.
		Stores results as numpy arrays in self.knn_neighbors and
		self.knn_distances for compatibility with FormatProjection.
		"""
		if self.verbose:
			print(f'{self.name}: FindNeighborsTorch()')

		trainEmbedding = self.Embedding[self.trainIndices, :]
		testEmbedding = self.Embedding[self.testIndices, :]

		trainTensor = torch.tensor(trainEmbedding, device=self.device, dtype=self.dtype)
		testTensor = torch.tensor(testEmbedding, device=self.device, dtype=self.dtype)

		# Pairwise squared Euclidean distances: [nTrain x nTest]
		# torch.cdist computes Euclidean (p=2), we square afterward for consistency
		# with the weight computation which expects Euclidean (not squared) distances
		distanceMatrix = torch.cdist(trainTensor, testTensor, p=2)

		# Apply exclusion mask: set excluded pairs to infinity
		exclusionMask = self._BuildExclusionMask()
		if exclusionMask.any():
			maskTensor = torch.tensor(exclusionMask, device=self.device, dtype=torch.bool)
			distanceMatrix[maskTensor] = float('inf')

		# topk on dim=0 finds k smallest distances per test point (columns)
		# distanceMatrix is [nTrain x nTest], so dim=0 selects across train rows
		topkDistances, topkIndices = torch.topk(distanceMatrix, self.knn, dim=0, largest=False)

		# Transpose to [nTest x knn] to match expected shape
		neighborDistances = topkDistances.t()
		neighborIndices = topkIndices.t()

		# Move results to CPU numpy
		self.knn_distances = neighborDistances.cpu().numpy()
		# We no long map the neighbor indices back to original data indices
		# because for predictions, we also just mask the Y data to the same indices
		self.knn_neighbors = neighborIndices.cpu().numpy()

		# Clean up GPU tensors
		del trainTensor, testTensor, distanceMatrix, topkDistances, topkIndices
		del neighborDistances, neighborIndices
		if exclusionMask.any():
			del maskTensor
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	#-------------------------------------------------------------------
	def Project(self):
	#-------------------------------------------------------------------
		"""
		Simplex Projection on GPU using torch.
		Sugihara & May (1990) doi.org/10.1038/344734a0
		"""
		if self.verbose:
			print(f'{self.name}: ProjectTorch()')

		distances = torch.tensor(self.knn_distances, device=self.device, dtype=self.dtype)
		neighbors = torch.tensor(self.knn_neighbors, device=self.device, dtype=torch.long)
		y_train = torch.tensor(self.targetVec[self.trainIndices + self.predictionHorizon].squeeze(), device=self.device, dtype=self.dtype)

		# Minimum distance per test row (first neighbor), floored at 1e-6
		minDistances = distances[:, 0]
		torch.clamp_min(minDistances, 1e-6, out=minDistances)

		# Scale distances and compute exponential weights
		scaledDistances = distances / minDistances.unsqueeze(1)
		weights = torch.exp(-scaledDistances)
		weightRowSum = torch.sum(weights, dim=1)

		# Library target values at neighbor + predictionHorizon
		libTargetValues = y_train[neighbors]

		# Weighted average prediction
		projection = torch.sum(weights * libTargetValues, dim=1) / weightRowSum

		# Variance estimate
		libTargetPredDiff = libTargetValues - projection.unsqueeze(1)
		deltaSqr = libTargetPredDiff ** 2
		variance = torch.sum(weights * deltaSqr, dim=1) / weightRowSum

		# Move results back to numpy
		self.projection = projection.cpu().numpy()
		self.variance = variance.cpu().numpy()

		# Clean up GPU tensors
		del distances, neighbors, y_train, minDistances, scaledDistances
		del weights, weightRowSum, libTargetValues
		del projection, libTargetPredDiff, deltaSqr, variance
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	#-------------------------------------------------------------------
	def Generate(self, return_predictions: bool = True):
	#-------------------------------------------------------------------
		"""
		Simplex Generation
		Given train: override test for single prediction at end of train
		Replace self.Projection with G.Projection

		:param return_predictions: If False, the predictions field of the result will not be populated
		"""
		if self.verbose:
			print(f'{self.name}: Generate()')

		N      = self.Data.shape[0]
		column = self.columns[0]
		target = self.target[0]
		train  = self.train

		if self.verbose:
			print(f'\tData shape: {self.Data.shape}')
			print(f'\ttrain: {train}')

		test = [train[-1] - 1, train[-1]]
		if self.verbose:
			print(f'\tGenerate(): test overriden to {test}')

		nOutRows  = self.generateSteps
		generated = full((nOutRows, 4), nan)

		columnData     = full(N + nOutRows, nan)
		columnData[:N] = self.Data[:, column]

		timeData = full(N + nOutRows, nan)
		if self.noTime:
			timeData[:N] = linspace(1, N, N)
			newData = column_stack([timeData[:N], columnData[:N]])
		else:
			timeData[:N] = self.time
			newData = column_stack([timeData[:N], columnData[:N]])

		for step in range(self.generateSteps):
			if self.verbose:
				print(f'{self.name}: Generate(): step {step} {"=" * 50}')

			G = Simplex(data=newData,
						columns=[column],
						target=target,
						train=train,
						test=test,
						embedDimensions=self.embedDimensions,
						predictionHorizon=self.predictionHorizon,
						knn=self.knn,
						step=self.step,
						exclusionRadius=self.exclusionRadius,
						embedded=self.embedded,
						validLib=self.validLib,
						noTime=self.noTime,
						generateSteps=self.generateSteps,
						generateConcat=self.generateConcat,
						ignoreNan=self.ignoreNan,
						verbose=self.verbose,
						device=self.device,
						dtype=self.dtype)

			G.Run()

			if self.verbose:
				print('1) G.Projection')
				print(G.Projection); print()

			newPrediction = G.Projection[-1, 2]
			newTime       = G.Projection[-1, 0]

			generated[step, 0] = newTime
			generated[step, 1] = nan
			generated[step, 2] = newPrediction
			generated[step, 3] = nan

			if self.verbose:
				print(f'2) generated step {step}')

			test = [p + 1 for p in test]

			if self.verbose:
				print(f'4) test {test}')

			columnData[N + step] = newPrediction
			timeData[N + step]   = newTime

			newData = column_stack([timeData[:(N + step + 1)],
									columnData[:(N + step + 1)]])

			if self.verbose:
				print(f'5) newData: {newData.shape}')

		if self.generateConcat:
			timeName = 0
			data_obs = column_stack([self.Data[:, timeName], self.Data[:, target]])
			self.Projection = numpy.vstack([data_obs, generated[:, [0, 2]]])
		else:
			self.Projection = generated

		return SimplexResult(
			time=self.Projection[:, 0],
			projection=self.Projection if return_predictions else None,
			embedDimensions=self.embedDimensions,
			predictionHorizon=self.predictionHorizon
		)
