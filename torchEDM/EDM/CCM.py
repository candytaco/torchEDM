# python modules
from multiprocessing import get_context

# package modules
import numpy
from numpy import array, exp, fmax, divide, mean, nan, roll, sum, zeros
from numpy.random import default_rng

from ..Utils import IsNonStringIterable
from ..Scoring import Correlation
# local modules
from .Simplex import Simplex as SimplexClass
from .Results import CCMResult
from .NeighborFinder import PairwiseDistanceNeighborFinder


# ------------------------------------------------------------
class CCM:
	"""
	CCM class : Base class. Contains two Simplex instances
	"""

	def __init__(self,
				 data,
				 columns = None,
				 target = None,
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
				 noTime = False,
				 ignoreNan = True,
				 trainBlockIndices = None,
				 testBlockIndices = None,
				 mpMethod = None,
				 sequential = False,
				 verbose = False,
				 kdTree: bool = False,
				 scoring_function = Correlation):
		"""
		Initialize CCM.

		:param data: 				2D numpy array where column 0 is time (unless noTime=True)
		:param columns: 			Column indices to use (defaults to all except time)
		:param target: 				Target column index (defaults to column 1)
		:param trainSizes: 			Library sizes to evaluate [start, stop, increment]. For example, [10, 100, 10] tests library sizes 10, 20, ..., 100.
		:param sample: 				Number of random samples at each library size. If 0, uses all available.
		:param embedDimensions: 	Embedding dimension (E). If 0, will be set by Validate()
		:param predictionHorizon: 	Prediction time horizon (Tp)
		:param knn: 				Number of nearest neighbors. If 0, will be set to E+1 by Validate()
		:param step: 				Time delay step size (tau). Negative values indicate lag
		:param exclusionRadius: 	Temporal exclusion radius for neighbors
		:param seed: 				Random seed for reproducible sampling
		:param embedded: 			Whether data is already embedded
		:param validLib:			Boolean mask for valid library points
		:param includeData: 		Whether to include detailed prediction statistics in results
		:param noTime: 				Whether first column is time or data
		:param ignoreNan: 			Remove NaN values from embedding
		:param trainBlockIndices: 	Train block index range [start, end]. If None, uses all data.
		:param testBlockIndices: 	Test block index range [start, end]. If None, uses all data.
		:param mpMethod: 			Multiprocessing context method (ExecutionMode.SPAWN, ExecutionMode.FORK, ExecutionMode.FORKSERVER). If None, uses platform default
		:param sequential: 			Use sequential execution instead of multiprocessing
		:param verbose: 			Print diagnostic messages
		:param kdTree:				use KDTree for neighbors? Else use pairwise distances
		"""

		# Assign parameters directly
		self.name = 'CCM'
		self.Data = data
		self.columns = columns
		self.target = target
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.noTime = noTime
		self.ignoreNan = ignoreNan
		self.verbose = verbose

		# Assign CCM parameters
		self.trainSizes = trainSizes if trainSizes is not None else []
		self.sample = sample
		self.seed = seed
		self.includeData = includeData

		# Assign execution parameters
		self.mpMethod = mpMethod
		self.sequential = sequential
		self.scoring_function = scoring_function

		# Set train & test block indices (0-based, half-open)
		if trainBlockIndices is not None:
			self.train = trainBlockIndices
		else:
			self.train = [(0, self.Data.shape[0])]

		if testBlockIndices is not None:
			self.test = testBlockIndices
		else:
			self.test = [(0, self.Data.shape[0])]

		self.CrossMapList = None  # List of CrossMap results
		self.libMeans = None  # DataFrame of CrossMap results
		self.PredictStats1 = None  # DataFrame of CrossMap stats
		self.PredictStats2 = None  # DataFrame of CrossMap stats

		self.kdtree = kdTree

		# Setup
		self.Validate()  # CCM Method

		# Instantiate Forward and Reverse Mapping objects using plain arguments
		self.FwdMap = SimplexClass(data = data,
								   columns = columns,
								   target = target,
								   train = self.train,
								   test = self.test,
								   embedDimensions = embedDimensions,
								   predictionHorizon = predictionHorizon,
								   knn = knn,
								   step = step,
								   exclusionRadius = exclusionRadius,
								   embedded = embedded,
								   validLib = validLib,
								   noTime = noTime,
								   ignoreNan = ignoreNan,
								   verbose = verbose)
		self.FwdMap.KDTree = kdTree

		# For reverse map, swap columns and target
		self.RevMap = SimplexClass(data = data,
								   columns = target,
								   target = columns,
								   train = self.train,
								   test = self.test,
								   embedDimensions = embedDimensions,
								   predictionHorizon = predictionHorizon,
								   knn = knn,
								   step = step,
								   exclusionRadius = exclusionRadius,
								   embedded = embedded,
								   validLib = validLib,
								   noTime = noTime,
								   ignoreNan = ignoreNan,
								   verbose = verbose)
		self.RevMap.KDTree = kdTree

	# -------------------------------------------------------------------
	# Methods
	# -------------------------------------------------------------------
	def Run(self):
		"""
		Execute CCM and return CCMResult.

		:return: CCM results with library means and optional detailed statistics
		"""
		self.Project()

		return CCMResult(
			libMeans = self.libMeans,
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			predictStats1 = self.PredictStats1 if self.includeData else None,
			predictStats2 = self.PredictStats2 if self.includeData else None
		)

	# -------------------------------------------------------------------
	def Project(self, sequential = False):
		"""
		CCM both directions with CrossMap()
		"""

		if self.verbose:
			print(f'{self.name}: Project()')

		if self.sequential:  # Sequential alternative to multiprocessing
			FwdCM = self.CrossMap(False)
			RevCM = self.CrossMap(True)
			self.CrossMapList = [FwdCM, RevCM]
		else:
			# multiprocessing Pool CrossMap both directions simultaneously
			poolArgs = [False, True]
			mpContext = get_context(self.mpMethod)
			with mpContext.Pool(processes = 2) as pool:
				CrossMapList = pool.map(self.CrossMap, poolArgs)

			self.CrossMapList = CrossMapList

		FwdCM, RevCM = self.CrossMapList

		# Create libMeans array: shape (n_lib_sizes, 3)
		# Column 0: LibSize, Column 1: Fwd correlation, Column 2: Rev correlation
		self.libMeans = numpy.zeros([len(self.trainSizes), 3])
		for i, size in enumerate(self.trainSizes):
			self.libMeans[i, :] = [size, FwdCM['libcorrelation'][size], RevCM['libcorrelation'][size]]

		if self.includeData:
			FwdCMStats = FwdCM['predictStats']  # key libSize : list of CE dicts
			RevCMStats = RevCM['predictStats']

			# Build PredictStats1 array
			# Each row is a sample with: LibSize, correlation, mae, rmse, mse, nrmse
			stats1_rows = []
			for libSize in FwdCMStats.keys():
				LibSize = [libSize] * self.sample  # this libSize sample times
				libStats = FwdCMStats[libSize]  # sample ComputeError dicts

				for s in range(self.sample):
					stats = libStats[s]
					row = [libSize[s], stats['correlation'], stats['mae'], stats['rmse'],
						   stats['mse'], stats['nrmse']]
					stats1_rows.append(row)

			self.PredictStats1 = array(stats1_rows)

			# Build PredictStats2 array
			stats2_rows = []
			for libSize in RevCMStats.keys():
				LibSize = [libSize] * self.sample  # this libSize sample times
				libStats = RevCMStats[libSize]  # sample ComputeError dicts

				for s in range(self.sample):
					stats = libStats[s]
					row = [libSize[s], stats['correlation'], stats['mae'], stats['rmse'],
						   stats['mse'], stats['nrmse']]
					stats2_rows.append(row)

			self.PredictStats2 = array(stats2_rows)

	# -------------------------------------------------------------------
	#
	# -------------------------------------------------------------------
	def CrossMap(self, reverse: bool = False):
		"""
		Perform cross-mapping in specified direction

		:param reverse: do reverse direction?
		:return: Dictionary containing cross-mapping results
		"""
		if self.verbose:
			print(f'{self.name}: CrossMap()')

		simplex = self.RevMap if reverse else self.FwdMap
		simplex.EmbedData()
		simplex.RemoveNan()

		# Create random number generator : None sets random state from OS
		RNG = default_rng(self.seed)

		# Copy S.lib_i since it's replaced every iteration
		lib_i = simplex.trainIndices.copy()
		N_lib_i = len(lib_i)

		libcorrelationMap = {}  # Output dict libSize key : mean correlation value
		libStatMap = {}  # Output dict libSize key : list of ComputeError dicts

		if not self.kdtree:
			simplex.FindNeighbors()

		# Loop for library sizes
		for libSize in self.trainSizes:
			correlations = zeros(self.sample)
			if self.includeData:
				predictStats = [None] * self.sample

			# Loop for subsamples
			for s in range(self.sample):
				if self.kdtree:
					# Generate library row indices for this subsample
					rng_i = RNG.choice(lib_i, size = min(libSize, N_lib_i),
									   replace = False)

					simplex.trainIndices = rng_i
					simplex.FindNeighbors()  # Depends on S.lib_i
					neighbor_distances = simplex.knn_distances
					neighbor_indices = simplex.knn_neighbors
				else:
					rng_i = RNG.choice(numpy.arange(simplex.neighborFinder.distanceMatrix.shape[0]), size = min(libSize, N_lib_i),
									   replace = False)
					d = simplex.neighborFinder.distanceMatrix[rng_i, :]
					neighbor_distances, sub_indices = PairwiseDistanceNeighborFinder.find_neighbors(d, simplex.knn)
					raw_indices = rng_i[sub_indices]
					neighbor_indices = simplex._MapKNNIndicesToLibraryIndices(raw_indices)

				# Code from Simplex:Project ---------------------------------
				# First column is minimum distance of all N test rows
				minDistances = neighbor_distances[:, 0]
				# In case there is 0 in minDistances: minWeight = 1E-6
				minDistances = fmax(minDistances, 1E-6)

				# Divide each column of N x k knn_distances by minDistances
				scaledDistances = divide(neighbor_distances, minDistances[:, None])
				weights = exp(-scaledDistances)  # Npred x k
				weightRowSum = sum(weights, axis = 1)  # Npred x 1

				# Matrix of knn_neighbors + predictionHorizon defines library target values
				knn_neighbors_Tp = neighbor_indices + self.predictionHorizon  # Npred x k

				libTargetValues = simplex.targetVec[knn_neighbors_Tp].squeeze()
				# Code from Simplex:Project ----------------------------------

				# Projection is average of weighted knn library target values
				projection_ = sum(weights * libTargetValues,
								  axis = 1) / weightRowSum

				# Align observations & predictions as in FormatProjection()
				# Shift projection_ by predictionHorizon
				projection_ = roll(projection_, simplex.predictionHorizon)
				if simplex.predictionHorizon > 0:
					projection_[:simplex.predictionHorizon] = nan
				elif simplex.predictionHorizon < 0:
					projection_[simplex.predictionHorizon:] = nan

				err = self.scoring_function(simplex.targetVec[simplex.testIndices, 0], projection_)

				correlations[s] = err

				if self.includeData:
					predictStats[s] = err

			libcorrelationMap[libSize] = mean(correlations)

			if self.includeData:
				libStatMap[libSize] = predictStats

		# Reset S.lib_i to original
		simplex.trainIndices = lib_i

		if self.includeData:
			return {'columns': simplex.columns, 'target': simplex.target,
					'libcorrelation': libcorrelationMap, 'predictStats': libStatMap}
		else:
			return {'columns': simplex.columns, 'target': simplex.target, 'libcorrelation': libcorrelationMap}

	# --------------------------------------------------------------------
	def Validate(self):
		# --------------------------------------------------------------------
		"""
		Validate CCM inputs and parameters

		:raises RuntimeError: if inputs are invalid
		"""
		if self.verbose:
			print(f'{self.name}: Validate()')

		if not len(self.trainSizes):
			raise RuntimeError(f'{self.name} Validate(): LibSizes required.')
		if not IsNonStringIterable(self.trainSizes):
			self.trainSizes = [int(L) for L in self.trainSizes.split()]

		if self.sample == 0:
			raise RuntimeError(f'{self.name} Validate(): ' + \
							   'sample must be non-zero.')

		# libSizes
		#   if 3 arguments presume [start, stop, increment]
		#      if increment < stop generate the library sequence.
		#      if increment > stop presume list of 3 library sizes.
		#   else: Already list of library sizes.
		if len(self.trainSizes) == 3:
			# Presume ( start, stop, increment ) sequence arguments
			start, stop, increment = [int(s) for s in self.trainSizes]

			# If increment < stop, presume start : stop : increment
			# and generate the sequence of library sizes
			if increment < stop:
				if increment < 1:
					msg = f'{self.name} Validate(): ' + \
						  f'libSizes increment {increment} is invalid.'
					raise RuntimeError(msg)

				if start > stop:
					msg = f'{self.name} Validate(): ' + \
						  f'libSizes start {start} stop {stop} are invalid.'
					raise RuntimeError(msg)

				if start < self.embedDimensions:
					msg = f'{self.name} Validate(): ' + \
						  f'libSizes start {start} less than E {self.embedDimensions}'
					raise RuntimeError(msg)
				elif start < 3:
					msg = f'{self.name} Validate(): ' + \
						  f'libSizes start {start} less than 3.'
					raise RuntimeError(msg)

				# Fill in libSizes sequence
				self.trainSizes = [i for i in range(start, stop + 1, increment)]

		if self.trainSizes[-1] > self.Data.shape[0]:
			msg = f'{self.name} Validate(): ' + \
				  f'Maximum libSize {self.trainSizes[-1]}' + \
				  f' exceeds data size {self.Data.shape[0]}.'
			raise RuntimeError(msg)

		if self.trainSizes[0] < self.embedDimensions + 2:
			msg = f'{self.name} Validate(): ' + \
				  f'Minimum libSize {self.trainSizes[0]}' + \
				  f' invalid for E={self.embedDimensions}. Minimum {self.embedDimensions + 2}.'
			raise RuntimeError(msg)

		if self.predictionHorizon < 0:
			embedShift = abs(self.step) * (self.embedDimensions - 1)
			maxLibSize = self.trainSizes[-1]
			maxAllowed = self.Data.shape[0] - embedShift + (self.predictionHorizon + 1)
			if maxLibSize > maxAllowed:
				msg = f'{self.name} Validate(): Maximum libSize {maxLibSize}' + \
					  f' too large for predictionHorizon {self.predictionHorizon}, E {self.embedDimensions}, step {self.step}' + \
					  f' Maximum is {maxAllowed}'
				raise RuntimeError(msg)
