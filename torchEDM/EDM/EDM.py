	# python modules
import datetime as dt
from datetime import datetime
from typing import List, Tuple
from warnings import warn

import numpy
# package modules
from numpy import any, concatenate, isnan
from numpy import append, array, column_stack, empty, floating, full, integer, nan
from numpy import delete, zeros, apply_along_axis
from scipy.spatial import KDTree

from ..Utils import IsNonStringIterable
# local modules
from .Embed import Embed
from .NeighborFinder import KDTreeNeighborFinder, PairwiseDistanceNeighborFinder


# --------------------------------------------------------------------
class EDM:
	# --------------------------------------------------------------------
	"""
	EDM class : data container
	Simplex, SMap, CCM inherited from EDM
	"""

	def __init__(self, data, isEmbedded = False, name = 'EDM'):
		self.knnThreads = -1
		self.predictionHorizon: int = None
		self.embedStep: int = None
		self.isEmbedded: bool = isEmbedded
		self.name = name
		self.embedDimensions: int = None

		self.Data = data  # DataFrame
		self.Embedding: numpy.ndarray = None  # DataFrame, includes nan
		self.Projection = None  # DataFrame Simplex & SMap output

		# Note: these two are over-written by the indexing-making function into list of actual indices
		self.trainIndices: List[Tuple[int, int]] = None  # ndarray library indices
		self.testIndices: List[Tuple[int, int]] = None  # ndarray prediction indices : nan removed

		self.pred_i_all = None  # ndarray prediction indices : nan included
		self.predList = []  # list of disjoint pred_i_all
		self.disjointLib = False  # True if disjoint library
		self.libOverlap = False  # True if train & test overlap
		self.ignoreNan = True  # Remove nan from embedding
		self.xRadKnnFactor = 5  # exlcusionRadius knn factor

		self.knn_neighbors = None  # ndarray (N_pred, knn) sorted
		self.knn_distances = None  # ndarray (N_pred, knn) sorted

		self.projection = None  # ndarray Simplex & SMap output
		self.variance = None  # ndarray Simplex & SMap output
		self.targetVec = None  # ndarray entire record
		self.targetVecNan = False  # True if targetVec has nan : SMap only
		self.time = None  # ndarray entire record numerically operable

		self.KDTree = False			# use KDTree? if false, use matrix neighbor finder
		self.neighborFinder = None
		self.requery = False
		
		self._exclusion = None
		self._knn = None

	
	@property
	def knn_(self):
		"""
		Returns knn value, inflated only for KDTree with exclusions.
		Matrix-based finder handles exclusions internally.
		"""
		if self._knn is None:
			self._knn = self.knn
			
			# Only inflate knn for KDTree with exclusions
			if self.KDTree:
				if self.libOverlap and not self.CheckExclusion:
					self._knn += 1
				elif self.CheckExclusion:
					self._knn = min(
						self._knn * self.xRadKnnFactor, 
						len(self.trainIndices)
					)
				elif len(self.validLib):
					self._knn = len(self.trainIndices)
		
		return self._knn

	def _BuildExclusionMask(self):
		"""
		Pre-compute boolean exclusion mask for neighbor queries.
		Encodes both degenerate neighbors (libOverlap) and exclusionRadius.
		
		:returns: (n_pred, n_lib) where True = exclude neighbor
		"""

		n_pred = len(self.testIndices)
		n_lib = len(self.trainIndices)
		mask = numpy.zeros((n_lib, n_pred), dtype = bool)
		if not self.libOverlap and not self.CheckExclusion:
			return mask

		# Initialize mask: False = include neighbor

		# Build index lookup for library
		lib_index_map = {index: i for i, index in enumerate(self.trainIndices)}

		for i, pred_index in enumerate(self.testIndices):
			# Handle degenerate neighbors (leave-one-out)
			if self.libOverlap and pred_index in lib_index_map:
				lib_pos = lib_index_map[pred_index]
				mask[lib_pos, i] = True

			# Handle exclusion radius
			if self.CheckExclusion:
				rowLow = max(numpy.min(self.trainIndices), pred_index - self.exclusionRadius)
				rowHi = min(numpy.max(self.trainIndices), pred_index + self.exclusionRadius)

				# Mark all library indices in exclusion window
				for lib_i, lib_idx in enumerate(self.trainIndices):
					if rowLow <= lib_idx <= rowHi:
						mask[lib_i, i] = True

		return mask
	
	
	def FindNeighbors(self, requery=False):
		"""
		Find closest neighbors with exclusions baked into distance computation.
		"""
		if self.verbose:
			print(f'{self.name}: FindNeighbors()')

		if (self.neighborFinder is None) or (not self.requery):
			if self.KDTree:
				self.neighborFinder = KDTreeNeighborFinder(
					self.Embedding[self.trainIndices, :]
				)
			else:
				self.neighborFinder = PairwiseDistanceNeighborFinder(
					self.Embedding[self.trainIndices, :],
					exclusion = self._BuildExclusionMask()
				)
		
		# Query with actual knn (no inflation needed)
		if requery or self.requery:
			knn_distances, knn_neighbors = self.neighborFinder.requery()
		else:
			knn_distances, knn_neighbors = self.neighborFinder.query(
				self.Embedding[self.testIndices, :],
				self.knn_ if self.KDTree else self.knn,  # Use actual knn when using pdist with baked-in exclusions
				workers=self.knnThreads
			)
		
		# Only need index mapping now (no filtering)
		if self.KDTree:
			# KDTree still needs full MapKNNIndicesToData
			self.knn_distances, self.knn_neighbors = self.MapKNNIndicesToData(
				knn_neighbors, knn_distances
			)
		else:
			# Matrix-based only needs index mapping
			self.knn_neighbors = self._MapKNNIndicesToLibraryIndices(knn_neighbors)
			self.knn_distances = knn_distances


	def MapKNNIndicesToData(self, raw_neighbors, raw_distances):
		"""
		Processes raw KDTree results through index mapping and neighbor filtering.
		
		Args:
			raw_neighbors: KDTree query neighbor indices
			raw_distances: KDTree query distances
			
		Returns:
			tuple: (processed_neighbors, processed_distances)
		"""
		# Handle edge case: knn=1 without libOverlap
		if self.knn == 1 and not self.libOverlap:
			raw_distances = raw_distances[:, None]
			raw_neighbors = raw_neighbors[:, None]
		
		# Step 1: Map KDTree indices to actual data indices
		knn_neighbors = self._MapKNNIndicesToLibraryIndices(raw_neighbors)
		knn_distances = raw_distances
		
		# Step 2: Remove degenerate neighbors (leave-one-out)
		knn_neighbors, knn_distances = self._RemoveDegenerateNeighbors(
			knn_neighbors, knn_distances
		)
		
		# Step 3: Apply exclusion radius filtering
		knn_neighbors, knn_distances = self._ApplyExclusionRadius(
			knn_neighbors, knn_distances
		)
		
		return knn_distances, knn_neighbors

	def _MapKNNIndicesToLibraryIndices(self, raw_neighbors):
		"""
		Maps KDTree indices (0-based) to actual library data row indices.
		
		Args:
			raw_neighbors: KDTree query results with indices relative to library data
			
		Returns:
			ndarray: Neighbor indices mapped to actual data row numbers
		"""
		# Handle contiguous library case
		if not self.disjointLib and \
				self.trainIndices[-1] - self.trainIndices[0] + 1 == len(self.trainIndices):
			return raw_neighbors + self.trainIndices[0]
		
		# Handle disjoint library or CCM subset
		knn_lib_map = {i: self.trainIndices[i] for i in range(len(self.trainIndices))}
		
		def knnMapFunc(knn):
			"""Maps KDTree indices to library indices"""
			return array([knn_lib_map[idx] for idx in knn], dtype=int)
		
		knn_neighbors_ = zeros(raw_neighbors.shape, dtype=int)
		for j in range(raw_neighbors.shape[1]):
			knn_neighbors_[:, j] = knnMapFunc(raw_neighbors[:, j])
		
		return knn_neighbors_
	
	
	def _RemoveDegenerateNeighbors(self, knn_neighbors, knn_distances):
		"""
		Removes self-matching neighbors when library overlaps with prediction set.
		Implements leave-one-out validation by shifting neighbors when first nn is degenerate.
		
		Args:
			knn_neighbors: Neighbor indices
			knn_distances: Neighbor distances
			
		Returns:
			tuple: (filtered_neighbors, filtered_distances)
		"""
		if not self.libOverlap:
			return knn_neighbors, knn_distances
		
		# Identify degenerate rows where pred_i == first neighbor
		knn_neighbors_0 = knn_neighbors[:, 0]
		i_overlap = [i == j for i, j in zip(self.testIndices, knn_neighbors_0)]
		
		# Shift columns: move col[1:knn_] into col[0:(knn_-1)]
		J = knn_distances.shape[1]
		knn_distances[i_overlap, 0:(J - 1)] = knn_distances[i_overlap, 1:J]
		knn_neighbors[i_overlap, 0:(J - 1)] = knn_neighbors[i_overlap, 1:J]
		
		# Remove extra column if not doing exclusion radius filtering
		if not self.CheckExclusion:
			knn_distances = delete(knn_distances, self.knn, axis=1)
			knn_neighbors = delete(knn_neighbors, self.knn, axis=1)
		
		return knn_neighbors, knn_distances
	
	
	def _ApplyExclusionRadius(self, knn_neighbors, knn_distances):
		"""
		Filters neighbors within temporal exclusion radius.
		For each prediction row, selects k neighbors outside exclusionRadius.
		
		Args:
			knn_neighbors: Neighbor indices
			knn_distances: Neighbor distances
			
		Returns:
			tuple: (filtered_neighbors, filtered_distances)
		"""
		if not self.CheckExclusion:
			return knn_neighbors, knn_distances
		
		def _SelectValidNeighbors(knnRow, knnDist, excludeRow):
			"""Select knn neighbors not in excludeRow"""
			valid_neighbors = full(self.knn, -1E6, dtype=int)
			valid_distances = full(self.knn, -1E6, dtype=float)
			
			k = 0
			for r in range(len(knnRow)):
				if knnRow[r] not in excludeRow:
					valid_neighbors[k] = knnRow[r]
					valid_distances[k] = knnDist[r]
					k += 1
					if k == self.knn:
						break
			
			# Warn if couldn't find enough valid neighbors
			if -1E6 in valid_neighbors:
				valid_neighbors = knnRow[:self.knn]
				valid_distances = knnDist[:self.knn]
				warn(f'{self.name}: Failed to find {self.knn} neighbors outside '
					 f'exclusionRadius {self.exclusionRadius}. Consider reducing knn.')
			
			return valid_neighbors, valid_distances
		
		# Apply exclusion for each prediction row
		for i in range(len(self.testIndices)):
			pred_i = self.testIndices[i]
			rowLow = max(self.trainIndices.min(), pred_i - self.exclusionRadius)
			rowHi = min(self.trainIndices.max(), pred_i + self.exclusionRadius)
			excludeRow = list(range(rowLow, rowHi + 1))
			
			knn_neighbors[i, :self.knn], knn_distances[i, :self.knn] = \
				_SelectValidNeighbors(knn_neighbors[i, :], knn_distances[i, :], excludeRow)
		
		# Remove extra knn_ columns
		extra_cols = list(range(self.knn, knn_distances.shape[1]))
		knn_distances = delete(knn_distances, extra_cols, axis=1)
		knn_neighbors = delete(knn_neighbors, extra_cols, axis=1)
		
		return knn_neighbors, knn_distances

	def CheckValidTrainSamples(self):
		if len(self.validLib):
			# Convert self.validLib boolean vector to data indices
			data_i = array([i for i in range(self.Data.shape[0])],
						   dtype = int)
			validLib_i = data_i[self.validLib]

			# Filter lib_i to only include valid library points
			lib_i_valid = array([i for i in self.trainIndices if i in validLib_i],
								dtype = int)

			if len(lib_i_valid) == 0:
				msg = f'{self.name}: FindNeighbors() : ' + \
					  'No valid library points found. ' + \
					  'All library points excluded by validLib.'
				raise ValueError(msg)

			if len(lib_i_valid) < self.knn:
				msg = f'{self.name}: FindNeighbors() : Only {len(lib_i_valid)} ' + \
					  f'valid library points found, but knn={self.knn}. ' + \
					  'Reduce knn or check validLib.'
				warn(msg)

			# Replace lib_ with lib_i_valid
			self.trainIndices = lib_i_valid

	@property
	def CheckExclusion(self):
		if self._exclusion is None:
			self._exclusion = False
			if self.exclusionRadius > 0:
				if self.libOverlap:
					self._exclusion = True
				else:
					# If no libOverlap and exclusionRadius is less than the
					# distance in rows between train : test, no library neighbor
					# exclusion needed.
					# Find row span between train & test
					excludeRow = 0
					if self.testIndices[0] > self.trainIndices[-1]:
						# test start is beyond train end
						excludeRow = self.testIndices[0] - self.trainIndices[-1]
					elif self.trainIndices[0] > self.testIndices[-1]:
						# train start row is beyond test end
						excludeRow = self.trainIndices[0] - self.testIndices[-1]
					if self.exclusionRadius >= excludeRow:
						self._exclusion = True
		return self._exclusion

	# --------------------------------------------------------------------
	# EDM Methods
	# -------------------------------------------------------------------
	def FormatProjection(self):
		# -------------------------------------------------------------------
		"""
		Create Projection, Coefficients, SingularValues DataFrames
		AddTime() attempts to extend forecast time if needed

		NOTE: self.pred_i had all nan removed for KDTree by RemoveNan().
		self.predList only had leading/trailing embedding nan removed.
		Here we want to include any nan observation rows so we
		process predList & pred_i_all, not self.pred_i.
		"""
		if self.verbose:
			print(f'{self.name}: FormatProjection()')

		if len(self.testIndices) == 0:
			msg = f'{self.name}: FormatProjection() No valid prediction indices.'
			warn(msg)

			# Return empty numpy array with shape (0, 4)
			# Columns: Time, Observations, Predictions, Pred_Variance
			self.Projection = empty((0, 4))
			return

		N_dim = self.embedDimensions + 1
		Tp_magnitude = abs(self.predictionHorizon)

		# ----------------------------------------------------
		# Observations: Insert target data in observations
		# ----------------------------------------------------
		outSize = 0

		# Create array of indices into self.targetVec for observations
		obs_i = array([], dtype = int)

		# Process each test segment in self.predList
		for pred_i in self.predList:
			N_pred = len(pred_i)
			outSize_i = N_pred + Tp_magnitude
			outSize = outSize + outSize_i
			append_i = array([], dtype = int)

			if N_pred == 0:
				# No prediction made for this test segment
				if self.verbose:
					msg = f'{self.name} FormatProjection(): No prediction made ' + \
					      f'for empty test in {self.predList}. ' + \
					      'Examine test, E, step, predictionHorizon parameters and/or nan.'
					print(msg)
				continue

			if self.predictionHorizon == 0:  # predictionHorizon = 0
				append_i = pred_i.copy()

			elif self.predictionHorizon > 0:  # Positive predictionHorizon
				if pred_i[-1] + self.predictionHorizon < self.targetVec.shape[0]:
					# targetVec data available before end of targetVec
					append_i = append(append_i, pred_i)
					Tp_i = [i for i in range(append_i[-1] + 1,
					                         append_i[-1] + self.predictionHorizon + 1)]
					append_i = append(append_i, array(Tp_i, dtype = int))
				else:
					# targetVec data not available at prediction end
					append_i = append(append_i, pred_i)

			else:  # Negative predictionHorizon
				if pred_i[0] + self.predictionHorizon > -1:
					# targetVec data available after begin of pred_i[0]
					append_i = append(append_i, pred_i)
					Tp_i = [i for i in range(pred_i[0] + self.predictionHorizon,
					                         pred_i[0])]
					append_i = append(array(Tp_i, dtype = int), append_i)
				else:
					# targetVec data not available before pred_i[0]
					append_i = append(append_i, pred_i)

			obs_i = append(obs_i, append_i)

		observations = self.targetVec[obs_i, 0]

		# ----------------------------------------------------
		# Projections & variance
		# ----------------------------------------------------
		# Define array's of indices predOut_i, obsOut_i for DataFrame vectors
		predOut_i = array([], dtype = int)
		predOut_0 = 0

		# Process each test segment in self.predList for predOut_i
		for pred_i in self.predList:
			N_pred = len(pred_i)
			outSize_i = N_pred + Tp_magnitude

			if N_pred == 0:
				# No prediction made for this test segment
				continue

			if self.predictionHorizon == 0:
				Tp_i = [i for i in range(predOut_0, predOut_0 + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + 1

			elif self.predictionHorizon > 0:  # Positive predictionHorizon
				Tp_i = [i for i in range(predOut_0 + self.predictionHorizon,
				                         predOut_0 + self.predictionHorizon + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + 1

			else:  # Negative predictionHorizon
				Tp_i = [i for i in range(predOut_0, predOut_0 + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + Tp_magnitude + 1

		# If nan are present, the foregoing can be wrong since it is not
		# known before prediction what train vectors will produce test
		# If len( pred_i ) != len( predOut_i ), nan resulted in missing test
		# Create a map between pred_i_all : predOut_i to create a new/shorter
		# predOut_i mapping pred_i to the output vector predOut_i
		if len(self.testIndices) != len(predOut_i):
			# Map the last predOut_i values since embed shift near data begining
			# can have (E-1)*step na, but still listed in pred_i_all
			N_ = len(predOut_i)

			if self.embedStep < 0:
				D = dict(zip(self.pred_i_all[-N_:], predOut_i))
			else:
				D = dict(zip(self.pred_i_all[:N_], predOut_i))

			# Reset predOut_i
			predOut_i = [D[i] for i in self.testIndices]

		# Create obsOut_i indices for output vectors in DataFrame
		if self.predictionHorizon > 0:  # Positive predictionHorizon
			if obs_i[-1] + self.predictionHorizon > self.Data.shape[0] - 1:
				# Edge case of end of data with positive predictionHorizon
				obsOut_i = [i for i in range(len(obs_i))]
			else:
				obsOut_i = [i for i in range(outSize)]

		elif self.predictionHorizon < 1:  # Negative or Zero predictionHorizon
			if self.testIndices[0] + self.predictionHorizon < 0:
				# Edge case of start of data with negative predictionHorizon
				obsOut_i = [i for i in range(len(obs_i))]

				# Shift obsOut_i values based on leading nan
				shift = Tp_magnitude - self.testIndices[0]
				obsOut_i = obsOut_i + shift
			else:
				obsOut_i = [i for i in range(len(obs_i))]

		obsOut_i = array(obsOut_i, dtype = int)

		# ndarray init to nan
		observationOut = full(outSize, nan)
		projectionOut = full(outSize, nan)
		varianceOut = full(outSize, nan)

		# fill *Out with observed & projected values
		observationOut[obsOut_i] = observations
		projectionOut[predOut_i] = self.projection
		varianceOut[predOut_i] = self.variance

		# ----------------------------------------------------
		# Time
		# ----------------------------------------------------
		self.ConvertTime()

		if self.predictionHorizon == 0 or \
				(self.predictionHorizon > 0 and (self.pred_i_all[-1] + self.predictionHorizon) < len(self.time)) or \
				(self.predictionHorizon < 0 and (self.testIndices[0] + self.predictionHorizon >= 0)):
			# All times present in self.time, copy them
			timeOut = empty(outSize, dtype = self.time.dtype)
			timeOut[obsOut_i] = self.time[obs_i]
		else:
			# Need to pre/append additional times
			timeOut = self.AddTime(Tp_magnitude, outSize, obs_i, obsOut_i)

		# ----------------------------------------------------
		# Output numpy array
		# Shape: (n_samples, 4)
		# Column 0: Time
		# Column 1: Observations
		# Column 2: Predictions
		# Column 3: Pred_Variance
		# ----------------------------------------------------
		self.Projection = column_stack([timeOut, observationOut, projectionOut, varianceOut])

		# ----------------------------------------------------
		# SMap coefficients and singular values
		# ----------------------------------------------------
		if self.name == 'SMap':
			# ndarray init to nan
			coefOut = full((outSize, N_dim), nan)
			SVOut = full((outSize, N_dim), nan)
			# fill coefOut, SVOut with projected values
			coefOut[predOut_i, :] = self.coefficients
			SVOut[predOut_i, :] = self.singularValues

			# Prepend time column to coefficients and singular values
			# Coefficients shape: (n_samples, N_dim + 1)
			# Column 0: Time, Columns 1-N_dim: coefficients
			self.Coefficients = column_stack([timeOut, coefOut])

			# SingularValues shape: (n_samples, N_dim + 1)
			# Column 0: Time, Columns 1-N_dim: singular values
			self.SingularValues = column_stack([timeOut, SVOut])

	# -------------------------------------------------------------------
	def ConvertTime(self):
		# -------------------------------------------------------------------
		"""
		Replace self.time with ndarray numerically operable values
		ISO 8601 formats are supported in the time & datetime modules
		"""
		if self.verbose:
			print(f'{self.name}: ConvertTime()')

		time0 = self.time[0]

		# If times are numerically operable, nothing to do.
		if isinstance(time0, int) or isinstance(time0, float) or \
				isinstance(time0, integer) or isinstance(time0, floating) or \
				isinstance(time0, dt.time) or isinstance(time0, dt.datetime):
			return

		# Local copy of time
		time_ = self.time.copy()  # ndarray

		# If times are strings, try to parse into time or datetime
		# If success, replace time with parsed time or datetime array
		if isinstance(time0, str):
			try:
				t0 = dt.time.fromisoformat(time0)
				# Parsed t0 into dt.time OK. Parse the whole vector
				time_ = array([dt.time.fromisoformat(t) for t in time_])
			except ValueError:
				try:
					t0 = dt.datetime.fromisoformat(time0)
					# Parsed t0 into dt.datetime OK. Parse the whole vector
					time_ = array([dt.datetime.fromisoformat(t) for t in time_])
				except ValueError:
					msg = f'{self.name} ConvertTime(): Time values are strings ' + \
					      'but are not ISO 8601 recognized time or datetime.'
					raise RuntimeError(msg)

		# If times were string they have been converted to time or datetime
		# Ensure times can be numerically manipulated, compute deltaT
		try:
			deltaT = time_[1] - time_[0]
		except TypeError:
			msg = f'{self.name} ConvertTime(): Time values not recognized.' + \
			      ' Accepted values are int, float, or string of ISO 8601' + \
			      ' compliant time or datetime.'
			raise RuntimeError(msg)

		# Replace DataFrame derived time with converted time_
		self.time = time_

	# -------------------------------------------------------------------
	def AddTime(self, Tp_magnitude, outSize, obs_i, obsOut_i):
		# -------------------------------------------------------------------
		"""
		Prepend or append time values to self.time if needed
		Return timeOut vector with additional predictionHorizon points
		"""
		if self.verbose:
			print(f'{self.name}: AddTime()')

		min_pred_i = self.testIndices[0]
		max_pred_i_all = self.pred_i_all[-1]
		deltaT = self.time[1] - self.time[0]

		# First, fill timeOut with times in time
		# timeOut should not be int (np.integer) since they cannot be nan
		time0 = self.time[0]
		if isinstance(time0, int) or isinstance(time0, integer):
			time_dtype = float
		else:
			time_dtype = type(time0)

		timeOut = full(outSize, nan, dtype = time_dtype)

		timeOut[obsOut_i] = self.time[obs_i]

		newTimes = full(Tp_magnitude, nan, dtype = time_dtype)

		if self.predictionHorizon > 0:
			# predictionHorizon introduces time values beyond the range of time
			# Generate future times
			lastTime = self.time[max_pred_i_all]
			newTimes[0] = lastTime + deltaT

			for i in range(1, self.predictionHorizon):
				newTimes[i] = newTimes[i - 1] + deltaT

			timeOut[-self.predictionHorizon:] = newTimes

		else:
			# predictionHorizon introduces time values before the range of time
			# Generate past times
			newTimes[0] = self.time[min_pred_i] - deltaT
			for i in range(1, Tp_magnitude):
				newTimes[i] = newTimes[i - 1] - deltaT

			newTimes = newTimes[::-1]  # Reverse

			# Shift timeOut values based on leading nan
			shift = Tp_magnitude - self.testIndices[0]
			timeOut[: Tp_magnitude] = newTimes

		return timeOut

	def EmbedData(self):
		"""
		Embed data : If not embedded call API.Embed()
		"""

		if not self.isEmbedded:
			self.Embedding = Embed(data = self.Data, embeddingDimensions = self.embedDimensions,
			                       stepSize = self.embedStep, columns = self.columns)
		else:
			self.Embedding = self.Data[:, self.columns]  # Already an embedding

	# TODO: change this to properly inherit and override
	def RemoveNan(self):
		"""
		KDTree in Neighbors does not accept nan
		If ignoreNan remove Embedding rows with nan from lib_i, pred_i
		"""

		if self.ignoreNan:
			# Check for nan in all Embedding columns (axis = 1) of lib_i...
			na_lib = numpy.isnan(self.Embedding[self.trainIndices, :]).any(axis = 1)
			na_pred = numpy.isnan(self.Embedding[self.testIndices, :]).any(axis = 1)

			if na_lib.any():
				## todo: factor this out into overrides
				if self.name == 'SMap':
					original_knn = self.knn
					original_lib_i_len = len(self.trainIndices)

				# Redefine lib_i excluding nan
				self.trainIndices = self.trainIndices[~na_lib]

				# lib_i resized, update SMap self.knn if not user provided
				if self.name == 'SMap':
					if original_knn == original_lib_i_len - 1:
						self.knn = len(self.trainIndices) - 1

			# Redefine pred_i excluding nan
			if any(na_pred):
				self.testIndices = self.testIndices[~na_pred]

			# If targetVec has nan, set flag for SMap internals
			if self.name == 'SMap':
				if any(isnan(self.targetVec)):
					self.targetVecNan = True

		self.PredictionValid()

	# TODO: properly override these in inheritance
	def CreateIndices(self):
		"""
		Populate array index vectors lib_i, pred_i
		Indices specified in list of pairs [ 1,10, 31,40... ]
		where each pair is start:stop span of data rows.
		"""

		# Convert self.train from flat list to list of (start, stop) pairs
		if len(self.train) % 2:
			# Odd number of train elements
			msg = f'{self.name}: CreateIndices() train must be an even ' + \
			      'number of elements. train start : stop pairs'
			raise RuntimeError(msg)

		libPairs = []  # List of 2-tuples of train indices
		for i in range(0, len(self.train), 2):
			libPairs.append((self.train[i], self.train[i+1]))

		# Validate end > start
		for libPair in libPairs:
			libStart, libEnd = libPair

			if self.name in ['Simplex', 'SMap', 'Multiview']:
				# Don't check if CCM since default of "1 1" is used.
				assert libStart < libEnd

			# Disallow indices < 1, the user may have specified 0 start
			# but why??
			assert libStart >= 0 and libEnd >= 0
			# for now, to prevent -1 indexing
			if libStart > 1:
				libStart = 1

		# Loop over each train pair
		# Add rows for library segments, disallowing vectors
		# in disjoint library gap accommodating embedding and predictionHorizon
		embedShift = abs(self.embedStep) * (self.embedDimensions - 1)
		lib_i_list = list()

		for r in range(len(libPairs)):
			start, stop = libPairs[r]

			# Adjust start, stop to enforce disjoint library gaps
			if not self.isEmbedded:
				if self.embedStep < 0:
					start = start + embedShift
				else:
					stop = stop - embedShift

			if self.predictionHorizon < 0:
				if not self.isEmbedded:
					start = max(start, start + abs(self.predictionHorizon) - 1)
			else:
				if (r == len(libPairs) - 1):
					stop = stop - self.predictionHorizon

			libPair_i = [i - 1 for i in range(start, stop + 1)]

			lib_i_list.append(array(libPair_i, dtype = int))

		# Concatenate lib_i_list into lib_i
		self.trainIndices = concatenate(lib_i_list)

		if len(lib_i_list) > 1: self.disjointLib = True

		# ------------------------------------------------
		# Validate lib_i: E, step, predictionHorizon combination
		# ------------------------------------------------
		if self.name in ['Simplex', 'SMap', 'CCM', 'Multiview']:
			if self.isEmbedded:
				assert len(self.trainIndices) >= abs(self.predictionHorizon)
			else:
				vectorStart = max([-embedShift, 0, self.predictionHorizon])
				vectorEnd = min([-embedShift, 0, self.predictionHorizon])
				vectorLength = abs(vectorStart - vectorEnd) + 1

				assert vectorLength <= len(self.trainIndices)

		# ------------------------------------------------
		# pred_i from test
		# ------------------------------------------------
		# Convert self.test from flat list to list of (start, stop) pairs
		if len(self.test) % 2:
			# Odd number of test elements
			msg = f'{self.name}: CreateIndices() test must be an even ' + \
			      'number of elements. test start : stop pairs'
			raise RuntimeError(msg)

		predPairs = []  # List of 2-tuples of test indices
		for i in range(0, len(self.test), 2):
			predPairs.append((self.test[i], self.test[i+1]))

		if len(predPairs) > 1: self.disjointPred = True

		# Validate end > start
		for predPair in predPairs:
			predStart, predEnd = predPair

			if self.name in ['Simplex', 'SMap', 'Multiview']:
				# Don't check CCM since default of "1 1" is used.
				if predStart >= predEnd:
					msg = f'{self.name}: CreateIndices() test start ' + \
					      f' {predStart} exceeds test end {predEnd}.'
					raise RuntimeError(msg)

			# Disallow indices < 1, the user may have specified 0 start
			if predStart < 1 or predEnd < 1:
				msg = f'{self.name}: CreateIndices() test indices ' + \
				      ' less than 1 not allowed.'
				raise RuntimeError(msg)

		# Create pred_i indices from predPairs
		for r in range(len(predPairs)):
			start, stop = predPairs[r]
			pred_i = zeros(stop - start + 1, dtype = int)

			i = 0
			for j in range(start, stop + 1):
				pred_i[i] = j - 1  # apply zero-offset
				i = i + 1

			self.predList.append(pred_i)  # Append disjoint segment(s)

		# flatten arrays in self.predList for single array self.pred_i
		pred_i_ = []
		for pred_i in self.predList:
			i_ = [i for i in pred_i]
			pred_i_ = pred_i_ + i_

		self.testIndices = array(pred_i_, dtype = int)

		self.PredictionValid()

		self.pred_i_all = self.testIndices.copy()  # Before nan are removed

		# Remove embedShift nan from predPairs
		# NOTE : This does NOT redefine self.pred_i, only self.predPairs
		#        self.pred_i is redefined to remove all nan in RemoveNan()
		#        at the API level.
		if not self.isEmbedded:
			# If [0, 1, ... embedShift] nan (negative step) or
			# [N - embedShift, ... N-1, N]  (positive step) nan
			# are in pred_i delete elements

			# TODO: this is like, 4 levels of index shadowing - needs to be fixed
			nan_i_start = [i for i in range(embedShift)]
			nan_i_end = [self.Data.shape[0] - 1 - i for i in range(embedShift)]

			for i in range(len(self.predList)):
				pred_i = self.predList[i]

				if self.embedStep > 0:
					if any([i in nan_i_end for i in pred_i]):
						pred_i_ = [i for i in pred_i if i not in nan_i_end]
						self.predList[i] = array(pred_i_, dtype = int)
				else:
					if any([i in nan_i_start for i in pred_i]):
						pred_i_ = [i for i in pred_i if i not in nan_i_start]
						self.predList[i] = array(pred_i_, dtype = int)

		# ------------------------------------------------
		# Validate lib_i pred_i do not exceed data
		# ------------------------------------------------
		assert self.trainIndices[-1] < self.Data.shape[0]
		assert self.testIndices[-1] < self.Data.shape[0]

		# ---------------------------------------------------
		# Check for train : test overlap for knn leave-one-out
		# ---------------------------------------------------
		if len(set(self.trainIndices).intersection(set(self.testIndices))):
			self.libOverlap = True

		if self.name == 'SMap':
			if self.knn < 1:  # default knn = 0, set knn value to full train
				self.knn = len(self.trainIndices) - 1

				if self.verbose:
					msg = f'{self.name} CreateIndices(): ' + \
					      f'Set knn = {self.knn} for SMap.'
					print(msg, flush = True)

		self.CheckValidTrainSamples()

	# --------------------------------------------------------------------
	def PredictionValid(self):
		# --------------------------------------------------------------------
		"""
		Validate there are pred_i to make a prediction
		"""

		if len(self.testIndices) == 0:
			raise ValueError("No valid predictions")

	# --------------------------------------------------------------------
	def Validate(self):
		# --------------------------------------------------------------------
		"""
		Validate inputs and parameters for EDM analysis

		:param self: EDM object
		:raises RuntimeError: if data is invalid
		"""
		if self.verbose:
			print(f'{self.name}: Validate()')

		if self.Data is None:
			raise RuntimeError(f'Validate() {self.name}: data array required.')
		else:
			if not isinstance(self.Data, numpy.ndarray):
				raise RuntimeError(f'Validate() {self.name}: data must be a numpy array.')
			if self.Data.ndim != 2:
				raise RuntimeError(f'Validate() {self.name}: data must be a 2D array.')

		# Convert columns to list of integers if needed
		if self.columns is None or not len(self.columns):
			# Default: use all columns except time (column 0)
			self.columns = list(range(1, self.Data.shape[1]))
		elif isinstance(self.columns, int):
			self.columns = [self.columns]
		elif not IsNonStringIterable(self.columns):
			raise RuntimeError(f'Validate() {self.name}: columns must be a list of integers or an integer.')

		# Validate all columns are integers within bounds
		for column in self.columns:
			if not isinstance(column, (int, numpy.integer)):
				raise RuntimeError(f'Validate() {self.name}: column indices must be integers, got {type(column)}.')
			if column < 0 or column >= self.Data.shape[1]:
				raise RuntimeError(f'Validate() {self.name}: column index {column} out of bounds for data shape {self.Data.shape}.')

		# Convert target to list of integers if needed
		if self.target is None:
			# Default: use first data column (column 1, since 0 is time)
			self.target = [self.columns[0]] if self.columns else [1]
		elif isinstance(self.target, int):
			self.target = [self.target]
		elif not IsNonStringIterable(self.target):
			raise RuntimeError(f'Validate() {self.name}: target must be a list of integers or an integer.')

		# Validate all targets are integers within bounds
		for target in self.target:
			if not isinstance(target, (int, numpy.integer)):
				raise RuntimeError(f'Validate() {self.name}: target indices must be integers, got {type(target)}.')
			if target < 0 or target >= self.Data.shape[1]:
				raise RuntimeError(f'Validate() {self.name}: target index {target} out of bounds for data shape {self.Data.shape}.')

		if not self.isEmbedded:
			if self.embedStep == 0:
				raise RuntimeError(f'Validate() {self.name}:' + \
				                   ' step must be non-zero.')
			if self.embedDimensions < 1:
				raise RuntimeError(f'Validate() {self.name}:' + \
				                   f' E = {self.embedDimensions} is invalid.')

		if self.name != 'CCM':
			if not len(self.train):
				raise RuntimeError(f'Validate() {self.name}: train required.')
			if not IsNonStringIterable(self.train):
				self.train = [int(i) for i in self.train.split()]

			if not len(self.test):
				raise RuntimeError(f'Validate() {self.name}: test required.')
			if not IsNonStringIterable(self.test):
				self.test = [int(i) for i in self.test.split()]

		# Set knn default based on E and train size, E embedded on num columns
		if self.name in ['Simplex', 'CCM', 'Multiview']:
			# embedded = true: Set E to number of columns
			if self.isEmbedded:
				self.embedDimensions = len(self.columns)

			# knn not specified : knn set to E+1
			if self.knn < 1:
				self.knn = self.embedDimensions + 1

				if self.verbose:
					msg = f'{self.name} Validate(): Set knn = {self.knn}'
					print(msg, flush = True)

		if self.name == 'SMap':
			# embedded = true: Set E to number of columns
			if self.isEmbedded and len(self.columns):
				self.embedDimensions = len(self.columns)

			if not self.isEmbedded and len(self.columns) > 1:
				msg = f'{self.name} Validate(): Multivariable S-Map ' + \
				      'must use embedded = True to ensure data/dimension ' + \
				      'correspondance.'
				raise RuntimeError(msg)

		if self.generateSteps > 0:
			# univariate only, embedded must be False
			if self.name in ['Simplex', 'SMap']:

				if self.isEmbedded:
					msg = f'{self.name} Validate(): generateSteps > 0 ' + \
					      'must use univariate embedded = False.'
					raise RuntimeError(msg)

				if self.target[0] != self.columns[0]:
					msg = f'{self.name} Validate(): generateSteps > 0 ' + \
					      f'must use univariate target ({self.target[0]}) ' + \
					      f' == columns ({self.columns[0]}).'
					raise RuntimeError(msg)

				# If times are datetime, AddTime() fails
				# EDM.time is ndarray storing python datetime
				# In AddTime() datetime, timedelta operations are not compatible
				# with numpy datetime64, timedelta64 : deltaT fails in conversion
				# If times are datetime: raise exception
				if not self.noTime:
					try:
						time0 = self.Data[0, 0]  # self.time not yet, column 0 is time
						dt0 = datetime.fromisoformat(time0)
					except:
						# dt0 is not a datetime assign for finally to pass
						dt0 = None
					finally:
						# if dt0 is datetime, raise exception for noTime = True
						if isinstance(dt0, datetime):
							msg = f'{self.name} Validate(): generateSteps ' + \
							      'with datetime needs to use noTime = True.'
							raise RuntimeError(msg)
