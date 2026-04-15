from abc import ABC as AbstractBaseClass
import numpy
from typing import Tuple, Union, List, Optional

from scipy.spatial import KDTree, distance


class NeighborFinderBase(AbstractBaseClass):
	"""
	Interface for describing a class used by EDM to find nearest neighbors
	"""
	def __init__(self, data: numpy.ndarray):
		"""
		Constructor
		:param data: data in the shape of [samples, dimensions]
		"""
		self.data = data

	def query(self, x: numpy.ndarray, k: int = 1) -> Tuple[Union[float, numpy.ndarray], Union[int, numpy.ndarray]]:
		"""
		Get nearest neighbors for k
		:param x:	data to query
		:param k: 	number of nearest neighbors to get
		:return: distance to each nearest neighbor and also the index in the data for each nearest neighbor
		"""
		raise NotImplementedError


class KDTreeNeighborFinder(NeighborFinderBase):
	"""
	Scipy KDTree neighbor finder. Used as in original EDM Implementation

	Note: If dimensionality is k, the number of points n in
	the data should be n >> 2^k, otherwise KDTree efficiency is low.
	k:2^k pairs { 4 : 16, 5 : 32, 7 : 128, 8 : 256, 10 : 1024 }
	"""

	def __init__(self, data: numpy.ndarray,
				 leafsize = 20, compact_nodes = True, copy_data = False, balanced_tree = True, boxsize = None):
		super().__init__(data)
		self.tree = KDTree(self.data, leafsize = leafsize, compact_nodes = compact_nodes, copy_data = copy_data,
						   balanced_tree = balanced_tree, boxsize = boxsize)

	def query(self, x: numpy.ndarray, k: int = 1, eps: float = 0.0,
			  distance_upper_bound = numpy.inf, p: float = 2.0, workers: int = 1) -> Tuple[Union[float, numpy.ndarray], Union[int, numpy.ndarray]]:
		return self.tree.query(x, k, eps, p, distance_upper_bound, workers)


class PairwiseDistanceNeighborFinder(NeighborFinderBase):
	"""
	Neighbor finder that uses pairwise euclidean distances and can be updated with new dimensions and re-queried
	"""
	@staticmethod
	def find_neighbors(distances, k):
		neighbors = numpy.argpartition(distances, k, axis = 0)[:k, :]
		indices = numpy.arange(distances.shape[1])[numpy.newaxis, :]
		neighbor_distances = distances[neighbors, indices]

		# Sort the top-k neighbors by distance, breaking ties by index
		sort_order = numpy.lexsort((neighbors, neighbor_distances), axis = 0)
		neighbors = numpy.take_along_axis(neighbors, sort_order, axis = 0)
		neighbor_distances = numpy.take_along_axis(neighbor_distances, sort_order, axis = 0)

		return numpy.sqrt(neighbor_distances).transpose().squeeze(), neighbors.transpose().squeeze()

	def __init__(self, data: numpy.ndarray, x: Optional[numpy.ndarray] = None, exclusion: Optional[numpy.ndarray] = None):
		"""

		:param data: data
		:param x: 	data that we might want to query
		"""
		super().__init__(data)
		self.distanceMatrix = None
		self.mask = exclusion
		if x is not None:
			self.distanceMatrix = distance.cdist(data, x, 'sqeuclidean')
			if self.mask is not None:
				self.distanceMatrix[self.mask] = numpy.inf
		self.numNeighbors = None

	def requery(self):
		return self.query(None, self.numNeighbors)

	def update(self, additional_distance: numpy.ndarray) -> 'PairwiseDistanceNeighborFinder':
		"""
		Update the distance matrix and return a new neighborfinder
		:param additional_distance:
		:return:
		"""
		out = PairwiseDistanceNeighborFinder(None, None)
		out.distanceMatrix = self.distanceMatrix + additional_distance
		out.numNeighbors = self.numNeighbors
		return out

	def query(self, x: numpy.ndarray = None, k: int = 1, workers = 1) -> Tuple[Union[float, numpy.ndarray], Union[int, numpy.ndarray]]:
		self.numNeighbors = k
		if x is not None:
			self.distanceMatrix = distance.cdist(self.data, x, 'sqeuclidean')
			if self.mask is not None:
				self.distanceMatrix[self.mask] = numpy.inf
		return PairwiseDistanceNeighborFinder.find_neighbors(self.distanceMatrix, self.numNeighbors)