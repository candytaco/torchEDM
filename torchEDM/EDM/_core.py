## core functions for torchEDM
from typing import Optional, Union, Callable

import torch


def ElementwisePairwiseDistance(a, b, out):
	"""
	Pairwise square euclidean distances between elements of a and b
	along every dimension. Basically an outer subtract.
	:param a:	[n1 x dims] tensor 1
	:param b:	[n2 x dims] tensor 2
	:param out:	out tensor to write to [dims x n1 x n2]
	"""
	dims = a.shape[1]

	for v in range(dims):
		diff = a[:, v].unsqueeze(1) - b[:, v].unsqueeze(0)
		out[v, :, :] = diff
	out.square_()


def IncrementPairwiseDistance(distances, increments, out):
	"""
	For a set of pairwise distances, increment each slice by the same amount
	i.e. a 2D array broadcast
	:param distances: 	[dims, n1 x n2] set of pairwise distances
	:param increments: 	[n1 x n2] increments
	:param out: 		[dims, n1 x n2] tensor to write into
	:return:
	"""
	out[:, :, :] = distances + increments.unsqueeze(0)


def MinAxis1(arr):
	"""
	Compute minimum along axis 1 of 3D tensor
	:param arr: [k x neighbors x dims] tensor
	:return: [k x dims] minimum values
	"""
	return torch.min(arr, dim = 1)[0]


def SumAxis1(arr):
	"""
	Sum along axis 1 of 3D tensor
	:param arr: [k x neighbors x dims] tensor
	:return: [k x dims] sum values
	"""
	return torch.sum(arr, dim = 1)


def ComputeWeights(neighborDistances, minDistances):
	"""
	Compute exponential weights
	:param neighborDistances: [k x neighbors x dims] distances
	:param minDistances: [k x dims] minimum distances
	:return: [k x neighbors x dims] weights
	"""
	return torch.exp(-neighborDistances / minDistances.unsqueeze(1))


def ComputePredictions(weights, select, weightSum):
	"""
	Compute weighted average predictions
	:param weights: [k x neighbors x dims] weights
	:param select: [k x neighbors x dims] selected values
	:param weightSum: [k x dims] sum of weights
	:return: [k x dims] predictions
	"""
	return (weights * select).sum(dim = 1) / weightSum


def _promoteDimensions(score_function: Callable[[torch.tensor, torch.tensor, Optional[torch.tensor]], torch.tensor]):
	"""
	Decorator that reshape score functions inputs to handle multiple prediction targets
	:param score_function:
	:return:
	"""
	def wrapper(target, predictions, out = None):
		target = torch.as_tensor(target)
		predictions = torch.as_tensor(predictions)
		# Integer input would reach torch.mean, which rejects integer dtypes
		if not target.is_floating_point():
			target = target.to(torch.get_default_dtype())
		if not predictions.is_floating_point():
			predictions = predictions.to(torch.get_default_dtype())
		isSingleSeries = predictions.ndim == 1
		if target.ndim < 2:
			target = target[:, None]
		if isSingleSeries:
			# A plain prediction vector is one source and one target series.
			predictions = predictions[None, :, None]
		elif predictions.ndim < 3:
			predictions = predictions[:, :, None]
		if out is not None and out.ndim < 2:
			out = out.unsqueeze(-1)
		result = score_function(target, predictions, out)
		if isSingleSeries and isinstance(result, torch.Tensor):
			return result.reshape(())
		return result
	return wrapper


@_promoteDimensions
def Correlation(target: torch.tensor, predictions: torch.tensor, out: Optional[torch.tensor] = None):
	"""
	Correlation between target time series and batched predictions.
	:param target:		[n_time, n_targets] tensor of true values
	:param predictions:	[n_sources, n_time, n_targets] tensor of predicted values
	:param out:			[n_sources, n_targets] output tensor
	:return: out tensor with correlations
	"""
	if out is None:
		out = torch.zeros(predictions.shape[0], predictions.shape[2], device = target.device)

	targetCentered = target - torch.mean(target, dim = 0, keepdim = True)
	targetStd = torch.sqrt(torch.sum(targetCentered ** 2, dim = 0))

	predictionsCentered = predictions - torch.mean(predictions, dim = 1, keepdim = True)
	predictionsStd = torch.sqrt(torch.sum(predictionsCentered ** 2, dim = 1))

	out[:] = torch.sum(targetCentered * predictionsCentered, dim = 1) / (targetStd * predictionsStd)

	return out.squeeze()

@_promoteDimensions
def CorrelationInPlace(target: torch.tensor, predictions: torch.tensor, out: torch.tensor):
	"""
	Correlation between target and batched predictions. Centers predictions in-place to avoid
	allocating a separate centered copy. Uses .norm() for std to avoid materializing the squared
	tensor in global memory. Caller must not use predictions after this call.

	Expects fully 3D inputs — no dimension promotion. Peak memory is 2x [n_sources, n_time, n_targets]
	instead of 3x for the standard Correlation.

	:param target:		[n_time, n_targets]
	:param predictions:	[n_sources, n_time, n_targets] — modified in-place
	:param out:			[n_sources, n_targets] output tensor
	"""
	targetCentered = target - torch.mean(target, dim = 0, keepdim = True)
	targetStd = targetCentered.norm(dim = 0)

	predictions.sub_(torch.mean(predictions, dim = 1, keepdim = True))
	predictionsStd = predictions.norm(dim = 1)

	out[:] = torch.sum(targetCentered * predictions, dim = 1) / (targetStd * predictionsStd).clamp(min = 1e-8)


@_promoteDimensions
def R2(target: torch.tensor, predictions: torch.tensor, out: Optional[torch.tensor] = None):
	"""
	R2 (variance explained) between target time series and batched predictions.
	:param target:		[n_time, n_targets] tensor of true values
	:param predictions:	[n_sources, n_time, n_targets] tensor of predicted values
	:param out:			[n_sources, n_targets] output tensor
	:return: out tensor with R2 values
	"""
	if out is None:
		out = torch.zeros(predictions.shape[0], predictions.shape[2], device = target.device)

	targetMean = torch.mean(target, dim = 0)
	totalSumOfSquares = torch.sum((target - targetMean) ** 2, dim = 0)

	residualSumOfSquares = torch.sum((target - predictions) ** 2, dim = 1)
	out[:] = 1 - residualSumOfSquares / totalSumOfSquares

	return out.squeeze()

def batch_simplex_predict_and_score(distanceMatrices: torch.tensor, numNeighbors: Union[int, torch.tensor],
									train_y: torch.tensor, test_y: torch.tensor, score_function: Callable,
									predictions: Optional[torch.tensor] = None,
									perf_out: Optional[torch.tensor] = None,
									train_indices: Optional[torch.tensor] = None):
	"""
	Batched multiple predictions and score via simplex. Each distance matrix is used to make a separate prediction on Y.
	These predictions are then scored
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use
	:param test_y:				test_y to compare against
	:param train_y:				train_y to predict from
	:param score_function:		score function to evaluate performance
	:param predictions:			tensor write prediction into
	:param perf_out:			array to write the performance into
	:param train_indices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return:
	"""
	predictions = batch_simplex_predict(distanceMatrices, numNeighbors, train_y, predictions, train_indices)
	return score_function(test_y, predictions, perf_out)


def batch_simplex_predict(distanceMatrices: torch.tensor, numNeighbors: Union[int, torch.tensor],
						  train_y: torch.tensor, predictions: Optional[torch.tensor] = None,
						  train_indices: Optional[torch.tensor] = None) -> torch.tensor:
	"""
	Batched multiple predictions via simplex. Each distance matrix is used to make a separate prediction on Y.
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use, can be a single shared n or one per distance matrix
	:param train_y:				train_y to predict from
	:param predictions:			array to write the predictions into
	:param train_indices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return: predicted Y in <source, n_test, target>
	"""
	neighbor_indices, weights = batch_get_simplex_weights(distanceMatrices, numNeighbors, train_indices)

	# force columns so we can do multi-target predictions
	if train_y.ndim < 2:
		train_y = train_y[:, None]

	select = train_y[neighbor_indices, :]
	if predictions is not None:
		predictions[:] = torch.sum(weights[:, :, :, None] * select, dim = 1)
	else:
		predictions = torch.sum(weights[:, :, :, None] * select, dim = 1)
	return predictions


def batch_get_simplex_weights(distanceMatrices, numNeighbors, train_indices = None):
	"""
	Given distance matrices, get neighbor indices and weights per timepoint in the test set.
	Useful for making custom predictions
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use, can be a single shared n or one per distance matrix
	:param train_indices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return: neighbor_dist and weights <source, k, n_test> nearst neighbors and weights in train for each test point
	"""
	sharedNeighbors = isinstance(numNeighbors, int)
	if sharedNeighbors:
		k = numNeighbors
	else:
		k = int(torch.max(numNeighbors))
		if len(torch.unique(numNeighbors)) < 2:	# Case degenerate vector that could've been an int
			sharedNeighbors = True

	neighbor_dist, neighbor_indices = torch.topk(distanceMatrices, k, dim = 1,
												 largest = False)

	neighbor_dist.sqrt_()
	torch.clamp_min(neighbor_dist, 1e-6, out = neighbor_dist)
	minDistances = torch.amin(neighbor_dist, dim = 1)
	weights = neighbor_dist / minDistances.unsqueeze(1)
	weights.neg_().exp_()

	# if different num neighbors per distance matrix, mask the extra ones to 0 weight
	if not sharedNeighbors:
		weights.masked_fill_(torch.arange(k, device = weights.device)[None, :, None] >= numNeighbors[:, None, None], 0)
	weights.div_(weights.sum(dim = 1, keepdim = True)) # normalize weights to sum to 1

	# in CCM, the distance matrices that this function sees are a view into a larger matrix along the train dimension
	# so we need the actual indices corresponding to the columns to properly index into the data
	if train_indices is not None:
		neighbor_indices = train_indices[neighbor_indices]
	return neighbor_indices, weights