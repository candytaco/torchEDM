"""
Tensor kernels shared by every predictor, batched scoring, and the batched neighbor-averaging
prediction used by the sweeps and the variable selection.
"""
from typing import Optional, Union, Callable

import torch


def _promoteDimensions(scoringFunction: Callable[[torch.tensor, torch.tensor, Optional[torch.tensor]], torch.tensor]):
	"""
	Decorator that reshape score functions inputs to handle multiple prediction targets
	:param scoringFunction:
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
		result = scoringFunction(target, predictions, out)
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

# ---------------------------------------------------------------------------
# Kernels shared by every predictor. Distance matrices are [..., nTrain, nTest]
# and neighbors run along dim -2, so a batch of matrices [nSources, nTrain, nTest]
# and a single matrix [nTrain, nTest] go through the same code.
# ---------------------------------------------------------------------------

def ComputePairwiseDistances(X_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
	"""
	Euclidean distance from every training state to every test state.
	:param X_train:	[nTrain, stateSize]
	:param X_test:	[nTest, stateSize]
	:return: [nTrain, nTest]
	"""
	return torch.cdist(X_train, X_test, p = 2)


def SelectNearestNeighbors(distanceMatrix: torch.Tensor, numNeighbors: int,
						   isTieBreakDeterministic: bool = False,
						   trainRows: Optional[torch.Tensor] = None,
						   testRows: Optional[torch.Tensor] = None):
	"""
	The numNeighbors smallest entries of every test column of a distance matrix.

	:param distanceMatrix:	[..., nTrain, nTest]; excluded pairs hold inf
	:param numNeighbors:	neighbors kept per test column
	:param isTieBreakDeterministic:	False: torch.topk, whose ordering of exactly tied
		distances is unspecified. True: tied distances are ordered by training position,
		and when trainRows and testRows are given (a shared sample axis, i.e. in-sample),
		by |testRow - trainRow| first, reproducing the reference selection. Needs a
		2-D matrix.
	:param trainRows:	[nTrain] sample positions of the training states on the shared axis
	:param testRows:	[nTest] sample positions of the test states on the same axis
	:return: (neighborDistances, neighborIndices), both [..., numNeighbors, nTest], nearest
		first; indices are positions along the nTrain axis
	"""
	if not isTieBreakDeterministic:
		return torch.topk(distanceMatrix, numNeighbors, dim = -2, largest = False)

	if distanceMatrix.ndim != 2:
		raise ValueError('Deterministic tie ordering needs a 2-D [nTrain, nTest] distance matrix')
	nTrain, nTest = distanceMatrix.shape
	device = distanceMatrix.device

	# Stable argsorts applied in reverse priority: the last sort decides, earlier
	# sorts settle its ties.
	order = torch.arange(nTrain, device = device)[:, None].expand(-1, nTest)
	if trainRows is not None and testRows is not None:
		trainRows = torch.as_tensor(trainRows, device = device, dtype = torch.long)
		testRows = torch.as_tensor(testRows, device = device, dtype = torch.long)
		temporalOffsets = (testRows[None, :] - trainRows[:, None]).abs()
		order = torch.argsort(trainRows, stable = True)[:, None].expand(-1, nTest)
		offsetsByOrder = torch.gather(temporalOffsets, 0, order)
		order = torch.gather(order, 0, torch.argsort(offsetsByOrder, dim = 0, stable = True))
	distancesByOrder = torch.gather(distanceMatrix, 0, order)
	selection = torch.gather(order, 0, torch.argsort(distancesByOrder, dim = 0, stable = True))[:numNeighbors, :]
	return torch.gather(distanceMatrix, 0, selection), selection


def ComputeSimplexWeights(neighborDistances: torch.Tensor) -> torch.Tensor:
	"""
	Exponential weights exp(-d / dNearest) for the neighbor-averaging predictor.
	The nearest distance is floored at 1e-6 for the division only; the distances
	themselves are not altered, so an exact zero-distance neighbor keeps weight 1.

	:param neighborDistances:	[..., k, nTest] Euclidean distances
	:return: [..., k, nTest] weights, not normalized
	"""
	nearest = torch.clamp_min(torch.amin(neighborDistances, dim = -2, keepdim = True), 1e-6)
	return torch.exp(-neighborDistances / nearest)


def ProjectSimplex(weights: torch.Tensor, neighborTargets: torch.Tensor):
	"""
	Weighted average of neighbor targets, and the weighted variance around it.

	:param weights:			[..., k, nTest]
	:param neighborTargets:	[..., k, nTest, nTargets]
	:return: (predictions, variance), both [..., nTest, nTargets]
	"""
	weightSum = weights.sum(dim = -2)[..., None]
	predictions = (weights[..., None] * neighborTargets).sum(dim = -3) / weightSum
	deviations = neighborTargets - predictions[..., None, :, :]
	variance = (weights[..., None] * deviations ** 2).sum(dim = -3) / weightSum
	return predictions, variance


def ComputeSMapWeights(neighborDistances: torch.Tensor, theta: float) -> torch.Tensor:
	"""
	Localization weights exp(-theta * d / mean(d)) for the locally linear predictor.

	:param neighborDistances:	[..., k, nTest]
	:param theta:				0 gives uniform weights (one global linear map)
	:return: [..., k, nTest]
	"""
	if theta == 0:
		return torch.ones_like(neighborDistances)
	meanDistance = torch.clamp_min(neighborDistances.mean(dim = -2, keepdim = True), 1e-10)
	return torch.exp(-theta * neighborDistances / meanDistance)


def SolveWeightedLinearMap(weights: torch.Tensor, neighborStates: torch.Tensor,
						   neighborTargets: torch.Tensor, X_test: torch.Tensor):
	"""
	Per test state, solve the weighted least-squares map from neighbor states to
	neighbor targets (with an intercept) and apply it to the test state.
	A NaN neighbor target drops that neighbor's equation for that target only; a target
	with no finite neighbor target has no equations, and its coefficients, prediction,
	variance, and singular values are NaN.

	:param weights:			[nTest, k]
	:param neighborStates:	[nTest, k, stateSize]
	:param neighborTargets:	[nTest, k, nTargets]
	:param X_test:		[nTest, stateSize]
	:return: coefficients [nTest, stateSize + 1, nTargets] (intercept first),
		predictions [nTest, nTargets], variance [nTest, nTargets],
		singularValues [nTest, stateSize + 1, nTargets] of the weighted design matrices,
		NaN-padded when k < stateSize + 1
	"""
	nTest, k, stateSize = neighborStates.shape
	nTargets = neighborTargets.shape[-1]

	isFinite = torch.isfinite(neighborTargets)
	maskedWeights = torch.where(isFinite, weights[:, :, None], torch.zeros_like(neighborTargets))	# [nTest, k, nTargets]
	maskedTargets = torch.where(isFinite, neighborTargets, torch.zeros_like(neighborTargets))

	# one design matrix per (test state, target): [nTest, nTargets, k, stateSize + 1]
	weightsByTarget = maskedWeights.permute(0, 2, 1)
	design = torch.cat([weightsByTarget[..., None],
						weightsByTarget[..., None] * neighborStates[:, None, :, :]], dim = -1)
	rightHandSide = (weightsByTarget * maskedTargets.permute(0, 2, 1))[..., None]

	coefficients = torch.linalg.lstsq(design, rightHandSide).solution[..., 0]	# [nTest, nTargets, stateSize + 1]
	predictions = coefficients[..., 0] + (coefficients[..., 1:] * X_test[:, None, :]).sum(dim = -1)

	residuals = maskedTargets - predictions[:, None, :]
	weightSum = maskedWeights.sum(dim = 1)
	variance = (maskedWeights * residuals ** 2).sum(dim = 1) / weightSum

	singularValues = torch.linalg.svdvals(design)	# [nTest, nTargets, min(k, stateSize + 1)]
	if singularValues.shape[-1] < stateSize + 1:
		padding = torch.full((nTest, nTargets, stateSize + 1 - singularValues.shape[-1]), float('nan'),
							 device = design.device, dtype = design.dtype)
		singularValues = torch.cat([singularValues, padding], dim = -1)

	# an all-NaN neighbor target leaves an all-zero system, which lstsq solves as zeros
	hasEquation = weightSum > 0	# [nTest, nTargets]
	nan = torch.tensor(float('nan'), device = design.device, dtype = design.dtype)
	predictions = torch.where(hasEquation, predictions, nan)
	variance = torch.where(hasEquation, variance, nan)
	coefficients = torch.where(hasEquation[..., None], coefficients, nan)
	singularValues = torch.where(hasEquation[..., None], singularValues, nan)

	return coefficients.permute(0, 2, 1), predictions, variance, singularValues.permute(0, 2, 1)


def batch_simplex_predict_and_score(distanceMatrices: torch.tensor, numNeighbors: Union[int, torch.tensor],
									Y_train: torch.tensor, Y_test: torch.tensor, scoringFunction: Callable,
									predictions: Optional[torch.tensor] = None,
									performanceOut: Optional[torch.tensor] = None,
									trainIndices: Optional[torch.tensor] = None):
	"""
	Batched multiple predictions and score via simplex. Each distance matrix is used to make a separate prediction on Y.
	These predictions are then scored
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use
	:param Y_train:				[nTrain] or [nTrain, nTargets] targets of the training rows
	:param Y_test:				[nTest] or [nTest, nTargets] truth for the test rows
	:param scoringFunction:		score function to evaluate performance
	:param predictions:			tensor write prediction into
	:param performanceOut:			array to write the performance into
	:param trainIndices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return:
	"""
	predictions = batch_simplex_predict(distanceMatrices, numNeighbors, Y_train, predictions, trainIndices)
	return scoringFunction(Y_test, predictions, performanceOut)


def batch_simplex_predict(distanceMatrices: torch.tensor, numNeighbors: Union[int, torch.tensor],
						  Y_train: torch.tensor, predictions: Optional[torch.tensor] = None,
						  trainIndices: Optional[torch.tensor] = None) -> torch.tensor:
	"""
	Batched multiple predictions via simplex. Each distance matrix is used to make a separate prediction on Y.
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use, can be a single shared n or one per distance matrix
	:param Y_train:				[nTrain] or [nTrain, nTargets] targets of the training rows
	:param predictions:			array to write the predictions into
	:param trainIndices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return: predicted Y in <source, n_test, target>
	"""
	neighbor_indices, weights = batch_get_simplex_weights(distanceMatrices, numNeighbors, trainIndices)

	# force columns so we can do multi-target predictions
	if Y_train.ndim < 2:
		Y_train = Y_train[:, None]

	select = Y_train[neighbor_indices, :]
	if predictions is not None:
		predictions[:] = torch.sum(weights[:, :, :, None] * select, dim = 1)
	else:
		predictions = torch.sum(weights[:, :, :, None] * select, dim = 1)
	return predictions


def batch_get_simplex_weights(distanceMatrices, numNeighbors, trainIndices = None):
	"""
	Given distance matrices, get neighbor indices and weights per timepoint in the test set.
	Useful for making custom predictions
	:param distanceMatrices:	distance matrices of shape <source, n_train, n_test>
	:param numNeighbors:		number of nearest neighbors to use, can be a single shared n or one per distance matrix
	:param trainIndices:		actual indices for each entry in the 2nd dim in the distance matrices; for CCM subsampling
	:return: neighbor_dist and weights <source, k, n_test> nearst neighbors and weights in train for each test point
	"""
	sharedNeighbors = isinstance(numNeighbors, int)
	if sharedNeighbors:
		k = numNeighbors
	else:
		k = int(torch.max(numNeighbors))
		if len(torch.unique(numNeighbors)) < 2:	# Case degenerate vector that could've been an int
			sharedNeighbors = True

	neighbor_dist, neighbor_indices = SelectNearestNeighbors(distanceMatrices, k)

	# the matrices hold squared distances; the weights want Euclidean ones
	weights = ComputeSimplexWeights(neighbor_dist.sqrt())

	# if different num neighbors per distance matrix, mask the extra ones to 0 weight
	if not sharedNeighbors:
		weights.masked_fill_(torch.arange(k, device = weights.device)[None, :, None] >= numNeighbors[:, None, None], 0)
	weights.div_(weights.sum(dim = 1, keepdim = True)) # normalize weights to sum to 1

	# in CCM, the distance matrices that this function sees are a view into a larger matrix along the train dimension
	# so we need the actual indices corresponding to the columns to properly index into the data
	if trainIndices is not None:
		neighbor_indices = trainIndices[neighbor_indices]
	return neighbor_indices, weights