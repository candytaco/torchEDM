"""
Tensor kernels shared by every predictor: batched performance metrics, distance and
neighbor selection, and the batched nearest-neighbor prediction used by the parameter
sweeps and the variable selection.
"""
from typing import Optional, Union, Callable

import torch


def _promoteDimensions(scoringFunction: Callable[[torch.tensor, torch.tensor, Optional[torch.tensor]], torch.tensor]):
	"""
	Give the torch performance metrics one calling convention. target may be [nTime] or
	[nTime, nTargets]; predictions may be [nTime], [nSources, nTime] or
	[nSources, nTime, nTargets]; integer inputs are converted to floats. Everything is
	promoted to the 3-D form the metric expects, and the score of a 1-D prediction vector
	comes back as a 0-d tensor.

	:param scoringFunction:	metric called as scoringFunction(target [nTime, nTargets], predictions [nSources, nTime, nTargets], out) -> out
	:return: the wrapped metric
	"""
	def wrapper(target, predictions, out = None):
		"""
		:param target:	true data, [nTime] or [nTime, nTargets]
		:param predictions:	predicted data, [nTime], [nSources, nTime] or [nSources, nTime, nTargets]
		:param out:	optional output buffer, promoted to [nSources, nTargets]
		:return: the metric's result, reduced to a 0-d tensor for a 1-D prediction vector
		"""
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
	Compute the Pearson correlation between the true data and each series of predicted data.

	:param target:	true data, [nTime, nTargets]
	:param predictions:	predicted data, [nSources, nTime, nTargets], one series per source
	:param out:	output buffer for the correlations, [nSources, nTargets]; allocated when None
	:return: out with singleton axes squeezed
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
	Compute the Pearson correlation between the true data and each series of predicted data,
	centering the predictions in place so no second copy is allocated. Peak memory is two
	[nSources, nTime, nTargets] tensors instead of three. The caller must not use predictions
	afterwards. Inputs must already be 3-D.

	:param target:	true data, [nTime, nTargets]
	:param predictions:	predicted data, [nSources, nTime, nTargets]; overwritten
	:param out:	output buffer for the correlations, [nSources, nTargets]
	"""
	targetCentered = target - torch.mean(target, dim = 0, keepdim = True)
	targetStd = targetCentered.norm(dim = 0)

	predictions.sub_(torch.mean(predictions, dim = 1, keepdim = True))
	predictionsStd = predictions.norm(dim = 1)

	out[:] = torch.sum(targetCentered * predictions, dim = 1) / (targetStd * predictionsStd).clamp(min = 1e-8)


@_promoteDimensions
def R2(target: torch.tensor, predictions: torch.tensor, out: Optional[torch.tensor] = None):
	"""
	Compute the variance explained in the true data by each series of predicted data (R squared).

	:param target:	true data, [nTime, nTargets]
	:param predictions:	predicted data, [nSources, nTime, nTargets], one series per source
	:param out:	output buffer for the values, [nSources, nTargets]; allocated when None
	:return: out with singleton axes squeezed
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
	Compute the Euclidean distance from every training state to every test state.

	:param X_train:	training states, [nTrain, stateSize]
	:param X_test:	test states, [nTest, stateSize]
	:return: [nTrain, nTest]
	"""
	return torch.cdist(X_train, X_test, p = 2)


def SelectNearestNeighbors(distanceMatrix: torch.Tensor, numNeighbors: int,
						   isTieBreakDeterministic: bool = False,
						   trainRows: Optional[torch.Tensor] = None,
						   testRows: Optional[torch.Tensor] = None):
	"""
	Select, for every test sample, the numNeighbors smallest distances in a distance matrix.

	:param distanceMatrix:	distances, [..., nTrain, nTest]; excluded pairs hold inf
	:param numNeighbors:	number of neighbors kept per test state
	:param isTieBreakDeterministic:	False uses torch.topk, whose ordering of exactly tied distances is unspecified. True orders tied distances by training position, and when trainRows and testRows are given (a shared sample axis, i.e. in-sample), by |testRow - trainRow| first, reproducing the reference selection. Needs a 2-D matrix
	:param trainRows:	positions of the training states on the shared sample axis, [nTrain]
	:param testRows:	positions of the test states on the same axis, [nTest]
	:return: (neighborDistances, neighborIndices), both [..., numNeighbors, nTest], nearest first; indices are positions along the nTrain axis
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
	Compute the exponential weights exp(-d / dNearest) of the nearest-neighbor predictor.
	The nearest distance is floored at 1e-6 for the division only; the distances themselves
	are not altered, so an exact zero-distance neighbor keeps weight 1.

	:param neighborDistances:	Euclidean distances of the selected neighbors, [..., k, nTest]
	:return: weights, [..., k, nTest], not normalized
	"""
	nearest = torch.clamp_min(torch.amin(neighborDistances, dim = -2, keepdim = True), 1e-6)
	return torch.exp(-neighborDistances / nearest)


def ProjectSimplex(weights: torch.Tensor, neighborTargets: torch.Tensor):
	"""
	Compute the weighted average of the neighbor targets and the weighted variance around it.

	:param weights:	neighbor weights, [..., k, nTest]
	:param neighborTargets:	targets of the selected neighbors, [..., k, nTest, nTargets]
	:return: (predictions, variance), both [..., nTest, nTargets]
	"""
	weightSum = weights.sum(dim = -2)[..., None]
	predictions = (weights[..., None] * neighborTargets).sum(dim = -3) / weightSum
	deviations = neighborTargets - predictions[..., None, :, :]
	variance = (weights[..., None] * deviations ** 2).sum(dim = -3) / weightSum
	return predictions, variance


def ComputeSMapWeights(neighborDistances: torch.Tensor, theta: float) -> torch.Tensor:
	"""
	Compute the localization weights exp(-theta * d / mean(d)) of the locally linear predictor.

	:param neighborDistances:	distances of the selected neighbors, [..., k, nTest]
	:param theta:	localization strength; 0 gives uniform weights, one global linear map
	:return: weights, [..., k, nTest]
	"""
	if theta == 0:
		return torch.ones_like(neighborDistances)
	meanDistance = torch.clamp_min(neighborDistances.mean(dim = -2, keepdim = True), 1e-10)
	return torch.exp(-theta * neighborDistances / meanDistance)


def SolveWeightedLinearMap(weights: torch.Tensor, neighborStates: torch.Tensor,
						   neighborTargets: torch.Tensor, X_test: torch.Tensor):
	"""
	Solve, per test state, the weighted least-squares map from neighbor states to neighbor
	targets (with an intercept) and apply it to the test state. A NaN neighbor target drops
	that neighbor's equation for that target only; a target with no finite neighbor target
	has no equations, and its coefficients, prediction, variance, and singular values are NaN.

	:param weights:	neighbor weights, [nTest, k]
	:param neighborStates:	states of the selected neighbors, [nTest, k, stateSize]
	:param neighborTargets:	targets of the selected neighbors, [nTest, k, nTargets]
	:param X_test:	test states, [nTest, stateSize]
	:return: coefficients [nTest, stateSize + 1, nTargets] with the intercept first, predictions [nTest, nTargets], variance [nTest, nTargets], and singularValues [nTest, stateSize + 1, nTargets] of the weighted design matrices, NaN-padded when k < stateSize + 1
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
	Predict with the nearest-neighbor predictor from each of a batch of distance matrices and
	score the predictions against the true data.

	:param distanceMatrices:	squared distances, [nMatrices, nTrain, nTest], one matrix per candidate variable or embedding dimension
	:param numNeighbors:	number of nearest neighbors per test sample: one int shared by every matrix, or a long tensor [nMatrices] with one count per matrix, in which case the largest count is selected and each matrix's surplus neighbors get zero weight
	:param Y_train:	targets of the training states, [nTrain] or [nTrain, nTargets]
	:param Y_test:	true data for the test states, [nTest] or [nTest, nTargets]
	:param scoringFunction:	one of the torch metrics above, called as scoringFunction(Y_test, predictions, performanceOut)
	:param predictions:	optional output buffer for the predictions, [nMatrices, nTest, nTargets]
	:param performanceOut:	optional output buffer for the scores, [nMatrices, nTargets]
	:param trainIndices:	optional map from an index along the training axis of the matrices to the training sample it stands for, [nTrain], when the matrices are a subset view of a larger training set
	:return: the scores, [nMatrices, nTargets] with singleton axes squeezed
	"""
	predictions = batch_simplex_predict(distanceMatrices, numNeighbors, Y_train, predictions, trainIndices)
	return scoringFunction(Y_test, predictions, performanceOut)


def batch_simplex_predict(distanceMatrices: torch.tensor, numNeighbors: Union[int, torch.tensor],
						  Y_train: torch.tensor, predictions: Optional[torch.tensor] = None,
						  trainIndices: Optional[torch.tensor] = None) -> torch.tensor:
	"""
	Predict with the nearest-neighbor predictor from each of a batch of distance matrices.

	:param distanceMatrices:	squared distances, [nMatrices, nTrain, nTest], one matrix per candidate variable or embedding dimension
	:param numNeighbors:	number of nearest neighbors per test sample: one int shared by every matrix, or a long tensor [nMatrices] with one count per matrix, in which case the largest count is selected and each matrix's surplus neighbors get zero weight
	:param Y_train:	targets of the training states, [nTrain] or [nTrain, nTargets]
	:param predictions:	optional output buffer for the predictions, [nMatrices, nTest, nTargets]
	:param trainIndices:	optional map from an index along the training axis of the matrices to the training sample it stands for, [nTrain], when the matrices are a subset view of a larger training set
	:return: predictions, [nMatrices, nTest, nTargets]
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
	Select the nearest neighbors and compute their normalized exponential weights per test
	sample of each distance matrix, for building custom predictions.

	:param distanceMatrices:	squared distances, [nMatrices, nTrain, nTest], one matrix per candidate variable or embedding dimension; the weights are computed from their square roots
	:param numNeighbors:	number of nearest neighbors per test sample: one int shared by every matrix, or a long tensor [nMatrices] with one count per matrix, in which case the largest count is selected and each matrix's surplus neighbors get zero weight
	:param trainIndices:	optional map from an index along the training axis of the matrices to the training sample it stands for, [nTrain], when the matrices are a subset view of a larger training set
	:return: (neighborIndices [nMatrices, k, nTest] into the training samples, weights [nMatrices, k, nTest] summing to one per test sample), nearest first
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