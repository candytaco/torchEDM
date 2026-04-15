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


def FloorArray(arr, floor_value):
	"""
	In-place minimum clamping
	:param arr: tensor to clamp
	:param floor_value: minimum value
	"""
	torch.clamp_min(arr, floor_value, out = arr)


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


def RowwiseCorrelation(vector, array, out):
	"""
	Correlation between a vector and columns of an array

	Because of the order of operations in the MDE tensors, things end up in rows instead of columns

	:param vector: [n] tensor
	:param array: [m x n] tensor
	:param out: [m] output tensor
	:return: out tensor with correlations
	"""
	vectorCentered = vector - torch.mean(vector)
	vectorStd = torch.sqrt(torch.sum(vectorCentered ** 2))

	arrayMeans = torch.mean(array, dim = 1, keepdim = True)
	arrayCentered = array - arrayMeans
	arrayStd = torch.sqrt(torch.sum(arrayCentered ** 2, dim = 1))

	out[:] = torch.sum(vectorCentered * arrayCentered, dim = 1) / (vectorStd * arrayStd)

	return out


def RowwiseR2(vector, array, out):
	"""
	R2 (coefficient of determination) between a vector (Y_true) and rows of an array (Y_pred)
	:param vector: [n] tensor of true values
	:param array: [m x n] tensor of predicted values (each row is a set of predictions)
	:param out: [m] output tensor
	:return: out tensor with R2 values
	"""
	vectorMean = torch.mean(vector)
	totalSumOfSquares = torch.sum((vector - vectorMean) ** 2)

	residualSumOfSquares = torch.sum((vector - array) ** 2, dim = 1)
	out[:] = 1 - residualSumOfSquares / totalSumOfSquares

	return out
