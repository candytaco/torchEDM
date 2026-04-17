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


def FloorArray(arr, floor_value):
	"""
	In-place minimum clamping
	:param arr: tensor to clamp
	:param floor_value: minimum value
	"""
	torch.clamp_min(arr, floor_value, out = arr)


def batched_simplex_predict(sq_distances, y_train, knn, knn_per_batch = None, target_per_batch = False):
	"""
	Core batched simplex projection on stacked squared-distance matrices.

	For each batch item the knn nearest training neighbours are found, exponential
	weights are computed from their Euclidean distances, and the weighted-average
	target value is returned as the prediction.

	Parameters
	----------
	sq_distances : Tensor [nBatch, nTrain, nTest]
		Squared pairwise Euclidean distances between training and test points,
		one matrix per batch item.
	y_train : Tensor
		Target values at training-set positions.  Three shapes are accepted:

		* [nTrain]           – a single target shared by all batch items;
		                       output shape is [nBatch, nTest].
		* [nTargets, nTrain] – multiple shared targets (requires target_per_batch=False);
		                       output shape is [nTargets, nBatch, nTest].
		* [nBatch, nTrain]   – a distinct target row for each batch item, e.g. for
		                       self-prediction or batched sampling (requires
		                       target_per_batch=True);
		                       output shape is [nBatch, nTest].

	knn : int
		Maximum number of nearest neighbours to fetch with topk.
	knn_per_batch : LongTensor [nBatch, 1, 1], optional
		Effective knn per batch item.  Weights for neighbours at position >= knn_per_batch[i]
		are set to zero, implementing variable-knn predictions (e.g. when each batch item
		corresponds to a different embedding dimension).  If all weights are zeroed for any
		(batch item, test point) pair, a RuntimeError is raised.  When None, all knn
		neighbours contribute to every batch item.
	target_per_batch : bool, optional
		When True, treat y_train as [nBatch, nTrain] (per-batch targets) even if
		y_train.shape[0] could otherwise be mistaken for nTargets.  Ignored when
		y_train is 1-D.  Default False.

	Returns
	-------
	Tensor [nBatch, nTest] or [nTargets, nBatch, nTest]
	"""
	nBatch, nTrain, nTest = sq_distances.shape
	device = sq_distances.device

	# ── 1. kNN search ───────────────────────────────────────────────────────
	neighbor_distances, neighbor_indices = torch.topk(sq_distances, knn, dim = 1, largest = False)
	# neighbor_distances, neighbor_indices: [nBatch, knn, nTest]

	# ── 2. Convert to Euclidean distances and apply floor ───────────────────
	neighbor_distances.sqrt_()
	FloorArray(neighbor_distances, 1e-6)

	# ── 3. Exponential weights w_i = exp(-d_i / d_min) ──────────────────────
	min_distances = neighbor_distances.amin(dim = 1, keepdim = True)  # [nBatch, 1, nTest]
	weights = torch.exp(-neighbor_distances / min_distances)           # [nBatch, knn, nTest]

	# ── 4. Optionally zero out extra neighbours (variable-knn) ──────────────
	if knn_per_batch is not None:
		k_indices = torch.arange(knn, device = device).view(1, knn, 1)
		weights.masked_fill_(k_indices >= knn_per_batch, 0.0)

	weight_sum = weights.sum(dim = 1)                                  # [nBatch, nTest]

	if (weight_sum == 0).any():
		raise RuntimeError(
			'batched_simplex_predict: all neighbours were excluded for at least one '
			'(batch item, test point) pair. Check knn_per_batch values.'
		)

	# ── 5. Gather targets and compute predictions ────────────────────────────
	if y_train.dim() == 1:
		# Single shared target [nTrain] → output [nBatch, nTest]
		selected = y_train[neighbor_indices]                           # [nBatch, knn, nTest]
		return (weights * selected).sum(dim = 1) / weight_sum

	if target_per_batch:
		# Per-batch target [nBatch, nTrain] → output [nBatch, nTest]
		flat_indices = neighbor_indices.reshape(nBatch, -1)            # [nBatch, knn*nTest]
		selected = y_train.gather(1, flat_indices).view(nBatch, knn, nTest)
		return (weights * selected).sum(dim = 1) / weight_sum

	# Multiple shared targets [nTargets, nTrain] → output [nTargets, nBatch, nTest]
	selected = y_train[:, neighbor_indices]                            # [nTargets, nBatch, knn, nTest]
	return (weights.unsqueeze(0) * selected).sum(dim = 2) / weight_sum.unsqueeze(0)


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
