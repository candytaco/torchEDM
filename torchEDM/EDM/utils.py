import numpy
from typing import List, Tuple
def BuildEmbeddingIndices(nSamples: int, nVariables: int,
						  train_start_stops: List[Tuple[int, int]], test_start_stops: List[Tuple[int, int]],
						  embedDimensions, predictionHorizon, step,
						  is_embedded = False, valid_train_samples = None):
	"""
	Compute train/test indices given data parameters.

	:param nSamples: number of total samples
	:param nVariables: number of variables
	:param train_start_stops: Training blocks as 0-based, half-open [start, stop) pairs
	:param test_start_stops: Test blocks as 0-based, half-open [start, stop) pairs
	:param embedDimensions: Embedding dimensions used when computing index offsets; ignored with is_embedded == True
	:param predictionHorizon: Prediction horizon applied to window bounds.
	:param step: Embedding step, where negative values indicate lagging.
	:param is_embedded: If True, skip the lag-shift offset when computing train start/stop.
	:param valid_train_samples : Optional boolean mask selecting valid library rows.
	:returns: Tuple (trainIndices, testIndices).
	"""
	if is_embedded:
		embedDimensions = nVariables

	embedding_offset = abs(step) * (embedDimensions - 1)

	train_indices = []
	for start, stop in train_start_stops:
		if not is_embedded:
			if step < 0:
				start = start + embedding_offset
			else:
				stop = stop - embedding_offset
		# trim exactly the rows whose target index row + predictionHorizon is out of bounds
		if predictionHorizon < 0:
			start = max(start, -predictionHorizon)
		else:
			stop = min(stop, nSamples - predictionHorizon)
		train_indices.append(numpy.arange(start, stop, dtype = int))

	train_indices = numpy.concatenate(train_indices)

	test_indices = []
	for start, stop in test_start_stops:
		if start < 0 or stop < 0:
			raise RuntimeError('test indices less than 0')
		# trim exactly the rows whose target index row + predictionHorizon is out of bounds
		if predictionHorizon < 0:
			start = max(start, -predictionHorizon)
		else:
			stop = min(stop, nSamples - predictionHorizon)
		test_indices.append(numpy.arange(start, stop, dtype = int))

	test_indices = numpy.concatenate(test_indices)

	if len(train_indices) == 0 or len(test_indices) == 0:
		raise ValueError('no valid train or test indices.')

	if train_indices[-1] >= nSamples or test_indices[-1] >= nSamples:
		raise RuntimeError('train or test indices exceed data bounds.')

	if valid_train_samples is None:
		valid_train_samples  = []
	if len(valid_train_samples):
		data_i = numpy.array([i for i in range(nSamples)], dtype = int)
		validLib_i = data_i[valid_train_samples]
		lib_i_valid = numpy.array([i for i in train_indices if i in validLib_i], dtype = int)
		if len(lib_i_valid) == 0:
			raise ValueError('no valid library points found after validLib filtering.')
		if len(lib_i_valid) < embedDimensions + 1:
			raise ValueError('Not enough training data for minimum nearest neighbors')
		train_indices = lib_i_valid

	return numpy.array(train_indices), numpy.array(test_indices)


def build_exclusion_mask(train_indices, test_indices, exclusionRadius):
	"""
	Build an exclusion mask for a distance matrix that will be built from the train and test indices
	THis accounts for re-using data for both train and test. Reconsider data choices if using this.
	:param train_indices:
	:param test_indices:
	:param exclusionRadius:
	:return: exclusion mask, None if nothing is to be excluded
	"""
	train_test_overlap = len(set(train_indices).intersection(set(test_indices))) > 0
	train_test_close = False
	if exclusionRadius > 0:
		if train_test_overlap:
			train_test_close = True
		else:
			excludeRow = 0
			if test_indices[0] > train_indices[-1]:
				excludeRow = test_indices[0] - train_indices[-1]
			elif train_indices[0] > test_indices[-1]:
				excludeRow = train_indices[0] - test_indices[-1]
			if exclusionRadius >= excludeRow:
				train_test_close = True
	exclusionMask = numpy.zeros((len(train_indices), len(test_indices)), dtype = bool)
	if train_test_overlap or train_test_close:
		lib_index_map = {index: i for i, index in enumerate(train_indices)}
		for i, pred_index in enumerate(test_indices):
			if train_test_overlap and pred_index in lib_index_map:
				lib_pos = lib_index_map[pred_index]
				exclusionMask[lib_pos, i] = True
			if train_test_close:
				rowLow = max(numpy.min(train_indices), pred_index - exclusionRadius)
				rowHi = min(numpy.max(train_indices), pred_index + exclusionRadius)
				for lib_i, lib_idx in enumerate(train_indices):
					if rowLow <= lib_idx <= rowHi:
						exclusionMask[lib_i, i] = True
	return exclusionMask if numpy.any(exclusionMask) else None


def MakeDelays(data, num_delays, stepSize = -1, fill = numpy.nan):
	"""
	Make delayed copies of the columns of the data
	"""
	if data.ndim < 2:
		data = data[:, None]

	if num_delays < 1:
		raise RuntimeError('Need at least 1 delay')
	if stepSize == 0:
		raise RuntimeError('Need non-zero delay size')

	n_rows, n_cols = data.shape

	# Setup shift indices
	shiftVec = [i for i in range(0, int(num_delays * (-stepSize)), -stepSize)]

	# Create embedded array
	embedded_cols = []
	for col_idx in range(n_cols):
		for shift in shiftVec:
			shifted_col = numpy.full(n_rows, fill)
			if shift >= 0:
				if shift < n_rows:
					shifted_col[shift:] = data[:n_rows - shift, col_idx]
			else:
				if -shift < n_rows:
					shifted_col[:shift] = data[-shift:, col_idx]
			embedded_cols.append(shifted_col)

	result = numpy.column_stack(embedded_cols)
	return result


def _get_embedding_dimension(embedDims, sourceIndex, targetIndex):
	"""
	Return the optimal embedding dimension for source sourceIndex predicting target targetIndex.
	Small handler to deal with both 1D and 2D source:target matrices and also just an int
	"""
	if isinstance(embedDims, int):
		return embedDims
	arr = numpy.asarray(embedDims)
	if arr.ndim == 1:
		return int(arr[sourceIndex])
	return int(arr[sourceIndex, targetIndex])  # [nVars, nTargets]
