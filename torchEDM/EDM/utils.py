import numpy
from typing import List, Tuple
def BuildEmbeddingIndices(data: numpy.ndarray,
						  train_start_stops: List[Tuple[int, int]], test_start_stops: List[Tuple[int, int]],
						  embedDimensions, predictionHorizon, step,
						  is_embedded = False, valid_train_samples = None):
	"""
	Compute train/test indices given data parameters.

	:param data: 2D numpy array containing the source data.
	:param train_start_stops: Training blocks start/stop index pairs
	:param test_start_stops: Test block start/stop indes pairs
	:param embedDimensions: Embedding dimension E used when computing index offsets; ignore with is_embedded == True
	:param predictionHorizon: Prediction horizon Tp applied to library bounds.
	:param step: Embedding step (tau), where negative values indicate lagging.
	:param is_embedded: If True, skip the lag-shift offset when computing train start/stop.
	:param valid_train_samples : Optional boolean mask selecting valid library rows.
	:returns: Tuple (trainIndices, testIndices, exclusionMask).
	"""
	nVars = data.shape[1]
	nSamples = data.shape[0]

	if is_embedded:
		embedDimensions = nVars

	embedding_offset = abs(step) * (embedDimensions - 1)

	train_indices = []
	for i, (start, stop) in enumerate(train_start_stops):
		if not is_embedded:
			if step < 0:
				start = start + embedding_offset
			else:
				stop = stop - embedding_offset
		if predictionHorizon < 0:
			if not is_embedded:
				start = max(start, start + abs(predictionHorizon) - 1)
		elif i == len(train_start_stops) - 1:
			stop = min(stop, nSamples - predictionHorizon)
		these_indices = [i - 1 for i in range(start, stop + 1)]
		train_indices.append(numpy.array(these_indices, dtype = int))

	train_indices = numpy.concatenate(train_indices)

	test_indices = []
	for start, stop in test_start_stops:
		if start < 1 or stop < 1:
			raise RuntimeError('test indices less than 1')
		test_indices.extend([j - 1 for j in range(start, stop + 1)])

	test_indices = numpy.array(test_indices, dtype = int)

	if len(train_indices) == 0 or len(test_indices) == 0:
		raise ValueError('no valid train or test indices.')

	if train_indices[-1] >= data.shape[0] or test_indices[-1] >= data.shape[0]:
		raise RuntimeError('train or test indices exceed data bounds.')

	if valid_train_samples is None:
		valid_train_samples  = []
	if len(valid_train_samples):
		data_i = numpy.array([i for i in range(data.shape[0])], dtype = int)
		validLib_i = data_i[valid_train_samples]
		lib_i_valid = numpy.array([i for i in train_indices if i in validLib_i], dtype = int)
		if len(lib_i_valid) == 0:
			raise ValueError('no valid library points found after validLib filtering.')
		knn_default = embedDimensions + 1
		if len(lib_i_valid) < knn_default:
			raise ValueError('Fewer valid train data points than desired nearest neighbors')
		train_indices = lib_i_valid

	return train_indices, test_indices


def build_exclusion_mask(train_indices, test_indices, exclusionRadius):
	"""
	Build an exclusion mask for a distance matrix that will be built from the train and test indices
	THis accounts for re-using data for both train and test. Reconsider data choises if using this.
	:param train_indices:
	:param test_indices:
	:param exclusionRadius:
	:return:
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
	return exclusionMask
