from warnings import warn

import numpy

from .Embed import Embed


def BuildEmbeddingIndices(data, columns, train, test, embedDimensions, predictionHorizon, step, exclusionRadius = 0, embedded = False, validLib = None, ignoreNan = True, removeNan = True):
	"""Build embedding, train/test indices, and exclusion mask without creating a dummy Simplex.

	:param data: 2D numpy array containing the source data.
	:param columns: Column indices used to build the embedding.
	:param train: Training index spans as [start1, end1, ...] with 1-based bounds.
	:param test: Test index spans as [start1, end1, ...] with 1-based bounds.
	:param embedDimensions: Embedding dimension E used when building lagged vectors.
	:param predictionHorizon: Prediction horizon Tp applied to library bounds.
	:param step: Embedding step (tau), where negative values indicate lagging.
	:param exclusionRadius: Temporal neighbor exclusion radius.
	:param embedded: If True, treat data[:, columns] as pre-embedded vectors.
	:param validLib: Optional boolean mask selecting valid library rows.
	:param ignoreNan: If True, allow NaN filtering when removeNan is enabled.
	:param removeNan: If True, remove NaN-containing rows from train/test indices.
	:returns: Tuple (embedding, trainIndices, testIndices, exclusionMask).
	"""
	if columns is None or not len(columns):
		columns = list(range(1, data.shape[1]))
	elif isinstance(columns, int):
		columns = [columns]
	else:
		columns = list(columns)

	if not hasattr(train, '__iter__') or isinstance(train, str):
		train = [int(i) for i in train.split()]
	else:
		train = list(train)

	if not hasattr(test, '__iter__') or isinstance(test, str):
		test = [int(i) for i in test.split()]
	else:
		test = list(test)

	if embedded:
		embedDimensions = len(columns)

	embedShift = abs(step) * (embedDimensions - 1)

	if len(train) % 2:
		raise RuntimeError('BuildEmbeddingIndices(): train must contain start/stop pairs.')
	if len(test) % 2:
		raise RuntimeError('BuildEmbeddingIndices(): test must contain start/stop pairs.')

	libPairs = [(train[i], train[i + 1]) for i in range(0, len(train), 2)]
	predPairs = [(test[i], test[i + 1]) for i in range(0, len(test), 2)]

	lib_i_list = []
	for r, libPair in enumerate(libPairs):
		start, stop = libPair
		if not embedded:
			if step < 0:
				start = start + embedShift
			else:
				stop = stop - embedShift
		if predictionHorizon < 0:
			if not embedded:
				start = max(start, start + abs(predictionHorizon) - 1)
		elif r == len(libPairs) - 1:
			stop = stop - predictionHorizon
		libPair_i = [i - 1 for i in range(start, stop + 1)]
		lib_i_list.append(numpy.array(libPair_i, dtype = int))

	trainIndices = numpy.concatenate(lib_i_list)

	pred_i_ = []
	for start, stop in predPairs:
		if start < 1 or stop < 1:
			raise RuntimeError('BuildEmbeddingIndices(): test indices less than 1 are not allowed.')
		pred_i_.extend([j - 1 for j in range(start, stop + 1)])

	testIndices = numpy.array(pred_i_, dtype = int)

	if len(trainIndices) == 0 or len(testIndices) == 0:
		raise ValueError('BuildEmbeddingIndices(): no valid train or test indices.')

	if trainIndices[-1] >= data.shape[0] or testIndices[-1] >= data.shape[0]:
		raise RuntimeError('BuildEmbeddingIndices(): train or test indices exceed data bounds.')

	if validLib is None:
		validLib = []
	if len(validLib):
		data_i = numpy.array([i for i in range(data.shape[0])], dtype = int)
		validLib_i = data_i[validLib]
		lib_i_valid = numpy.array([i for i in trainIndices if i in validLib_i], dtype = int)
		if len(lib_i_valid) == 0:
			raise ValueError('BuildEmbeddingIndices(): no valid library points found after validLib filtering.')
		knn_default = embedDimensions + 1
		if len(lib_i_valid) < knn_default:
			warn(f'BuildEmbeddingIndices(): only {len(lib_i_valid)} valid library points found, but default knn={knn_default}.')
		trainIndices = lib_i_valid

	if embedded:
		embedding = data[:, columns]
	else:
		embedding = Embed(data = data, embeddingDimensions = embedDimensions, stepSize = step, columns = columns)

	if removeNan and ignoreNan:
		na_lib = numpy.isnan(embedding[trainIndices, :]).any(axis = 1)
		na_pred = numpy.isnan(embedding[testIndices, :]).any(axis = 1)
		if na_lib.any():
			trainIndices = trainIndices[~na_lib]
		if na_pred.any():
			testIndices = testIndices[~na_pred]
		if len(testIndices) == 0:
			raise ValueError('BuildEmbeddingIndices(): No valid test indices after NaN removal.')

	libOverlap = len(set(trainIndices).intersection(set(testIndices))) > 0
	checkExclusion = False
	if exclusionRadius > 0:
		if libOverlap:
			checkExclusion = True
		else:
			excludeRow = 0
			if testIndices[0] > trainIndices[-1]:
				excludeRow = testIndices[0] - trainIndices[-1]
			elif trainIndices[0] > testIndices[-1]:
				excludeRow = trainIndices[0] - testIndices[-1]
			if exclusionRadius >= excludeRow:
				checkExclusion = True

	exclusionMask = numpy.zeros((len(trainIndices), len(testIndices)), dtype = bool)
	if libOverlap or checkExclusion:
		lib_index_map = {index: i for i, index in enumerate(trainIndices)}
		for i, pred_index in enumerate(testIndices):
			if libOverlap and pred_index in lib_index_map:
				lib_pos = lib_index_map[pred_index]
				exclusionMask[lib_pos, i] = True
			if checkExclusion:
				rowLow = max(numpy.min(trainIndices), pred_index - exclusionRadius)
				rowHi = min(numpy.max(trainIndices), pred_index + exclusionRadius)
				for lib_i, lib_idx in enumerate(trainIndices):
					if rowLow <= lib_idx <= rowHi:
						exclusionMask[lib_i, i] = True

	return embedding, trainIndices, testIndices, exclusionMask
