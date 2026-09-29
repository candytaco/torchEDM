"""
Stack lagged copies of variables, and read a per-pair embedding dimension out of the
layouts ConvergentCrossMap accepts.
"""
import numpy


def MakeDelays(data, num_delays, stepSize = -1, fill = numpy.nan):
	"""
	Stack lagged copies of every variable, grouped by variable: all lags of variable 0, then
	of variable 1, and so on.

	:param data:	the series, [nSamples, nVariables], or [nSamples] for one variable
	:param num_delays:	number of copies of each variable, the first unshifted
	:param stepSize:	sample offset between consecutive copies; a negative offset shifts each copy further into the past
	:param fill:	value of the samples a shift moves outside the series
	:return: [nSamples, nVariables * num_delays]
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
	Return the embedding dimension of one (source, target) pair from any of the layouts
	ConvergentCrossMap accepts.

	:param embedDims:	an int for every pair, [nSources] per source, or [nSources, nTargets] per pair
	:param sourceIndex:	index of the source variable
	:param targetIndex:	index of the target variable
	:return: int
	"""
	if isinstance(embedDims, int):
		return embedDims
	arr = numpy.asarray(embedDims)
	if arr.ndim == 1:
		return int(arr[sourceIndex])
	return int(arr[sourceIndex, targetIndex])  # [nVars, nTargets]
