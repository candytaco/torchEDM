"""
Stacking shifted copies of columns, and reading a per-pair embedding dimension out of the shapes
ConvergentCrossMap accepts.
"""
import numpy


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
