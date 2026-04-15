import numpy


def Embed(data,
          columns,
          embeddingDimensions = 0,
          stepSize = -1,
          includeTime = False, ):
	"""
	Takens time-delay embedding on columns via pandas DataFrame.shift()
	if includeTime True : insert dataFrame column 0 in first column
	nan will be present in |step| * (E-1) rows.
	"""

	if embeddingDimensions < 1:
		raise RuntimeError('Embed(): E must be positive.')
	if stepSize == 0:
		raise RuntimeError('Embed(): step must be non-zero.')

	selected_data = data[:, columns]
	n_rows, n_cols = selected_data.shape

	# Setup shift indices
	shiftVec = [i for i in range(0, int(embeddingDimensions * (-stepSize)), -stepSize)]

	# Create embedded array
	embedded_cols = []
	for col_idx in range(n_cols):
		for shift in shiftVec:
			shifted_col = numpy.full(n_rows, numpy.nan)
			if shift >= 0:
				if shift < n_rows:
					shifted_col[shift:] = selected_data[:n_rows - shift, col_idx]
			else:
				if -shift < n_rows:
					shifted_col[:shift] = selected_data[-shift:, col_idx]
			embedded_cols.append(shifted_col)

	result = numpy.column_stack(embedded_cols)

	if includeTime:
		# Prepend first column of original data
		result = numpy.column_stack([data[:, 0], result])

	return result
