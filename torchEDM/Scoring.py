from math import sqrt

import numpy
from numpy import isfinite, any, corrcoef, max, absolute, mean


def ComputeError(actual, predicted, metric, digits = 6):
	"""
	Compute performance metrics. This is a bad function because it computes errors,
	which we want to minimize, and also performance, which we want to maximize.

	:param actual: Actual values
	:param predicted: Predicted values
	:param metric: Metric to use (None for correlation, 'MAE', 'CAE', 'RMSE')
	:param digits: Number of decimal digits to round to
	:return: Computed metric value
	"""

	notNan = isfinite(predicted)
	if any( ~notNan ) :
		predicted = predicted[ notNan]
		actual  = actual [ notNan]

	notNan = isfinite(actual)
	if any( ~notNan ) :
		predicted = predicted[ notNan]
		actual  = actual [ notNan]

	if len(predicted) < 5 :
		msg = f'ComputeError(): Not enough data ({len(predicted)}) to ' +\
			   ' compute error statistics.'
		print( msg )
		return None

	if metric is None:
		return numpy.nan_to_num(corrcoef(actual, predicted)[0,1])

	err  = actual - predicted
	if metric == 'MAE':
		return max( err )
	if metric == 'CAE':
		return absolute( err ).sum()
	if metric == 'RMSE':
		return sqrt( mean( err**2 ))

	raise ValueError('Unknown metric {}'.format(metric))
