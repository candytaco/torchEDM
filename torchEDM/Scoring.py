import functools
import warnings

import numpy


def _FilterNonFinite(actual, predicted):
	notNan = numpy.isfinite(predicted)
	if numpy.any(~notNan):
		predicted = predicted[notNan]
		actual = actual[notNan]

	notNan = numpy.isfinite(actual)
	if numpy.any(~notNan):
		predicted = predicted[notNan]
		actual = actual[notNan]

	return actual, predicted


def _CheckLength(function):
	@functools.wraps(function)
	def wrapper(actual, predicted):
		actual, predicted = _FilterNonFinite(actual, predicted)
		if len(predicted) < 5:
			print('{}: Not enough data ({}) to compute error statistics.'.format(function.__name__, len(predicted)))
			return None
		return function(actual, predicted)

	return wrapper


@_CheckLength
def Correlation(actual, predicted):
	actual_centered = actual - numpy.mean(actual)
	predicted_centered = predicted - numpy.mean(predicted)
	numerator = numpy.sum(actual_centered * predicted_centered)
	denominator = numpy.sqrt(numpy.sum(actual_centered ** 2) * numpy.sum(predicted_centered ** 2))
	return numpy.nan_to_num(numerator / denominator)


@_CheckLength
def MaxAbsoluteError(actual, predicted):
	error = numpy.abs(actual - predicted)
	return numpy.max(error)


@_CheckLength
def MaxError(actual, predicted):
	# The reference package reports the maximum of the signed error
	# under its MAE label; this reproduces that statistic.
	error = actual - predicted
	return numpy.max(error)


@_CheckLength
def SumAbsoluteError(actual, predicted):
	error = actual - predicted
	return numpy.absolute(error).sum()


@_CheckLength
def RootMeanSquareError(actual, predicted):
	error = actual - predicted
	return numpy.sqrt(numpy.mean(error ** 2))


@_CheckLength
def R2(actual, predicted):
	residual_sum_of_squares = numpy.sum((actual - predicted) ** 2)
	total_sum_of_squares = numpy.sum((actual - numpy.mean(actual)) ** 2)
	return numpy.nan_to_num(1 - residual_sum_of_squares / total_sum_of_squares)


def ComputeError(actual, predicted, metric):
	"""
	Leftover old style function
	"""
	warnings.warn('ComputeError is deprecated; call the individual metric functions directly.',
	              DeprecationWarning,
	              stacklevel = 2)
	if metric is None:
		return Correlation(actual, predicted)
	if metric == 'MAE':
		return MaxAbsoluteError(actual, predicted)
	if metric == 'CAE':
		return SumAbsoluteError(actual, predicted)
	if metric == 'RMSE':
		return RootMeanSquareError(actual, predicted)
	raise ValueError('Unknown metric {}'.format(metric))
