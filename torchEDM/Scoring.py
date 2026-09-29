"""
Scores of a prediction against its truth, on numpy arrays. Every metric first drops the
pairs where either value is not finite and declines (returns None) when fewer than five
pairs remain.
"""
import functools
import warnings

import numpy


def _FilterNonFinite(actual, predicted):
	"""
	Keep only the pairs where both values are finite.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: (actual, predicted) filtered the same way
	"""
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
	"""
	Decorator applying _FilterNonFinite before a metric and declining when fewer than five
	pairs remain.

	:param function:	function(actual, predicted) -> float on filtered 1-D arrays
	:return: the wrapped function, which returns None instead of a score when it declines
	"""
	@functools.wraps(function)
	def wrapper(actual, predicted):
		"""
		:param actual:	[n] true values
		:param predicted:	[n] predictions aligned with actual
		:return: the metric over the finite pairs, or None when fewer than five remain
		"""
		actual, predicted = _FilterNonFinite(actual, predicted)
		if len(predicted) < 5:
			print('{}: Not enough data ({}) to compute error statistics.'.format(function.__name__, len(predicted)))
			return None
		return function(actual, predicted)

	return wrapper


@_CheckLength
def Correlation(actual, predicted):
	"""
	Pearson correlation; a constant series gives 0 rather than NaN.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: float in [-1, 1], or None when fewer than five finite pairs remain
	"""
	actual_centered = actual - numpy.mean(actual)
	predicted_centered = predicted - numpy.mean(predicted)
	numerator = numpy.sum(actual_centered * predicted_centered)
	denominator = numpy.sqrt(numpy.sum(actual_centered ** 2) * numpy.sum(predicted_centered ** 2))
	return numpy.nan_to_num(numerator / denominator)


@_CheckLength
def MaxAbsoluteError(actual, predicted):
	"""
	Largest absolute difference between a prediction and its truth.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: float, or None when fewer than five finite pairs remain
	"""
	error = numpy.abs(actual - predicted)
	return numpy.max(error)


@_CheckLength
def SumAbsoluteError(actual, predicted):
	"""
	Sum of the absolute differences between predictions and truth.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: float, or None when fewer than five finite pairs remain
	"""
	error = actual - predicted
	return numpy.absolute(error).sum()


@_CheckLength
def RootMeanSquareError(actual, predicted):
	"""
	Root mean square of the differences between predictions and truth.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: float, or None when fewer than five finite pairs remain
	"""
	error = actual - predicted
	return numpy.sqrt(numpy.mean(error ** 2))


@_CheckLength
def R2(actual, predicted):
	"""
	Variance of the truth explained by the predictions (1 minus residual over total sum of
	squares); the undefined ratio of a constant truth passes through numpy.nan_to_num.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:return: float, or None when fewer than five finite pairs remain
	"""
	residual_sum_of_squares = numpy.sum((actual - predicted) ** 2)
	total_sum_of_squares = numpy.sum((actual - numpy.mean(actual)) ** 2)
	return numpy.nan_to_num(1 - residual_sum_of_squares / total_sum_of_squares)


def ComputeError(actual, predicted, metric):
	"""
	Deprecated dispatcher kept for old callers; call the metric functions directly.

	:param actual:	[n] true values
	:param predicted:	[n] predictions aligned with actual
	:param metric:	None for Correlation, 'MAE' for MaxAbsoluteError, 'CAE' for SumAbsoluteError, 'RMSE' for RootMeanSquareError
	:return: the chosen metric's score
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
