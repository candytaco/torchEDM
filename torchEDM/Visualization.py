"""
Plots of predictions and sweeps. Rows are sample positions; there is no time axis.
"""
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def _FirstRun(arrays):
	return arrays[0] if isinstance(arrays, list) else arrays


def _Column(values, column = 0):
	values = np.asarray(values)
	return values if values.ndim == 1 else values[:, column]


def plot_prediction(Y_true: np.ndarray, Y_pred, title: str = "", block: bool = True):
	"""
	Observations and predictions against row index, with correlation and RMSE in the title.

	:param Y_true:	[nRows] or [nRows, nTargets]; the first target is plotted
	:param Y_pred:	a result record with Y_pred, or an array shaped like Y_true (the first run of a list)
	"""
	from .Scoring import Correlation, RootMeanSquareError
	predicted = _Column(_FirstRun(getattr(Y_pred, 'Y_pred', Y_pred)))
	actual = _Column(_FirstRun(Y_true))
	corr = Correlation(actual, predicted)
	rmse = RootMeanSquareError(actual, predicted)
	plot_title = (title + "\n" if title else "") + f"correlation={corr}  RMSE={rmse}"
	plt.figure()
	rows = np.arange(len(actual))
	plt.plot(rows, actual, label = 'Observations', linewidth = 3)
	plt.plot(rows, predicted, label = 'Predictions', linewidth = 3)
	plt.xlabel('Row')
	plt.title(plot_title)
	plt.legend()
	plt.show(block = block)


def plot_smap_coefficients(result, title: str = "", block: bool = True):
	"""
	Each coefficient of the locally weighted linear map against row index, first target.

	:param result:	an SMapResult, or an array [nRows, stateSize + 1] (or [nRows, stateSize + 1, nTargets])
	"""
	coefficients = np.asarray(_FirstRun(getattr(result, 'coefficients', result)))
	if coefficients.ndim == 3:
		coefficients = coefficients[:, :, 0]
	numCoefficients = coefficients.shape[1]
	rows = np.arange(coefficients.shape[0])
	plt.figure()
	for i in range(numCoefficients):
		plt.subplot(numCoefficients, 1, i + 1)
		plt.plot(rows, coefficients[:, i], linewidth = 3)
		plt.title('Intercept' if i == 0 else f'Coefficient {i}')
	plt.suptitle((title + "\n" if title else "") + "S-Map Coefficients")
	plt.tight_layout()
	plt.show(block = block)


def plot_ccm(result, title: str = "", block: bool = True):
	"""
	Cross-map skill against training-subset size, one line per source (and target).

	:param result:	a BatchedCCMResult, or an array [nSizes, 1 + nLines] with the sizes in column 0
	"""
	if hasattr(result, 'forward_performance'):
		sizes = np.asarray(result.library_sizes)
		skill = np.asarray(result.forward_performance).reshape(len(sizes), -1)
	else:
		data = np.asarray(result)
		sizes, skill = data[:, 0], data[:, 1:]
	fig, ax = plt.subplots()
	for column in range(skill.shape[1]):
		ax.plot(sizes, skill[:, column], linewidth = 3, label = f'source {column}' if skill.shape[1] > 1 else None)
	if skill.shape[1] > 1:
		ax.legend()
	ax.set(xlabel = "Training-subset size", ylabel = "Cross-map correlation", title = title)
	plt.axhline(y = 0, linewidth = 1)
	plt.show(block = block)


def plot_multiview(Y_true: np.ndarray, result, title: str = "", block: bool = True):
	"""Ensemble prediction against observations; see plot_prediction."""
	plot_prediction(Y_true, result, title = title, block = block)


def _plot_sweep(result: np.ndarray, xlabel: str, title: str, block: bool):
	result = np.asarray(result)
	plt.figure()
	for column in range(1, result.shape[1]):
		plt.plot(result[:, 0], result[:, column], 'o-', linewidth = 2, markersize = 8,
				 label = f'target {column - 1}' if result.shape[1] > 2 else None)
	if result.shape[1] > 2:
		plt.legend()
	plt.xlabel(xlabel)
	plt.ylabel('Prediction skill')
	plt.title(title)
	plt.grid(True, alpha = 0.3)
	plt.show(block = block)


def plot_embed_dimension(result: np.ndarray, title: str = "", block: bool = True):
	"""
	:param result:	[maxDims] scores from FindOptimalEmbeddingDimensionality, or [maxDims, 1 + nTargets] with the embedding dimensions in column 0
	"""
	result = np.asarray(result)
	if result.ndim == 1:
		result = np.column_stack([np.arange(1, len(result) + 1), result])
	_plot_sweep(result, 'Embedding dimensions', title or "Embedding dimensions", block)


def plot_predict_interval(result: np.ndarray, title: str = "", block: bool = True):
	"""
	:param result:	[maxTp, 1 + nTargets] from FindOptimalPredictionHorizon
	"""
	_plot_sweep(result, 'Prediction horizon', title or "Prediction horizon", block)


def plot_predict_nonlinear(result: np.ndarray, title: str = "", block: bool = True):
	"""
	:param result:	[nTheta, 1 + nTargets] from FindSMapNeighborhood
	"""
	_plot_sweep(result, 'Localization (theta)', title or "Localization", block)
