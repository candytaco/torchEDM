"""
Plot torchEDM results. These helpers live outside the package so that matplotlib is not a
dependency of the core. Every x axis counts samples; there is no time axis.

Run alongside fitter_examples.py, or import from a script started in this directory.
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from torchEDM.Scoring import Correlation, RootMeanSquareError


def _FirstRun(arrays):
	"""
	Return the first run of a list or tuple of runs, or a single array as is.

	:param arrays:	an array or a collection of per-run arrays
	"""
	return arrays[0] if isinstance(arrays, (list, tuple)) else arrays


def _Column(values, column = 0):
	"""
	Return one variable of a 2-D array, or a 1-D array as is.

	:param values:	[nSamples] or [nSamples, nVariables]
	:param column:	index of the variable taken from a 2-D array
	"""
	values = np.asarray(values)
	return values if values.ndim == 1 else values[:, column]


def plot_prediction(Y_true, Y_pred, title: str = "", block: bool = True):
	"""
	Plot the true data and the predicted data against sample index, with the correlation and
	RMSE in the title.

	:param Y_true:	true (observed) data, [nSamples] or [nSamples, nTargets]; a list of runs plots its first run; the first target is plotted
	:param Y_pred:	predicted data with the same shape as the true data, or a result record whose Y_pred has that shape
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
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
	Plot each coefficient of the locally weighted linear map against sample index, first target.

	:param result:	an SMapResult, or a coefficient array [nSamples, stateSize + 1] or [nSamples, stateSize + 1, nTargets]
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
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
	Plot the cross-map performance against training-subset size, one line per source (and target).

	:param result:	a BatchedCCMResult, or an array [nSizes, 1 + nLines] holding, per size, the size itself and then one performance value per line
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
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


def plot_multiview(Y_true, result, title: str = "", block: bool = True):
	"""
	Plot the ensemble prediction against the true data, drawn by plot_prediction.

	:param Y_true:	true (observed) data, [nSamples] or [nSamples, nTargets]; a list of runs plots its first run
	:param result:	a MultiviewResult, or predicted data with the same shape as the true data
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	plot_prediction(Y_true, result, title = title, block = block)


def _plot_sweep(x: np.ndarray, lines: np.ndarray, labels: Optional[List[str]], xlabel: str, title: str, block: bool):
	"""
	Plot one performance curve per line against the sweep values.

	:param x:	sweep values, [nX]
	:param lines:	one performance curve per line, [nLines, nX]
	:param labels:	one legend entry per line; None for no legend
	:param xlabel:	name of the sweep value
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	plt.figure()
	for lineIndex in range(lines.shape[0]):
		plt.plot(x, lines[lineIndex], 'o-', linewidth = 2, markersize = 8,
				 label = labels[lineIndex] if labels is not None else None)
	if labels is not None:
		plt.legend()
	plt.xlabel(xlabel)
	plt.ylabel('Prediction skill')
	plt.title(title)
	plt.grid(True, alpha = 0.3)
	plt.show(block = block)


def _SweepWithLeadingColumn(result, xlabel: str, title: str, block: bool):
	"""
	Plot a sweep table holding, per sweep value, the value itself and then the performance per target.

	:param result:	sweep table, [nX, 1 + nTargets]
	:param xlabel:	name of the sweep value
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	result = np.asarray(result)
	lines = result[:, 1:].T
	labels = [f'target {t}' for t in range(lines.shape[0])] if lines.shape[0] > 1 else None
	_plot_sweep(result[:, 0], lines, labels, xlabel, title, block)


def plot_embed_dimension(scores, title: str = "", block: bool = True):
	"""
	Plot the performance from FindOptimalEmbeddingDimensionality against embedding dimension
	1..maxDims.

	:param scores:	[maxDims] for one target; [nTargets, maxDims] for several targets, or [nVariables, maxDims] from the per-variable sweep of one target; [nTargets, nVariables, maxDims] from the per-variable sweep of several targets. One line per leading index
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	scores = np.asarray(scores)
	if scores.ndim == 1:
		lines, labels = scores[None, :], None
	elif scores.ndim == 2:
		lines = scores
		labels = [f'line {i}' for i in range(lines.shape[0])] if lines.shape[0] > 1 else None
	elif scores.ndim == 3:
		nTargets, nColumns, maxDims = scores.shape
		lines = scores.reshape(nTargets * nColumns, maxDims)
		labels = [f'target {t}, column {c}' for t in range(nTargets) for c in range(nColumns)]
	else:
		raise ValueError(f'scores has {scores.ndim} axes; expected 1, 2, or 3')
	x = np.arange(1, lines.shape[1] + 1)
	_plot_sweep(x, lines, labels, 'Embedding dimensions', title or "Embedding dimensions", block)


def plot_predict_interval(result, title: str = "", block: bool = True):
	"""
	Plot the performance from FindOptimalPredictionHorizon against the horizon.

	:param result:	sweep table, [maxHorizon, 1 + nTargets]: per horizon, the horizon itself and then the performance per target
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	_SweepWithLeadingColumn(result, 'Prediction horizon', title or "Prediction horizon", block)


def plot_predict_nonlinear(result, title: str = "", block: bool = True):
	"""
	Plot the performance from FindSMapNeighborhood against the localization strength.

	:param result:	sweep table, [nTheta, 1 + nTargets]: per localization strength, theta itself and then the performance per target
	:param title:	plot title
	:param block:	True keeps the script at plt.show until the window closes; False returns at once
	"""
	_SweepWithLeadingColumn(result, 'Localization (theta)', title or "Localization", block)
