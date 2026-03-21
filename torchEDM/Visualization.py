"""
Visualization functions for results.
These plotting functions are mostly factored out of the pyEDM math functions
to separate the math and visualization
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import show, axhline
from typing import Union
import numpy as np

def plot_prediction(result: Union['SimplexResult', 'SMapResult', 'MultiviewResult', np.ndarray],
				   title: str = "",
				   embedDimensions: int = None,
				   predictionHorizon: int = None,
				   block: bool = True):
	"""
	Plot observations vs predictions.

	:param result: Result object or legacy numpy array with columns [Time, Observations, Predictions]
	:param title: Additional title text
	:param embedDimensions: Embedding dimension (only needed if passing numpy array)
	:param predictionHorizon: Prediction horizon (only needed if passing numpy array)
	:param block: Whether to block execution when showing plot
	"""
	from .Scoring import Correlation, RootMeanSquareError

	# Handle both Result objects and numpy arrays
	if hasattr(result, 'projection'):
		# It's a Result object
		data = result.projection
		E = result.embedDimensions
		Tp = result.predictionHorizon
	else:
		# It's a numpy array (legacy)
		data = result
		E = embedDimensions or 0
		Tp = predictionHorizon or 0

	# Compute error statistics
	corr = Correlation(data[:, 1], data[:, 2])
	RMSE = RootMeanSquareError(data[:, 1], data[:, 2])

	# Build title
	plot_title = title
	if plot_title:
		plot_title += "\n"
	plot_title += f"Embedding Dims = {E}  predictionHorizon={Tp}  " \
				  f"correlation={round(corr, 3)}  " \
				  f"RMSE={round(RMSE, 3)}"

	# Create plot
	plt.figure()
	plt.plot(data[:, 0], data[:, 1], label='Observations', linewidth=3)
	plt.plot(data[:, 0], data[:, 2], label='Predictions', linewidth=3)
	plt.title(plot_title)
	plt.legend()
	plt.show(block=block)

def plot_smap_coefficients(result: Union['SMapResult', np.ndarray],
						  title: str = "",
						  embedDimensions: int = None,
						  predictionHorizon: int = None,
						  block: bool = True):
	"""
	Plot S-Map coefficients over time.

	:param result: SMap result object or legacy numpy array with columns [Time, Coeff_0, Coeff_1, ...]
	:param title: Additional title text
	:param embedDimensions: Embedding dimension (only needed if passing numpy array)
	:param predictionHorizon: Prediction horizon (only needed if passing numpy array)
	:param block: Whether to block execution when showing plot
	"""
	# Handle both SMapResult and numpy arrays
	if hasattr(result, 'coefficients'):
		# It's an SMapResult object
		data = result.coefficients
		E = result.embedDimensions
		Tp = result.predictionHorizon
	else:
		# It's a numpy array (legacy)
		data = result
		E = embedDimensions or 0
		Tp = predictionHorizon or 0

	# Build title
	plot_title = title
	if plot_title:
		plot_title += "\n"
	plot_title += f"Embedding Dims = {E}  predictionHorizon={Tp}  S-Map Coefficients"

	# Create subplots for each coefficient
	n_coeff = data.shape[1] - 1 if data.shape[1] > 1 else data.shape[1]

	plt.figure()
	for i in range(1, data.shape[1]):
		plt.subplot(data.shape[1] - 1, 1, i)
		plt.plot(data[:, 0], data[:, i], linewidth=3)
		plt.title(f'Coefficient {i-1}')

	plt.suptitle(plot_title)
	plt.tight_layout()
	plt.show(block=block)

def plot_ccm(result: Union['CCMResult', np.ndarray],
			title: str = "",
			embedDimensions: int = None,
			block: bool = True):
	"""
	Plot CCM convergence.

	:param result: CCM result object or legacy numpy array with columns [LibSize, Correlation_1, Correlation_2]
	:param title: Additional title text
	:param embedDimensions: Embedding dimension (only needed if passing numpy array)
	:param block: Whether to block execution when showing plot
	"""
	# Handle both CCMResult and numpy arrays
	if hasattr(result, 'libMeans'):
		# It's a CCMResult object
		data = result.libMeans
		E = result.embedDimensions
	else:
		# It's a numpy array (legacy)
		data = result
		E = embedDimensions or 0

	# Build title
	plot_title = title or f'E = {E}'

	fig, ax = plt.subplots()

	# Check if we have two directions or one
	if data.shape[1] == 3:
		# CCM of two different variables
		ax.plot(data[:, 0], data[:, 1], linewidth=3, label='Direction 1')
		ax.plot(data[:, 0], data[:, 2], linewidth=3, label='Direction 2')
		ax.legend()
	elif data.shape[1] == 2:
		# CCM of degenerate columns (single direction)
		ax.plot(data[:, 0], data[:, 1], linewidth=3)

	ax.set(xlabel="Library Size",
		  ylabel="CCM correlation",
		  title=plot_title)
	axhline(y=0, linewidth=1)
	show(block=block)

def plot_multiview(result: Union['MultiviewResult', np.ndarray],
				  title: str = "",
				  block: bool = True):
	"""
	Plot Multiview ensemble prediction.

	:param result: Multiview result object or legacy numpy array
	:param title: Additional title text
	:param block: Whether to block execution when showing plot
	"""
	# Use plot_prediction for the ensemble result
	if hasattr(result, 'projection'):
		plot_prediction(result, title=title, block=block)
	else:
		plot_prediction(result, title=title, block=block)

# Legacy function names for backward compatibility
def PlotObsPred(data, dataName="", embedDimensions=0, predictionHorizon=0, block=True):
	"""
	Legacy function for plotting observations vs predictions.

	.. deprecated:: Use plot_prediction() instead.
	"""
	plot_prediction(data, title=dataName, embedDimensions=embedDimensions,
				   predictionHorizon=predictionHorizon, block=block)

def PlotCoeff(data, dataName="", embedDimensions=0, predictionHorizon=0, block=True):
	"""
	Legacy function for plotting S-Map coefficients.

	.. deprecated:: Use plot_smap_coefficients() instead.
	"""
	plot_smap_coefficients(data, title=dataName, embedDimensions=embedDimensions,
						  predictionHorizon=predictionHorizon, block=block)

def plot_embed_dimension(result: np.ndarray,
						title: str = "",
						block: bool = True):
	"""
	Plot embedding dimension vs prediction skill.

	:param result: Array with columns [E, correlation]
	:param title: Plot title
	:param block: Whether to block execution when showing plot
	"""
	plot_title = title or "Embedding Dimension"

	plt.figure()
	plt.plot(result[:, 0], result[:, 1], 'o-', linewidth=2, markersize=8)
	plt.xlabel('Embedding Dimension (E)')
	plt.ylabel('Prediction Skill (correlation)')
	plt.title(plot_title)
	plt.grid(True, alpha=0.3)
	plt.show(block=block)

def plot_predict_interval(result: np.ndarray,
						 title: str = "",
						 block: bool = True):
	"""
	Plot prediction interval vs prediction skill.

	:param result: Array with columns [predictionHorizon, correlation]
	:param title: Plot title
	:param block: Whether to block execution when showing plot
	"""
	plot_title = title or "Prediction Interval"

	plt.figure()
	plt.plot(result[:, 0], result[:, 1], 'o-', linewidth=2, markersize=8)
	plt.xlabel('Prediction Horizon (Tp)')
	plt.ylabel('Prediction Skill (correlation)')
	plt.title(plot_title)
	plt.grid(True, alpha=0.3)
	plt.show(block=block)

def plot_predict_nonlinear(result: np.ndarray,
						  title: str = "",
						  block: bool = True):
	"""
	Plot theta vs prediction skill for S-Map.

	:param result: Array with columns [theta, correlation]
	:param title: Plot title
	:param block: Whether to block execution when showing plot
	"""
	plot_title = title or "S-Map Localization (theta)"

	plt.figure()
	plt.plot(result[:, 0], result[:, 1], 'o-', linewidth=2, markersize=8)
	plt.xlabel('S-Map Localization (theta)')
	plt.ylabel('Prediction Skill (correlation)')
	plt.title(plot_title)
	plt.grid(True, alpha=0.3)
	plt.show(block=block)
