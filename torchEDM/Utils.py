"""
Auxiliary functions.

ComputeError    Pearson correlation, RMSE, MAE, CAE
Iterable        Is an object iterable?
IsIterable      Is an object iterable and not a string?
SurrogateData   ebisuzaki, random shuffle, seasonal
PlotObsPred     Plot observations & predictions
PlotCoef        Plot s-map coefficients
Examples        Canonical examples
"""

from cmath import exp
# python modules
from math import floor, pi, sqrt, cos
from random import sample, uniform, normalvariate

import numpy as np
# package modules
from numpy import absolute, arange, fft
from numpy import mean, ptp, std, sqrt, zeros
from scipy.interpolate import UnivariateSpline

from torchEDM.Scoring import Correlation, MaxAbsoluteError, SumAbsoluteError, RootMeanSquareError


def Iterable( obj ):
	"""
	Is an object iterable?

	:param obj: Object to check
	:return: True if object is iterable, False otherwise
	"""

	try:
		it = iter( obj )
	except TypeError: 
		return False
	return True



def IsNonStringIterable(obj):
	"""
	Is an object iterable and not a string?

	:param obj: Object to check
	:return: True if object is iterable and not a string, False otherwise
	"""

	if Iterable( obj ) :
		if isinstance( obj, str ) :
			return False
		else :
			return True
	return False



def SurrogateData( data     = None,
				   column        = None,
				   method        = 'ebisuzaki',
				   numSurrogates = 10,
				   alpha         = None,
				   smooth        = 0.8,
				   outputFile    = None ):
	"""
	Three methods:

	random_shuffle :
	  Sample the data with a uniform distribution.

	ebisuzaki :
	  Journal of Climate. A Method to Estimate the Statistical Significance
	  of a Correlation When the Data Are Serially Correlated.
	  https://doi.org/10.1175/1520-0442(1997)010<2147:AMTETS>2.0.CO;2

	  Presumes data are serially correlated with low pass coherence. It is:
	  "resampling in the frequency domain. This procedure will not preserve
	  the distribution of values but rather the power spectrum (periodogram).
	  The advantage of preserving the power spectrum is that resampled series
	  retains the same autocorrelation as the original series."

	seasonal :
	  Presume a smoothing spline represents the seasonal trend.
	  Each surrogate is a summation of the trend, resampled residuals,
	  and possibly additive Gaussian noise. Default noise has a standard
	  deviation that is the data range / 5.

	:param data: Data array
	:param column: Column index
	:param method: Method to use ('random_shuffle', 'ebisuzaki', 'seasonal')
	:param numSurrogates: Number of surrogates to generate
	:param alpha: Standard deviation for Gaussian noise
	:param smooth: Smoothing factor for seasonal method
	:param outputFile: Output file path (optional)
	:return: Result array with time column and surrogate data
	"""

	if data is None :
		raise RuntimeError( "SurrogateData() empty data array." )

	if column is None :
		raise RuntimeError( "SurrogateData() must specify column index." )

	# Extract time column (column 0) and data column
	time_col = data[:, 0]
	data_col = data[:, column]

	# Initialize result array: (n_samples, numSurrogates + 1)
	# Column 0: time, Columns 1+: surrogate data
	result = zeros((data.shape[0], numSurrogates + 1))
	result[:, 0] = time_col  # Time column

	if method.lower() == "random_shuffle" :
		for s in range( numSurrogates ) :
			# Random shuffle of the data column
			surr = data_col.copy()
			np.random.shuffle(surr)
			result[:, s + 1] = surr

	elif method.lower() == "ebisuzaki" :
		n             = data.shape[0]
		n2            = floor( n/2 )
		mu            = mean   ( data_col )
		sigma         = std    ( data_col )
		a             = fft.fft( data_col )
		amplitudes    = absolute( a )
		amplitudes[0] = 0

		for s in range( numSurrogates ) :
			thetas      = [ 2 * pi * uniform( 0, 1 ) for x in range( n2 - 1 )]
			revThetas   = thetas[::-1]
			negThetas   = [ -x for x in revThetas ]
			angles      = [0] + thetas + [0] + negThetas
			surrogate_z = [ A * exp( complex( 0, theta ) )
							for A, theta in zip( amplitudes, angles ) ]

			if n % 2 == 0 : # even length
				surrogate_z[-1] = complex( sqrt(2) * amplitudes[-1] *
										   cos( 2 * pi * uniform(0,1) ) )

			ifft = fft.ifft( surrogate_z ) / n

			realifft = [ x.real for x in ifft ]
			sdevifft = std( realifft )

			# adjust variance of surrogate time series to match original
			scaled = [ sigma * x / sdevifft for x in realifft ]

			result[:, s + 1] = scaled

	elif method.lower() == "seasonal" :
		y = data_col
		n = data.shape[0]

		# Presume a spline captures the seasonal cycle
		x      = arange( n )
		spline = UnivariateSpline( x, y )
		spline.set_smoothing_factor( smooth )
		y_spline = spline( x )

		# Residuals of the smoothing
		residual = list( y - y_spline )

		# spline plus shuffled residuals plus Gaussian noise
		noise = zeros( n )

		# If no noise specified, set std dev to data range / 5
		if alpha is None :
			alpha = ptp( y ) / 5

		for s in range( numSurrogates ) :
			noise = [ normalvariate( 0, alpha ) for z in range( n ) ]

			result[:, s + 1] = y_spline + sample( residual, n ) + noise

	else :
		raise RuntimeError( "SurrogateData() invalid method." )

	# Round to 8 decimal places
	result = result.round( 8 )

	if outputFile :
		# Save as CSV with column names
		import csv
		with open(outputFile, 'w', newline='') as f:
			writer = csv.writer(f)
			# Write header
			header = ['Time'] + [f'Column_{column}_{s+1}' for s in range(numSurrogates)]
			writer.writerow(header)
			# Write data
			writer.writerows(result)

	return result



def PlotObsPred( data, dataName = "", embedDimensions = 0, predictionHorizon = 0, block = True ):
	"""
	Plot observations and predictions.

	:param data: numpy array with shape (n_samples, 4). 
		Column 0: Time, 
		Column 1: Observations, 
		Column 2: Predictions, 
		Column 3: Pred_Variance
	:param dataName: Plot title
	:param embedDimensions: Embedding dimensions
	:param predictionHorizon: Prediction horizon
	:param block: Whether to block execution
	"""
	import matplotlib.pyplot as plt

	# stats: {'MAE': 0., 'RMSE': 0., 'correlation': 0. }
	stats = [Correlation(data[:, 1], data[:, 2]),
			 MaxAbsoluteError(data[:, 1], data[:, 2]),
			 SumAbsoluteError(data[:, 1], data[:, 2]),
			 RootMeanSquareError(data[:, 1], data[:, 2])]

	title = dataName + "\nEmbedding Dims = " + str(embedDimensions) + " predictionHorizon=" + str(predictionHorizon) +\
			"  correlation="   + str( round( stats[0],  3 ) )   +\
			" RMSE=" + str( round( stats[3], 3 ) )

	plt.figure()
	plt.plot(data[:, 0], data[:, 1], label='Observations', linewidth=3)
	plt.plot(data[:, 0], data[:, 2], label='Predictions', linewidth=3)
	plt.title(title)
	plt.legend()
	plt.show(block=block)



def PlotCoeff( data, dataName = "", embedDimensions = 0, predictionHorizon = 0, block = True ):
	"""
	Plot S-Map coefficients.

	:param data: numpy array with shape (n_samples, n_coeff + 1). Column 0: Time, Columns 1+: coefficients
	:param dataName: Plot title
	:param embedDimensions: Embedding dimensions
	:param predictionHorizon: Prediction horizon
	:param block: Whether to block execution
	"""
	import matplotlib.pyplot as plt

	title = dataName + "\nEmbedding Dims = " + str(embedDimensions) + " predictionHorizon=" + str(predictionHorizon) +\
			"  S-Map Coefficients"

	plt.figure()
	for i in range(1, data.shape[1]):
		plt.subplot(data.shape[1] - 1, 1, i)
		plt.plot(data[:, 0], data[:, i], linewidth=3)
		plt.title(f'Coefficient {i-1}')
	plt.suptitle(title)
	plt.tight_layout()
	plt.show(block=block)
