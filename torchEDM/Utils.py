"""
Auxiliary functions.

Iterable        Is an object iterable?
IsNonStringIterable   Is an object iterable and not a string?
SurrogateData   ebisuzaki, random shuffle, seasonal
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



def Iterable( obj ):
	"""
	Whether an object can be iterated over.

	:param obj:	any object
	:return: True when iter(obj) succeeds
	"""

	try:
		it = iter( obj )
	except TypeError: 
		return False
	return True



def IsNonStringIterable(obj):
	"""
	Whether an object can be iterated over and is not a string.

	:param obj:	any object
	:return: True for iterables other than str
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
	Surrogate series of one column for significance testing, by one of three methods:

	random_shuffle: the column's values in random order, which keeps the distribution and
	destroys the serial correlation.

	ebisuzaki (Journal of Climate, doi.org/10.1175/1520-0442(1997)010<2147:AMTETS>2.0.CO;2):
	the column's Fourier amplitudes with random phases, which keeps the power spectrum, and so
	the autocorrelation, but not the distribution of values; each surrogate is rescaled to the
	column's standard deviation.

	seasonal: a smoothing spline is taken as the seasonal trend; each surrogate is the trend
	plus the residuals in random order plus Gaussian noise.

	:param data:	[nSamples, nColumns] array whose column 0 is copied into the output as is
	:param column:	index of the column to resample
	:param method:	'random_shuffle', 'ebisuzaki', or 'seasonal'
	:param numSurrogates:	surrogate series to generate
	:param alpha:	standard deviation of the Gaussian noise in 'seasonal'; None uses the column's range divided by 5
	:param smooth:	smoothing factor of the spline in 'seasonal'
	:param outputFile:	optional CSV path the result is also written to, with a header
	:return: [nSamples, 1 + numSurrogates]: column 0 of data, then one surrogate per column
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
