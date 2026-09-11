"""
Examples of the parameter-holding wrappers on the packaged sample data.
"""
from .ExampleData import sampleData
from .Fitters.CCMFitter import CCMFitter
from .Fitters.MultiviewFitter import MultiviewFitter
from .Fitters.SimplexFitter import SimplexFitter
from .Fitters.SMapFitter import SMapFitter
from .Visualization import plot_prediction, plot_smap_coefficients, plot_ccm


def FitterExamples():
	"""
	Every sample array has a time column at index 0, which the predictors never see.
	The test arrays start two rows before the rows of interest so those rows have history.
	"""
	# 1: three columns as the state, no history stacking
	data = sampleData["block_3sp"]
	X_train, Y_train = data[0:100, [1, 4, 7]], data[0:100, 1]
	X_test, Y_test = data[100:196, [1, 4, 7]], data[100:196, 1]
	result = SimplexFitter(EmbedDimensions = 1, PredictionHorizon = 1).Fit(X_train, Y_train, X_test, Y_test)
	plot_prediction(Y_test, result, "Simplex: block_3sp, three columns as the state")

	# 2: one column stacked to three copies
	X_train, Y_train = data[0:100, 1], data[0:100, 1]
	X_test, Y_test = data[98:196, 1], data[98:196, 1]
	result = SimplexFitter(EmbedDimensions = 3, PredictionHorizon = 1).Fit(X_train, Y_train, X_test, Y_test)
	plot_prediction(Y_test, result, "Simplex: block_3sp, one column stacked to 3 embedding dimensions")

	# 3: ensemble over combinations of the stacked columns
	X_train, Y_train = data[0:100, [1, 4, 7]], data[0:100, 1]
	X_test, Y_test = data[98:199, [1, 4, 7]], data[98:199, 1]
	result = MultiviewFitter(EmbedDimensions = 3, PredictionHorizon = 1, IsRankedInSample = False).Fit(X_train, Y_train, X_test, Y_test)
	plot_prediction(Y_test, result, "Multiview: block_3sp")

	# 4: locally weighted linear map on two columns
	data = sampleData["circle"]
	X_train, Y_train = data[0:100, [1, 2]], data[0:100, 1]
	X_test, Y_test = data[100:190, [1, 2]], data[100:190, 1]
	result = SMapFitter(EmbedDimensions = 1, PredictionHorizon = 1, Theta = 4.0).Fit(X_train, Y_train, X_test, Y_test)
	plot_prediction(Y_test, result, "S-Map: circle")
	plot_smap_coefficients(result, "S-Map coefficients: circle")

	# 5: cross-map skill of anchovy onto sea-surface temperature across training-subset sizes
	data = sampleData["sardine_anchovy_sst"]
	result = CCMFitter(TrainSizes = [10, 20, 30, 40, 50, 60, 70, 75], numRepeats = 50, EmbedDimensions = 3,
					   PredictionHorizon = 0, progressBar = False).Fit(data[:, 1], data[:, 4])
	plot_ccm(result, "CCM: anchovy -> sst")
