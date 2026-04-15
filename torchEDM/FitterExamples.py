"""
Examples demonstrating the sklearn-like wrapper classes
"""
from .Fitters.CCMFitter import CCMFitter
from .ExampleData import sampleData
from .Fitters.MultiviewFitter import MultiviewFitter
from .Fitters.SMapFitter import SMapFitter
from .Fitters.SimplexFitter import SimplexFitter
from .Visualization import (plot_prediction, plot_smap_coefficients, plot_ccm)


def FitterExamples():
	"""
	Examples using the new wrapper classes with sklearn-like separate arrays.
	"""


	# Example 1: SimplexWrapper with block_3sp data (embedded = True)
	print("Example 1: Simplex with block_3sp data (embedded = True)")

	# Split data into separate arrays
	data = sampleData["block_3sp"]
	XTrain = data[1:100, [1, 4, 7]]  # Columns 1, 4, 7 (features), rows 1-99
	YTrain = data[1:100, [1]]  # Target column 1, rows 1-99
	XTest = data[100:196, [1, 4, 7]]  # Columns 1, 4, 7, rows 100-195
	YTest = data[100:196, [1]]  # Target column 1, rows 100-195

	# Create and run SimplexWrapper
	simplexWrapper = SimplexFitter(
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Verbose = False,
		Embedded = True
	)

	result = simplexWrapper.Fit(XTrain = XTrain,
								YTrain = YTrain,
								XTest = XTest,
								YTest = YTest, )
	plot_prediction(result.projection, "Simplex: block_3sp embedded", embedDimensions = 3)

	# Example 2: SimplexWrapper with block_3sp data (embedded = False)
	print("\nExample 2: Simplex with block_3sp data (embedded = False)")

	data = sampleData["block_3sp"]
	XTrain = data[1:100, [1]]  # Column 1 only, rows 1-99
	YTrain = data[1:100, [1]]  # Target column 1, rows 1-99
	XTest = data[100:191, [1]]  # Column 1 only, rows 105-190
	YTest = data[100:191, [1]]  # Target column 1, rows 105-190

	simplexWrapper2 = SimplexFitter(
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Verbose = False,
		Embedded = False,
	)

	result = simplexWrapper2.Fit(XTrain = XTrain,
								 YTrain = YTrain,
								 XTest = XTest,
								 YTest = YTest,
								 TestStart = 5,  # the first 5 samples are to provide a history for the first real test sample
								 )
	plot_prediction(result.projection, "Simplex: block_3sp", embedDimensions = 3)

	# Example 3: MultiviewWrapper with block_3sp data
	print("\nExample 3: Multiview with block_3sp data")

	data = sampleData["block_3sp"]
	XTrain = data[1:101, [1, 4, 7]]  # Columns 1, 4, 7, rows 1-100
	YTrain = data[1:101, [1]]  # Target column 1, rows 1-100
	XTest = data[101:199, [1, 4, 7]]  # Columns 1, 4, 7, rows 101-198
	YTest = data[101:199, [1]]  # Target column 1, rows 101-198

	multiviewWrapper = MultiviewFitter(
		dimensions = 0,
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		NumMultiview = 0,
		ExclusionRadius = 0,
		TrainLib = False,
		ExcludeTarget = False,
		Verbose = False
	)

	result = multiviewWrapper.Fit(XTrain = XTrain,
								  YTrain = YTrain,
								  XTest = XTest,
								  YTest = YTest, )
	plot_prediction(result.projection, "Multiview: block_3sp", embedDimensions = 3)

	# Example 4: SMapWrapper with circle data
	print("\nExample 4: SMap with circle data")

	data = sampleData["circle"]
	XTrain = data[1:101, [1, 2]]  # Columns 1-2 (features), rows 1-100
	YTrain = data[1:101, [1]]  # Target column 1, rows 1-100
	XTest = data[101:201, [1, 2]]  # Columns 1-2, rows 110-190
	YTest = data[101:201, [1]]  # Target column 1, rows 110-190

	# note SMAP appears to have some sort of tail-of-data problems and needs additional
	# data beyond the end of the test data

	smapWrapper = SMapFitter(
		EmbedDimensions = 2,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Theta = 4,
		Verbose = False,
		Embedded = True,
	)

	result = smapWrapper.Fit(XTrain = XTrain,
							 YTrain = YTrain,
							 XTest = XTest,
							 YTest = YTest,
							 TestStart = 9,
							 TestEnd = 10,)
	plot_prediction(result.projection, "SMap: circle", embedDimensions = 2)
	plot_smap_coefficients(result.coefficients, "SMap Coefficients", embedDimensions = 2)

	# Example 5: CCMWrapper with sardine_anchovy_sst data
	print("\nExample 5: CCM with sardine_anchovy_sst data")

	data = sampleData["sardine_anchovy_sst"]
	XTrain = data[:, [1]]  # Column 1 (sardine), all rows
	YTrain = data[:, [4]]  # Target is same

	ccmWrapper = CCMFitter(
		TrainSizes = [10, 70, 10],
		numRepeats = 50,
		EmbedDimensions = 3,
		PredictionHorizon = 0,
		Verbose = False
	)

	result = ccmWrapper.Fit(
		XTrain = XTrain,
		YTrain = YTrain,)
	plot_ccm(result, "CCM: sardine anchovy sst", embedDimensions = 3)
