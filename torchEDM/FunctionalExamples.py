import functools
import inspect
import numpy as np

import torchEDM.Hyperparameters
import torchEDM.Utils
from .ExampleData import sampleData
from . import Functions as EDM
from .Visualization import (plot_prediction, plot_smap_coefficients, plot_ccm,
                                 plot_embed_dimension, plot_predict_interval,
                                 plot_predict_nonlinear)


def print_call(func):
	"""
	Decorator that prints function calls with their arguments.

	:param func: Function to decorate
	:return: Wrapped function
	"""

	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		func_name = func.__name__

		arg_parts = []

		if args:
			sig = inspect.signature(func)
			param_names = list(sig.parameters.keys())
			for i, arg in enumerate(args):
				if i < len(param_names):
					if isinstance(arg, np.ndarray):
						arg_parts.append(f"{param_names[i]} = array(shape={arg.shape}, dtype={arg.dtype})")
					else:
						arg_parts.append(f"{param_names[i]} = {repr(arg)}")
				else:
					if isinstance(arg, np.ndarray):
						arg_parts.append(f"array(shape={arg.shape}, dtype={arg.dtype})")
					else:
						arg_parts.append(repr(arg))

		for key, value in kwargs.items():
			if isinstance(value, np.ndarray):
				arg_parts.append(f"{key} = array(shape={value.shape}, dtype={value.dtype})")
			else:
				arg_parts.append(f"{key} = {repr(value)}")

		args_str = ", ".join(arg_parts)
		print(f"EDM.{func_name}({args_str})")
		print()

		return func(*args, **kwargs)

	return wrapper


def FunctionalExamples():
	"""
	Canonical EDM API examples using new Result objects and Visualization.
	"""

	EmbedDimension = print_call(torchEDM.Hyperparameters.FindOptimalEmbeddingDimensionality)
	PredictInterval = print_call(torchEDM.Hyperparameters.FindOptimalPredictionHorizon)
	PredictNonlinear = print_call(torchEDM.Hyperparameters.FindSMapNeighborhood)
	Simplex = print_call(EDM.FitSimplex)
	Multiview = print_call(EDM.FitMultiview)
	SMap = print_call(EDM.FitSMap)
	CCM = print_call(EDM.FitCCM)

	sampleDataNames = \
		["TentMap", "TentMapNoise", "circle", "block_3sp", "sardine_anchovy_sst"]

	for dataName in sampleDataNames:
		if dataName not in sampleData:
			raise Exception("Examples(): Failed to find sample data " + \
			                dataName + " in EDM package")

	embed_result = EmbedDimension(data = sampleData["TentMap"],
	                              columns = [1], target = 1,
	                              train = [1, 100], test = [201, 500])
	plot_embed_dimension(embed_result, "TentMap Embedding Dimension")

	interval_result = PredictInterval(data = sampleData["TentMap"],
	                                  columns = [1], target = 1,
	                                  train = [1, 100], test = [201, 500], embedDimensions = 2)
	plot_predict_interval(interval_result, "TentMap Prediction Interval")

	nonlinear_result = PredictNonlinear(data = sampleData["TentMapNoise"],
	                                    columns = [1], target = 1,
	                                    train = [1, 100], test = [201, 500], embedDimensions = 2)
	plot_predict_nonlinear(nonlinear_result, "TentMapNoise Nonlinearity (theta)")

	# Tent map simplex : specify multivariable columns embedded = True
	projection1 = Simplex(data = sampleData["block_3sp"],
	                      columns = [1, 4, 7], target = 1,
	                      train = [1, 99], test = [100, 195],
	                      embedDimensions = 3, embedded = True)
	plot_prediction(projection1, "Simplex: block_3sp embedded", embedDimensions = 3)

	# Tent map simplex : Embed column x_t to embedDimensions=3, embedded = False
	projection2 = Simplex(data = sampleData["block_3sp"],
	                      columns = [1], target = 1,
	                      train = [1, 99], test = [105, 190],
	                      embedDimensions = 3)
	plot_prediction(projection2, "Simplex: block_3sp", embedDimensions = 3)

	# Multiview
	mv_result = Multiview(data = sampleData["block_3sp"],
	                      columns = [1, 4, 7], target = 1,
	                      train = [1, 100], test = [101, 198],
	                      D = 0, embedDimensions = 3, predictionHorizon = 1, multiview = 0,
	                      trainLib = False)
	plot_prediction(mv_result['Predictions'], "Multiview: block_3sp", embedDimensions = 3)

	# S-map circle : specify multivariable columns embedded = True
	smap_result = SMap(data = sampleData["circle"],
	                   columns = [1, 2], target = 1,
	                   train = [1, 100], test = [110, 190], theta = 4, embedDimensions = 2,
	                   verbose = False, embedded = True)
	plot_prediction(smap_result['predictions'], "S-Map: circle", embedDimensions = 2)
	plot_smap_coefficients(smap_result['coefficients'], "S-Map Coefficients", embedDimensions = 2)

	# CCM
	ccm_result = CCM(data = sampleData["sardine_anchovy_sst"],
	                 columns = [1], target = [4],
	                 trainSizes = [10, 70, 10], sample = 50,
	                 embedDimensions = 3, predictionHorizon = 0, verbose = False)
	plot_ccm(ccm_result, "CCM: sardine anchovy sst", embedDimensions = 3)
