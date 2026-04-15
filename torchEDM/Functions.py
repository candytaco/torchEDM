"""
Function API that follows the syntax of the pyEDM API
While this package has been refactored to take/return numpy arrays, these arrays
are roughly structured to be similar to the dataframes returned by the original pyEDM functions
"""

# python modules
from typing import Any, Dict, List, Tuple, Union
import numpy

# local modules
from .EDM.CCM import CCM
from .EDM.Multiview import Multiview
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex


def FitSimplex(data: numpy.ndarray,
			   columns: List[int] = None,
			   target: int = None,
			   train: Tuple[int, int] = None,
			   test: Tuple[int, int] = None,
			   embedDimensions: int = 0,
			   predictionHorizon: int = 1,
			   knn: int = 0,
			   step: int = -1,
			   exclusionRadius: float = 0,
			   embedded: bool = False,
			   validLib: List = [],
			   noTime: bool = False,
			   generateSteps: int = 0,
			   generateConcat: bool = False,
			   verbose: bool = False,
			   ignoreNan: bool = True,
			   returnObject: bool = False) -> Union[numpy.ndarray, Simplex]:
	"""
	Simplex prediction.

	:param data: 			2D numpy array where column 0 is time
	:param columns: 		Column indices to use for embedding (defaults to all except time)
	:param target: 			Target column index (defaults to column 1)
	:param train: 			Train indices [start, end]
	:param test: 			Test indices [start, end]
	:param embedDimensions: Embedding dimension
	:param predictionHorizon: Prediction horizon
	:param knn: 			Number of nearest neighbors
	:param step: 			Step size for embedding
	:param exclusionRadius: Exclusion radius
	:param embedded: 		Whether data is already embedded
	:param validLib: 		Valid library indices
	:param noTime: 			Whether to exclude time column
	:param generateSteps: 	Number of generation steps
	:param generateConcat: 	Whether to concatenate generated predictions
	:param verbose: 		Print diagnostic messages
	:param ignoreNan: 		Whether to ignore NaN values
	:param returnObject: 	Whether to return Simplex object instead of projection
	:return: Prediction projection array or Simplex object
	"""

	# Instantiate SimplexClass object
	S = Simplex(data = data,
				columns = columns,
				target = target,
				train = train,
				test = test,
				embedDimensions = embedDimensions,
				predictionHorizon = predictionHorizon,
				knn = knn,
				step = step,
				exclusionRadius = exclusionRadius,
				embedded = embedded,
				validLib = validLib,
				noTime = noTime,
				generateSteps = generateSteps,
				generateConcat = generateConcat,
				ignoreNan = ignoreNan,
				verbose = verbose)

	if generateSteps:
		result = S.Generate()
	else:
		result = S.Run()

	if returnObject:
		return S
	else:
		return result.projection


def FitSMap(data: numpy.ndarray,
			columns: List[int] = None,
			target: int = None,
			train: Tuple[int, int] = None,
			test: Tuple[int, int] = None,
			embedDimensions: int = 0,
			predictionHorizon: int = 1,
			knn: int = 0,
			step: int = -1,
			theta: float = 0,
			exclusionRadius: float = 0,
			embedded: bool = False,
			validLib: List = [],
			noTime: bool = False,
			generateSteps: int = 0,
			generateConcat: bool = False,
			ignoreNan: bool = True,
			verbose: bool = False,
			returnObject: bool = False) -> Union[Dict[str, Any], SMap]:
	"""
	S-Map prediction.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use for embedding (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param theta: 				Localization parameter
	:param exclusionRadius: 	Exclusion radius
	:param solver: 				solver
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param generateSteps: 		Number of generation steps
	:param generateConcat: 		Whether to concatenate generated predictions
	:param verbose: 			Print diagnostic messages
	:param ignoreNan: 			Whether to ignore NaN values
	:param returnObject: 		Whether to return SMap object instead of prediction dict
	:return: Dictionary with predictions, coefficients, and singular values or SMap object
	"""

	# Instantiate SMapClass object
	S = SMap(data = data,
			 columns = columns,
			 target = target,
			 train = train,
			 test = test,
			 embedDimensions = embedDimensions,
			 predictionHorizon = predictionHorizon,
			 knn = knn,
			 step = step,
			 theta = theta,
			 exclusionRadius = exclusionRadius,
			 embedded = embedded,
			 validLib = validLib,
			 noTime = noTime,
			 generateSteps = generateSteps,
			 generateConcat = generateConcat,
			 ignoreNan = ignoreNan,
			 verbose = verbose)

	if generateSteps:
		result = S.Generate()
	else:
		result = S.Run()

	if returnObject:
		return S
	else:
		SMapDict = {'predictions': result.projection,
					'coefficients': S.Coefficients,
					'singularValues': S.SingularValues}
		return SMapDict


def FitCCM(data: numpy.ndarray,
		   columns: List[int] = None,
		   target: int = None,
		   trainSizes: Any = None,
		   sample: int = 0,
		   embedDimensions: int = 0,
		   predictionHorizon: int = 0,
		   knn: int = 0,
		   step: int = -1,
		   exclusionRadius: float = 0,
		   seed: Any = None,
		   embedded: bool = False,
		   validLib: List = [],
		   includeData: bool = False,
		   noTime: bool = False,
		   ignoreNan: bool = True,
		   mpMethod: Any = None,
		   sequential: bool = False,
		   verbose: bool = False,
		   returnObject: bool = False,
		   kdTree: bool = False) -> Union[Dict[str, Any], CCM]:
	"""
	Convergent Cross Mapping.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param trainSizes: 			Library sizes to evaluate
	:param sample: 				Sample size for each library
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param seed: 				Random seed
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param includeData: 		Whether to include detailed prediction statistics
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param mpMethod: 			Multiprocessing method
	:param sequential: 			Whether to run sequentially
	:param verbose: 			Print diagnostic messages
	:param returnObject: 		Whether to return CCM object instead of libMeans
	:param kdTree:				Use kdtree for neighbors
	:return: Library means array or CCM object
	"""

	# Instantiate CCMClass object
	C = CCM(data = data,
			columns = columns,
			target = [target],
			trainSizes = trainSizes,
			sample = sample,
			embedDimensions = embedDimensions,
			predictionHorizon = predictionHorizon,
			knn = knn,
			step = step,
			exclusionRadius = exclusionRadius,
			seed = seed,
			embedded = embedded,
			validLib = validLib,
			includeData = includeData,
			noTime = noTime,
			ignoreNan = ignoreNan,
			mpMethod = mpMethod,
			sequential = sequential,
			verbose = verbose,
			kdTree = kdTree)

	# Embedding of Forward & Reverse mapping
	C.FwdMap.EmbedData()
	C.FwdMap.RemoveNan()
	C.RevMap.EmbedData()
	C.RevMap.RemoveNan()

	result = C.Run()

	if returnObject:
		return C
	else:
		if includeData:
			return {'LibMeans': result.libMeans,
					'PredictStats1': result.predictStats1,
					'PredictStats2': result.predictStats2}
		else:
			return result.libMeans


def FitMultiview(data: numpy.ndarray,
				 columns: List[int] = None,
				 target: int = None,
				 train: Tuple[int, int] = None,
				 test: Tuple[int, int] = None,
				 D: int = 0,
				 embedDimensions: int = 1,
				 predictionHorizon: int = 1,
				 knn: int = 0,
				 step: int = -1,
				 multiview: int = 0,
				 exclusionRadius: float = 0,
				 trainLib: bool = True,
				 excludeTarget: bool = False,
				 ignoreNan: bool = True,
				 verbose: bool = False,
				 device: Any = None,
				 dtype: Any = None,
				 batchSize: int = 1000,
				 returnObject: bool = False) -> Union[Dict[str, Any], Multiview]:
	"""
	Multiview prediction.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param D: 					State-space dimension
	:param embedDimensions: 	Embedding dimension for each variable
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param multiview: 			Multiview parameter
	:param exclusionRadius: 	Exclusion radius
	:param trainLib: 			Whether to use training library
	:param excludeTarget: 		Whether to exclude target from columns
	:param ignoreNan: 			Whether to ignore NaN values
	:param verbose: 			Print diagnostic messages
	:param device: 				torch device to use (None for auto-detect)
	:param dtype: 				torch dtype to use (None for float64)
	:param batchSize: 			Number of column combinations to process per batch
	:param returnObject: 		Whether to return Multiview object instead of prediction dict
	:return: Dictionary with predictions and view rankings or Multiview object
	"""

	# Instantiate MultiviewClass object
	M = Multiview(data = data,
				  columns = columns,
				  target = target,
				  train = train,
				  test = test,
				  D = D,
				  embedDimensions = embedDimensions,
				  predictionHorizon = predictionHorizon,
				  knn = knn,
				  step = step,
				  multiview = multiview,
				  exclusionRadius = exclusionRadius,
				  trainLib = trainLib,
				  excludeTarget = excludeTarget,
				  ignoreNan = ignoreNan,
				  verbose = verbose,
				  # device = device,
				  # dtype = dtype,
				  # batchSize = batchSize
				  )

	result = M.Run()

	if returnObject:
		return M
	else:
		return {'Predictions': result.projection, 'View': result.view}
