"""
Result records returned by the predictors, the variable selection, and the cross-map
screen, plus ResultsIO for saving and loading them.

Every prediction field is named Y_pred and has the shape of the Y_test it was scored
against (a list of arrays when the test data came as a list of runs). Nothing here
carries time; positions count samples.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

ArrayOrList = Union[np.ndarray, List[np.ndarray]]


@dataclass(frozen = True)
class SimplexResult:
	"""
	Hold the output of the nearest-neighbor weighted-average predictor.

	:param Y_pred:	predicted data with the same shape as Y_test (a list for multiple test runs); NaN where no complete state predicts the sample
	:param variance:	weighted spread of the neighbor targets around each prediction, same shape as Y_pred
	:param score:	performance per target, [nTargets], from scoringFunction over the (Y_test, Y_pred) pairs; None when Y_test was not given
	:param embedDimensions:	number of lagged copies of each input variable in the state
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param knn:	number of nearest neighbors used
	"""
	Y_pred: ArrayOrList
	variance: ArrayOrList
	score: Optional[np.ndarray]
	embedDimensions: int
	predictionHorizon: int
	knn: int


@dataclass(frozen = True)
class SMapResult:
	"""
	Hold the output of the locally weighted linear predictor.

	:param Y_pred:	predicted data with the same shape as Y_test (a list for multiple test runs); NaN where no complete state predicts the sample
	:param variance:	weighted spread of the neighbor targets around each prediction, same shape as Y_pred
	:param coefficients:	coefficients of each predicted sample's fit, [nSamples, stateSize + 1, nTargets], intercept first; NaN where nothing was predicted
	:param singularValues:	singular values of each sample's weighted design matrix, [nSamples, stateSize + 1, nTargets]
	:param score:	performance per target, [nTargets], from scoringFunction over the (Y_test, Y_pred) pairs; None when Y_test was not given
	:param embedDimensions:	number of lagged copies of each input variable in the state
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param knn:	number of nearest neighbors used
	:param theta:	localization strength used
	"""
	Y_pred: ArrayOrList
	variance: ArrayOrList
	coefficients: ArrayOrList
	singularValues: ArrayOrList
	score: Optional[np.ndarray]
	embedDimensions: int
	predictionHorizon: int
	knn: int
	theta: float


@dataclass(frozen = True)
class MultiviewResult:
	"""
	Hold the ensemble prediction over the top-ranked combinations of state dimensions.

	:param Y_pred:	ensemble-averaged predicted data with the same shape as Y_test
	:param view:	one entry per top-ranked combination: [combination, correlation, max abs error, sum abs error, RMSE]
	:param topRankPredictions:	combination tuple -> that combination's Y_pred
	:param topRankStats:	combination tuple -> [correlation, max abs error, sum abs error, RMSE]
	:param columnsPerView:	number of state dimensions in each combination
	:param embedDimensions:	number of lagged copies of each input variable in the state
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param score:	performance per target, [nTargets], from scoringFunction over the ensemble (Y_test, Y_pred) pairs; None when Y_test was not given
	"""
	Y_pred: ArrayOrList
	view: List
	topRankPredictions: Dict
	topRankStats: Dict
	columnsPerView: int
	embedDimensions: int
	predictionHorizon: int
	score: Optional[np.ndarray] = None

	@property
	def top_combinations(self) -> List:
		"""
		Return the top-ranked combinations in rank order.

		:return: list of tuples of state-dimension indices
		"""
		return list(self.topRankPredictions.keys())

	def get_combination_stats(self, combo: tuple) -> List[float]:
		"""
		Return the statistics of one top-ranked combination.

		:param combo:	tuple of state-dimension indices, as listed by top_combinations
		:return: [correlation, max abs error, sum abs error, RMSE]; raises ValueError for a combination outside the top ranks
		"""
		if combo not in self.topRankStats:
			raise ValueError(f'Combination {combo} not in top-ranked results')
		return self.topRankStats[combo]


@dataclass(frozen = True)
class MDEResult:
	"""
	Hold the output of the greedy variable selection.

	Variable indices refer to the candidate view [input variables | target variables], so
	nCandidates is nFeatures + nTargets and a target's own index is nFeatures + targetIndex.

	:param Y_pred:	final predicted data with the same shape as Y_test, one variable per target
	:param selected_variables:	indices of the selected variables per target, [nTargets, maxVariables], padded with -1
	:param performance:	performance after each addition, [nTargets, maxVariables], padded with NaN
	:param ccm_values:	convergence slopes of the selected variables, [nTargets, maxVariables], padded with NaN
	:param stepwise_performance:	performance of every candidate at every step, [nTargets, maxVariables, nCandidates]
	:param candidate_embed_dimensions:	best embedding dimension per (target, candidate), [nTargets, nCandidates]; -1 where the search did not run
	:param candidate_peak_scores:	performance at that embedding dimension, [nTargets, nCandidates]; NaN where the search did not run
	:param candidate_slopes:	convergence slope per (target, candidate), [nTargets, nCandidates]; NaN for a candidate the run never checked, -inf for a check whose slope was NaN
	:param selected_z_scores:	z-score of each selected candidate's performance among every candidate evaluated at that step, [nTargets, maxVariables], padded with NaN; NaN also where fewer than two candidates were evaluated or they all scored alike
	:param score:	performance per target, [nTargets], from scoringFunction over the final (Y_test, Y_pred) pairs, or None
	"""
	Y_pred: Optional[ArrayOrList]
	selected_variables: np.ndarray
	performance: np.ndarray
	ccm_values: np.ndarray
	stepwise_performance: np.ndarray
	candidate_embed_dimensions: Optional[np.ndarray] = None
	candidate_peak_scores: Optional[np.ndarray] = None
	candidate_slopes: Optional[np.ndarray] = None
	selected_z_scores: Optional[np.ndarray] = None
	score: Optional[np.ndarray] = None


@dataclass(frozen = True)
class MDECVResults:
	"""
	Hold the output of the cross-validated variable selection of MDEFitterCV.

	:param fold_selected_variables:	indices of the selected variables per fold, [nFolds, nTargets, maxVariables], padded with -1
	:param fold_stepwise_performances:	performance of every candidate per fold, [nFolds, nTargets, maxVariables, nCandidates]
	:param fold_accuracies:	performance per fold and target, [nFolds, nTargets]
	:param fold_Y_pred:	predicted data of each fold's held-out data
	:param fold_selected_z_scores:	z-score of each selected candidate per fold, [nFolds, nTargets, maxVariables], padded with NaN; None in files that predate it
	:param best_fold:	index of the best fold per target, [nTargets]
	:param selected_variables:	indices of the final selected variables, [nTargets, maxVariables], padded with -1
	:param Y_pred:	final predicted data with the same shape as Y_test, or None
	:param score:	performance per target, [nTargets], from scoringFunction over the final (Y_test, Y_pred) pairs, or None
	"""
	fold_selected_variables: np.ndarray
	fold_stepwise_performances: np.ndarray
	fold_accuracies: np.ndarray
	fold_Y_pred: Optional[List[ArrayOrList]]
	best_fold: np.ndarray
	selected_variables: np.ndarray
	Y_pred: Optional[ArrayOrList] = None
	score: Optional[np.ndarray] = None
	fold_selected_z_scores: Optional[np.ndarray] = None

	@property
	def selected_stepwise_performances(self) -> np.ndarray:
		"""
		Return the performance of the variable selected at each step, per fold and target:
		fold_stepwise_performances indexed by fold_selected_variables, NaN where nothing was
		selected.

		:return: [nFolds, nTargets, maxVariables]
		"""
		selected = self.fold_selected_variables
		mask = selected >= 0
		performances = np.full(selected.shape, np.nan)
		foldIndices, targetIndices, stepIndices = np.where(mask)
		variableIndices = selected[foldIndices, targetIndices, stepIndices]
		performances[foldIndices, targetIndices, stepIndices] = self.fold_stepwise_performances[
			foldIndices, targetIndices, stepIndices, variableIndices]
		return performances


@dataclass(frozen = True)
class BatchedCCMResult:
	"""
	Hold the cross-map performance of every source variable against every target variable
	across training-subset sizes.

	:param forward_performance:	mean performance per subset size, [nSizes, nSources, nTargets] with singleton axes squeezed
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param library_sizes:	the training-subset sizes evaluated
	:param forward_embed_dimensions:	embedding dimensions used per source, [nSources] or [nSources, nTargets] when searched, else the scalar given
	"""
	forward_performance: np.ndarray
	predictionHorizon: int
	library_sizes: Union[np.ndarray, List]
	forward_embed_dimensions: Optional[Union[int, np.ndarray]] = None

	def GetVariableCorrelations(self, variableIndex: int) -> np.ndarray:
		"""
		Return the performance curve of one source variable across the subset sizes.

		:param variableIndex:	index of the source variable
		:return: [nSizes], or [nSizes, nTargets] with several targets
		"""
		return self.forward_performance[:, variableIndex]


@dataclass(frozen = True)
class CCMCVResult:
	"""
	Hold the cross-validated cross-map screen.

	:param fold_results:	BatchedCCMResult per fold
	:param fold_performances:	performance per fold, [nFolds, nSources] or [nFolds, nSources, nTargets]
	:param mean_performance:	mean over folds
	:param std_performance:	standard deviation over folds
	:param predictionHorizon:	number of samples between a state and the target it predicts
	:param fold_forward_embed_dimensions:	embedding dimensions per fold
	"""
	fold_results: List['BatchedCCMResult']
	fold_performances: Optional[np.ndarray]
	mean_performance: Optional[np.ndarray]
	std_performance: Optional[np.ndarray]
	predictionHorizon: int
	fold_forward_embed_dimensions: Optional[List] = None


class ResultsIO:
	"""
	Save and load result records as npz files or as folders of npy objects in the cloud.
	The record type is stored under 'result_type' so Load rebuilds the right class.
	"""

	@staticmethod
	def _Arrays(result) -> dict:
		"""
		Convert a record to the arrays it is stored as.

		:param result:	any result record
		:return: dict of name -> numpy array
		"""
		if isinstance(result, SimplexResult):
			return ResultsIO._SimplexArrays(result)
		if isinstance(result, SMapResult):
			return ResultsIO._SMapArrays(result)
		if isinstance(result, MultiviewResult):
			return ResultsIO._MultiviewArrays(result)
		if isinstance(result, MDEResult):
			return ResultsIO._MDEArrays(result)
		if isinstance(result, MDECVResults):
			return ResultsIO._MDECVResultsArrays(result)
		if isinstance(result, BatchedCCMResult):
			return ResultsIO._BatchedCCMArrays(result)
		if isinstance(result, CCMCVResult):
			return ResultsIO._CCMCVArrays(result)
		raise TypeError(f'Unsupported result type: {type(result).__name__}')

	@staticmethod
	def _FromData(data):
		"""
		Rebuild the record stored in a mapping of arrays.

		:param data:	name -> array, as loaded from a file or the cloud, including 'result_type'
		:return: the result record
		"""
		result_type = str(data['result_type'])
		loaders = {
			'SimplexResult': ResultsIO._LoadSimplex,
			'SMapResult': ResultsIO._LoadSMap,
			'MultiviewResult': ResultsIO._LoadMultiview,
			'MDEResult': ResultsIO._LoadMDE,
			'MDECVResults': ResultsIO._LoadMDECVResults,
			'BatchedCCMResult': ResultsIO._LoadBatchedCCM,
			'CCMCVResult': ResultsIO._LoadCCMCV}
		if result_type not in loaders:
			raise ValueError(f'Unknown result type: {result_type}')
		return loaders[result_type](data)

	@staticmethod
	def Save(result, path: str) -> None:
		"""
		Write a record to an npz file.

		:param result:	any result record
		:param path:	output file path (.npz is appended when absent)
		"""
		arrays = ResultsIO._Arrays(result)
		arrays['result_type'] = np.array(type(result).__name__)
		np.savez(path, **arrays)

	@staticmethod
	def Load(path: str):
		"""
		Read a record written by Save.

		:param path:	the npz file
		:return: the result record
		"""
		return ResultsIO._FromData(np.load(path, allow_pickle = True))

	@staticmethod
	def SaveToCloud(result, path: str, cloud = None) -> None:
		"""
		Upload a record as one npy object per array.

		:param result:	any result record
		:param path:	folder-like S3 path; every array is uploaded as path/key.npy
		:param cloud:	a cottoncandy interface; created when None
		"""
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface()
		arrays = ResultsIO._Arrays(result)
		arrays['result_type'] = np.array(type(result).__name__)
		path = path.rstrip('/')
		for key, value in arrays.items():
			cloud.upload_npy_array(f'{path}/{key}.npy', value)

	@staticmethod
	def DownloadFromCloud(path: str, cloud = None):
		"""
		Read a record written by SaveToCloud.

		:param path:	the folder-like S3 path given to SaveToCloud
		:param cloud:	a cottoncandy interface; created when None
		:return: the result record
		"""
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface(verbose = False)
		path = path.rstrip('/')
		data = {name.split('/')[-1].split('.')[0]: cloud.download_npy_array(name) for name in cloud.ls(path)}
		return ResultsIO._FromData(data)

	# --- run-list packing: a list of per-run arrays is stored as key_0, key_1, ... plus n_key ---

	@staticmethod
	def _PackRuns(arrays: dict, key: str, value) -> None:
		"""
		Store a per-run list as key_0, key_1, ... plus n_key, or a single array under key.

		:param arrays:	the dict being filled
		:param key:	field name
		:param value:	an array, a list of per-run arrays, or None (stored as nothing)
		"""
		if value is None:
			return
		if isinstance(value, list):
			arrays[f'n_{key}'] = np.array(len(value))
			for i, run in enumerate(value):
				arrays[f'{key}_{i}'] = run
		else:
			arrays[key] = value

	@staticmethod
	def _UnpackRuns(data, key: str):
		"""
		Read back what _PackRuns stored.

		:param data:	name -> array
		:param key:	field name
		:return: the list of per-run arrays, the single array, or None when absent
		"""
		if f'n_{key}' in data:
			return [data[f'{key}_{i}'] for i in range(int(data[f'n_{key}']))]
		return data[key] if key in data else None

	# --- arrays from records ---

	@staticmethod
	def _SimplexArrays(result: SimplexResult) -> dict:
		"""
		Convert a SimplexResult to the arrays it is stored as.

		:param result:	the SimplexResult
		:return: dict of name -> numpy array
		"""
		arrays = dict(embedDimensions = np.array(result.embedDimensions),
					  predictionHorizon = np.array(result.predictionHorizon),
					  knn = np.array(result.knn))
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		ResultsIO._PackRuns(arrays, 'variance', result.variance)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		return arrays

	@staticmethod
	def _SMapArrays(result: SMapResult) -> dict:
		"""
		Convert a SMapResult to the arrays it is stored as.

		:param result:	the SMapResult
		:return: dict of name -> numpy array
		"""
		arrays = ResultsIO._SimplexArrays(result)
		arrays['theta'] = np.array(result.theta)
		ResultsIO._PackRuns(arrays, 'coefficients', result.coefficients)
		ResultsIO._PackRuns(arrays, 'singularValues', result.singularValues)
		return arrays

	@staticmethod
	def _MultiviewArrays(result: MultiviewResult) -> dict:
		"""
		Convert a MultiviewResult to the arrays it is stored as.

		:param result:	the MultiviewResult
		:return: dict of name -> numpy array
		"""
		combos = list(result.topRankPredictions.keys())
		combo_keys = np.empty(len(combos), dtype = object)
		stats_values = np.empty(len(combos), dtype = object)
		for i, combo in enumerate(combos):
			combo_keys[i] = combo
			stats_values[i] = result.topRankStats[combo]
		view_array = np.empty(len(result.view), dtype = object)
		for i, entry in enumerate(result.view):
			view_array[i] = entry
		arrays = dict(view = view_array, combo_keys = combo_keys, topRankStats_values = stats_values,
					  n_combos = np.array(len(combos)), columnsPerView = np.array(result.columnsPerView),
					  embedDimensions = np.array(result.embedDimensions),
					  predictionHorizon = np.array(result.predictionHorizon))
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		for i, combo in enumerate(combos):
			ResultsIO._PackRuns(arrays, f'topRankPred_{i}', result.topRankPredictions[combo])
		return arrays

	@staticmethod
	def _MDEArrays(result: MDEResult) -> dict:
		"""
		Convert a MDEResult to the arrays it is stored as.

		:param result:	the MDEResult
		:return: dict of name -> numpy array
		"""
		arrays = dict(selected_variables = result.selected_variables,
					  accuracy = result.performance,
					  ccm_values = result.ccm_values,
					  stepwise_performance = result.stepwise_performance)
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		for key in ('candidate_embed_dimensions', 'candidate_peak_scores', 'candidate_slopes', 'selected_z_scores'):
			value = getattr(result, key)
			if value is not None:
				arrays[key] = np.asarray(value)
		return arrays

	@staticmethod
	def _MDECVResultsArrays(result: MDECVResults) -> dict:
		"""
		Convert a MDECVResults to the arrays it is stored as.

		:param result:	the MDECVResults
		:return: dict of name -> numpy array
		"""
		arrays = dict(fold_selected_variables = result.fold_selected_variables,
					  fold_stepwise_performances = result.fold_stepwise_performances,
					  fold_accuracies = result.fold_accuracies,
					  best_fold = result.best_fold,
					  selected_variables = result.selected_variables)
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		if result.fold_selected_z_scores is not None:
			arrays['fold_selected_z_scores'] = np.asarray(result.fold_selected_z_scores)
		if result.fold_Y_pred is not None:
			arrays['n_fold_Y_pred'] = np.array(len(result.fold_Y_pred))
			for i, foldPrediction in enumerate(result.fold_Y_pred):
				ResultsIO._PackRuns(arrays, f'fold_Y_pred_{i}', foldPrediction)
		return arrays

	@staticmethod
	def _BatchedCCMArrays(result: BatchedCCMResult) -> dict:
		"""
		Convert a BatchedCCMResult to the arrays it is stored as.

		:param result:	the BatchedCCMResult
		:return: dict of name -> numpy array
		"""
		arrays = dict(forward_performance = result.forward_performance,
					  predictionHorizon = np.array(result.predictionHorizon),
					  library_sizes = np.array(result.library_sizes))
		if result.forward_embed_dimensions is not None:
			arrays['forward_embed_dimensions'] = np.array(result.forward_embed_dimensions)
		return arrays

	@staticmethod
	def _CCMCVArrays(result: CCMCVResult) -> dict:
		"""
		Convert a CCMCVResult to the arrays it is stored as.

		:param result:	the CCMCVResult
		:return: dict of name -> numpy array
		"""
		arrays = dict(predictionHorizon = np.array(result.predictionHorizon),
					  n_folds = np.array(len(result.fold_results)))
		if result.fold_performances is not None:
			arrays['fold_performances'] = result.fold_performances
		if result.mean_performance is not None:
			arrays['mean_performance'] = result.mean_performance
		if result.std_performance is not None:
			arrays['std_performance'] = result.std_performance
		if result.fold_forward_embed_dimensions is not None:
			for i, embedDimensions in enumerate(result.fold_forward_embed_dimensions):
				arrays[f'fold_forward_embed_dimensions_{i}'] = np.array(embedDimensions)
		for i, foldResult in enumerate(result.fold_results):
			for key, value in ResultsIO._BatchedCCMArrays(foldResult).items():
				arrays[f'fold_{i}_{key}'] = value
		return arrays

	# --- records from loaded data ---

	@staticmethod
	def _LoadSimplex(data) -> SimplexResult:
		"""
		Rebuild a SimplexResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: SimplexResult
		"""
		return SimplexResult(Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
							 variance = ResultsIO._UnpackRuns(data, 'variance'),
							 score = data['score'] if 'score' in data else None,
							 embedDimensions = int(data['embedDimensions']),
							 predictionHorizon = int(data['predictionHorizon']),
							 knn = int(data['knn']))

	@staticmethod
	def _LoadSMap(data) -> SMapResult:
		"""
		Rebuild a SMapResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: SMapResult
		"""
		return SMapResult(Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
						  variance = ResultsIO._UnpackRuns(data, 'variance'),
						  coefficients = ResultsIO._UnpackRuns(data, 'coefficients'),
						  singularValues = ResultsIO._UnpackRuns(data, 'singularValues'),
						  score = data['score'] if 'score' in data else None,
						  embedDimensions = int(data['embedDimensions']),
						  predictionHorizon = int(data['predictionHorizon']),
						  knn = int(data['knn']),
						  theta = float(data['theta']))

	@staticmethod
	def _LoadMultiview(data) -> MultiviewResult:
		"""
		Rebuild a MultiviewResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: MultiviewResult
		"""
		n_combos = int(data['n_combos'])
		combo_keys = list(data['combo_keys'])
		stats_values = list(data['topRankStats_values'])
		return MultiviewResult(
			Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
			view = list(data['view']),
			topRankPredictions = {combo_keys[i]: ResultsIO._UnpackRuns(data, f'topRankPred_{i}') for i in range(n_combos)},
			topRankStats = {combo_keys[i]: stats_values[i] for i in range(n_combos)},
			columnsPerView = int(data['columnsPerView'] if 'columnsPerView' in data else data['D']),
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			score = data['score'] if 'score' in data else None)

	@staticmethod
	def _LoadMDE(data) -> MDEResult:
		"""
		Rebuild a MDEResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: MDEResult
		"""
		def optional(key):
			"""
			Read a field that older files may lack.

			:param key:	array name
			:return: the array, or None when absent
			"""
			return data[key] if key in data else None
		return MDEResult(Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
						 selected_variables = data['selected_variables'],
						 performance = data['accuracy'],
						 ccm_values = data['ccm_values'],
						 stepwise_performance = data['stepwise_performance'],
						 candidate_embed_dimensions = optional('candidate_embed_dimensions'),
						 candidate_peak_scores = optional('candidate_peak_scores'),
						 candidate_slopes = optional('candidate_slopes'),
						 selected_z_scores = optional('selected_z_scores'),
						 score = optional('score'))

	@staticmethod
	def _LoadMDECVResults(data) -> MDECVResults:
		"""
		Rebuild a MDECVResults from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: MDECVResults
		"""
		foldPredictions = None
		if 'n_fold_Y_pred' in data:
			foldPredictions = [ResultsIO._UnpackRuns(data, f'fold_Y_pred_{i}') for i in range(int(data['n_fold_Y_pred']))]
		return MDECVResults(fold_selected_variables = data['fold_selected_variables'],
							fold_stepwise_performances = data['fold_stepwise_performances'],
							fold_accuracies = data['fold_accuracies'],
							fold_Y_pred = foldPredictions,
							best_fold = data['best_fold'],
							selected_variables = data['selected_variables'],
							Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
							score = data['score'] if 'score' in data else None,
							fold_selected_z_scores = data['fold_selected_z_scores'] if 'fold_selected_z_scores' in data else None)

	@staticmethod
	def _LoadBatchedCCM(data) -> BatchedCCMResult:
		"""
		Rebuild a BatchedCCMResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: BatchedCCMResult
		"""
		return BatchedCCMResult(forward_performance = data['forward_performance'],
								predictionHorizon = int(data['predictionHorizon']),
								library_sizes = data['library_sizes'],
								forward_embed_dimensions = data['forward_embed_dimensions'] if 'forward_embed_dimensions' in data else None)

	@staticmethod
	def _LoadCCMCV(data) -> CCMCVResult:
		"""
		Rebuild a CCMCVResult from its stored arrays.

		:param data:	name -> array as written by Save or SaveToCloud
		:return: CCMCVResult
		"""
		nFolds = int(data['n_folds'])
		foldResults = []
		for i in range(nFolds):
			prefix = f'fold_{i}_'
			foldResults.append(ResultsIO._LoadBatchedCCM({key[len(prefix):]: data[key] for key in list(data) if key.startswith(prefix)}))
		foldForwardEmbedDimensions = None
		if 'fold_forward_embed_dimensions_0' in data:
			foldForwardEmbedDimensions = [data[f'fold_forward_embed_dimensions_{i}'] for i in range(nFolds)]
		return CCMCVResult(fold_results = foldResults,
						   fold_performances = data['fold_performances'] if 'fold_performances' in data else None,
						   mean_performance = data['mean_performance'] if 'mean_performance' in data else None,
						   std_performance = data['std_performance'] if 'std_performance' in data else None,
						   predictionHorizon = int(data['predictionHorizon']),
						   fold_forward_embed_dimensions = foldForwardEmbedDimensions)
