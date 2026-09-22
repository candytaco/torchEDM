"""
Result records returned by the predictors, the variable-selection drivers, and the
cross-map screen, plus ResultsIO for saving and loading them.

Every prediction field is named Y_pred and has the shape of the Y_test it was scored
against (a list of arrays when the test data came as a list of runs). Nothing here
carries time; rows are sample positions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

ArrayOrList = Union[np.ndarray, List[np.ndarray]]


@dataclass(frozen = True)
class SimplexResult:
	"""
	Nearest-neighbor weighted-average prediction.

	:param Y_pred:		predictions shaped like Y_test (a list for multiple test runs); NaN where no complete state predicts the row
	:param variance:	weighted spread of the neighbor targets around each prediction, same shape as Y_pred
	:param score:		[nTargets] scoringFunction over the (Y_test, Y_pred) pairs; None when Y_test was not given
	:param embedDimensions:	copies of each feature column in the state
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors used
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
	Locally weighted linear prediction.

	:param Y_pred:		predictions shaped like Y_test (a list for multiple test runs); NaN where no complete state predicts the row
	:param variance:	weighted spread of the neighbor targets around each prediction, same shape as Y_pred
	:param coefficients:	[nRows, stateSize + 1, nTargets] per predicted row, intercept first; NaN rows where nothing was predicted
	:param singularValues:	[nRows, stateSize + 1, nTargets] of the weighted design matrices
	:param score:		[nTargets] scoringFunction over the (Y_test, Y_pred) pairs; None when Y_test was not given
	:param embedDimensions:	copies of each feature column in the state
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors used
	:param theta:		localization used
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
	Ensemble of the top-ranked feature combinations.

	:param Y_pred:		ensemble-averaged predictions shaped like Y_test
	:param view:		one row per top-ranked combination: [combination, correlation, max abs error, sum abs error, RMSE]
	:param topRankPredictions:	combination tuple -> that combination's Y_pred
	:param topRankStats:		combination tuple -> [correlation, max abs error, sum abs error, RMSE]
	:param columnsPerView:	columns of the stacked state used per combination
	:param embedDimensions:	copies of each feature column in the stacked state
	:param predictionHorizon:	rows between a state and the target it predicts
	:param score:		[nTargets] scoringFunction over the ensemble (Y_test, Y_pred) pairs; None when Y_test was not given
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
		return list(self.topRankPredictions.keys())

	def get_combination_stats(self, combo: tuple) -> List[float]:
		if combo not in self.topRankStats:
			raise ValueError(f'Combination {combo} not in top-ranked results')
		return self.topRankStats[combo]


@dataclass(frozen = True)
class MDEResult:
	"""
	Greedy variable selection (manifold dimensional expansion).

	Column indices refer to the candidate view [X columns | target columns], so nColumns is
	nFeatures + nTargets and a target's own column is nFeatures + targetIndex.

	:param Y_pred:		final predictions shaped like Y_test, one column per target
	:param selected_variables:	selected columns per target, shape [nTargets, maxVariables], padded with -1
	:param performance:	score after each addition, shape [nTargets, maxVariables], padded with NaN
	:param ccm_values:	convergence slopes of the selected variables, shape [nTargets, maxVariables], padded with NaN
	:param stepwise_performance:	score of every candidate at every step, shape [nTargets, maxVariables, nColumns]
	:param candidate_embed_dimensions:	best embedding dimension per (target, candidate), shape [nTargets, nColumns];
		-1 where the search did not run
	:param candidate_peak_scores:	the score at that embedding dimension, shape [nTargets, nColumns]; NaN where the search did not run
	:param candidate_slopes:	convergence slope per (target, candidate), shape [nTargets, nColumns]; NaN for a
		candidate the run never checked, -inf for a check whose slope was NaN
	:param score:		[nTargets] scoringFunction over the final (Y_test, Y_pred) pairs, or None
	"""
	Y_pred: Optional[ArrayOrList]
	selected_variables: np.ndarray
	performance: np.ndarray
	ccm_values: np.ndarray
	stepwise_performance: np.ndarray
	candidate_embed_dimensions: Optional[np.ndarray] = None
	candidate_peak_scores: Optional[np.ndarray] = None
	candidate_slopes: Optional[np.ndarray] = None
	score: Optional[np.ndarray] = None


@dataclass(frozen = True)
class MDECVResults:
	"""
	Cross-validated variable selection from MDEFitterCV.

	:param fold_selected_variables:	selected X columns per fold, shape [nFolds, nTargets, maxVariables], padded with -1
	:param fold_stepwise_performances:	candidate scores per fold, shape [nFolds, nTargets, maxVariables, nColumns]
	:param fold_accuracies:	score per fold and target, shape [nFolds, nTargets]
	:param fold_Y_pred:		Y_pred of each fold's held-out data
	:param best_fold:		best fold per target, shape [nTargets]
	:param selected_variables:	final selected X columns, shape [nTargets, maxVariables], padded with -1
	:param Y_pred:			final predictions shaped like Y_test, or None
	:param score:			[nTargets] scoringFunction over the final (Y_test, Y_pred) pairs, or None
	"""
	fold_selected_variables: np.ndarray
	fold_stepwise_performances: np.ndarray
	fold_accuracies: np.ndarray
	fold_Y_pred: Optional[List[ArrayOrList]]
	best_fold: np.ndarray
	selected_variables: np.ndarray
	Y_pred: Optional[ArrayOrList] = None
	score: Optional[np.ndarray] = None

	@property
	def selected_stepwise_performances(self) -> np.ndarray:
		"""
		Score of the variable actually selected at each step, per fold and target:
		fold_stepwise_performances [nFolds, nTargets, maxVariables, nColumns] indexed by
		fold_selected_variables, NaN where nothing was selected.
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
	Cross-map skill of every source column against every target across training-subset sizes.

	:param forward_performance:	mean skill per subset size, shape [nSizes, nSources, nTargets] with singleton axes squeezed
	:param predictionHorizon:	rows between a state and the target it predicts
	:param library_sizes:	the training-subset sizes evaluated
	:param forward_embed_dimensions:	embedding dimensions used per source ([nSources] or [nSources, nTargets] when searched, else the scalar given)
	"""
	forward_performance: np.ndarray
	predictionHorizon: int
	library_sizes: Union[np.ndarray, List]
	forward_embed_dimensions: Optional[Union[int, np.ndarray]] = None

	def GetVariableCorrelations(self, variableIndex: int) -> np.ndarray:
		return self.forward_performance[:, 1 + variableIndex]


@dataclass(frozen = True)
class CCMCVResult:
	"""
	Cross-validated cross-map screen.

	:param fold_results:	BatchedCCMResult per fold
	:param fold_performances:	[nFolds, nSources] or [nFolds, nSources, nTargets]
	:param mean_performance:	mean over folds
	:param std_performance:	standard deviation over folds
	:param predictionHorizon:	rows between a state and the target it predicts
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
	The record type is stored under 'result_type' so Load reconstructs the right class.
	"""

	@staticmethod
	def _Arrays(result) -> dict:
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
		:param result:	any result record
		:param path:	output file path (.npz is appended when absent)
		"""
		arrays = ResultsIO._Arrays(result)
		arrays['result_type'] = np.array(type(result).__name__)
		np.savez(path, **arrays)

	@staticmethod
	def Load(path: str):
		return ResultsIO._FromData(np.load(path, allow_pickle = True))

	@staticmethod
	def SaveToCloud(result, path: str, cloud = None) -> None:
		"""
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
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface(verbose = False)
		path = path.rstrip('/')
		data = {name.split('/')[-1].split('.')[0]: cloud.download_npy_array(name) for name in cloud.ls(path)}
		return ResultsIO._FromData(data)

	# --- run-list packing: a list of per-run arrays is stored as key_0, key_1, ... plus n_key ---

	@staticmethod
	def _PackRuns(arrays: dict, key: str, value) -> None:
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
		if f'n_{key}' in data:
			return [data[f'{key}_{i}'] for i in range(int(data[f'n_{key}']))]
		return data[key] if key in data else None

	# --- arrays from records ---

	@staticmethod
	def _SimplexArrays(result: SimplexResult) -> dict:
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
		arrays = ResultsIO._SimplexArrays(result)
		arrays['theta'] = np.array(result.theta)
		ResultsIO._PackRuns(arrays, 'coefficients', result.coefficients)
		ResultsIO._PackRuns(arrays, 'singularValues', result.singularValues)
		return arrays

	@staticmethod
	def _MultiviewArrays(result: MultiviewResult) -> dict:
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
		arrays = dict(selected_variables = result.selected_variables,
					  accuracy = result.performance,
					  ccm_values = result.ccm_values,
					  stepwise_performance = result.stepwise_performance)
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		for key in ('candidate_embed_dimensions', 'candidate_peak_scores', 'candidate_slopes'):
			value = getattr(result, key)
			if value is not None:
				arrays[key] = np.asarray(value)
		return arrays

	@staticmethod
	def _MDECVResultsArrays(result: MDECVResults) -> dict:
		arrays = dict(fold_selected_variables = result.fold_selected_variables,
					  fold_stepwise_performances = result.fold_stepwise_performances,
					  fold_accuracies = result.fold_accuracies,
					  best_fold = result.best_fold,
					  selected_variables = result.selected_variables)
		ResultsIO._PackRuns(arrays, 'Y_pred', result.Y_pred)
		if result.score is not None:
			arrays['score'] = np.asarray(result.score)
		if result.fold_Y_pred is not None:
			arrays['n_fold_Y_pred'] = np.array(len(result.fold_Y_pred))
			for i, foldPrediction in enumerate(result.fold_Y_pred):
				ResultsIO._PackRuns(arrays, f'fold_Y_pred_{i}', foldPrediction)
		return arrays

	@staticmethod
	def _BatchedCCMArrays(result: BatchedCCMResult) -> dict:
		arrays = dict(forward_performance = result.forward_performance,
					  predictionHorizon = np.array(result.predictionHorizon),
					  library_sizes = np.array(result.library_sizes))
		if result.forward_embed_dimensions is not None:
			arrays['forward_embed_dimensions'] = np.array(result.forward_embed_dimensions)
		return arrays

	@staticmethod
	def _CCMCVArrays(result: CCMCVResult) -> dict:
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
		return SimplexResult(Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
							 variance = ResultsIO._UnpackRuns(data, 'variance'),
							 score = data['score'] if 'score' in data else None,
							 embedDimensions = int(data['embedDimensions']),
							 predictionHorizon = int(data['predictionHorizon']),
							 knn = int(data['knn']))

	@staticmethod
	def _LoadSMap(data) -> SMapResult:
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
		def optional(key):
			return data[key] if key in data else None
		return MDEResult(Y_pred = ResultsIO._UnpackRuns(data, 'Y_pred'),
						 selected_variables = data['selected_variables'],
						 performance = data['accuracy'],
						 ccm_values = data['ccm_values'],
						 stepwise_performance = data['stepwise_performance'],
						 candidate_embed_dimensions = optional('candidate_embed_dimensions'),
						 candidate_peak_scores = optional('candidate_peak_scores'),
						 candidate_slopes = optional('candidate_slopes'),
						 score = optional('score'))

	@staticmethod
	def _LoadMDECVResults(data) -> MDECVResults:
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
							score = data['score'] if 'score' in data else None)

	@staticmethod
	def _LoadBatchedCCM(data) -> BatchedCCMResult:
		return BatchedCCMResult(forward_performance = data['forward_performance'],
								predictionHorizon = int(data['predictionHorizon']),
								library_sizes = data['library_sizes'],
								forward_embed_dimensions = data['forward_embed_dimensions'] if 'forward_embed_dimensions' in data else None)

	@staticmethod
	def _LoadCCMCV(data) -> CCMCVResult:
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
