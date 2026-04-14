"""
Result classes for torchEDM predictions.

This module provides dataclasses for structured prediction results from different EDM methods.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class SimplexResult:
	"""
	Results from Simplex prediction.

	:param time: Time values, shape [N]
	:param projection: Array with columns [Time, Observations, Predictions, Variance], or None if predictions were not requested
	:param embedDimensions: Embedding dimension used
	:param predictionHorizon: Prediction horizon used
	:param score: Prediction score computed by the calling function, or None if not computed
	"""
	time: np.ndarray
	projection: Optional[np.ndarray]
	embedDimensions: int
	predictionHorizon: int
	score: Optional[float] = None

	@property
	def predictions(self) -> Optional[np.ndarray]:
		"""
		Predicted values from projection, or None if not populated.
		"""
		if self.projection is None:
			return None
		return self.projection[:, 2]


@dataclass(frozen=True)
class SMapResult:
	"""
	Results from S-Map prediction.

	:param time: Time values, shape [N]
	:param projection: Array with columns [Time, Observations, Predictions, Variance], or None if predictions were not requested
	:param coefficients: S-Map coefficients for each prediction (N_pred, E+1)
	:param singularValues: Singular values from SVD for each prediction (N_pred, E+1)
	:param embedDimensions: Embedding dimension used
	:param predictionHorizon: Prediction horizon used
	:param theta: Localization parameter used
	:param score: Prediction score computed by the calling function, or None if not computed
	"""
	time: np.ndarray
	projection: Optional[np.ndarray]
	coefficients: np.ndarray
	singularValues: np.ndarray
	embedDimensions: int
	predictionHorizon: int
	theta: float
	score: Optional[float] = None

	@property
	def predictions(self) -> Optional[np.ndarray]:
		"""
		Predicted values from projection, or None if not populated.
		"""
		if self.projection is None:
			return None
		return self.projection[:, 2]

	@property
	def prediction_result(self) -> SimplexResult:
		"""
		Get prediction as SimplexResult for compatibility.
		"""
		return SimplexResult(
			time=self.time,
			projection=self.projection,
			embedDimensions=self.embedDimensions,
			predictionHorizon=self.predictionHorizon
		)


@dataclass(frozen=True)
class CCMResult:
	"""
	Results from Convergent Cross Mapping.

	:param libMeans: Mean correlation at each library size. Shape (n_lib_sizes, 2 or 3):
		Column 0: Library size,
		Column 1: Mean correlation for first direction,
		Column 2: Mean correlation for second direction (if applicable)
	:param embedDimensions: Embedding dimension used
	:param predictionHorizon: Prediction horizon used
	:param predictStats1: Detailed prediction statistics for first direction (only if includeData=True)
	:param predictStats2: Detailed prediction statistics for second direction (only if includeData=True)
	"""
	libMeans: np.ndarray
	embedDimensions: int
	predictionHorizon: int
	predictStats1: Optional[np.ndarray] = None
	predictStats2: Optional[np.ndarray] = None

	@property
	def library_sizes(self) -> np.ndarray:
		"""
		Library sizes evaluated.
		"""
		return self.libMeans[:, 0]

	@property
	def correlations(self) -> np.ndarray:
		"""
		Correlation values (excludes library size column).
		"""
		return self.libMeans[:, 1:]


@dataclass(frozen=True)
class MultiviewResult:
	"""
	Results from Multiview prediction.

	:param time: Time values, shape [N]
	:param view: Rankings of column combinations. Each element is [combo_string, correlation, MAE, CAE, RMSE]
	:param topRankProjections: Dictionary mapping column combinations (tuples) to their prediction arrays [Time, Observations, Predictions, Variance]
	:param topRankStats: Dictionary mapping column combinations (tuples) to their error statistics {'correlation', 'MAE', 'CAE', 'RMSE'}
	:param D: State-space dimension used
	:param embedDimensions: Embedding dimension for each variable
	:param predictionHorizon: Prediction horizon used
	:param predictions: Ensemble-averaged predictions, or None if predictions were not requested
	:param score: Prediction score computed by the calling function, or None if not computed
	"""
	time: np.ndarray
	view: List
	topRankProjections: Dict
	topRankStats: Dict
	D: int
	embedDimensions: int
	predictionHorizon: int
	predictions: Optional[np.ndarray] = None
	score: Optional[float] = None

	@property
	def top_combinations(self) -> List:
		"""
		Get list of top-ranked column combinations.
		"""
		return list(self.topRankProjections.keys())

	def get_combination_stats(self, combo: tuple) -> Dict[str, float]:
		"""
		Get error statistics for a specific column combination.

		:param combo: Column combination (e.g., (0, 2, 4))
		:return: Error statistics for this combination
		:raises ValueError: if combination not in top-ranked results
		"""
		if combo not in self.topRankStats:
			raise ValueError(f"Combination {combo} not in top-ranked results")
		return self.topRankStats[combo]


@dataclass(frozen=True)
class MDEResult:
	"""
	Results from Multivariate Dimensional expansion.

	:param time: Time values, shape [N]
	:param predictions: Predicted values, shape [N, K], or None if predictions were not requested
	:param selected_variables: Selected variable column indices, shape [K, maxD], padded with -1
	:param accuracy: Accuracy at each variable addition step, shape [K, maxD], padded with NaN
	:param ccm_values: CCM convergence slopes for selected variables, shape [K, maxD], padded with NaN
	:param stepwise_performance: Performance of adding each candidate at each step, shape [target, dimensions, variables]
	:param timeDelayResults: Time delay analysis results as list of (variable, delay, improvement, score) tuples
	:param score: Per-target prediction score computed by the calling function, shape [K], or None if not computed
	"""
	time: np.ndarray
	predictions: Optional[np.ndarray]
	selected_variables: np.ndarray
	accuracy: np.ndarray
	ccm_values: np.ndarray
	stepwise_performance: np.ndarray
	timeDelayResults: List[Tuple[int, int, float, float]] = None
	score: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MDECVResult:
	"""
	Results from MDE Cross-Validation.

	TODO: this should be combined with MDECVResults - this is from the EDM-style class, the other is from fitters

	:param time: Time values, shape [N]
	:param predictions: Predicted values, shape [N, K], or None if predictions were not requested
	:param selected_variables: Final selected variable indices, shape [K, maxD] padded with -1
	:param fold_results: Results from each cross-validation fold
	:param fold_accuracies: Per-fold accuracy for the first target, shape [nFolds]
	:param best_fold: Index of best performing fold
	:param score: Per-target prediction score computed by the calling function, shape [K], or None if not computed
	"""
	time: np.ndarray
	predictions: Optional[np.ndarray]
	selected_variables: np.ndarray
	fold_results: List[MDEResult]
	fold_accuracies: np.ndarray
	best_fold: np.ndarray
	score: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MDECVResults:
	"""
	Results from MDE Cross-Validation fitting.

	:param fold_selected_variables: Selected features per fold, shape [nFolds, nTargets, maxD] padded with -1
	:param fold_stepwise_performances: Stepwise candidate performance per fold, shape [nFolds, nTargets, maxD, nCandidates]
	:param fold_accuracies: Per-fold, per-target accuracy, shape [nFolds, nTargets]
	:param best_fold: Index of best performing fold per target, shape [nTargets]
	:param fold_predictions: predictions for each cross-validation fold
	:param selected_variables: Final selected feature column indices, shape [nTargets, maxD] padded with -1
	:param time: Time values for final prediction, shape [N] (None if no prediction computed)
	:param predictions: Predicted values for final prediction, shape [N, nTargets] (None if no prediction computed)
	:param score: Per-target prediction score computed by the calling function, shape [nTargets], or None if not computed
	"""
	fold_selected_variables: np.ndarray
	fold_stepwise_performances: np.ndarray
	fold_accuracies: np.ndarray
	fold_predictions: Optional[List[np.ndarray]]
	best_fold: np.ndarray
	selected_variables: np.ndarray
	time: Optional[np.ndarray] = None
	predictions: Optional[np.ndarray] = None
	score: Optional[np.ndarray] = None

	@property
	def selected_stepwise_performances(self) -> np.ndarray:
		"""
		Performance of the actually selected variable at each step, for each fold and target.
		Condenses fold_stepwise_performances from shape [nFolds, nTargets, maxD, nCandidates]
		to [nFolds, nTargets, maxD] by indexing into the candidate axis with the selected variable index.
		Entries are NaN where fold_selected_variables is -1 (padding, i.e. no variable was selected at that step).
		"""
		selected = self.fold_selected_variables  # [nFolds, nTargets, maxD]
		mask = selected >= 0
		performances = np.full(selected.shape, np.nan)
		foldIndices, targetIndices, stepIndices = np.where(mask)
		variableIndices = selected[foldIndices, targetIndices, stepIndices]
		performances[foldIndices, targetIndices, stepIndices] = self.fold_stepwise_performances[foldIndices, targetIndices, stepIndices, variableIndices]
		return performances


@dataclass(frozen=True)
class BatchedCCMResult:
	"""
	Results from Batched Convergent Cross Mapping.

	:param forward_performance: Forward direction correlations. Shape (n_lib_sizes, 1+M):
		Column 0: Library size
		Columns 1-M: Mean correlation for each predictor variable
	:param reverse_performance: Reverse direction correlations. Shape (n_lib_sizes, 1+M) or None:
		Column 0: Library size
		Columns 1-M: Mean correlation for each predictor variable
	:param embedDimensions: Embedding dimension used
	:param predictionHorizon: Prediction horizon used
	"""
	forward_performance: np.ndarray
	reverse_performance: Optional[np.ndarray]
	embedDimensions: int
	predictionHorizon: int
	library_sizes: Union[np.ndarray, List]

	def GetVariableCorrelations(self, variableIndex: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		"""
		Get correlations for a specific variable across all library sizes.

		:param variableIndex: Index of the variable (0-based)
		:return: Tuple of (forward_correlations, reverse_correlations) as 1D arrays
		"""
		forwardCorr = self.forward_performance[:, 1 + variableIndex]
		reverseCorr = self.reverse_performance[:, 1 + variableIndex] if self.reverse_performance is not None else None
		return forwardCorr, reverseCorr


class ResultsIO:
	"""
	Static class for saving and loading result objects to and from npz files.

	The result type is stored in the file under the key 'result_type', allowing
	Load to reconstruct the correct object without the caller specifying the type.
	"""

	@staticmethod
	def Save(result, path: str) -> None:
		"""
		Save any result object to an npz file.

		:param result: A result object (SimplexResult, SMapResult, etc.)
		:param path: Output file path (the .npz extension is added automatically if absent)
		"""
		result_type = type(result).__name__

		if isinstance(result, SimplexResult):
			arrays = ResultsIO._SimplexArrays(result)
		elif isinstance(result, SMapResult):
			arrays = ResultsIO._SMapArrays(result)
		elif isinstance(result, CCMResult):
			arrays = ResultsIO._CCMArrays(result)
		elif isinstance(result, MultiviewResult):
			arrays = ResultsIO._MultiviewArrays(result)
		elif isinstance(result, MDEResult):
			arrays = ResultsIO._MDEArrays(result)
		elif isinstance(result, MDECVResult):
			arrays = ResultsIO._MDECVArrays(result)
		elif isinstance(result, MDECVResults):
			arrays = ResultsIO._MDECVResultsArrays(result)
		elif isinstance(result, BatchedCCMResult):
			arrays = ResultsIO._BatchedCCMArrays(result)
		else:
			raise TypeError(f"Unsupported result type: {result_type}")

		arrays['result_type'] = np.array(result_type)
		np.savez(path, **arrays)

	@staticmethod
	def Load(path: str):
		"""
		Load a result object from an npz file.

		:param path: Path to the npz file
		:return: The reconstructed result object
		"""
		data = np.load(path, allow_pickle = True)
		result_type = str(data['result_type'])

		if result_type == 'SimplexResult':
			return ResultsIO._LoadSimplex(data)
		elif result_type == 'SMapResult':
			return ResultsIO._LoadSMap(data)
		elif result_type == 'CCMResult':
			return ResultsIO._LoadCCM(data)
		elif result_type == 'MultiviewResult':
			return ResultsIO._LoadMultiview(data)
		elif result_type == 'MDEResult':
			return ResultsIO._LoadMDE(data)
		elif result_type == 'MDECVResult':
			return ResultsIO._LoadMDECV(data)
		elif result_type == 'MDECVResults':
			return ResultsIO._LoadMDECVResults(data)
		elif result_type == 'BatchedCCMResult':
			return ResultsIO._LoadBatchedCCM(data)
		else:
			raise ValueError(f"Unknown result type in file: {result_type}")

	@staticmethod
	def SaveToCloud(result, path: str, cloud = None) -> None:
		"""
		Save any result object to S3, with each array stored as a separate object.

		:param result: A result object (SimplexResult, SMapResult, etc.)
		:param path: Folder-like S3 path; each array is uploaded as path/key
		:param cloud: A cottoncandy interface object. If None, one is created via cottoncandy.get_interface()
		"""
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface()

		result_type = type(result).__name__

		if isinstance(result, SimplexResult):
			arrays = ResultsIO._SimplexArrays(result)
		elif isinstance(result, SMapResult):
			arrays = ResultsIO._SMapArrays(result)
		elif isinstance(result, CCMResult):
			arrays = ResultsIO._CCMArrays(result)
		elif isinstance(result, MultiviewResult):
			arrays = ResultsIO._MultiviewArrays(result)
		elif isinstance(result, MDEResult):
			arrays = ResultsIO._MDEArrays(result)
		elif isinstance(result, MDECVResult):
			arrays = ResultsIO._MDECVArrays(result)
		elif isinstance(result, MDECVResults):
			arrays = ResultsIO._MDECVResultsArrays(result)
		elif isinstance(result, BatchedCCMResult):
			arrays = ResultsIO._BatchedCCMArrays(result)
		else:
			raise TypeError(f"Unsupported result type: {result_type}")

		arrays['result_type'] = np.array(result_type)
		path = path.rstrip('/')
		for key, value in arrays.items():
			cloud.upload_npy_array(f'{path}/{key}.npy', value)

	@staticmethod
	def DownloadFromCloud(path: str, cloud = None):
		"""
		Load a result object from S3.

		:param path: Folder-like S3 path used when saving
		:param cloud: A cottoncandy interface object. If None, one is created via cottoncandy.get_interface()
		:return: The reconstructed result object
		"""
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface(verbose = False)

		path = path.rstrip('/')
		object_names = cloud.ls(path)
		data = {name.split('/')[-1].split('.')[0]: cloud.download_npy_array(name)
		        for name in object_names}

		result_type = str(data['result_type'])

		if result_type == 'SimplexResult':
			return ResultsIO._LoadSimplex(data)
		elif result_type == 'SMapResult':
			return ResultsIO._LoadSMap(data)
		elif result_type == 'CCMResult':
			return ResultsIO._LoadCCM(data)
		elif result_type == 'MultiviewResult':
			return ResultsIO._LoadMultiview(data)
		elif result_type == 'MDEResult':
			return ResultsIO._LoadMDE(data)
		elif result_type == 'MDECVResult':
			return ResultsIO._LoadMDECV(data)
		elif result_type == 'MDECVResults':
			return ResultsIO._LoadMDECVResults(data)
		elif result_type == 'BatchedCCMResult':
			return ResultsIO._LoadBatchedCCM(data)
		else:
			raise ValueError(f"Unknown result type in cloud folder: {result_type}")

	# --- internal helpers: arrays from result ---

	@staticmethod
	def _SimplexArrays(result: SimplexResult) -> dict:
		arrays = dict(
			time = result.time,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon))
		if result.projection is not None:
			arrays['projection'] = result.projection
		if result.score is not None:
			arrays['score'] = np.array(result.score)
		return arrays

	@staticmethod
	def _SMapArrays(result: SMapResult) -> dict:
		arrays = dict(
			time = result.time,
			coefficients = result.coefficients,
			singularValues = result.singularValues,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon),
			theta = np.array(result.theta))
		if result.projection is not None:
			arrays['projection'] = result.projection
		if result.score is not None:
			arrays['score'] = np.array(result.score)
		return arrays

	@staticmethod
	def _CCMArrays(result: CCMResult) -> dict:
		arrays = dict(
			libMeans = result.libMeans,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon))
		if result.predictStats1 is not None:
			arrays['predictStats1'] = result.predictStats1
		if result.predictStats2 is not None:
			arrays['predictStats2'] = result.predictStats2
		return arrays

	@staticmethod
	def _MultiviewArrays(result: MultiviewResult) -> dict:
		combos = list(result.topRankProjections.keys())

		combo_keys = np.empty(len(combos), dtype = object)
		for i, combo in enumerate(combos):
			combo_keys[i] = combo

		stats_values = np.empty(len(combos), dtype = object)
		for i, combo in enumerate(combos):
			stats_values[i] = result.topRankStats[combo]

		view_array = np.empty(len(result.view), dtype = object)
		for i, entry in enumerate(result.view):
			view_array[i] = entry

		arrays = dict(
			time = result.time,
			view = view_array,
			combo_keys = combo_keys,
			topRankStats_values = stats_values,
			n_combos = np.array(len(combos)),
			D = np.array(result.D),
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon))

		if result.predictions is not None:
			arrays['predictions'] = result.predictions
		if result.score is not None:
			arrays['score'] = np.array(result.score)

		for i, combo in enumerate(combos):
			arrays[f'topRankProj_{i}'] = result.topRankProjections[combo]

		return arrays

	@staticmethod
	def _MDEArrays(result: MDEResult) -> dict:
		arrays = dict(
			time = result.time,
			selected_variables = result.selected_variables,
			accuracy = result.accuracy,
			ccm_values = result.ccm_values,
			stepwise_performance = result.stepwise_performance)
		if result.predictions is not None:
			arrays['predictions'] = result.predictions
		if result.score is not None:
			arrays['score'] = result.score
		if result.timeDelayResults is not None:
			arrays['timeDelayResults'] = np.array(result.timeDelayResults, dtype = float)
		return arrays

	@staticmethod
	def _MDECVArrays(result: MDECVResult) -> dict:
		n_folds = len(result.fold_results)
		arrays = dict(
			time = result.time,
			selected_features = result.selected_variables,
			fold_accuracies = result.fold_accuracies,
			best_fold = result.best_fold,
			n_folds = np.array(n_folds))

		if result.predictions is not None:
			arrays['predictions'] = result.predictions
		if result.score is not None:
			arrays['score'] = result.score

		for i, fold in enumerate(result.fold_results):
			fold_arrays = ResultsIO._MDEArrays(fold)
			for key, value in fold_arrays.items():
				arrays[f'fold_{i}_{key}'] = value

		return arrays

	@staticmethod
	def _MDECVResultsArrays(result: MDECVResults) -> dict:
		arrays = dict(
			fold_selected_variables = result.fold_selected_variables,
			fold_stepwise_performances = result.fold_stepwise_performances,
			fold_accuracies = result.fold_accuracies,
			best_fold = result.best_fold,
			selected_variables = result.selected_variables)
		if result.time is not None:
			arrays['time'] = result.time
		if result.predictions is not None:
			arrays['predictions'] = result.predictions
		if result.score is not None:
			arrays['score'] = result.score
		if result.fold_predictions is not None:
			arrays['n_fold_predictions'] = np.array(len(result.fold_predictions))
			for i, foldPrediction in enumerate(result.fold_predictions):
				arrays['fold_predictions_{}'.format(i)] = foldPrediction
		return arrays

	@staticmethod
	def _BatchedCCMArrays(result: BatchedCCMResult) -> dict:
		arrays = dict(
			forward_performance = result.forward_performance,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon),
			library_sizes = np.array(result.library_sizes))
		if result.reverse_performance is not None:
			arrays['reverse_performance'] = result.reverse_performance
		return arrays

	# --- internal helpers: result from loaded data ---

	@staticmethod
	def _LoadSimplex(data) -> SimplexResult:
		return SimplexResult(
			time = data['time'],
			projection = data['projection'] if 'projection' in data else None,
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			score = float(data['score']) if 'score' in data else None)

	@staticmethod
	def _LoadSMap(data) -> SMapResult:
		return SMapResult(
			time = data['time'],
			projection = data['projection'] if 'projection' in data else None,
			coefficients = data['coefficients'],
			singularValues = data['singularValues'],
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			theta = float(data['theta']),
			score = float(data['score']) if 'score' in data else None)

	@staticmethod
	def _LoadCCM(data) -> CCMResult:
		return CCMResult(
			libMeans = data['libMeans'],
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			predictStats1 = data['predictStats1'] if 'predictStats1' in data else None,
			predictStats2 = data['predictStats2'] if 'predictStats2' in data else None)

	@staticmethod
	def _LoadMultiview(data) -> MultiviewResult:
		n_combos = int(data['n_combos'])
		combo_keys = list(data['combo_keys'])
		stats_values = list(data['topRankStats_values'])

		topRankProjections = {
			combo_keys[i]: data[f'topRankProj_{i}']
			for i in range(n_combos)}

		topRankStats = {
			combo_keys[i]: stats_values[i]
			for i in range(n_combos)}

		return MultiviewResult(
			time = data['time'],
			view = list(data['view']),
			topRankProjections = topRankProjections,
			topRankStats = topRankStats,
			D = int(data['D']),
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			predictions = data['predictions'] if 'predictions' in data else None,
			score = float(data['score']) if 'score' in data else None)

	@staticmethod
	def _LoadMDE(data) -> MDEResult:
		time_delay = None
		if 'timeDelayResults' in data:
			time_delay = [
				(int(row[0]), int(row[1]), float(row[2]), float(row[3]))
				for row in data['timeDelayResults']]
		return MDEResult(
			time = data['time'],
			predictions = data['predictions'] if 'predictions' in data else None,
			selected_variables = data['selected_variables'],
			accuracy = data['accuracy'],
			ccm_values = data['ccm_values'],
			stepwise_performance = data['stepwise_performance'],
			timeDelayResults = time_delay,
			score = data['score'] if 'score' in data else None)

	@staticmethod
	def _LoadMDECV(data) -> MDECVResult:
		n_folds = int(data['n_folds'])

		fold_results = []
		for i in range(n_folds):
			fold_data = {key[len(f'fold_{i}_'):]: data[key]
			             for key in list(data)
			             if key.startswith(f'fold_{i}_')}
			fold_results.append(ResultsIO._LoadMDE(fold_data))

		return MDECVResult(
			time = data['time'],
			predictions = data['predictions'] if 'predictions' in data else None,
			selected_variables = data['selected_variables'],
			fold_results = fold_results,
			fold_accuracies = data['fold_accuracies'],
			best_fold = data['best_fold'],
			score = data['score'] if 'score' in data else None)

	@staticmethod
	def _LoadMDECVResults(data) -> MDECVResults:
		foldPredictions = None
		if 'n_fold_predictions' in data:
			numFolds = int(data['n_fold_predictions'])
			foldPredictions = [data['fold_predictions_{}'.format(i)] for i in range(numFolds)]
		return MDECVResults(
			fold_selected_variables = data['fold_selected_variables'],
			fold_stepwise_performances = data['fold_stepwise_performances'],
			fold_accuracies = data['fold_accuracies'],
			best_fold = data['best_fold'],
			fold_predictions = foldPredictions,
			selected_variables = data['selected_variables'],
			time = data['time'] if 'time' in data else None,
			predictions = data['predictions'] if 'predictions' in data else None,
			score = data['score'] if 'score' in data else None)

	@staticmethod
	def _LoadBatchedCCM(data) -> BatchedCCMResult:
		return BatchedCCMResult(
			forward_performance = data['forward_performance'],
			reverse_performance = data['reverse_performance'] if 'reverse_performance' in data else None,
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			library_sizes = data['library_sizes'])
