"""
Result classes for torchEDM predictions.

This module provides dataclasses for structured prediction results from different EDM methods.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

from ..Scoring import Correlation


@dataclass(frozen=True)
class SimplexResult:
    """
    Results from Simplex prediction.

    :param projection: Array with columns [Time, Observations, Predictions]
    :param embedDimensions: Embedding dimension used
    :param predictionHorizon: Prediction horizon used
    """
    projection: np.ndarray
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self) -> np.ndarray:
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from projection.
        """
        return self.projection[:, 2]

    def score(self, scoring_function = Correlation) -> float:
        """
        Compute prediction error statistics.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :return: Computed metric value
        """
        return scoring_function(self.observations, self.predictions)


@dataclass(frozen=True)
class SMapResult:
    """
    Results from S-Map prediction.

    :param projection: Array with columns [Time, Observations, Predictions]
    :param coefficients: S-Map coefficients for each prediction (N_pred, E+1)
    :param singularValues: Singular values from SVD for each prediction (N_pred, E+1)
    :param embedDimensions: Embedding dimension used
    :param predictionHorizon: Prediction horizon used
    :param theta: Localization parameter used
    """
    projection: np.ndarray
    coefficients: np.ndarray
    singularValues: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    theta: float

    @property
    def time(self) -> np.ndarray:
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from projection.
        """
        return self.projection[:, 2]

    @property
    def prediction_result(self) -> SimplexResult:
        """
        Get prediction as SimplexResult for compatibility.
        """
        return SimplexResult(
            projection=self.projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    def score(self, scoring_function = Correlation) -> float:
        """
        Compute prediction error statistics.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :return: Computed metric value
        """
        return scoring_function(self.observations, self.predictions)


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

    :param projection: Ensemble-averaged prediction array [Time, Observations, Predictions]
    :param view: Rankings of column combinations. Each element is [combo_string, correlation, MAE, CAE, RMSE]
    :param topRankProjections: Dictionary mapping column combinations (tuples) to their prediction arrays [Time, Observations, Predictions, Variance]
    :param topRankStats: Dictionary mapping column combinations (tuples) to their error statistics {'correlation', 'MAE', 'CAE', 'RMSE'}
    :param D: State-space dimension used
    :param embedDimensions: Embedding dimension for each variable
    :param predictionHorizon: Prediction horizon used
    """
    projection: np.ndarray
    view: List
    topRankProjections: Dict
    topRankStats: Dict
    D: int
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self) -> np.ndarray:
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Ensemble-averaged predictions.
        """
        return self.projection[:, 2]

    @property
    def top_combinations(self) -> List:
        """
        Get list of top-ranked column combinations.
        """
        return list(self.topRankProjections.keys())

    def score(self, scoring_function = Correlation) -> float:
        """
        Compute prediction error statistics for ensemble prediction.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :return: Computed metric value
        """
        return scoring_function(self.observations, self.predictions)

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
    Results from Multivariate Delay Embedding.

    :param time: Time values, shape [N]
    :param observations: Observed values, shape [N, K] where K is the number of targets
    :param predictions: Predicted values, shape [N, K]
    :param selected_features: Selected feature column indices, shape [K, maxD], padded with -1
    :param accuracy: Accuracy at each feature addition step, shape [K, maxD], padded with NaN
    :param ccm_values: CCM convergence slopes for selected features, shape [K, maxD], padded with NaN
    :param stepwise_performance: Performance of adding each candidate at each step, shape [target, dimensions, variables]
    :param timeDelayResults: Time delay analysis results as list of (variable, delay, improvement, score) tuples
    """
    time: np.ndarray
    observations: np.ndarray
    predictions: np.ndarray
    selected_features: np.ndarray
    accuracy: np.ndarray
    ccm_values: np.ndarray
    stepwise_performance: np.ndarray
    timeDelayResults: List[Tuple[int, int, float, float]] = None

    def score(self, scoring_function = Correlation, target: int = 0) -> float:
        """
        Compute prediction error statistics for one target.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :param target: Target index (default 0)
        :return: Computed metric value
        """
        return scoring_function(self.observations[:, target], self.predictions[:, target])


@dataclass(frozen=True)
class MDECVResult:
    """
    Results from MDE Cross-Validation.

    :param time: Time values, shape [N]
    :param observations: Observed values, shape [N, K]
    :param predictions: Predicted values, shape [N, K]
    :param selected_features: Final selected feature indices, shape [K, maxD] padded with -1
    :param fold_results: Results from each cross-validation fold
    :param fold_accuracies: Per-fold accuracy for the first target, shape [nFolds]
    :param best_fold: Index of best performing fold
    """
    time: np.ndarray
    observations: np.ndarray
    predictions: np.ndarray
    selected_features: np.ndarray
    fold_results: List[MDEResult]
    fold_accuracies: np.ndarray
    best_fold: np.ndarray

    def compute_error(self, scoring_function = Correlation, target: int = 0) -> float:
        """
        Compute prediction error statistics for one target.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :param target: Target index (default 0)
        :return: Computed metric value
        """
        return scoring_function(self.observations[:, target], self.predictions[:, target])

@dataclass(frozen=True)
class MDECVResults:
    """
    Results from MDE Cross-Validation fitting.

    :param fold_selected_features: Selected features per fold, shape [nFolds, nTargets, maxD] padded with -1
    :param fold_stepwise_performances: Stepwise candidate performance per fold, shape [nFolds, nTargets, maxD, nCandidates]
    :param fold_accuracies: Per-fold, per-target accuracy, shape [nFolds, nTargets]
    :param best_fold: Index of best performing fold per target, shape [nTargets]
    :param selected_features: Final selected feature column indices, shape [nTargets, maxD] padded with -1
    :param time: Time values for final prediction, shape [N] (None if no prediction computed)
    :param observations: Observed values for final prediction, shape [N, nTargets] (None if no prediction computed)
    :param predictions: Predicted values for final prediction, shape [N, nTargets] (None if no prediction computed)
    """
    fold_selected_features: np.ndarray
    fold_stepwise_performances: np.ndarray
    fold_accuracies: np.ndarray
    best_fold: np.ndarray
    selected_features: np.ndarray
    time: Optional[np.ndarray] = None
    observations: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None

    def score(self, scoring_function = Correlation, target: int = 0) -> float:
        """
        Compute prediction score for one target.

        :param scoring_function: Scoring function taking (actual, predicted) and returning a scalar
        :param target: Target index (default 0)
        :return: Computed metric value
        :raises RuntimeError: if no predictions have been computed
        """
        if self.observations is None or self.predictions is None:
            raise RuntimeError("No predictions available. Call Predict() to compute final predictions.")
        return scoring_function(self.observations[:, target], self.predictions[:, target])


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
		elif isinstance(result, BatchedCCMResult):
			arrays = ResultsIO._BatchedCCMArrays(result)
		else:
			raise TypeError(f"Unsupported result type: {result_type}")

		arrays['result_type'] = np.array(result_type)
		path = path.rstrip('/')
		for key, value in arrays.items():
			cloud.upload_npy_array(f'{path}/{key}', value)

	@staticmethod
	def LoadFromCloud(path: str, cloud = None):
		"""
		Load a result object from S3.

		:param path: Folder-like S3 path used when saving
		:param cloud: A cottoncandy interface object. If None, one is created via cottoncandy.get_interface()
		:return: The reconstructed result object
		"""
		if cloud is None:
			import cottoncandy
			cloud = cottoncandy.get_interface()

		path = path.rstrip('/')
		object_names = cloud.ls(path)
		data = {name[len(path) + 1:]: cloud.download_npy_array(name)
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
		elif result_type == 'BatchedCCMResult':
			return ResultsIO._LoadBatchedCCM(data)
		else:
			raise ValueError(f"Unknown result type in cloud folder: {result_type}")

	# --- internal helpers: arrays from result ---

	@staticmethod
	def _SimplexArrays(result: SimplexResult) -> dict:
		return dict(
			projection = result.projection,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon))

	@staticmethod
	def _SMapArrays(result: SMapResult) -> dict:
		return dict(
			projection = result.projection,
			coefficients = result.coefficients,
			singularValues = result.singularValues,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon),
			theta = np.array(result.theta))

	@staticmethod
	def _CCMArrays(result: CCMResult) -> dict:
		arrays = dict(
			libMeans = result.libMeans,
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon),
			has_predictStats1 = np.array(result.predictStats1 is not None),
			has_predictStats2 = np.array(result.predictStats2 is not None))
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
			projection = result.projection,
			view = view_array,
			combo_keys = combo_keys,
			topRankStats_values = stats_values,
			n_combos = np.array(len(combos)),
			D = np.array(result.D),
			embedDimensions = np.array(result.embedDimensions),
			predictionHorizon = np.array(result.predictionHorizon))

		for i, combo in enumerate(combos):
			arrays[f'topRankProj_{i}'] = result.topRankProjections[combo]

		return arrays

	@staticmethod
	def _MDEArrays(result: MDEResult) -> dict:
		has_time_delay = result.timeDelayResults is not None
		arrays = dict(
			time = result.time,
			observations = result.observations,
			predictions = result.predictions,
			selected_features = result.selected_features,
			accuracy = result.accuracy,
			ccm_values = result.ccm_values,
			stepwise_performance = result.stepwise_performance,
			has_timeDelayResults = np.array(has_time_delay))
		if has_time_delay:
			arrays['timeDelayResults'] = np.array(result.timeDelayResults, dtype = float)
		return arrays

	@staticmethod
	def _MDECVArrays(result: MDECVResult) -> dict:
		n_folds = len(result.fold_results)
		arrays = dict(
			time = result.time,
			observations = result.observations,
			predictions = result.predictions,
			selected_features = result.selected_features,
			fold_accuracies = result.fold_accuracies,
			best_fold = result.best_fold,
			n_folds = np.array(n_folds))

		for i, fold in enumerate(result.fold_results):
			fold_arrays = ResultsIO._MDEArrays(fold)
			for key, value in fold_arrays.items():
				arrays[f'fold_{i}_{key}'] = value

		return arrays

	@staticmethod
	def _BatchedCCMArrays(result: BatchedCCMResult) -> dict:
		arrays = dict(
			forward_performance = result.forward_performance,
			has_reverse_performance = np.array(result.reverse_performance is not None),
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
			projection = data['projection'],
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']))

	@staticmethod
	def _LoadSMap(data) -> SMapResult:
		return SMapResult(
			projection = data['projection'],
			coefficients = data['coefficients'],
			singularValues = data['singularValues'],
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			theta = float(data['theta']))

	@staticmethod
	def _LoadCCM(data) -> CCMResult:
		return CCMResult(
			libMeans = data['libMeans'],
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			predictStats1 = data['predictStats1'] if bool(data['has_predictStats1']) else None,
			predictStats2 = data['predictStats2'] if bool(data['has_predictStats2']) else None)

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
			projection = data['projection'],
			view = list(data['view']),
			topRankProjections = topRankProjections,
			topRankStats = topRankStats,
			D = int(data['D']),
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']))

	@staticmethod
	def _LoadMDE(data) -> MDEResult:
		time_delay = None
		if bool(data['has_timeDelayResults']):
			time_delay = [
				(int(row[0]), int(row[1]), float(row[2]), float(row[3]))
				for row in data['timeDelayResults']]
		return MDEResult(
			time = data['time'],
			observations = data['observations'],
			predictions = data['predictions'],
			selected_features = data['selected_features'],
			accuracy = data['accuracy'],
			ccm_values = data['ccm_values'],
			stepwise_performance = data['stepwise_performance'],
			timeDelayResults = time_delay)

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
			observations = data['observations'],
			predictions = data['predictions'],
			selected_features = data['selected_features'],
			fold_results = fold_results,
			fold_accuracies = data['fold_accuracies'],
			best_fold = data['best_fold'])

	@staticmethod
	def _LoadBatchedCCM(data) -> BatchedCCMResult:
		return BatchedCCMResult(
			forward_performance = data['forward_performance'],
			reverse_performance = data['reverse_performance'] if bool(data['has_reverse_performance']) else None,
			embedDimensions = int(data['embedDimensions']),
			predictionHorizon = int(data['predictionHorizon']),
			library_sizes = data['library_sizes'])
