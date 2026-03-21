"""
Result classes for torchEDM predictions.

This module provides dataclasses for structured prediction results from different EDM methods.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

from ..Utils import ComputeError


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

    def compute_error(self, metric = None) -> float:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)


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

    def compute_error(self, metric = None) -> float:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)


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

    def compute_error(self, metric = None) -> float:
        """
        Compute prediction error statistics for ensemble prediction.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)

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
    :param stepwise_performance: Performance of adding each candidate at each step, shape [K, maxD, nVars]
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

    def compute_error(self, metric = None, target: int = 0) -> float:
        """
        Compute prediction error statistics for one target.

        :param metric: Error metric to compute
        :param target: Target index (default 0)
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations[:, target], self.predictions[:, target], metric)


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

    def compute_error(self, metric = None, target: int = 0) -> float:
        """
        Compute prediction error statistics for one target.

        :param metric: Error metric to compute
        :param target: Target index (default 0)
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations[:, target], self.predictions[:, target], metric)

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
