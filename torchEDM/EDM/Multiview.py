"""
Predict with an ensemble over the top-ranked combinations of state dimensions
(Ye & Sugihara 2016, doi.org/10.1126/science.aag0863).
"""
from itertools import combinations
from math import floor, sqrt
from typing import Callable, Optional
from warnings import warn

import numpy
import torch

from .Predictors import SimplexPredict
from .Results import MultiviewResult
from .Setup import ArrayOrRuns, AsRuns, IsListOfRuns, StackHistory, ScorePredictions
from ..Scoring import Correlation, MaxAbsoluteError, SumAbsoluteError, RootMeanSquareError


def MultiviewPredict(X_train: ArrayOrRuns,
					 Y_train: ArrayOrRuns,
					 X_test: Optional[ArrayOrRuns] = None,
					 Y_test: Optional[ArrayOrRuns] = None,
					 embedDimensions: int = 1,
					 step: int = -1,
					 predictionHorizon: int = 1,
					 knn: int = 0,
					 exclusionRadius: int = 0,
					 columnsPerView: int = 0,
					 numTopViews: int = 0,
					 isRankedInSample: bool = True,
					 isTieBreakDeterministic: bool = False,
					 scoringFunction: Callable = Correlation,
					 device = None,
					 dtype: torch.dtype = torch.float64) -> MultiviewResult:
	"""
	Predict with an ensemble of nearest-neighbor predictors: every input variable is stacked to
	embedDimensions lagged copies, every combination of columnsPerView of those state
	dimensions predicts the target with SimplexPredict and is ranked by performance, and the
	predictions of the top numTopViews combinations are averaged.

	:param X_train:	training input data, [nTrain, nFeatures] or a list of such arrays with one per run; the states are built from these variables
	:param Y_train:	training target data, [nTrain, nTargets] (1-D for one target) or a list of runs aligned sample by sample with X_train
	:param X_test:	test input data, [nTest, nFeatures] or a list of runs; these samples are predicted. None predicts the training samples in-sample
	:param Y_test:	true data for the test samples, same layout as Y_train; required with X_test, and a NaN sample is predicted but neither scored nor ranked on
	:param embedDimensions:	number of lagged copies of each input variable; the combinations are drawn from the nFeatures * embedDimensions state dimensions
	:param step:	sample offset between consecutive lagged copies; a negative offset reaches into the past
	:param predictionHorizon:	number of samples between a state and the target value it predicts
	:param knn:	number of nearest neighbors per combination; 0 means columnsPerView + 1
	:param exclusionRadius:	in-sample only: training states within this many samples of a test state are excluded from its neighbors; the test state itself is always excluded
	:param columnsPerView:	number of state dimensions in each combination; 0 means the number of input variables
	:param numTopViews:	number of top-ranked combinations whose predictions are averaged; 0 means the square root of the number of combinations
	:param isRankedInSample:	True ranks the combinations on the training samples predicting themselves (faster, optimistic); False ranks them on X_test and Y_test
	:param isTieBreakDeterministic:	True orders exactly tied neighbor distances by sample position (in-sample, by proximity to the test sample first) so repeated runs pick the same neighbors; False leaves the order to torch.topk and is faster
	:param scoringFunction:	performance metric called as scoringFunction(trueData, predictedData), used for the ranking (first target) and the final score per target
	:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
	:param dtype:	torch dtype of the distances, weights and predictions
	:return: MultiviewResult holding the ensemble Y_pred, each top combination's own Y_pred and statistics, and the performance per target
	"""
	if X_test is not None and Y_test is None:
		raise ValueError('Y_test is needed to score predictions on X_test')
	xRuns = AsRuns(X_train)
	stackedTrain = [StackHistory(x, embedDimensions, step) for x in xRuns]
	stackedTest = None if X_test is None else [StackHistory(x, embedDimensions, step) for x in AsRuns(X_test)]
	isTrainList = IsListOfRuns(X_train)
	isTestList = X_test is not None and IsListOfRuns(X_test)
	Y_true = Y_test if X_test is not None else Y_train

	nFeatures = xRuns[0].shape[1]
	nStacked = stackedTrain[0].shape[1]
	if columnsPerView <= 0:
		columnsPerView = nFeatures
	if columnsPerView > nStacked:
		warn(f'MultiviewPredict: columnsPerView = {columnsPerView} exceeds the {nStacked} stacked columns; set to {nStacked}')
		columnsPerView = nStacked
	combos = list(combinations(range(nStacked), columnsPerView))
	if numTopViews < 1:
		numTopViews = floor(sqrt(len(combos)))
	if numTopViews > len(combos):
		warn(f'MultiviewPredict: numTopViews = {numTopViews} exceeds the {len(combos)} combinations; set to {len(combos)}')
		numTopViews = len(combos)

	def selectColumns(stacked, isList, combo):
		"""
		Extract the state dimensions of one combination from every stacked run.

		:param stacked:	list of stacked state arrays, one per run
		:param isList:	True returns a list of arrays (the caller passed runs), False the single array
		:param combo:	indices of the state dimensions
		"""
		selected = [s[:, list(combo)] for s in stacked]
		return selected if isList else selected[0]

	def predict(combo, isInSample):
		"""
		Run SimplexPredict with the combination's state dimensions as the state.

		:param combo:	indices of the state dimensions
		:param isInSample:	True predicts the training samples from themselves (with exclusionRadius), False predicts X_test
		"""
		return SimplexPredict(
			X_train = selectColumns(stackedTrain, isTrainList, combo), Y_train = Y_train,
			X_test = None if isInSample else selectColumns(stackedTest, isTestList, combo),
			Y_test = Y_train if isInSample else Y_test,
			embedDimensions = 1, step = step, predictionHorizon = predictionHorizon, knn = knn,
			exclusionRadius = exclusionRadius if isInSample else 0,
			isTieBreakDeterministic = isTieBreakDeterministic, scoringFunction = scoringFunction,
			device = device, dtype = dtype)

	# rank: first target's score, NaN last
	rankInSample = isRankedInSample or X_test is None
	rankScores = numpy.array([predict(combo, rankInSample).score[0] for combo in combos], dtype = float)
	order = numpy.argsort(numpy.where(numpy.isnan(rankScores), -numpy.inf, rankScores))[::-1]
	topCombos = [combos[i] for i in order[:numTopViews]]

	# predict the requested rows with the top combinations and average
	topResults = {combo: predict(combo, X_test is None) for combo in topCombos}
	firstPrediction = next(iter(topResults.values())).Y_pred
	if isinstance(firstPrediction, list):
		Y_pred = [numpy.mean([topResults[c].Y_pred[r] for c in topCombos], axis = 0) for r in range(len(firstPrediction))]
	else:
		Y_pred = numpy.mean([topResults[c].Y_pred for c in topCombos], axis = 0)

	yTrueColumns = numpy.concatenate(AsRuns(Y_true))[:, 0]
	def firstColumn(prediction):
		"""
		Extract the first target of a prediction with its runs concatenated, for the per-combination statistics.

		:param prediction:	predicted data, an array or a list of runs
		"""
		flat = numpy.concatenate(AsRuns(prediction))
		return flat[:, 0]
	topRankStats = {}
	view = []
	for combo in topCombos:
		predicted = firstColumn(topResults[combo].Y_pred)
		stats = [Correlation(yTrueColumns, predicted), MaxAbsoluteError(yTrueColumns, predicted),
				 SumAbsoluteError(yTrueColumns, predicted), RootMeanSquareError(yTrueColumns, predicted)]
		topRankStats[combo] = stats
		view.append([str(combo)] + stats)

	return MultiviewResult(
		Y_pred = Y_pred,
		view = view,
		topRankPredictions = {combo: topResults[combo].Y_pred for combo in topCombos},
		topRankStats = topRankStats,
		columnsPerView = columnsPerView,
		embedDimensions = embedDimensions,
		predictionHorizon = predictionHorizon,
		score = ScorePredictions(scoringFunction, Y_true, Y_pred))
