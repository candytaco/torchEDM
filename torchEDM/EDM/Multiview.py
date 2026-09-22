"""
Ensemble prediction over the top-ranked combinations of stacked feature columns
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
	Every feature column is stacked to embedDimensions copies; every combination of
	columnsPerView stacked columns predicts the target with SimplexPredict and is ranked by
	score; the top numTopViews combinations are averaged.

	:param X_train:		[nTrain, nFeatures] or a list of runs
	:param Y_train:		[nTrain, nTargets] or a list of runs
	:param X_test:		[nTest, nFeatures] or a list of runs; None predicts the training rows in-sample
	:param Y_test:		targets for X_test; required with X_test
	:param embedDimensions:	copies of each feature column in the stacked state
	:param step:		row offset between copies; negative reaches into the past
	:param predictionHorizon:	rows between a state and the target it predicts
	:param knn:			neighbors per combination; 0 means columnsPerView + 1
	:param exclusionRadius:	in-sample only; training states this close in rows are not neighbors
	:param columnsPerView:	stacked columns per combination; 0 means the number of feature columns
	:param numTopViews:	combinations averaged; 0 means the square root of the number of combinations
	:param isRankedInSample:	True ranks combinations on the training rows predicting themselves
						(faster, optimistic); False ranks them on X_test/Y_test
	:param scoringFunction:	scoringFunction(actual, predicted) -> float used for ranking and the score
	:param isTieBreakDeterministic:	order exactly tied neighbor distances reproducibly
	:param device:		torch device; None picks cuda when available
	:param dtype:		torch dtype for the computation
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
		selected = [s[:, list(combo)] for s in stacked]
		return selected if isList else selected[0]

	def predict(combo, isInSample):
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
