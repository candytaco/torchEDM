from typing import List, Optional, Union

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .CVSplitter import RunSplitter, SliceRuns
from .EDMFitter import EDMFitter
from ..EDM.MDE import MDE
from ..EDM.Predictors import SimplexPredict
from ..EDM.Results import MDEResult, MDECVResults
from ..EDM.Setup import AsRuns, IsListOfRuns
from ..Scoring import Correlation


class MDEFitterCV(EDMFitter):
	"""
	MDE across leave-one-run-out or n-fold splits of the training runs, then a final
	prediction of the test data with the chosen variables.
	"""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'post',
				 Metric: str = "correlation",
				 BatchSize: int = 10000,
				 dtype: torch.dtype = torch.float32,
				 Folds: int = 5,
				 LeaveOneRunOut: bool = True,
				 FinalVariableSelection: str = "best_fold",
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 IterativeDimensionSearch: bool = False,
				 progressBar: bool = True,
				 device = None):
		"""
		:param MaxD:	columns to select per target; the target's own column counts when IncludeTarget
		:param IncludeTarget:	start with the target series itself selected, so its own value is part of every state; it then appears in selected_variables as column nFeatures + targetIndex
		:param Convergent:	'pre' screens every candidate for cross-map convergence before selection, 'post' checks candidates in score order at each step, False skips the check
		:param Metric:	'correlation' or 'r2': the score that ranks candidates at each step
		:param BatchSize:	candidate columns whose distance matrices are held on the device at once
		:param dtype:	torch dtype of the selection tensors
		:param EmbedDimensions:	fixed embedding dimensions of the target's history in the convergence check; 0 searches each candidate's own
		:param PredictionHorizon:	rows between a state and the target value it predicts
		:param KNN:	neighbors in the convergence check, in a final SMapPredict within each fold, and in Predict, 0 meaning each one's default (the target's embedding dimensions + 1, every training state, the number of selected columns plus one); each fold's final SimplexPredict always uses the number of selected columns plus one
		:param Step:	row offset between the stacked copies in the convergence check and the embedding-dimension search; negative reaches into the past
		:param ExclusionRadius:	training states within this many rows of a test state may not be its neighbors; always applied in the convergence check, which runs on the training rows, and to selection and prediction only in-sample
		:param Verbose:	print when a target stops expanding and other progress details
		:param UseSMap:	final prediction with SMapPredict instead of SimplexPredict
		:param Theta:	localization of that final SMapPredict; 0 fits one global linear map
		:param stdThreshold:	candidates whose standard deviation over the training rows is below this are dropped from the pool
		:param CCMLibraryPercentiles:	training-subset sizes of the convergence check, as percentages of the training states
		:param CCMNumSamples:	random subsets drawn per size in the convergence check
		:param CCMConvergenceThreshold:	minimum slope of cross-map skill against subset fraction for a candidate to count as convergent
		:param CCMSeed:	seed of the convergence check's subset draws; None draws fresh subsets
		:param CCMMaxEmbeddingDimensions:	largest embedding dimension tried in the per-candidate search
		:param MinPredictionThreshold:	minimum score a candidate must reach at a step to be selectable
		:param MinCandidatePerformance:	minimum peak score a candidate alone (at its best embedding dimension) must reach predicting the target to stay in the pool; 0 disables
		:param IterativeDimensionSearch:	True evaluates each embedding dimension of the per-candidate search on its own complete rows (slower, reproduces the reference); False shares the rows complete at the largest one in one pass
		:param Folds:	contiguous blocks each run is cut into when LeaveOneRunOut is False; fold k holds out block k of every run
		:param LeaveOneRunOut:	True holds out one whole run per split; False uses the n-fold blocks
		:param FinalVariableSelection:	how Predict chooses its columns: 'best_fold' takes the selection of the fold with the best score per target, 'frequency' the columns selected in the most folds, 'reselect' reruns the selection on all training runs over the union of the fold selections, without the convergence check
		:param progressBar:	show a progress bar over the folds
		:param device:	torch device for the computation; None picks cuda when available and cpu otherwise
		"""
		super().__init__(progressBar)
		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.dtype = dtype
		self.Folds = Folds
		self.LeaveOneRunOut = LeaveOneRunOut
		self.FinalVariableSelection = FinalVariableSelection
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.stdThreshold = stdThreshold
		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxEmbeddingDimensions = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.MinCandidatePerformance = MinCandidatePerformance
		self.IterativeDimensionSearch = IterativeDimensionSearch
		self.device = device

		self.xRuns = None
		self.yRuns = None
		self.X_test = None
		self.Y_test = None
		self.splitter = None
		self.foldResults = []
		self.foldAccuracies = []
		self.bestFold = None
		self.bestVariablesInFold = None
		self.bestFoldAccuracy = None

	def MDEKeywords(self, candidateColumns = None, convergenceCheck = None) -> dict:
		"""
		The keyword arguments of MDE that these settings map onto.

		:param candidateColumns:	candidate X columns for this run; None uses all
		:param convergenceCheck:	override of Convergent for this run; None keeps it
		:return: dict ready to unpack into MDE(X_train, Y_train, X_test, Y_test, **keywords)
		"""
		return dict(maxVariables = self.MaxD, isTargetIncluded = self.IncludeTarget,
					convergenceCheck = self.Convergent if convergenceCheck is None else convergenceCheck,
					candidateMetric = self.Metric, batchSize = self.BatchSize, dtype = self.dtype, candidateColumns = candidateColumns,
					embedDimensions = self.EmbedDimensions, predictionHorizon = self.PredictionHorizon,
					knn = self.KNN, step = self.Step, exclusionRadius = self.ExclusionRadius, isVerbose = self.Verbose,
					isUsingSMap = self.UseSMap, theta = self.Theta, stdThreshold = self.stdThreshold,
					convergenceSubsetPercentiles = self.CCMLibraryPercentiles, convergenceRepeats = self.CCMNumSamples,
					convergenceSlopeThreshold = self.CCMConvergenceThreshold, convergenceSeed = self.CCMSeed,
					convergenceMaxEmbedDimensions = self.CCMMaxEmbeddingDimensions,
					minPredictionScore = self.MinPredictionThreshold,
					minCandidateScore = self.MinCandidatePerformance,
					isIterativeDimensionSearch = self.IterativeDimensionSearch,
					hasProgressBar = False,
					device = self.device)

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None, initialVariables: Optional[List[int]] = None,
			scoringFunction = Correlation) -> MDECVResults:
		"""
		Run MDE on every split of the training runs and keep the per-fold selections in Result.

		:param X_train:	[nTrain, nFeatures] or a list of runs: the candidate columns; the splits are made over these runs
		:param Y_train:	[nTrain, nTargets] (1-D for one target) or a list of runs aligned with X_train
		:param X_test:	kept for Predict, which uses it when called without test data; optional
		:param Y_test:	kept for Predict; optional
		:param initialVariables:	candidate X columns; None uses all
		:param scoringFunction:	scoringFunction(actual, predicted) -> float scoring each fold's held-out prediction per target
		:return: MDECVResults with the selected columns, candidate scores, held-out predictions and score of every fold, and the best fold's selection per target
		"""
		self.xRuns = AsRuns(X_train)
		self.yRuns = AsRuns(Y_train)
		self.X_test = X_test
		self.Y_test = Y_test
		self.splitter = RunSplitter([run.shape[0] for run in self.xRuns], self.Folds, self.LeaveOneRunOut)
		nTargets = self.yRuns[0].shape[1]
		nFeatures = self.xRuns[0].shape[1]

		self.foldResults = []
		foldAccuracyRows = []
		progressBar = ProgressBar(total = self.splitter.GetNSplits(), desc = 'MDE CV Fold', leave = False, disable = self.hideProgress)
		for trainSlices, testSlices in self.splitter.Split():
			foldResult = self.FitSingleFold(SliceRuns(self.xRuns, trainSlices), SliceRuns(self.yRuns, trainSlices),
											SliceRuns(self.xRuns, testSlices), SliceRuns(self.yRuns, testSlices),
											initialVariables, scoringFunction = scoringFunction)
			self.foldResults.append(foldResult)
			foldAccuracyRows.append(foldResult.score)
			progressBar.update(1)
		progressBar.close()

		self.foldAccuracies = numpy.array(foldAccuracyRows)						# [nFolds, nTargets]
		self.bestFold = numpy.argmax(self.foldAccuracies, axis = 0)					# [nTargets]
		self.bestFoldAccuracy = self.foldAccuracies[self.bestFold, numpy.arange(nTargets)]

		self.bestVariablesInFold = numpy.full([nTargets, self.MaxD], -1, dtype = int)
		for j in range(nTargets):
			self.bestVariablesInFold[j, :] = self.foldResults[self.bestFold[j]].selected_variables[j, :]

		foldSelectedVariables = numpy.stack([r.selected_variables for r in self.foldResults], axis = 0)
		foldStepwisePerformances = numpy.stack([r.stepwise_performance for r in self.foldResults], axis = 0)[:, :, :, :nFeatures]

		self.Result = MDECVResults(
			fold_selected_variables = foldSelectedVariables,
			fold_stepwise_performances = foldStepwisePerformances,
			fold_accuracies = self.foldAccuracies,
			fold_Y_pred = [res.Y_pred for res in self.foldResults],
			best_fold = self.bestFold,
			selected_variables = self.bestVariablesInFold)
		return self.Result

	def FitSingleFold(self, X_train, Y_train, X_test, Y_test, candidateColumns = None, convergenceCheck = None,
					  scoringFunction = Correlation) -> MDEResult:
		"""
		Run MDE on one split.

		:param X_train:	the split's training runs of candidate columns
		:param Y_train:	the split's training runs of targets
		:param X_test:	the held-out runs; None scores the training rows in-sample
		:param Y_test:	targets of the held-out runs; None with X_test None
		:param candidateColumns:	candidate X columns; None uses all
		:param convergenceCheck:	override of Convergent; None keeps it
		:param scoringFunction:	scoringFunction(actual, predicted) -> float for the final score
		:return: MDEResult of the split
		"""
		return MDE(X_train, Y_train, X_test, Y_test, scoringFunction = scoringFunction,
				   **self.MDEKeywords(candidateColumns, convergenceCheck))

	def SelectedVariables(self) -> numpy.ndarray:
		"""
		The final selection per FinalVariableSelection.

		:return: [nTargets, MaxD] column indices, padded with -1
		"""
		nTargets = self.yRuns[0].shape[1]
		if self.FinalVariableSelection == 'frequency':
			return self.GetMostFrequentVariables()
		if self.FinalVariableSelection == 'reselect':
			allSelected = set()
			for foldSelection in self.Result.fold_selected_variables:
				for j in range(nTargets):
					allSelected |= set(int(v) for v in foldSelection[j] if v >= 0)
			result = self.FitSingleFold(self.xRuns if len(self.xRuns) > 1 else self.xRuns[0],
										self.yRuns if len(self.yRuns) > 1 else self.yRuns[0],
										None, None, sorted(allSelected), convergenceCheck = False)
			return result.selected_variables
		return self.bestVariablesInFold

	def Predict(self, X_test = None, Y_test = None, scoringFunction = Correlation) -> MDECVResults:
		"""
		Predict test data from all training runs with the final variables, one SimplexPredict per
		target over its selected columns, and keep the updated MDECVResults in Result.

		:param X_test:	[nTest, nFeatures] or a list of runs; None uses the X_test given to Fit
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; None uses the Y_test given to Fit
		:param scoringFunction:	scoringFunction(actual, predicted) -> float applied per target
		:return: MDECVResults carrying the fold fields of Fit plus the final selection, Y_pred shaped like Y_test, and the score per target
		"""
		if len(self.foldResults) == 0:
			raise RuntimeError('Model not fitted. Call Fit() first.')
		X_test = self.X_test if X_test is None else X_test
		Y_test = self.Y_test if Y_test is None else Y_test
		if X_test is None or Y_test is None:
			raise ValueError('No test data provided')

		variablesPerTarget = self.SelectedVariables()
		isSingleTestRun = not IsListOfRuns(X_test)
		xTestRuns = AsRuns(X_test)
		yTestRuns = AsRuns(Y_test)
		nTargets = self.yRuns[0].shape[1]
		Y_pred = [numpy.full((run.shape[0], nTargets), numpy.nan) for run in yTestRuns]
		scores = numpy.full(nTargets, numpy.nan)

		for j in range(nTargets):
			variables = [int(v) for v in variablesPerTarget[j] if v >= 0]
			if len(variables) == 0:
				continue
			result = SimplexPredict(
				[run[:, variables] for run in self.xRuns], [run[:, [j]] for run in self.yRuns],
				[run[:, variables] for run in xTestRuns], [run[:, [j]] for run in yTestRuns],
				embedDimensions = 1, step = self.Step, predictionHorizon = self.PredictionHorizon,
				knn = self.KNN if self.KNN > 0 else len(variables) + 1,
				scoringFunction = scoringFunction, device = self.device, dtype = self.dtype)
			for run, values in zip(Y_pred, result.Y_pred):
				run[:, j] = values[:, 0]
			scores[j] = result.score[0]

		self.Result = MDECVResults(
			fold_selected_variables = self.Result.fold_selected_variables,
			fold_stepwise_performances = self.Result.fold_stepwise_performances,
			fold_accuracies = self.foldAccuracies,
			fold_Y_pred = self.Result.fold_Y_pred,
			best_fold = self.bestFold,
			selected_variables = variablesPerTarget,
			Y_pred = Y_pred[0] if isSingleTestRun else Y_pred,
			score = scores)
		return self.Result

	def ReconstructFoldPredictions(self, results: MDECVResults, X_train, Y_train, scoringFunction = Correlation):
		"""
		Recompute each fold's held-out predictions from the variable selections stored in a
		saved MDECVResults, for files that predate fold_Y_pred.

		:param results:	the saved MDECVResults whose fold_selected_variables drive the predictions
		:param X_train:	the training runs Fit was given, so the same splits are rebuilt
		:param Y_train:	the matching target runs
		:param scoringFunction:	scoringFunction(actual, predicted) -> float applied per target and fold
		:return: (foldPredictions, foldAccuracies): per fold the Y_pred of its held-out runs, and [nFolds, nTargets] scores
		"""
		xRuns = AsRuns(X_train)
		yRuns = AsRuns(Y_train)
		splitter = RunSplitter([run.shape[0] for run in xRuns], self.Folds, self.LeaveOneRunOut)
		nTargets = yRuns[0].shape[1]

		foldPredictions = []
		foldAccuracyRows = []
		for foldIndex, (trainSlices, testSlices) in enumerate(splitter.Split()):
			testX = SliceRuns(xRuns, testSlices)
			testY = SliceRuns(yRuns, testSlices)
			foldPrediction = [numpy.full((run.shape[0], nTargets), numpy.nan) for run in testY]
			foldAccuracyRow = numpy.full(nTargets, numpy.nan)
			for j in range(nTargets):
				variables = [int(v) for v in results.fold_selected_variables[foldIndex, j, :] if v >= 0]
				if len(variables) == 0:
					continue
				result = SimplexPredict(
					[run[:, variables] for run in SliceRuns(xRuns, trainSlices)],
					[run[:, [j]] for run in SliceRuns(yRuns, trainSlices)],
					[run[:, variables] for run in testX], [run[:, [j]] for run in testY],
					embedDimensions = 1, step = self.Step, predictionHorizon = self.PredictionHorizon,
					knn = len(variables) + 1, scoringFunction = scoringFunction, device = self.device, dtype = self.dtype)
				for run, values in zip(foldPrediction, result.Y_pred):
					run[:, j] = values[:, 0]
				foldAccuracyRow[j] = result.score[0]
			foldPredictions.append(foldPrediction)
			foldAccuracyRows.append(foldAccuracyRow)
		return foldPredictions, numpy.array(foldAccuracyRows)

	def GetMostFrequentVariables(self) -> numpy.ndarray:
		"""
		Per target, the MaxD columns selected in the most folds.

		:return: [nTargets, MaxD] column indices, padded with -1
		"""
		nTargets = self.yRuns[0].shape[1]
		result = numpy.full([nTargets, self.MaxD], -1, dtype = int)
		for j in range(nTargets):
			counts = {}
			for foldResult in self.foldResults:
				for column in foldResult.selected_variables[j]:
					if column >= 0:
						counts[int(column)] = counts.get(int(column), 0) + 1
			ranked = sorted(counts.items(), key = lambda item: item[1], reverse = True)[:self.MaxD]
			for k, (column, _) in enumerate(ranked):
				result[j, k] = column
		return result
