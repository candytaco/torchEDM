from typing import Optional, List, Union

import numpy
from tqdm import tqdm as ProgressBar

from .DataAdapter import DataAdapter
from .EDMFitter import EDMFitter
from .CVSplitter import EDMCVSplitter
from ..EDM.MDE import MDE
from ..EDM.Results import MDEResult, MDECVResults
from ..EDM.Simplex import Simplex


class MDEFitterCV(EDMFitter):
	"""
	MDE with cross-validation that supports both n-fold and leave-one-run-out CV.
	"""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'pre',
				 Metric: str = "correlation",
				 BatchSize: int = 10000,
				 HalfPrecision: bool = False,
				 Folds: int = 5,
				 LeaveOneRunOut: bool = True,
				 FinalFeatureMode: str = "best_fold",
				 Embed: bool = False,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 knn: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 stdThreshold: float = 1e-2,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0,
				 progressBar: bool = True):
		"""
		Initialize MDE cross-validation fitter.

		:param MaxD: 				Maximum number of features to select
		:param IncludeTarget: 		Whether to start with target in feature list
		:param Convergent: 			Whether to use convergence checking
		:param Metric: 				Metric to use: "correlation" or "MAE"
		:param BatchSize: 			Number of features to process in each batch
		:param HalfPrecision: 		Use float16 instead of float32 for GPU tensors
		:param Folds: 				Number of cross-validation folds (ignored if LeaveOneRunOut is True)
		:param LeaveOneRunOut: 		If True, use leave-one-run-out CV instead of n-fold
		:param FinalFeatureMode: 	Method for selecting final features: "best_fold" or "frequency"
		:param Embed:				Whether to embed the data
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param knn: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Verbose: 			Print diagnostic messages
		:param UseSMap: 			Whether to use SMap instead of Simplex
		:param Theta: 				S-Map localization parameter
		:param stdThreshold:		Stdev threshold below which to ignore variables
		"""
		super().__init__(progressBar)

		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.HalfPrecision = HalfPrecision
		self.Folds = Folds
		self.LeaveOneRunOut = LeaveOneRunOut
		self.FinalFeatureMode = FinalFeatureMode
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = knn
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.embed = Embed
		self.stdThreshold = stdThreshold

		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.MinPredictionThreshold = MinPredictionThreshold
		self.EmbedDimCorrelationMin = EmbedDimCorrelationMin
		self.FirstEMax = FirstEMax
		self.TimeDelay = TimeDelay

		self.trainDataAdapter = None
		self.cvSplitter = None
		self.foldResults = []
		self.foldAccuracies = []
		self.bestFold = None
		self.bestFoldFeatures = None
		self.bestFoldAccuracy = None

	def Fit(self,
			XTrain: Union[numpy.ndarray, List[numpy.ndarray]],
			YTrain: Union[numpy.ndarray, List[numpy.ndarray]],
			XTest: Optional[numpy.ndarray] = None,
			YTest: Optional[numpy.ndarray] = None,
			TrainStart: int = 0,
			TrainEnd: int = 0,
			TestStart: int = 0,
			TestEnd: int = 0,
			TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None,
			initialVariables: Optional[List[int]] = None):
		"""
		Fit the model using cross-validation.

		:param XTrain:				Training features (single array or list of arrays for multiple runs)
		:param YTrain:				Training target (single array or list of arrays for multiple runs)
		:param XTest:				Test features (optional, for final prediction)
		:param YTest:				Test target (optional, for final prediction)
		:param TrainStart:			Samples to exclude at start of each run
		:param TrainEnd:			Samples to exclude at end of each run
		:param TestStart:			Samples to exclude at start of test data
		:param TestEnd:				Samples to exclude at end of test data
		:param TrainTime:			Time labels for train data
		:param TestTime:			Time labels for test data
		:param initialVariables: 	Initial columns to use
		"""
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		self.trainDataAdapter = DataAdapter.MakeDataAdapter(
			XTrain, YTrain, None, None, TrainStart, TrainEnd, 0, 0, TrainTime, None
		)

		self.cvSplitter = EDMCVSplitter(
			dataAdapter = self.trainDataAdapter,
			nFolds = self.Folds,
			leaveOneRunOut = self.LeaveOneRunOut,
			edmStyleIndices = True
		)

		trainData = self.trainDataAdapter.fullData
		target = self.trainDataAdapter.YIndex
		nTargets = len(target)

		xStart, xEnd = self.trainDataAdapter.XIndices
		effectiveColumns = initialVariables if initialVariables is not None else list(range(xStart, xEnd + 1))

		self.foldResults = []
		fold_accuracy_rows = []

		numSplits = self.cvSplitter.GetNSplits()
		progressBar = ProgressBar(total = numSplits, desc = 'MDE CV Fold', leave = False)

		for trainIndices, testIndices in self.cvSplitter.Split():
			foldResult = self.FitSingleFold(trainData, trainIndices, testIndices, target, effectiveColumns)
			self.foldResults.append(foldResult)
			fold_accuracy_rows.append(numpy.array([foldResult.score(target = j) for j in range(nTargets)]))
			progressBar.update(1)

		# foldAccuracies: [nFolds, nTargets]
		self.foldAccuracies = numpy.array(fold_accuracy_rows)

		# bestFold: [nTargets] — best fold index per target
		self.bestFold = numpy.argmax(self.foldAccuracies, axis = 0)
		self.bestFoldAccuracy = self.foldAccuracies[self.bestFold, numpy.arange(nTargets)]

		# bestFoldFeatures: [nTargets, maxD]
		self.bestFoldFeatures = numpy.full([nTargets, self.MaxD], -1, dtype = int)
		for j in range(nTargets):
			self.bestFoldFeatures[j, :] = self.foldResults[self.bestFold[j]].selected_features[j, :]

		foldSelectedFeatures = numpy.stack([r.selected_features for r in self.foldResults], axis = 0)
		foldStepwisePerformances = numpy.stack([r.stepwise_performance for r in self.foldResults], axis = 0)

		self.Result = MDECVResults(
			fold_selected_features = foldSelectedFeatures,
			fold_stepwise_performances = foldStepwisePerformances,
			fold_accuracies = self.foldAccuracies,
			best_fold = self.bestFold,
			selected_features = self.bestFoldFeatures
		)
		return self.Result

	def FitSingleFold(self,
					  data: numpy.ndarray,
					  trainIndices: List[int],
					  testIndices: List[int],
					  target: Union[int, List[int]],
					  initialVariables: Optional[List[int]] = None,
					  convergent: bool = None,
					  return_predictions: bool = True) -> MDEResult:
		"""
		Fit MDE on a single cross-validation fold.

		:param data: 			Full data array
		:param trainIndices: 	EDM-style train indices [start1, end1, start2, end2, ...]
		:param testIndices: 	EDM-style test indices [start1, end1, start2, end2, ...]
		:param target: 			Target column index
		:param initialVariables: Initial columns to use
		:param return_predictions: If False, the predictions field of the result will not be populated
		:return: 				MDEResult for this fold
		"""
		mde = MDE(
			data = data,
			target = target,
			maxD = self.MaxD,
			include_target = self.IncludeTarget,
			convergent = convergent if convergent is not None else self.Convergent, # for final call without convergence
			metric = self.Metric,
			batch_size = self.BatchSize,
			use_half_precision = self.HalfPrecision,
			columns = initialVariables,
			train = trainIndices,
			test = testIndices,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			embedded = not self.embed,
			noTime = not self.trainDataAdapter.HasTime,
			verbose = self.Verbose,
			useSMap = self.UseSMap,
			theta = self.Theta,
			stdThreshold = self.stdThreshold,
			CCMLibraryPercentiles = self.CCMLibraryPercentiles,
			CCMNumSamples = self.CCMNumSamples,
			CCMConvergenceThreshold = self.CCMConvergenceThreshold,
			MinPredictionThreshold = self.MinPredictionThreshold,
			EmbedDimCorrelationMin = self.EmbedDimCorrelationMin,
			FirstEMax = self.FirstEMax,
			TimeDelay = self.TimeDelay
		)

		return mde.Run(return_predictions = return_predictions)

	def Predict(self, XTest: numpy.ndarray = None, YTest: numpy.ndarray = None,
				TestStart = None, TestEnd = None,
				testTime: Optional[numpy.ndarray] = None,
				return_predictions: bool = True
				) -> MDECVResults:
		"""
		Predict using the final chosen feature set on test data.

		:param XTest: 	Test features (uses stored test data if None)
		:param YTest: 	Test target (uses stored test data if None)
		:param return_predictions: If False, the predictions field of the result will not be populated
		:return: 		Cross-validation results including final prediction
		"""
		if len(self.foldResults) == 0:
			raise RuntimeError("Model not fitted. Call Fit() first.")

		if XTest is None:
			XTest = self.DataAdapter.XTest
		if YTest is None:
			YTest = self.DataAdapter.YTest

		if TestStart is None:
			TestStart = self.DataAdapter.TestStart
		if TestEnd is None:
			TestEnd = self.DataAdapter.TestEnd
		if testTime is None:
			testTime = self.DataAdapter.testTime

		if XTest is None or YTest is None:
			raise ValueError("No test data provided")

		self.DataAdapter = DataAdapter.MakeDataAdapter(self.trainDataAdapter.XTrain, self.trainDataAdapter.YTrain,
													   XTest, YTest, self.trainDataAdapter.TrainStart,
													   self.trainDataAdapter.TrainEnd, TestStart, TestEnd,
													   self.trainDataAdapter.trainTime, testTime)

		yIndices = self.DataAdapter.YIndex
		nTargets = len(yIndices)

		if self.FinalFeatureMode == "frequency":
			featuresPerTarget = self.GetFrequencyFeatures()
		elif self.FinalFeatureMode == 'reselect':
			allSelected = set()
			for i in range(self.Result.fold_selected_features.shape[0]):
				for j in range(nTargets):
					row = self.Result.fold_selected_features[i, j, :]
					allSelected |= set(int(f) for f in row[row != -1])
			allSelected -= set(yIndices)
			allSelectedList = sorted(allSelected)
			reselectColumns = allSelectedList + yIndices
			reselectData = self.DataAdapter.fullData[:, reselectColumns]
			reselectTargets = list(range(len(allSelectedList), len(reselectColumns)))
			res = self.FitSingleFold(reselectData, self.DataAdapter.TrainIndices, self.DataAdapter.TestIndices,
									 reselectTargets, convergent = False)
			# Map indices back to original columns
			featuresPerTarget = numpy.full_like(res.selected_features, -1)
			for j in range(nTargets):
				for k in range(res.selected_features.shape[1]):
					f = res.selected_features[j, k]
					if f != -1:
						featuresPerTarget[j, k] = reselectColumns[f]
		else:
			featuresPerTarget = self.bestFoldFeatures

		timeValues = None
		predictions = None

		if return_predictions:
			for j in range(nTargets):
				featuresJ = featuresPerTarget[j, :]
				featuresJ = featuresJ[featuresJ != -1].tolist()

				simplex = Simplex(
					data = self.DataAdapter.fullData,
					columns = featuresJ,
					target = yIndices[j],
					train = self.DataAdapter.TrainIndices,
					test = self.DataAdapter.TestIndices,
					embedDimensions = self.EmbedDimensions,
					predictionHorizon = self.PredictionHorizon,
					knn = self.KNN,
					step = self.Step,
					exclusionRadius = self.ExclusionRadius,
					noTime = not self.DataAdapter.HasTime,
					verbose = self.Verbose,
					embedded = True
				)

				resultJ = simplex.Run()

				if j == 0:
					timeValues = resultJ.time
					nPredictions = len(timeValues)
					predictions = numpy.zeros([nPredictions, nTargets])

				predictions[:, j] = resultJ.projection[:, 2]

		self.Result = MDECVResults(
			fold_selected_features = self.Result.fold_selected_features,
			fold_stepwise_performances = self.Result.fold_stepwise_performances,
			fold_accuracies = self.foldAccuracies,
			best_fold = self.bestFold,
			selected_features = featuresPerTarget,
			time = timeValues,
			predictions = predictions
		)
		return self.Result


	def GetFrequencyFeatures(self) -> numpy.ndarray:
		"""
		Get most frequent features across folds, per target.

		:return: Selected features array [nTargets, maxD] padded with -1
		"""
		nTargets = len(self.trainDataAdapter.YIndex)
		result = numpy.full([nTargets, self.MaxD], -1, dtype = int)

		for j in range(nTargets):
			featureCounts = {}
			for i in range(self.Result.fold_selected_features.shape[0]):
				row = self.Result.fold_selected_features[i, j, :]
				for feature in row[row != -1]:
					feature = int(feature)
					featureCounts[feature] = featureCounts.get(feature, 0) + 1

			sortedFeatures = sorted(featureCounts.items(), key = lambda x: x[1], reverse = True)
			topFeatures = [feature for feature, count in sortedFeatures[:self.MaxD]]
			result[j, :len(topFeatures)] = topFeatures

		return result
