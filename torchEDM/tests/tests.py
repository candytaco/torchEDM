"""
torchEDM test suite.

Reference values come from pyEDM 2.5.7 (the validation CSVs). The reference addressed
rows through windows into one array; the tests reproduce each configuration with
X_train/Y_train/X_test/Y_test arrays:
- a test array starts early enough that the reference's first test state has complete
  history, and rows that the reference did not score get NaN in Y_test (predicted but
  never scored);
- a test window that lay inside the training window is run in-sample (X_test omitted);
- where the reference's training rows differ from the bounds-only rule (negative
  horizons, targets reaching past a window edge), trainRowMask reproduces them.
"""
import importlib.resources
import os
import unittest
from warnings import filterwarnings

import numpy
import torch
from numpy import nan
from pandas import read_csv

from torchEDM.EDM.Predictors import SimplexPredict, SMapPredict, SimplexGenerate, SMapGenerate
from torchEDM.EDM.Multiview import MultiviewPredict
from torchEDM.EDM.ConvergentCrossMap import ConvergentCrossMap
from torchEDM.EDM.MDE import MDE
from torchEDM.EDM.MDECV import MDECV
from torchEDM.EDM.Results import ResultsIO
from torchEDM.EDM.Setup import PreparePrediction
from torchEDM.EDM._core import ComputePairwiseDistances, SelectNearestNeighbors
from torchEDM.Fitters import SimplexFitter, SMapFitter, MultiviewFitter, CCMFitter, MDEFitter, MDEFitterCV
from torchEDM.ExampleData import dataFileNames
import torchEDM.Hyperparameters as Hyperparameters

filterwarnings('ignore', category = DeprecationWarning)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

sampleDataFrames = {}
for fileName, dataName in dataFileNames:
	ref = importlib.resources.files('torchEDM') / ('data/' + fileName)
	with importlib.resources.as_file(ref) as filePath:
		sampleDataFrames[dataName] = read_csv(filePath)


def Frame(name):
	"""A fresh copy of a sample frame, so tests that write NaN never touch each other."""
	return sampleDataFrames[name].copy()


def Validation(name):
	return read_csv(os.path.join(_TESTS_DIR, 'validation', name))


def CircleWithNaN():
	df = Frame('circle')
	df.iloc[[5, 6, 12], 1] = nan
	df.iloc[[10, 11, 17], 2] = nan
	return df.values.astype(float)


class test_Predictors(unittest.TestCase):

	# ---------------------------------------------------------------- Simplex

	def test_simplex_stacked_history(self):
		"""One column stacked to three copies, out of sample (Smplx_E3_block_3sp)."""
		d = Frame('block_3sp').values.astype(float)
		x = Frame('block_3sp').columns.get_loc('x_t')
		# reference test rows 100..194 predict 101..195; the two extra leading rows carry the history
		result = SimplexPredict(d[0:100, x], d[0:100, x], d[98:196, x], d[98:196, x], embedDimensions = 3, predictionHorizon = 1)
		reference = Validation('Smplx_E3_block_3sp_valid.csv')['Predictions'].values	# row k <-> data row 100 + k
		self.assertTrue(numpy.allclose(reference[1:96], result.Y_pred[3:98], atol = 1e-6))
		self.assertTrue(numpy.isnan(result.Y_pred[:3]).all())
		self.assertEqual(result.Y_pred.shape, d[98:196, x].shape)
		self.assertEqual(result.knn, 4)

		fitted = SimplexFitter(EmbedDimensions = 3, PredictionHorizon = 1).Fit(d[0:100, x], d[0:100, x], d[98:196, x], d[98:196, x])
		self.assertTrue(numpy.allclose(fitted.Y_pred, result.Y_pred, equal_nan = True))
		self.assertTrue(numpy.allclose(fitted.score, result.score))

	def test_simplex_columns_as_state(self):
		"""Three columns as the state, no stacking (Smplx_E3_embd_block_3sp)."""
		df = Frame('block_3sp'); d = df.values.astype(float)
		cols = [df.columns.get_loc(c) for c in ('x_t', 'y_t', 'z_t')]; x = cols[0]
		result = SimplexPredict(d[0:99, cols], d[0:99, x], d[99:198, cols], d[99:198, x], embedDimensions = 1, predictionHorizon = 1)
		reference = Validation('Smplx_E3_embd_block_3sp_valid.csv')['Predictions'].values	# row k <-> data row 99 + k
		self.assertTrue(numpy.allclose(reference[1:98], result.Y_pred[1:98], atol = 1e-6))
		self.assertEqual(result.knn, 4)

	def test_simplex_negative_horizon(self):
		"""Predicting two rows back, in-sample (Smplx_negTp_block_3sp)."""
		df = Frame('block_3sp'); d = df.values.astype(float)
		x, y = df.columns.get_loc('x_t'), df.columns.get_loc('y_t')
		# the reference drops one more leading training row than the bounds-only rule keeps
		mask = numpy.ones(100, bool); mask[:3] = False
		result = SimplexPredict(d[0:100, x], d[0:100, y], embedDimensions = 3, predictionHorizon = -2, trainRowMask = mask)
		reference = Validation('Smplx_negTp_block_3sp_valid.csv')['Predictions'].values	# row k <-> data row 47 + k, last two unpredicted
		self.assertTrue(numpy.allclose(reference[:31], result.Y_pred[47:78], atol = 1e-6))

	def test_simplex_train_row_mask(self):
		"""Only rows of large amplitude may serve as training states, in-sample (Smplx_validLib)."""
		df = Frame('circle'); d = df.values.astype(float); x = df.columns.get_loc('x')
		mask = df.eval('x > 0.5 | x < -0.5').values
		result = SimplexPredict(d[0:200, x], d[0:200, x], embedDimensions = 2, predictionHorizon = 1, trainRowMask = mask)
		reference = Validation('Smplx_validLib_valid.csv')['Predictions'].values	# row k <-> data row 1 + k
		self.assertTrue(numpy.allclose(reference[1:195], result.Y_pred[2:196], atol = 1e-6, equal_nan = True))

	def test_simplex_two_training_runs(self):
		"""Two training runs and step -3 (Smplx_disjointLib). The reference let the first window's
		last state target the row after the window, so that row is included and masked."""
		df = Frame('circle'); d = df.values.astype(float); x = df.columns.get_loc('x')
		firstRunMask = numpy.ones(41, bool); firstRunMask[40] = False
		X_train = [d[0:41, x], d[49:130, x]]
		reference = Validation('Smplx_disjointLib_valid.csv')['Predictions'].values	# row k <-> data row 79 + k
		# out of sample beyond the second training run: states 129..169 predict rows 130..170
		out = SimplexPredict(X_train, X_train, d[126:171, x], d[126:171, x], embedDimensions = 2, step = -3,
							 predictionHorizon = 1, trainRowMask = [firstRunMask, None])
		self.assertTrue(numpy.allclose(reference[51:92], out.Y_pred[4:45], atol = 1e-6))
		# in-sample inside the second training run: rows 80..129
		inSample = SimplexPredict(X_train, X_train, embedDimensions = 2, step = -3, predictionHorizon = 1,
								  trainRowMask = [firstRunMask, None])
		self.assertEqual(len(inSample.Y_pred), 2)
		self.assertTrue(numpy.allclose(reference[1:51], inSample.Y_pred[1][31:81], atol = 1e-6))

	def test_simplex_disjoint_test_runs_with_nan(self):
		"""Three training runs, five test runs, NaN in the data (Smplx_disjointPred_nan)."""
		df = Frame('Lorenz5D'); df.iloc[[8, 50, 501], [1, 2]] = nan
		d = df.values.astype(float); v1, v2 = df.columns.get_loc('V1'), df.columns.get_loc('V2')
		# training windows (0,50), (100,200), (250,500); the first two target into the gap after them
		X_train = [d[0:52, v1], d[100:202, v1], d[250:500, v1]]
		Y_train = [d[0:52, v2], d[100:202, v2], d[250:500, v2]]
		masks = [numpy.r_[numpy.ones(50, bool), False, False], numpy.r_[numpy.ones(100, bool), False, False], None]
		reference = Validation('Smplx_disjointPred_nan_valid.csv')['Predictions'].values
		# reference rows: window (0,10) -> data rows 4..11, (150,155) -> 150..156, (550,555) -> 550..556,
		# (880,885) -> 880..886, (990,1000) -> 990..1001; the first two rows of every window are unpredicted
		inSample = SimplexPredict(X_train, Y_train, embedDimensions = 5, predictionHorizon = 2, trainRowMask = masks)
		out = SimplexPredict(X_train, Y_train, [d[546:557, v1], d[876:887, v1], d[986:1000, v1]],
							 [d[546:557, v2], d[876:887, v2], d[986:1000, v2]], embedDimensions = 5, predictionHorizon = 2,
							 trainRowMask = masks)
		expected, actual = [], []
		for k in range(2, 8):		# window 1: rows 6..11 in run 0
			expected.append(reference[k]); actual.append(inSample.Y_pred[0][4 + k])
		for k in range(2, 7):		# window 2: rows 152..156 in run 1 (offset 100)
			expected.append(reference[8 + k]); actual.append(inSample.Y_pred[1][50 + k])
		for w, (start, base) in enumerate([(546, 15), (876, 22), (986, 29)]):
			count = 7 if w < 2 else 12
			for k in range(2, count):
				row = [550, 880, 990][w] + k
				if row >= 1000:
					continue
				expected.append(reference[base + k]); actual.append(out.Y_pred[w][row - start])
		expected, actual = numpy.array(expected), numpy.array(actual)
		self.assertTrue(numpy.array_equal(numpy.isnan(expected), numpy.isnan(actual)))
		self.assertTrue(numpy.allclose(expected, actual, atol = 1e-5, equal_nan = True))

	def test_simplex_exclusion_radius(self):
		"""In-sample with training states within five rows barred (Smplx_exclRadius)."""
		df = Frame('circle'); d = df.values.astype(float); x, y = df.columns.get_loc('x'), df.columns.get_loc('y')
		result = SimplexPredict(d[0:100, x], d[0:100, y], embedDimensions = 2, predictionHorizon = 1, exclusionRadius = 5)
		reference = Validation('Smplx_exclRadius_valid.csv')['Predictions'].values	# row k <-> data row 20 + k
		self.assertTrue(numpy.allclose(reference[1:61], result.Y_pred[21:81], atol = 1e-6))

	def test_simplex_exclusion_radius_needs_in_sample(self):
		d = Frame('circle').values.astype(float)
		with self.assertRaises(ValueError):
			SimplexPredict(d[0:100, 1], d[0:100, 2], d[100:150, 1], embedDimensions = 2, exclusionRadius = 5)

	def test_simplex_nan(self):
		"""NaN in the feature and target columns, in-sample (Smplx_nan)."""
		d = CircleWithNaN()
		result = SimplexPredict(d[0:100, 1], d[0:100, 2], embedDimensions = 2, predictionHorizon = 1)
		reference = Validation('Smplx_nan_valid.csv')['Predictions'].values	# row k <-> data row 1 + k
		self.assertTrue(numpy.allclose(reference[1:90], result.Y_pred[2:91], atol = 1e-6, equal_nan = True))
		self.assertTrue(numpy.array_equal(numpy.isnan(reference[1:90]), numpy.isnan(result.Y_pred[2:91])))

	def test_simplex_nan2(self):
		d = CircleWithNaN()
		result = SimplexPredict(d[0:200, 2], d[0:200, 1], embedDimensions = 2, predictionHorizon = 1)
		reference = Validation('Smplx_nan2_valid.csv')['Predictions'].values	# row k <-> data row 1 + k
		self.assertTrue(numpy.allclose(reference[1:190], result.Y_pred[2:191], atol = 1e-6, equal_nan = True))

	def test_simplex_deterministic_tie_order(self):
		"""Flow data with long runs of repeated values, in-sample: exactly tied distances are ordered
		by temporal proximity then row, reproducing the reference (Smplx_SumFlow_inSample)."""
		x = Frame('SumFlow_1980-2005').values[:, 1].astype(float)
		reference = Validation('Smplx_SumFlow_inSample_valid.csv')
		result = SimplexPredict(x, x, embedDimensions = 3, predictionHorizon = 1, isTieBreakDeterministic = True)
		rows = reference['Row'].values
		self.assertTrue(numpy.allclose(reference['Predictions'].values, result.Y_pred[rows], atol = 1e-6, equal_nan = True))

	def test_simplex_neighbor_rows(self):
		"""With one neighbor, the neighbor of each in-sample state is the reference's row (knn = 1 fixture)."""
		df = Frame('Lorenz5D'); d = df.values.astype(float); v5 = df.columns.get_loc('V5')
		inputs = PreparePrediction(d[300:399, v5], d[300:399, v5], None, 1, -1, 1)
		distances = ComputePairwiseDistances(torch.tensor(inputs.trainStates), torch.tensor(inputs.testStates))
		distances[torch.tensor(inputs.exclusionMask)] = float('inf')
		_, neighbors = SelectNearestNeighbors(distances, 1)
		neighborRows = inputs.trainRows[neighbors[0].numpy()] + 300
		testColumns = [int(numpy.where(inputs.testRows == row - 300)[0][0]) for row in range(349, 355)]
		self.assertEqual(neighborRows[testColumns].tolist(), [322, 334, 362, 387, 356, 355])

	def test_exclusion_radius_neighbors(self):
		"""On a linear ramp the nearest allowed neighbor of every in-sample state sits exactly one row
		outside the exclusion radius."""
		ramp = numpy.arange(1.0, 1001.0)
		inputs = PreparePrediction(ramp, ramp, None, 5, -1, 1, exclusionRadius = 10)
		distances = ComputePairwiseDistances(torch.tensor(inputs.trainStates), torch.tensor(inputs.testStates))
		distances[torch.tensor(inputs.exclusionMask)] = float('inf')
		_, neighbors = SelectNearestNeighbors(distances, 1, isTieBreakDeterministic = True, trainRows = inputs.trainRows, testRows = inputs.testRows)
		gaps = numpy.abs(inputs.trainRows[neighbors[0].numpy()] - inputs.testRows)
		self.assertTrue((gaps == 11).all())

	def test_simplex_multiple_targets_and_runs(self):
		"""Two targets give the two single-target answers; two test runs give the two single-run answers."""
		df = Frame('block_3sp'); d = df.values.astype(float)
		x, y = df.columns.get_loc('x_t'), df.columns.get_loc('y_t')
		both = SimplexPredict(d[0:100, x], d[0:100, [x, y]], d[98:150, x], d[98:150, [x, y]], embedDimensions = 3)
		for column, target in enumerate((x, y)):
			single = SimplexPredict(d[0:100, x], d[0:100, target], d[98:150, x], d[98:150, target], embedDimensions = 3)
			self.assertTrue(numpy.allclose(both.Y_pred[:, column], single.Y_pred, equal_nan = True))
			self.assertAlmostEqual(both.score[column], single.score[0])
		self.assertEqual(both.Y_pred.shape, (52, 2))
		runs = SimplexPredict(d[0:100, x], d[0:100, x], [d[98:150, x], d[150:198, x]], [d[98:150, x], d[150:198, x]], embedDimensions = 3)
		second = SimplexPredict(d[0:100, x], d[0:100, x], d[150:198, x], d[150:198, x], embedDimensions = 3)
		self.assertEqual(len(runs.Y_pred), 2)
		self.assertTrue(numpy.allclose(runs.Y_pred[1], second.Y_pred, equal_nan = True))
		with self.assertRaises(ValueError):
			SimplexPredict(d[0:100, x], d[0:100, x], d[98:150, x], d[98:151, x], embedDimensions = 3)

	def test_simplex_default_neighbor_count(self):
		"""State size plus one: two columns stacked to 5 embedding dimensions give eleven neighbors."""
		df = Frame('Lorenz5D'); d = df.values.astype(float)
		cols = [df.columns.get_loc('V1'), df.columns.get_loc('V3')]; v5 = df.columns.get_loc('V5')
		result = SimplexPredict(d[1:301, cols], d[1:301, v5], d[297:311, cols], d[297:311, v5], embedDimensions = 5)
		self.assertEqual(result.knn, 11)
		self.assertEqual(result.Y_pred.shape, (14,))

	# ---------------------------------------------------------------- generation

	def test_generate_simplex(self):
		d = Frame('circle').values.astype(float)
		generated = SimplexGenerate(d[0:200, 1], 100, embedDimensions = 2)
		self.assertEqual(generated.shape, (100, 1))
		self.assertTrue(numpy.isfinite(generated).all())
		self.assertTrue((numpy.abs(generated) < 1.1).all())
		L = Frame('Lorenz5D').values.astype(float)
		self.assertEqual(SimplexGenerate(L[0:1000, 1], 100, embedDimensions = 5).shape, (100, 1))
		self.assertEqual(SimplexGenerate(L[0:1000, 1:4], 20, embedDimensions = 2).shape, (20, 3))

	def test_generate_smap(self):
		d = Frame('circle').values.astype(float)
		generated = SMapGenerate(d[0:200, 1], 100, embedDimensions = 2, theta = 3.0)
		self.assertEqual(generated.shape, (100, 1))
		self.assertTrue(numpy.isfinite(generated).all())

	# ---------------------------------------------------------------- SMap

	def test_smap_stacked_history(self):
		"""One column stacked to four copies, theta 3 (SMap_circle_E4). The reference used every
		training state but one as neighbors."""
		d = Frame('circle').values.astype(float)
		result = SMapPredict(d[0:100, 1], d[0:100, 1], d[106:161, 1], d[106:161, 1], embedDimensions = 4, predictionHorizon = 1,
							 theta = 3.0, knn = 95)
		reference = Validation('SMap_circle_E4_valid.csv')['Predictions'].values	# row k <-> data row 109 + k
		self.assertTrue(numpy.allclose(reference[1:50], result.Y_pred[4:53], atol = 1e-6))
		self.assertEqual(result.coefficients.shape, (55, 5))
		self.assertEqual(result.singularValues.shape, (55, 5))
		fitted = SMapFitter(EmbedDimensions = 4, PredictionHorizon = 1, Theta = 3.0, KNN = 95).Fit(d[0:100, 1], d[0:100, 1], d[106:161, 1], d[106:161, 1])
		self.assertTrue(numpy.allclose(fitted.Y_pred, result.Y_pred, equal_nan = True))

	def test_smap_columns_as_state(self):
		"""Two columns as the state, in-sample (SMap_circle_E2_embd)."""
		d = Frame('circle').values.astype(float)
		result = SMapPredict(d[0:199, [1, 2]], d[0:199, 1], embedDimensions = 1, predictionHorizon = 1, theta = 3.0)
		reference = Validation('SMap_circle_E2_embd_valid.csv')['Predictions'].values
		self.assertTrue(numpy.allclose(reference[1:195], result.Y_pred[1:195], atol = 1e-6))
		self.assertEqual(result.knn, 197)
		coefficientMeans = numpy.nanmean(result.coefficients, axis = 0)
		self.assertTrue(numpy.allclose(coefficientMeans[1:3], [0.99801, 0.06311], atol = 1e-3))

	def test_smap_nan(self):
		"""NaN in the feature and target columns, in-sample (SMap_nan)."""
		d = CircleWithNaN()
		result = SMapPredict(d[0:50, 1], d[0:50, 2], embedDimensions = 2, predictionHorizon = 1, theta = 3.0)
		reference = Validation('SMap_nan_valid.csv')['Predictions'].values	# row k <-> data row 1 + k
		self.assertTrue(numpy.allclose(reference[1:49], result.Y_pred[2:50], atol = 1e-6, equal_nan = True))

	def test_smap_out_of_sample(self):
		"""(SMap_noTime) The reference used every training state but one as neighbors."""
		d = Frame('circle_noTime').values.astype(float)
		result = SMapPredict(d[0:100, 0], d[0:100, 1], d[99:151, 0], d[99:151, 1], embedDimensions = 2, predictionHorizon = 1,
							 theta = 3.0, knn = 97)
		reference = Validation('SMap_noTime_valid.csv')['Predictions'].values	# row k <-> data row 100 + k
		self.assertTrue(numpy.allclose(reference[1:50], result.Y_pred[2:51], atol = 1e-6))

	def test_results_io_roundtrip(self):
		d = Frame('circle').values.astype(float)
		path = os.path.join(_TESTS_DIR, '_roundtrip.npz')
		try:
			result = SMapPredict(d[0:100, 1], d[0:100, 1], [d[99:130, 1], d[130:160, 1]], [d[99:130, 1], d[130:160, 1]],
								 embedDimensions = 2, theta = 2.0)
			ResultsIO.Save(result, path)
			loaded = ResultsIO.Load(path)
			self.assertEqual(len(loaded.Y_pred), 2)
			self.assertTrue(numpy.allclose(loaded.Y_pred[1], result.Y_pred[1], equal_nan = True))
			self.assertTrue(numpy.allclose(loaded.coefficients[0], result.coefficients[0], equal_nan = True))
			self.assertEqual(loaded.knn, result.knn)
		finally:
			if os.path.exists(path):
				os.remove(path)


class test_Multiview(unittest.TestCase):

	def test_multiview(self):
		"""(Multiview_pred, Multiview_combos) Three rows of Y_test are NaN so that only the rows the
		reference scored count for ranking and statistics."""
		df = Frame('block_3sp'); d = df.values.astype(float)
		cols = [df.columns.get_loc(c) for c in ('x_t', 'y_t', 'z_t')]; x = cols[0]
		Y_test = d[98:198, x].copy(); Y_test[:3] = nan
		result = MultiviewPredict(d[0:100, cols], d[0:100, x], d[98:198, cols], Y_test, D = 0, embedDimensions = 3,
								  predictionHorizon = 1, isRankedInSample = False)
		reference = Validation('Multiview_pred_valid.csv')['Predictions'].values	# row k <-> data row 100 + k
		self.assertTrue(numpy.allclose(reference[1:98], result.Y_pred[3:100], atol = 1e-4))
		combos = Validation('Multiview_combos_valid.csv')
		self.assertTrue(numpy.allclose(combos['correlation'].values, [row[1] for row in result.view], atol = 1e-4))
		self.assertTrue(numpy.allclose(combos['RMSE'].values, [row[4] for row in result.view], atol = 1e-4))
		self.assertEqual(result.D, 3)
		self.assertEqual(len(result.topRankPredictions), 9)

		fitted = MultiviewFitter(EmbedDimensions = 3, PredictionHorizon = 1, IsRankedInSample = False).Fit(d[0:100, cols], d[0:100, x], d[98:198, cols], Y_test)
		self.assertTrue(numpy.allclose(fitted.Y_pred, result.Y_pred, equal_nan = True))


class test_CrossMap(unittest.TestCase):
	"""Random training subsets differ from the reference's draws, so tolerances are loose."""

	sizes = [10, 20, 30, 40, 50, 60, 70, 75]

	def test_forward_and_reverse(self):
		df = Frame('sardine_anchovy_sst'); d = df.values.astype(float)
		a, t = df.columns.get_loc('anchovy'), df.columns.get_loc('np_sst')
		reference = Validation('CCM_anch_sst_valid.csv').values
		forward = ConvergentCrossMap(d[:, [a, a]], d[:, t], trainSizes = self.sizes, repeats = 100, embedDimensions = 3,
									 predictionHorizon = 0, showProgress = False).Run()
		self.assertTrue(numpy.allclose(forward.forward_performance[:, 0], reference[:, 1], atol = 5e-2))
		self.assertTrue(numpy.allclose(forward.forward_performance[:, 1], reference[:, 1], atol = 5e-2))
		# the reverse direction is the same call with X and Y exchanged
		reverse = ConvergentCrossMap(d[:, t], d[:, a], trainSizes = self.sizes, repeats = 100, embedDimensions = 3,
									 predictionHorizon = 0, showProgress = False).Run()
		self.assertTrue(numpy.allclose(reverse.forward_performance, reference[:, 2], atol = 5e-2))
		fitted = CCMFitter(TrainSizes = self.sizes, numRepeats = 100, EmbedDimensions = 3, PredictionHorizon = 0, progressBar = False).Fit(d[:, a], d[:, t])
		self.assertTrue(numpy.allclose(fitted.forward_performance, reference[:, 1], atol = 5e-2))

	def test_sample_mode(self):
		df = Frame('sardine_anchovy_sst'); d = df.values.astype(float)
		a, t = df.columns.get_loc('anchovy'), df.columns.get_loc('np_sst')
		reference = Validation('CCM_anch_sst_valid.csv').values
		result = ConvergentCrossMap(d[:, a], d[:, t], trainSizes = self.sizes, repeats = 100, embedDimensions = 3,
									predictionHorizon = 0, batchMode = 'sample', seed = 3, showProgress = False).Run()
		self.assertTrue(numpy.allclose(result.forward_performance, reference[:, 1], atol = 5e-2))

	def test_nan(self):
		d = CircleWithNaN()
		reference = Validation('CCM_nan_valid.csv').values
		result = ConvergentCrossMap(d[:, 1], d[:, 2], trainSizes = list(range(10, 191, 10)), repeats = 20, embedDimensions = 2,
									predictionHorizon = 5, seed = 777, showProgress = False).Run()
		self.assertTrue(numpy.allclose(result.forward_performance, reference[:, 1], atol = 5e-2))

	def test_many_sources_and_targets(self):
		df = Frame('columnNameSpace'); d = df.values.astype(float)
		sources = [df.columns.get_loc(c) for c in ('Var 1', 'Var3', 'Var 5 1')]
		targets = [df.columns.get_loc(c) for c in ('Var 2', 'Var 4 A')]
		result = ConvergentCrossMap(d[:, sources], d[:, targets], trainSizes = [20, 50, 90], repeats = 3, embedDimensions = 5,
									predictionHorizon = 0, seed = 777, showProgress = False).Run()
		self.assertEqual(result.forward_performance.shape, (3, 3, 2))
		self.assertTrue(numpy.isfinite(result.forward_performance).all())
		searched = ConvergentCrossMap(d[:, sources], trainSizes = [20, 50, 90], repeats = 3, maxEmbedDimensions = 4,
									  showProgress = False).Run()
		self.assertEqual(len(searched.forward_embed_dimensions), 3)
		self.assertEqual(searched.forward_performance.shape, (3, 3, 3))


class test_Hyperparameters(unittest.TestCase):

	def lorenzArrays(self):
		df = Frame('Lorenz5D'); d = df.values.astype(float); v1 = df.columns.get_loc('V1')
		# reference test states 500..799 at embedding dimensions up to 12 with step -5: the test arrays start 55 rows
		# early for the history, and the targets of states before row 500 are NaN so they are not scored
		Y_test = d[445:815, v1].copy(); Y_test[:70] = nan
		return d[0:485, v1], d[445:815, v1], Y_test

	def test_embed_dimension(self):
		"""(EmbedDim) Each embedding dimension on its own complete rows reproduces the reference."""
		X_train, X_test, Y_test = self.lorenzArrays()
		scores = Hyperparameters.FindOptimalEmbeddingDimensionality(X_train, X_train, X_test, Y_test, maxDims = 12,
																	predictionHorizon = 15, step = -5, batched = False)
		reference = Validation('EmbedDim_valid.csv').values[:, 1]
		self.assertTrue(numpy.allclose(scores, reference, atol = 1e-6))

	def test_embed_dimension_batched(self):
		"""The shared-row pass trims the training rows at the largest embedding dimension, so it is close, not exact."""
		X_train, X_test, Y_test = self.lorenzArrays()
		scores = Hyperparameters.FindOptimalEmbeddingDimensionality(X_train, X_train, X_test, Y_test, maxDims = 12,
																	predictionHorizon = 15, step = -5, batched = True)
		reference = Validation('EmbedDim_valid.csv').values[:, 1]
		self.assertTrue(numpy.allclose(scores, reference, atol = 5e-2))

	def test_embed_dimension_variants(self):
		"""Per-column, self-prediction, and multi-target sweeps reduce to the joint single-column sweep."""
		X_train, X_test, Y_test = self.lorenzArrays()
		common = dict(maxDims = 12, predictionHorizon = 15, step = -5, batched = True)
		joint = Hyperparameters.FindOptimalEmbeddingDimensionality(X_train, X_train, X_test, Y_test, **common)
		separate = Hyperparameters.FindOptimalEmbeddingDimensionality(numpy.column_stack([X_train, X_train]), X_train,
																	  numpy.column_stack([X_test, X_test]), Y_test, joint = False, **common)
		self.assertEqual(separate.shape, (2, 12))
		self.assertTrue(numpy.allclose(separate[0], joint) and numpy.allclose(separate[1], joint))
		selfPrediction = Hyperparameters.FindOptimalEmbeddingDimensionality(X_train[:, None], None, X_test[:, None], **common)
		self.assertEqual(selfPrediction.shape, (1, 12))
		multiTarget = Hyperparameters.FindOptimalEmbeddingDimensionality(X_train[:, None], numpy.column_stack([X_train, X_train]),
																		 X_test[:, None], numpy.column_stack([Y_test, Y_test]), **common)
		self.assertEqual(multiTarget.shape, (2, 12))
		self.assertTrue(numpy.allclose(multiTarget[0], joint) and numpy.allclose(multiTarget[1], joint))
		embedDimensions = Hyperparameters.FindSelfPredictionEmbeddingDimension(Frame('Lorenz5D').values[0:800, 1:6].astype(float), maxDims = 8,
																	  device = 'cpu', dtype = torch.float32, showProgress = False)
		self.assertEqual(embedDimensions.shape, (5,))
		self.assertTrue(((embedDimensions >= 1) & (embedDimensions <= 8)).all())

	def test_prediction_horizon(self):
		"""(PredictInterval) Every horizon refitted on its own rows reproduces the reference."""
		df = Frame('block_3sp'); d = df.values.astype(float); x = df.columns.get_loc('x_t')
		scores = Hyperparameters.FindOptimalPredictionHorizon(d[0:150, x], d[0:150, x], d[148:200, x], d[148:200, x], maxTp = 15,
															  embedDimensions = 3, isScoringFinitePairsOnly = True)
		reference = Validation('PredictInterval_valid.csv').values
		self.assertTrue(numpy.allclose(scores, reference, atol = 1e-6))
		shared = Hyperparameters.FindOptimalPredictionHorizon(d[0:150, x], d[0:150, x], d[148:200, x], d[148:200, x], maxTp = 15,
															  embedDimensions = 3, batched = True, isScoringFinitePairsOnly = True)
		self.assertEqual(shared.shape, (15, 2))
		self.assertTrue(numpy.allclose(shared[:, 0], numpy.arange(1, 16)))

	def test_smap_neighborhood(self):
		"""(PredictNonlinear) The reference used every training state but one as neighbors."""
		df = Frame('TentMapNoise'); d = df.values.astype(float); c = df.columns.get_loc('TentMap')
		thetas = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
		scores = Hyperparameters.FindSMapNeighborhood(d[0:500, c], d[0:500, c], d[497:801, c], d[497:801, c], theta = thetas,
													  embedDimensions = 4, predictionHorizon = 1, knn = 495, isScoringFinitePairsOnly = True)
		reference = Validation('PredictNonlinear_valid.csv').values
		self.assertTrue(numpy.allclose(scores, reference, atol = 1e-6))


class test_MDE(unittest.TestCase):
	"""Variable selection against the dimx reference on Fly80XY."""

	@classmethod
	def setUpClass(cls):
		df = read_csv(os.path.join(_TESTS_DIR, 'validation', 'Fly80XY_norm_1061.csv'))
		cls.colNames = [c for c in df.columns if c not in ['index', 'Left_Right', 'FWD']]
		cls.X = df[cls.colNames].values.astype(float)
		cls.y = df['FWD'].values.astype(float)
		cls.truth = read_csv(os.path.join(_TESTS_DIR, 'validation', 'MDE_Fly80XY_valid.csv'))

	def test_mde1(self):
		"""Selected variables and scores match the reference (lib rows 0..299, pred rows 300..600)."""
		mde = MDE(self.X[0:300], self.y[0:300], self.X[300:601], self.y[300:601], maxD = 5, convergent = False, predictionHorizon = 1)
		result = mde.Run()
		selected = [self.colNames[i] for i in result.selected_variables[0] if i >= 0]
		self.assertEqual(selected, self.truth['variables'].tolist())
		performance = result.performance[0][~numpy.isnan(result.performance[0])]
		for computed, expected in zip(performance, self.truth['rho']):
			self.assertAlmostEqual(float(computed), float(expected), places = 4)
		self.assertEqual(result.Y_pred.shape, (301, 1))
		self.assertTrue(numpy.isnan(result.Y_pred[0, 0]) and numpy.isfinite(result.Y_pred[1:, 0]).all())
		fitted = MDEFitter(MaxD = 5, Convergent = False, PredictionHorizon = 1, progressBar = False).Fit(self.X[0:300], self.y[0:300], self.X[300:601], self.y[300:601])
		self.assertTrue(numpy.array_equal(fitted.selected_variables, result.selected_variables))

	def test_mde2(self):
		"""A duplicated target selects the same variables for both columns."""
		Y = numpy.column_stack([self.y, self.y])
		result = MDE(self.X[0:300], Y[0:300], self.X[300:601], Y[300:601], maxD = 5, convergent = False, predictionHorizon = 1).Run()
		self.assertEqual(result.selected_variables.shape[0], 2)
		for j in range(2):
			selected = [self.colNames[i] for i in result.selected_variables[j] if i >= 0]
			self.assertEqual(selected, self.truth['variables'].tolist())
			performance = result.performance[j, ~numpy.isnan(result.performance[j])]
			for computed, expected in zip(performance, self.truth['rho']):
				self.assertAlmostEqual(float(computed), float(expected), places = 4)
		self.assertEqual(result.Y_pred.shape, (301, 2))

	def test_mde_convergent_in_sample(self):
		"""The convergence gate runs and records slopes for the selected variables."""
		result = MDE(self.X[0:300], self.y[0:300], maxD = 2, convergent = 'post', CCMSeed = 1, MinCandidatePerformance = 0.3).Run()
		self.assertEqual(int((result.selected_variables[0] >= 0).sum()), 2)
		self.assertTrue(numpy.isfinite(result.ccm_values[0, :2]).all())

	def test_mde_cross_validation(self):
		"""Leave-one-run-out selection over three runs and a final prediction."""
		runs = [(0, 300), (300, 600), (600, 900)]
		X = [self.X[a:b] for a, b in runs]; Y = [self.y[a:b] for a, b in runs]
		fitter = MDEFitterCV(MaxD = 2, Convergent = False, LeaveOneRunOut = True, progressBar = False)
		result = fitter.Fit(X, Y)
		self.assertEqual(result.fold_selected_variables.shape, (3, 1, 2))
		self.assertEqual(result.fold_accuracies.shape, (3, 1))
		predicted = fitter.Predict(self.X[900:1061], self.y[900:1061])
		self.assertEqual(predicted.Y_pred.shape, (161, 1))
		self.assertTrue(numpy.isfinite(predicted.score).all())
		cv = MDECV(self.X[0:600], self.y[0:600], maxD = 2, convergent = False, folds = 3, include_target = False)
		cv.fit()
		final = cv.predict(self.X[600:800], self.y[600:800])
		self.assertEqual(final.Y_pred.shape, (200, 1))
		self.assertEqual(len(final.fold_results), 3)


if __name__ == '__main__':
	unittest.main()
