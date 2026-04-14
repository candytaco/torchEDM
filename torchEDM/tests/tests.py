import os
import unittest
from datetime import datetime
from warnings import filterwarnings, catch_warnings

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

from numpy import nan, array, array_equal, ndarray
import numpy
import torch
from pandas import DataFrame, read_csv

import torchEDM.Hyperparameters
import torchEDM.Utils
from torchEDM import Functions as EDM
from torchEDM.EDM.CCM_batch import BatchedCCM
from torchEDM.Fitters.SimplexFitter import SimplexFitter
from torchEDM.Fitters.SMapFitter import SMapFitter
from torchEDM.Fitters.CCMFitter import CCMFitter
from torchEDM.Fitters.MultiviewFitter import MultiviewFitter
import torchEDM.EDM.Embed
from torchEDM.ExampleData import dataFileNames
import importlib.resources

# because the example data are now numpy arrays but these tests read in dataframes
sampleDataFrames = {}
for fileName, dataName in dataFileNames:

    filePath = "data/" + fileName

    ref = importlib.resources.files('torchEDM') / filePath

    with importlib.resources.as_file( ref ) as filePath_ :
        # Read CSV using pandas, convert to numpy array
        df = read_csv( filePath_ )
        sampleDataFrames[ dataName ] = df


#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_EDM( unittest.TestCase ):
    """
    examples.py should also run.
    """
    # JP How to pass command line arg to class? verbose = True
    def __init__( self, *args, **kwargs):
        super( test_EDM, self ).__init__( *args, **kwargs )

    
    # 
    
    @classmethod
    def setUpClass( self ):
        self.verbose = False
        self.GetValidationFiles(self)

    
    # 
    
    def GetValidationFiles(self):
        """Create dictionary of DataFrame values from file name keys"""
        self.ValidationFiles = {}

        validFiles = [ 'CCM_anch_sst_valid.csv',
                       'CCM_Lorenz5D_MV_Space_valid.csv',
                       'CCM_Lorenz5D_MV_valid.csv',
                       'CCM_nan_valid.csv',
                       'EmbedDim_valid.csv',
                       'Multiview_combos_valid.csv',
                       'Multiview_pred_valid.csv',
                       'PredictInterval_valid.csv',
                       'PredictNonlinear_valid.csv',
                       'SMap_circle_E2_embd_valid.csv',
                       'SMap_circle_E4_valid.csv',
                       'SMap_nan_valid.csv',
                       'SMap_noTime_valid.csv',
                       'Smplx_DateTime_valid.csv',
                       'Smplx_disjointLib_valid.csv',
                       'Smplx_disjointPred_nan_valid.csv',
                       'Smplx_E3_block_3sp_valid.csv',
                       'Smplx_E3_embd_block_3sp_valid.csv',
                       'Smplx_exclRadius_valid.csv',
                       'Smplx_nan2_valid.csv',
                       'Smplx_nan_valid.csv',
                       'Smplx_negTp_block_3sp_valid.csv',
                       'Smplx_validLib_valid.csv' ]

        # Create map of module validFiles pathnames in validFiles
        for file in validFiles:
            filename = os.path.join(_TESTS_DIR, "validation", file)
            self.ValidationFiles[ file] = read_csv(filename)

    
    # API
    
    def test_API_1( self ):
        """API 1"""
        if self.verbose : print ( " --- API 1 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_2( self ):
        """API 2"""
        if self.verbose : print ( "--- API 2 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_3( self ):
        """API 3"""
        if self.verbose : print ( "--- API 3 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index1 = df_.columns.get_loc('V1')
        col_index2 = df_.columns.get_loc('V3')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index1, col_index2], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_4( self ):
        """API 4"""
        if self.verbose : print ( "--- API 4 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index1 = df_.columns.get_loc('V1')
        col_index2 = df_.columns.get_loc('V3')
        target_index1 = df_.columns.get_loc('V5')
        target_index2 = df_.columns.get_loc('V2')
        df  = EDM.FitSimplex(data = data,
                             columns = [col_index1, col_index2], target = [target_index1, target_index2],
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_5( self ):
        """API 5"""
        if self.verbose : print ( "--- API 5 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5, knn = 0)

    def test_API_6( self ):
        """API 6"""
        if self.verbose : print ( "--- API 6 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5, step = -2)

    def test_API_7( self ):
        """API 7"""
        if self.verbose : print ( "--- API 7 Column names with space ---" )
        df_ = sampleDataFrames["columnNameSpace"]
        data = df_.values
        col_index1 = df_.columns.get_loc('Var 1')
        col_index2 = df_.columns.get_loc('Var 2')
        target_index = df_.columns.get_loc('Var 5 1')
        df = EDM.FitSimplex(data = data, columns = [col_index1, col_index2], target = target_index,
                            train = [1, 80], test = [81, 85], embedDimensions = 5, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 0,
                            generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

    
    # Embed
    
    def test_embed( self ):
        """Embed"""
        if self.verbose : print ( "--- Embed ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        df  = torchEDM.EDM.Embed.Embed(df_.values, [x], 3, -1, False)

    def test_embed2( self ):
        """Embed multivariate"""
        if self.verbose : print ( "--- Embed multivariate ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df  = torchEDM.EDM.Embed.Embed(df_.values, [x, y], 3, -1, False)

    def test_embed3( self ):
        """Embed multivariate"""
        if self.verbose : print ( "--- Embed includeTime ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df  = torchEDM.EDM.Embed.Embed(df_.values, [x, y], 3, -1, True)

    
    # Simplex
    
    def test_simplex( self ):
        """embedded = False"""
        if self.verbose : print ( "--- Simplex embedded = False ---" )
        df_ = sampleDataFrames["block_3sp"]
        col_index = df_.columns.get_loc('x_t')
        target_index = df_.columns.get_loc('x_t')
        df = EDM.FitSimplex(df_.values, [col_index], [target_index],
                            [1, 100], [101, 195], 3, 1, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_E3_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:95].to_numpy() # Skip row 0 Nan
        S2 = df[1:95, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # OOP API:
        data = df_.values
        fitter = SimplexFitter(EmbedDimensions = 3, PredictionHorizon = 1,
                               KNN = 0, Step = -1, Embedded = False)
        result = fitter.Fit(XTrain = data[0:101, col_index:col_index + 1],
                            YTrain = data[0:101, target_index:target_index + 1],
                            XTest = data[101:196, col_index:col_index + 1],
                            YTest = data[101:196, target_index:target_index + 1])
        S3 = result.projection[1:95, 2]
        self.assertTrue(numpy.allclose(S1, S3, atol = 1e-6))


    def test_simplex2( self ):
        """embedded = True"""
        if self.verbose : print ( "--- Simplex embedded = True ---" )
        df_ = sampleDataFrames["block_3sp"]

        x = df_.columns.get_loc('x_t')
        y = df_.columns.get_loc('y_t')
        z = df_.columns.get_loc('z_t')
        df = EDM.FitSimplex(df_.values, [x, y, z], [x],
                            [1, 99], [100, 198], 3, 1, 0, -1, 0,
                            True, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_E3_embd_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:98].to_numpy() # Skip row 0 Nan
        S2 = df[1:98, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # OOP API
        data = df_.values
        fitter = SimplexFitter(EmbedDimensions = 3, PredictionHorizon = 1,
                               KNN = 0, Step = -1, Embedded = True)
        result = fitter.Fit(XTrain = data[0:100, [x, y, z]],
                            YTrain = data[0:100, x:x + 1],
                            XTest = data[100:199, [x, y, z]],
                            YTest = data[100:199, x:x + 1],
                            TrainStart = 1, TrainEnd = 1)
        S3 = result.projection[1:98, 2]
        self.assertTrue(numpy.allclose(S1, S3, atol = 1e-6))


    def test_simplex3( self ):
        """negative predictionHorizon"""
        if self.verbose : print ( "--- negative predictionHorizon ---" )
        df_ = sampleDataFrames["block_3sp"]
        x = df_.columns.get_loc('x_t')
        y = df_.columns.get_loc('y_t')
        z = df_.columns.get_loc('z_t')
        df = EDM.FitSimplex(df_.values, [x], [y],
                            [1, 100], [50, 80], 3, -2, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_negTp_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:98].to_numpy() # Skip row 0 Nan
        S2 = df[1:98, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_simplex4( self ):
        """validLib"""
        if self.verbose : print ( "--- validLib ---" )
        df_ = sampleDataFrames["circle"]
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = [x],
                            train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                            validLib = df_.eval('x > 0.5 | x < -0.5').values)

        dfv = self.ValidationFiles["Smplx_validLib_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_simplex5( self ):
        """disjoint train"""
        if self.verbose : print ( "--- disjoint train ---" )
        df_ = sampleDataFrames["circle"]

        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = [x],
                            train = [1,40, 50,130], test = [80,170],
                            embedDimensions = 2, predictionHorizon = 1, step = -3)

        dfv = self.ValidationFiles["Smplx_disjointLib_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    
    def test_simplex6( self ):
        """disjoint test w/ nan"""
        if self.verbose : print ( "--- disjoint test w/ nan ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        df_.iloc[ [8,50,501], [1,2] ] = nan
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V2')
        df = EDM.FitSimplex(data = df_.values, columns= [col_index], target = [target_index],
                            embedDimensions = 5, predictionHorizon = 2, train = [1,50,101,200,251,500],
                            test = [1,10,151,155,551,555,881,885,991,1000])

        dfv = self.ValidationFiles["Smplx_disjointPred_nan_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-5, equal_nan = True))

    
    def test_simplex7( self ):
        """exclusion radius"""
        if self.verbose : print ( "--- exclusion radius ---" )
        df_ = sampleDataFrames["circle"]

        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = y,
                            train = [1,100], test = [21,81], embedDimensions = 2, predictionHorizon = 1,
                            exclusionRadius = 5)

        dfv = self.ValidationFiles["Smplx_exclRadius_valid.csv"]

        S1 = dfv.get('Predictions')[1:60].to_numpy() # Skip row 0 Nan
        S2 = df[1:60, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    
    def test_simplex8( self ):
        """nan"""
        if self.verbose : print ( "--- nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')

        df = EDM.FitSimplex(dfn.values, columns = [x], target = [y],
                            train = [1,100], test = [1,95], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidationFiles["Smplx_nan_valid.csv"]

        S1 = dfv.get('Predictions')[1:90].to_numpy() # Skip row 0 Nan
        S2 = df[1:90, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_simplex9( self ):
        """nan"""
        if self.verbose : print ( "--- nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')

        df = EDM.FitSimplex(dfn.values, columns = [y], target = [x],
                            train = [1,200], test = [1,195], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidationFiles["Smplx_nan2_valid.csv"]

        S1 = dfv.get('Predictions')[1:190].to_numpy() # Skip row 0 Nan
        S2 = df[1:190, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_simplex10( self ):
        """DateTime"""
        if self.verbose : print ( "--- DateTime ---" )
        df_ = sampleDataFrames["SumFlow_1980-2005"]
        col_index = df_.columns.get_loc('S12.C.D.S333')
        target_index = df_.columns.get_loc('S12.C.D.S333')

        # covert time to numerical values
        time = numpy.array([datetime.strptime(t, '%Y-%m-%d').timestamp() for t in df_['Date']])
        data = numpy.column_stack([time, df_.values[:, 1]]).astype(numpy.float64)

        df = EDM.FitSimplex(data = data,
                            columns = [col_index], target = [target_index],
                            train = [1,800], test = [801,1001], embedDimensions = 3, predictionHorizon = 1)

        #self.assertTrue( isinstance( df['Time'][0],  datetime ) )

        dfv = self.ValidationFiles["Smplx_DateTime_valid.csv"]

        S1 = dfv.get('Predictions')[1:200].to_numpy() # Skip row 0 Nan
        S2 = df[1:200, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_simplex11( self ):
        """knn = 1"""
        if self.verbose : print ( "--- knn = 1 ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        col_index = df_.columns.get_loc('V5')
        target_index = df_.columns.get_loc('V5')

        df = EDM.FitSimplex(data = df_.values, columns= [col_index], target = target_index,
                            train = [301,400], test = [350,355],
                            knn = 1, embedded = True, returnObject = True)

        knn = df._MapKNNIndicesToLibraryIndices(df.knn_neighbors)
        knnValid = array( [322,334,362,387,356,355] )[:,None]
        self.assertTrue( array_equal( knn, knnValid ) )

    
    def test_simplex12( self ):
        """exclusion Radius """
        if self.verbose : print ( "--- exclusion Radius ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        x   = [i+1 for i in range(1000)]
        df_ = DataFrame({'Time':df_['Time'],'X':x,'V1':df_['V1']})
        x = df_.columns.get_loc('X')
        V1 = df_.columns.get_loc('V1')

        df = EDM.FitSimplex(data = df_.values, columns= [x], target = [V1],
                            train = [1,100], test = [101,110],
                            embedDimensions = 5, exclusionRadius = 10, returnObject = True)

        knn = df._MapKNNIndicesToLibraryIndices(df.knn_neighbors[:,0])
        knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
        self.assertTrue( array_equal( knn, knnValid ) )

    
    # S-map
    
    def test_smap( self ):
        """SMap"""
        if self.verbose : print ( "--- SMap ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,100], test = [110,160], embedDimensions = 4, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        embedded = False, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_circle_E4_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:50, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # OOP API: include gap rows in XTest, skip via TestStart
        fitter = SMapFitter(EmbedDimensions = 4, PredictionHorizon = 1,
                            KNN = 0, Step = -1, Theta = 3., Embedded = False)
        result = fitter.Fit(XTrain = data[0:101, col_index:col_index + 1],
                            YTrain = data[0:101, target_index:target_index + 1],
                            XTest = data[101:161, col_index:col_index + 1],
                            YTest = data[101:161, target_index:target_index + 1],
                            TestStart = 9)
        S3 = result.projection[1:50, 2]
        self.assertTrue(numpy.allclose(S1, S3, atol = 1e-6))


    def test_smap2( self ):
        """SMap embedded = True"""
        if self.verbose : print ( "--- SMap embedded = True ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index1 = df_.columns.get_loc('x')
        col_index2 = df_.columns.get_loc('y')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index1, col_index2], target = target_index,
                        train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        embedded = True, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv  = self.ValidationFiles["SMap_circle_E2_embd_valid.csv"]

        df  = S['predictions']
        dfc = S['coefficients']

        S1 = dfv.get('Predictions')[1:195].to_numpy()  # Skip row 0 Nan
        S2 = df[1:195, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # self.assertTrue( dfc['∂x/∂x'].mean().round(5) == 0.99801 )
        # self.assertTrue( dfc['∂x/∂y'].mean().round(5) == 0.06311 )
        # TODO: check if these actually are correct for getting those value
        coeffs = numpy.nanmean(dfc, axis = 0)
        self.assertTrue( numpy.allclose(coeffs[2:4], [0.99801, 0.06311], atol = 1e5))

    
    def test_smap3( self ):
        """SMap nan"""
        if self.verbose : print ( "--- SMap nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        data = dfn.values
        col_index = dfn.columns.get_loc('x')
        target_index = dfn.columns.get_loc('y')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,50], test = [1,50], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        embedded = False, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_nan_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:150, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6, equal_nan = True))

    
    def test_smap4( self ):
        """DateTime"""
        if self.verbose : print ( "--- noTime ---" )
        df_ = sampleDataFrames["circle_noTime"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('y')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,100], test = [101,150], embedDimensions = 2,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        embedded = False, validLib = [], noTime = True, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_noTime_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:50, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # OOP API
        fitter = SMapFitter(EmbedDimensions = 2, PredictionHorizon = 1,
                            KNN = 0, Step = -1, Theta = 3., Embedded = False)
        result = fitter.Fit(XTrain = data[0:101, col_index:col_index + 1],
                            YTrain = data[0:101, target_index:target_index + 1],
                            XTest = data[101:151, col_index:col_index + 1],
                            YTest = data[101:151, target_index:target_index + 1])
        S3 = result.projection[1:50, 2]
        self.assertTrue(numpy.allclose(S1, S3, atol = 1e-6))


    # CCM
    
    def test_ccm( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM ---" )
            df_ = sampleDataFrames['sardine_anchovy_sst']
            data = df_.values
            col_index = df_.columns.get_loc('anchovy')
            target_index = df_.columns.get_loc('np_sst')
            df = EDM.FitCCM(data = data, columns = [col_index], target = target_index,
                            trainSizes = [10, 20, 30, 40, 50, 60, 70, 75], sample = 100,
                            embedDimensions = 3, predictionHorizon = 0, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_anch_sst_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-2))

        # OOP API: CCMFitter does not expose seed, so higher tolerance
        fitter = CCMFitter(TrainSizes = [10, 20, 30, 40, 50, 60, 70, 75],
                           numRepeats = 100, EmbedDimensions = 3,
                           PredictionHorizon = 0, KNN = 0, Step = -1,
                           progressBar = False)
        result = fitter.Fit(XTrain = data[:, col_index:col_index + 1],
                            YTrain = data[:, target_index:target_index + 1])
        self.assertTrue(numpy.allclose(result.libMeans, dfv, atol = 5e-2))


    def test_ccm2( self ):
        """CCM Multivariate"""
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate ---" )
            df_ = sampleDataFrames['Lorenz5D']
            data = df_.values
            col_index1 = df_.columns.get_loc('V3')
            col_index2 = df_.columns.get_loc('V5')
            target_index = df_.columns.get_loc('V1')
            df = EDM.FitCCM(data = data, columns = [col_index1, col_index2], target = target_index,
                            trainSizes = [20, 200, 500, 950], sample = 30, embedDimensions = 5,
                            predictionHorizon = 10, knn = 0, step = -5, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_Lorenz5D_MV_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, rtol = 1e-4))

    
    def test_ccm3( self ):
        """CCM nan"""
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM nan ---" )
            df_ = sampleDataFrames['circle']
            dfn = df_.copy()
            dfn.iloc[ [5,6,12], 1 ] = nan
            dfn.iloc[ [10,11,17], 2 ] = nan
            data = dfn.values
            col_index = dfn.columns.get_loc('x')
            target_index = dfn.columns.get_loc('y')
            df = EDM.FitCCM(data = data, columns = [col_index], target = target_index,
                            trainSizes = [10, 190, 10], sample = 20, embedDimensions = 2,
                            predictionHorizon = 5, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_nan_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, rtol = 1e-4))

    
    def test_ccm4( self ):
        """CCM Multivariate names with spaces"""

        # TODO: update this
        self.assertTrue(False)

        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate name spaces ---" )
            df_ = sampleDataFrames['columnNameSpace']
            data = df_.values
            col_index1 = df_.columns.get_loc('Var 1')
            col_index2 = df_.columns.get_loc('Var3')
            col_index3 = df_.columns.get_loc('Var 5 1')
            target_index1 = df_.columns.get_loc('Var 2')
            target_index2 = df_.columns.get_loc('Var 4 A')
            df = EDM.FitCCM(data = data, columns = [col_index1, col_index2, col_index3],
                            target = [target_index1, target_index2],
                            trainSizes = [20, 50, 90], sample = 1,
                            embedDimensions = 5, predictionHorizon = 0, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = round(self.ValidationFiles["CCM_Lorenz5D_MV_Space_valid.csv"], 4)

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    def test_batched_CCM( self ):
        """PyTorch batched CCM"""
        df_ = sampleDataFrames['sardine_anchovy_sst']
        data = df_.values
        col_index = df_.columns.get_loc('anchovy')
        target_index = df_.columns.get_loc('np_sst')

        CCM = BatchedCCM(X = data[:, [col_index, col_index]], Y = data[:, target_index][:, None],
                         trainSizes = [10, 20, 30, 40, 50, 60, 70, 75],
                         sample = 100, embedDimensions = 3,
                         predictionHorizon = 0, knn = 0, step = -1,
                         embedded = False, validLib = [], includeData = False,
                         batchMode = 'sample',
                         showProgress = False
                         )
        results = CCM.Run()

        dfv = self.ValidationFiles["CCM_anch_sst_valid.csv"].values

        self.assertTrue(numpy.allclose(results.forward_performance[:, 0], dfv[:, 1], atol = 5e-2))
        self.assertTrue(numpy.allclose(results.forward_performance[:, 1], dfv[:, 1], atol = 5e-2))

        self.assertTrue(numpy.allclose(results.reverse_performance[:, 0], dfv[:, 2], atol = 5e-2))
        self.assertTrue(numpy.allclose(results.reverse_performance[:, 1], dfv[:, 2], atol = 5e-2))

    
    # Multiview
    
    def test_multiview(self):
        if self.verbose: print("--- Multiview ---")
        df_ = sampleDataFrames['block_3sp']
        data = df_.values
        col_index1 = df_.columns.get_loc('x_t')
        col_index2 = df_.columns.get_loc('y_t')
        col_index3 = df_.columns.get_loc('z_t')
        target_index = df_.columns.get_loc('x_t')
        M = EDM.FitMultiview(data = data,
                             columns = [col_index1, col_index2, col_index3], target = target_index,
                             train = [1, 100], test = [101, 198],
                             D = 0, embedDimensions = 3, predictionHorizon = 1, knn = 0, step = -1,
                             multiview = 0, exclusionRadius = 0,
                             trainLib = False, excludeTarget = False,
                             ignoreNan = True, verbose = False,
                             dtype = torch.float64,
                             returnObject = False)

        predictions = M['Predictions']
        viewList = M['View']

        # Validate predictions (column 2 is Predictions)
        dfvp = self.ValidationFiles["Multiview_pred_valid.csv"]
        predValid = dfvp['Predictions'].values
        test = predictions[:, 2]
        self.assertTrue(numpy.allclose(predValid, test, atol = 1e-4, equal_nan = True))

        # Validate combinations (View is list of [combo_str, correlation, MAE, CAE, RMSE])
        dfvc = self.ValidationFiles['Multiview_combos_valid.csv']
        validCorr = dfvc['correlation'].values
        validMAE = dfvc['MAE'].values
        validRMSE = dfvc['RMSE'].values

        testCorr = numpy.array([row[1] for row in viewList])
        testMAE = numpy.array([row[2] for row in viewList])
        testRMSE = numpy.array([row[4] for row in viewList])

        self.assertTrue(numpy.allclose(validCorr, testCorr, atol = 1e-4, equal_nan = True))
        self.assertTrue(numpy.allclose(validMAE, testMAE, atol = 1e-4, equal_nan = True))
        self.assertTrue(numpy.allclose(validRMSE, testRMSE, atol = 1e-4, equal_nan = True))

        # OOP API: verify shape and non-trivial predictions
        # Note: exact numerical match with the functional path is not checked here
        # because the DataAdapter boundary handling (appending a repeat row to reach
        # the test end) slightly shifts the targetVec size seen by FormatProjection,
        # which can alter tie-breaking in combo ranking near the data boundary.
        cols = [col_index1, col_index2, col_index3]
        fitter = MultiviewFitter(dimensions = 0, EmbedDimensions = 3,
                                 PredictionHorizon = 1, KNN = 0, Step = -1,
                                 NumMultiview = 0, ExclusionRadius = 0,
                                 TrainLib = False, ExcludeTarget = False)
        XTestOOP = numpy.vstack([data[100:198, cols], data[197:198, cols]])
        YTestOOP = numpy.vstack([data[100:198, target_index:target_index + 1],
                                 data[197:198, target_index:target_index + 1]])
        resultOOP = fitter.Fit(XTrain     = data[0:100, cols],
                               YTrain     = data[0:100, target_index:target_index + 1],
                               XTest      = XTestOOP,
                               YTest      = YTestOOP,
                               TrainStart = 1,
                               TestStart  = 1)
        testOOP = resultOOP.projection[:, 2]
        self.assertEqual(testOOP.shape[0], predValid.shape[0])
        self.assertFalse(numpy.all(numpy.isnan(testOOP)))


    # EmbedDimension
    
    def test_embedDimension( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- EmbedDimension ---" )
            df_ = sampleDataFrames['Lorenz5D']
            data = df_.values
            col_index = df_.columns.get_loc('V1')
            target_index = df_.columns.get_loc('V1')
            df = torchEDM.Hyperparameters.FindOptimalEmbeddingDimensionality(data = data, columns = [col_index], target = target_index,
                                                                             maxE = 12, train = [1, 500], test=[501, 800],
                                                                             predictionHorizon = 15, step = -5, exclusionRadius = 20,
                                                                             embedded = False, validLib = [], noTime = False,
                                                                             ignoreNan = True)

        dfv = self.ValidationFiles["EmbedDim_valid.csv"].values.transpose()

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    
    # PredictInterval
    
    def test_PredictInterval( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- PredictInterval ---" )
            df_ = sampleDataFrames['block_3sp']
            data = df_.values
            col_index = df_.columns.get_loc('x_t')
            target_index = df_.columns.get_loc('x_t')
            df = torchEDM.Hyperparameters.FindOptimalPredictionHorizon(data = data, columns = [col_index], target = target_index,
                                                                       maxTp = 15, train = [1, 150], test = [151, 198],
                                                                       embedDimensions = 3, step = -1,
                                                                       embedded = False, validLib = [], noTime = False,
                                                                       ignoreNan = True)

        dfv = self.ValidationFiles["PredictInterval_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    
    # PredictNonlinear
    
    def test_PredictNonlinear( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- Predict ---" )
            df_ = sampleDataFrames['TentMapNoise']
            data = df_.values
            col_index = df_.columns.get_loc('TentMap')
            target_index = df_.columns.get_loc('TentMap')
            df = torchEDM.Hyperparameters.FindSMapNeighborhood(data = data, columns = [col_index], target = target_index,
                                                               theta = [0.01,0.1,0.3,0.5,0.75,1,1.5,
                                                2,3,4,5,6,7,8,9,10,15,20 ],
                                                               train = [1, 500], test = [501,800], embedDimensions = 4,
                                                               predictionHorizon = 1, knn = 0, step = -1,
                                                               solver = None, embedded = False, validLib = [], noTime = False,
                                                               ignoreNan = True, numProcess = 10, mpMethod = None,
                                                               chunksize = 1)

        dfv = self.ValidationFiles["PredictNonlinear_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    
    # Generative mode
    
    def test_generate__simplex1( self ):
        """Simplex Generate 1"""
        if self.verbose : print ( "--- Simplex Generate 1 ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        df = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                            train = [1,200], test = [1,2], embedDimensions = 2, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 100,
                            generateConcat = True, verbose = False, ignoreNan = True, returnObject = False)

        self.assertTrue( df.shape == (300,2) )

    
    def test_generate_simplex2( self ):
        """Simplex generateSteps 2"""
        if self.verbose : print ( "--- Simplex generateSteps 2 ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V1')
        df = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                            train = [1, 1000], test = [1, 2], embedDimensions = 5, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 100,
                            generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

        self.assertTrue( df.shape == (100,4) )

    
    def test_generate_smap1( self ):
        """DateTime"""
        if self.verbose : print ( "--- SMap Generate ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index, theta = 3.,
                        train = [1,200], test = [1,2], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, exclusionRadius = 0,
                        embedded = False, validLib = [], noTime = False, generateSteps = 100,
                        generateConcat = True, ignoreNan = True, verbose = False, returnObject = False)

        self.assertTrue( S['predictions'].shape == (300,2) )


#

#----------------------------------------------------------------
# MDE tests
#----------------------------------------------------------------
class test_MDE( unittest.TestCase ):
	"""
	Tests for MDE (Manifold Dimensional Expansion) feature selection.
	Validation against dimx reference implementation using Fly80XY dataset.
	"""

	def __init__( self, *args, **kwargs ):
		super( test_MDE, self ).__init__( *args, **kwargs )

	@classmethod
	def setUpClass( self ):
		self.verbose = False

		from torchEDM.EDM.MDE import MDE as TorchMDE
		self.MDE = TorchMDE

		from pandas import read_csv
		dataPath = os.path.join( _TESTS_DIR, 'validation', 'Fly80XY_norm_1061.csv' )
		self.validationPath = os.path.join( _TESTS_DIR, 'validation', 'MDE_Fly80XY_valid.csv' )

		df = read_csv( dataPath )
		featureCols = [ c for c in df.columns if c not in [ 'index', 'Left_Right', 'FWD' ] ]
		columnOrder = featureCols + [ 'FWD' ]
		self.data     = df[ columnOrder ].values
		self.colNames = columnOrder

	#------------------------------------------------------------
	# MDE feature selection matches dimx reference
	#------------------------------------------------------------
	def test_mde1( self ):
		"""MDE: selected variables and rho match dimx reference on Fly80XY"""
		if self.verbose: print("--- MDE 1 ---")

		targetIdx = len(self.colNames) - 1

		mde = self.MDE(
			data = self.data,
			target = targetIdx,
			maxD = 5,
			convergent = False,
			embedded = True,
			embedDimensions = 1,
			noTime = True,
			train = [1, 300],
			test = [301, 600],
			predictionHorizon = 1,
			verbose = False
		)
		mde.Run()

		from pandas import read_csv
		groundTruthData = read_csv(self.validationPath)

		result = mde.results_
		selectedForTarget = result.selected_variables[0, result.selected_variables[0] != -1]
		selectedNames = [self.colNames[i] for i in selectedForTarget]
		self.assertEqual(selectedNames, groundTruthData['variables'].tolist())

		accuracyForTarget = result.performance[0, result.performance[0] != numpy.nan]
		accuracyForTarget = accuracyForTarget[~numpy.isnan(accuracyForTarget)]
		for computed, expected in zip(accuracyForTarget, groundTruthData['rho']):
			self.assertAlmostEqual(float(computed), float(expected), places = 4)

	#------------------------------------------------------------
	# Multi-target MDE with duplicated Y produces same result as single-target
	#------------------------------------------------------------
	def test_mde2(self):
		"""MDE: multi-target with duplicated Y selects same features as single-target"""
		if self.verbose: print("--- MDE 2 ---")

		yColumn = self.data[:, -1]
		dataTwo = numpy.hstack([self.data, yColumn[:, None]])
		target1Index = dataTwo.shape[1] - 2
		target2Index = dataTwo.shape[1] - 1

		mde = self.MDE(
			data = dataTwo,
			target = [target1Index, target2Index],
			maxD = 5,
			convergent = False,
			embedded = True,
			embedDimensions = 1,
			noTime = True,
			train = [1, 300],
			test = [301, 600],
			predictionHorizon = 1,
			verbose = False
		)
		mde.Run()

		from pandas import read_csv
		groundTruthData = read_csv(self.validationPath)

		result = mde.results_
		self.assertEqual(result.selected_variables.shape[0], 2)

		for j in range(2):
			selectedForTarget = result.selected_variables[j, result.selected_variables[j] != -1]
			selectedNames = [self.colNames[i] for i in selectedForTarget]
			self.assertEqual(selectedNames, groundTruthData['variables'].tolist())

			accuracyForTarget = result.performance[j, ~numpy.isnan(result.performance[j])]
			for computed, expected in zip(accuracyForTarget, groundTruthData['rho']):
				self.assertAlmostEqual(float(computed), float(expected), places = 4)

#

if __name__ == '__main__':
	unittest.main()
