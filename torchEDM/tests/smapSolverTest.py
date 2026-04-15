#! /usr/bin/env python3

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import SGDRegressor, Lars, LassoLars

import torchEDM as EDM


#------------------------------------------------------------
#------------------------------------------------------------
def main():
    """Test sklearn.linear_model solvers"""

    circle = EDM.sampleData['circle']
    data = circle.values
    col_index1 = circle.columns.get_loc('x')
    col_index2 = circle.columns.get_loc('y')
    target_index = circle.columns.get_loc('x')

    lmSolvers = {
        'SVD'              : None,
        'LinearRegression' : LinearRegression(),
        'SGDRegressor'     : SGDRegressor( alpha = 0.005 ),
        'Ridge'            : Ridge( alpha = 0.05 ),
        'Lasso'            : Lasso( alpha = 0.005 ),
        'Lars'             : Lars(),
        'LassoLars'        : LassoLars( alpha = 0.005 ),
        'ElasticNet'       : ElasticNet( alpha = 0.001, l1_ratio = 0.001 ),
        'RidgeCV'          : RidgeCV(),
        'LassoCV'          : LassoCV( cv = 5 ),
        'ElasticNetCV'     : ElasticNetCV(l1_ratio = [.05,0.2,.5,.9,1], cv = 5)
    }

    smapResults = {}

    for solverName in lmSolvers.keys() :
        print( solverName )
        result = EDM.FitSMap(data = data, columns = [col_index1, col_index2], target = target_index,
                             train = [1, 100], test = [101, 198],
                             embedded = True, embedDimensions = 2, predictionHorizon = 1, knn = 0, step = -1,
                             theta = 3.14, exclusionRadius = 0,
                             solver = lmSolvers[ solverName ], validLib = [], noTime = False, generateSteps = 0,
                             generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)
        smapResults[ solverName ] =  result

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
