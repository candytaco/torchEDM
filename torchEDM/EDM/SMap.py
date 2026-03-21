# python modules

# package modules
import numpy
import torch
from numpy import apply_along_axis, insert, isnan, isfinite, exp
from numpy import column_stack
from numpy import full, linspace, mean, nan, power, sum, array

# local modules
from .EDM import EDM
from .Results import SMapResult


#-----------------------------------------------------------
class SMap(EDM):
    """
    SMap class : child of EDM
    """

    def __init__(self,
                 data,
                 columns=None,
                 target=None,
                 train=None,
                 test=None,
                 embedDimensions=0,
                 predictionHorizon=1,
                 knn=0,
                 step=-1,
                 theta=0.0,
                 exclusionRadius=0,
                 embedded=False,
                 validLib=None,
                 noTime=False,
                 ignoreNan=True,
                 verbose=False,
                 generateSteps=0,
                 generateConcat=False,
                 device=None,
                 dtype=None):
        """
        Initialize SMap as child of EDM.

        :param data: 2D numpy array where column 0 is time (unless noTime=True)
        :param columns: Column indices to use for embedding (defaults to all except time)
        :param target: Target column index (defaults to column 1)
        :param train: Training set indices [start, end]
        :param test: Test set indices [start, end]
        :param embedDimensions: Embedding dimension (E). If 0, will be set by Validate()
        :param predictionHorizon: Prediction time horizon (Tp)
        :param knn: Number of nearest neighbors. If 0, will be set to E+1 by Validate()
        :param step: Time delay step size (tau). Negative values indicate lag
        :param theta: S-Map localization parameter. theta=0 is global linear map, larger values increase localization
        :param exclusionRadius: Temporal exclusion radius for neighbors
        :param embedded: Whether data is already embedded
        :param validLib: Boolean mask for valid library points
        :param noTime: Whether first column is time or data
        :param ignoreNan: Remove NaN values from embedding
        :param verbose: Print diagnostic messages
        :param generateSteps: Number of iterative generation steps. If 0, uses standard prediction.
        :param generateConcat: Whether to concatenate generated predictions
        :param device: torch device to use (None for auto-detect)
        :param dtype: torch dtype to use (None for float64)
        """

        # Instantiate EDM class: inheret all members to self
        super(SMap, self).__init__(data, isEmbedded=False, name='SMap')

        self.columns         = columns
        self.target          = target
        self.embedDimensions = embedDimensions
        self.predictionHorizon = predictionHorizon
        self.knn             = knn
        self.step            = step
        self.theta           = theta
        self.exclusionRadius = exclusionRadius
        self.embedded        = embedded
        self.validLib        = validLib if validLib is not None else []
        self.noTime          = noTime
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Assign split parameters
        self.train = train if train is not None else []
        self.test = test if test is not None else []

        # Assign generation parameters
        self.generateSteps = generateSteps
        self.generateConcat = generateConcat

        # Map API parameter names to EDM base class names
        self.embedStep         = self.step
        self.isEmbedded        = self.embedded

        # GPU setup
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.dtype = dtype if dtype is not None else torch.float64

        # SMap storage
        self.Coefficients   = None # DataFrame SMap API output
        self.SingularValues = None # DataFrame SMap API output

        # TODO: check what these actually do - they shadow the capitalized things but are written into differently??
        self.coefficients   = None # ndarray SMap output (N_pred, E+1)
        self.singularValues = None # ndarray SMap output (N_pred, E+1)

        # Setup
        self.Validate()      # EDM Method: set knn default, E if embedded
        self.CreateIndices() # Generate lib_i & pred_i, validLib

        self.targetVec = self.Data[:, [self.target[0]]]

        if self.noTime :
            # Generate a time/index vector, store as ndarray
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.time = array( timeIndex, dtype = int )
        else :
            # 1st data column is time
            self.time = self.Data[:, 0]

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run(self, return_predictions: bool = True):
    #-------------------------------------------------------------------
        """
        Execute S-Map prediction and return SMapResult.

        :param return_predictions: If False, the predictions field of the result will not be populated
        :return: Prediction results with projection array, coefficients, singular values, and metadata
        """
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighborsTorch()
        self.ProjectTorch()
        self.FormatProjection()

        return SMapResult(
            time=self.Projection[:, 0],
            projection=self.Projection if return_predictions else None,
            coefficients=self.Coefficients,
            singularValues=self.SingularValues,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon,
            theta=self.theta
        )

    #-------------------------------------------------------------------
    def FindNeighborsTorch(self):
    #-------------------------------------------------------------------
        """
        Find k nearest neighbors using torch.
        Computes pairwise Euclidean distances, applies exclusion
        mask, and selects k nearest via torch.topk.
        """
        if self.verbose:
            print(f'{self.name}: FindNeighborsTorch()')

        trainEmbedding = self.Embedding[self.trainIndices, :]
        testEmbedding = self.Embedding[self.testIndices, :]

        trainTensor = torch.tensor(trainEmbedding, device=self.device, dtype=self.dtype)
        testTensor = torch.tensor(testEmbedding, device=self.device, dtype=self.dtype)

        # Pairwise Euclidean distances: [nTrain x nTest]
        distanceMatrix = torch.cdist(trainTensor, testTensor, p=2)

        # Apply exclusion mask: set excluded pairs to infinity
        exclusionMask = self._BuildExclusionMask()
        if exclusionMask.any():
            maskTensor = torch.tensor(exclusionMask, device=self.device, dtype=torch.bool)
            distanceMatrix[maskTensor] = float('inf')

        # topk on dim=0 finds k smallest distances per test point (columns)
        topkDistances, topkIndices = torch.topk(distanceMatrix, self.knn, dim=0, largest=False)

        # Transpose to [nTest x knn] to match expected shape
        neighborDistances = topkDistances.t()
        neighborIndices = topkIndices.t()

        # Move results to CPU numpy
        self.knn_distances = neighborDistances.cpu().numpy()
        neighborIndicesNumpy = neighborIndices.cpu().numpy()

        # Map neighbor indices from library-local to data-space indices
        self.knn_neighbors = self._MapKNNIndicesToLibraryIndices(neighborIndicesNumpy)

        # Clean up GPU tensors
        del trainTensor, testTensor, distanceMatrix, topkDistances, topkIndices
        del neighborDistances, neighborIndices
        if exclusionMask.any():
            del maskTensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #-------------------------------------------------------------------
    def ProjectTorch(self):
    #-------------------------------------------------------------------
        """
        Vectorized S-Map projection using batched torch.linalg.lstsq.

        For each prediction row compute projection as the linear
        combination of regression coefficients (C) of weighted
        embedding vectors (A) against target vector (B) : AC = B.

        Weights reflect the SMap theta localization of the knn
        for each prediction. Default knn = len( lib_i ).

        Matrix A has (weighted) constant (1) first column
        to enable a linear intercept/bias term.

        Sugihara (1994) doi.org/10.1098/rsta.1994.0106
        """

        if self.verbose:
            print(f'{self.name}: ProjectTorch()')

        numberOfPredictions = len(self.testIndices)
        numberOfDimensions = self.embedDimensions + 1

        # Convert data to tensors
        distances = torch.tensor(self.knn_distances, device=self.device, dtype=self.dtype)
        neighbors = torch.tensor(self.knn_neighbors, device=self.device, dtype=torch.long)
        embedding = torch.tensor(self.Embedding, device=self.device, dtype=self.dtype)
        targetVector = torch.tensor(self.targetVec.squeeze(), device=self.device, dtype=self.dtype)
        testIndices = torch.tensor(self.testIndices, device=self.device, dtype=torch.long)

        # Compute weights: W[i,j] = exp(-theta * d[i,j] / mean(d[i,:]))
        distanceRowMean = torch.mean(distances, dim=1, keepdim=True)
        torch.clamp_min_(distanceRowMean, 1e-10)

        if self.theta == 0:
            weights = torch.ones_like(distances)
        else:
            distanceRowScale = self.theta / distanceRowMean
            weights = torch.exp(-distanceRowScale * distances)

        # Target values at neighbor indices + predictionHorizon: shape (N_pred, knn)
        neighborsPlusTp = neighbors + self.predictionHorizon
        targetValues = targetVector[neighborsPlusTp]

        # Handle NaN in target values by zeroing out those entries in both
        # the design matrix and target vector (effectively removing those equations)
        validMask = torch.isfinite(targetValues)
        maskedWeights = torch.where(validMask, weights, torch.zeros_like(weights))

        # Weighted target vector: shape (N_pred, knn)
        # Replace NaN with 0 before multiplying (masked out anyway)
        maskedTargetValues = torch.where(validMask, targetValues, torch.zeros_like(targetValues))
        weightedTargets = maskedWeights * maskedTargetValues

        # Build batched design matrix A: shape (N_pred, knn, E+1)
        # Column 0: weights (intercept term)
        # Columns 1:E+1: weighted embedding vectors of neighbors
        designMatrix = torch.zeros(numberOfPredictions, self.knn, numberOfDimensions,
                                   device=self.device, dtype=self.dtype)

        # Column 0: intercept weights (masked)
        designMatrix[:, :, 0] = maskedWeights

        # Columns 1:E+1: gather neighbor embeddings and weight them (masked)
        # neighbors has shape (N_pred, knn), need to gather from embedding (N_data, E)
        neighborEmbeddings = embedding[neighbors]  # shape (N_pred, knn, E)
        designMatrix[:, :, 1:] = maskedWeights.unsqueeze(2) * neighborEmbeddings

        # Solve batched least squares: A @ C = wB
        # torch.linalg.lstsq expects A of shape (..., m, n) and B of shape (..., m) or (..., m, k)
        # We have A: (N_pred, knn, E+1), wB: (N_pred, knn)
        # lstsq returns solution of shape (..., n) or (..., n, k)
        lstsqResult = torch.linalg.lstsq(designMatrix, weightedTargets)
        coefficients = lstsqResult.solution  # shape (N_pred, E+1)

        # Compute predictions: prediction = C[0] + sum(C[1:] * embedding[test_row, :])
        # testEmbeddings: shape (N_pred, E)
        testEmbeddings = embedding[testIndices]
        # prediction = C[:,0] + sum(C[:,1:] * testEmbeddings, dim=1)
        predictions = coefficients[:, 0] + torch.sum(coefficients[:, 1:] * testEmbeddings, dim=1)

        # Compute variance estimate: sum(W * (B - prediction)^2) / sum(W)
        # Use masked values for variance computation
        residuals = maskedTargetValues - predictions.unsqueeze(1)
        residualsSquared = residuals ** 2
        weightSum = torch.sum(maskedWeights, dim=1)
        variance = torch.sum(maskedWeights * residualsSquared, dim=1) / weightSum

        # Compute singular values via SVD of the design matrix
        # For each prediction row, compute SVD of A to get singular values
        singularValues = torch.linalg.svdvals(designMatrix)  # shape (N_pred, min(knn, E+1))

        # Pad singular values to match numberOfDimensions if needed
        if singularValues.shape[1] < numberOfDimensions:
            padding = torch.full((numberOfPredictions, numberOfDimensions - singularValues.shape[1]),
                                 float('nan'), device=self.device, dtype=self.dtype)
            singularValues = torch.cat([singularValues, padding], dim=1)

        # Move results to CPU numpy
        self.projection = predictions.cpu().numpy()
        self.variance = variance.cpu().numpy()
        self.coefficients = coefficients.cpu().numpy()
        self.singularValues = singularValues.cpu().numpy()

        # Clean up GPU tensors
        del distances, neighbors, embedding, targetVector, testIndices
        del distanceRowMean, weights, neighborsPlusTp, targetValues
        del validMask, maskedWeights, maskedTargetValues, weightedTargets
        del designMatrix, neighborEmbeddings, lstsqResult, coefficients
        del testEmbeddings, predictions, residuals, residualsSquared, weightSum, variance
        del singularValues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #-------------------------------------------------------------------
    def Generate( self, return_predictions: bool = True ) :
    #-------------------------------------------------------------------
        """
        SMap Generation
        Given train: override test to be single prediction at end of train
        Replace self.Projection with G.Projection

        Note: Generation with datetime time values fails: incompatible
        numpy.datetime64, timedelta64 and python datetime, timedelta

        :param return_predictions: If False, the predictions field of the result will not be populated
        """
        if self.verbose:
            print( f'{self.name}: Generate()' )

        # Local references for convenience
        N      = self.Data.shape[0]
        column = self.columns[0]
        target = self.target[0]
        train    = self.train

        # Override test for single prediction at end of train
        test = [ train[-1] - 1, train[-1] ]
        if self.verbose:
            print(f'{self.name}: Generate(): test overriden to {test}')

        # Output numpy arrays to replace self.Projection, self.Coefficients...
        nOutRows  = self.generateSteps

        # Projection array: shape (n_samples, 4)
        # Column 0: Time, Column 1: Observations, Column 2: Predictions, Column 3: Pred_Variance
        generated = full( (nOutRows, 4), nan )

        # Coefficients array: shape (n_samples, E+2)
        # Column 0: Time, Column 1: C0, Columns 2-E+1: coefficients
        genCoeff = full((nOutRows, self.embedDimensions + 2), nan)

        # SingularValues array: shape (n_samples, E+2)
        # Column 0: Time, Columns 1-E+1: singular values
        genSV = full((nOutRows, self.embedDimensions + 2), nan)

        # Allocate vector for univariate column data
        # At each iteration the prediction is stored in columnData
        # timeData and columnData are copied to newData for next iteration
        columnData     = full( N + nOutRows, nan )
        columnData[:N] = self.Data[:, column] # First col only

        # Allocate output time vector & newData DataFrame
        timeData = full( N + nOutRows, nan )
        if self.noTime :
            # If noTime create a time vector and join into self.Data
            timeData[:N] = linspace( 1, N, N )
            # Create new data array with time column
            newData = column_stack([timeData[:N], columnData[:N]])
        else :
            timeData[:N] = self.time # Presume column 0 is time
            # Create new data array with time column
            newData = column_stack([timeData[:N], columnData[:N]])

        #-------------------------------------------------------------------
        # Loop for each feedback generation step
        #-------------------------------------------------------------------
        for step in range( self.generateSteps ) :
            if self.verbose :
                print( f'{self.name}: Generate(): step {step} {"="*50}')

            # Local SMapClass for generation
            G = SMap(data = newData,
                     columns = [column],
                     target = target,
                     train = train,
                     test = test,
                     embedDimensions = self.embedDimensions,
                     predictionHorizon = self.predictionHorizon,
                     knn = self.knn,
                     step = self.step,
                     theta = self.theta,
                     exclusionRadius = self.exclusionRadius,
                     embedded = self.embedded,
                     validLib = self.validLib,
                     noTime = self.noTime,
                     generateSteps = self.generateSteps,
                     generateConcat = self.generateConcat,
                     ignoreNan = self.ignoreNan,
                     verbose = self.verbose,
                     device = self.device,
                     dtype = self.dtype)

            # 1) Generate prediction ----------------------------------
            G.Run()

            if self.verbose :
                print( 'G.Projection' )
                print( G.Projection ); print()

            newPrediction = G.Projection[-1, 2]  # Column 2 is Predictions
            newTime       = G.Projection[-1, 0]  # Column 0 is time

            # 2) Save prediction in generated --------------------------
            generated[step, 0] = newTime
            generated[step, 1] = nan  # Observations (not applicable for generation)
            generated[step, 2] = newPrediction
            generated[step, 3] = nan  # Pred_Variance (not applicable for generation)

            # Save coefficients and singular values
            genCoeff[step, 0] = newTime
            genCoeff[step, 1:] = G.Coefficients[-1, 1:]  # Skip time column
            genSV[step, 0] = newTime
            genSV[step, 1:] = G.SingularValues[-1, 1:]  # Skip time column

            if self.verbose :
                print( f'2) generated step {step}' )

            # 3) Increment library by adding another row index ---------
            # Dynamic library not implemented

            # 4) Increment prediction indices --------------------------
            test = [ p + 1 for p in test ]

            if self.verbose:
                print(f'4) test {test}')

            # 5) Add 1-step ahead projection to newData for next Project()
            columnData[ N + step ] = newPrediction
            timeData  [ N + step ] = newTime

            # JP : for big data this is likely not efficient
            newData = column_stack([timeData[:(N + step + 1)],
                                   columnData[:(N + step + 1)]])

            if self.verbose:
                print(f'5) newData: {newData.shape}')
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Loop for each feedback generation step
        #----------------------------------------------------------

        # Replace self.Projection with generated
        if self.generateConcat :
            # Concatenate original data observations with generated predictions
            # Original data: columns 0 (time), 1 (observations)
            # Generated: columns 0 (time), 1 (obs), 2 (test), 3 (var)
            # Result: columns 0 (time), 1 (obs), 2 (test), 3 (var)
            timeName = 0  # Column 0 is time
            data_obs = column_stack([self.Data[:, timeName], self.Data[:, column]])
            self.Projection = numpy.vstack([data_obs, generated[:, [0, 2]]])

        else :
            self.Projection = generated

        self.Coefficients   = genCoeff
        self.SingularValues = genSV

        return SMapResult(
            time=self.Projection[:, 0],
            projection=self.Projection if return_predictions else None,
            coefficients=genCoeff,
            singularValues=genSV,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon,
            theta=self.theta
        )
