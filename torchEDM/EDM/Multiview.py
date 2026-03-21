
# python modules
from itertools import combinations
from math import floor, sqrt
from warnings import warn

# package modules
from numpy import argsort, array, column_stack, mean

# local modules
from .Embed import Embed
from .. import Functions, ComputeError
from ..Utils import IsNonStringIterable
from .Results import MultiviewResult


#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :
	"""
	Function to evaluate multiview predictions top combos

	:param combo: Column combination tuple
	:param data: Embedded data array
	:param args: Dictionary of Simplex arguments
	:return: Prediction projection array
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = list( combo ),
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = args['embedDims'],
                                predictionHorizon              = args['predictionHorizon'],
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])
	return projection

#----------------------------------------------------
# Function to evaluate combo rank (correlation)
#----------------------------------------------------
def MultiviewSimplexcorrelation( combo, data, args ) :
	"""
	Function to evaluate combo rank (correlation)

	:param combo: Column combination tuple
	:param data: Embedded data array
	:param args: Dictionary of Simplex arguments
	:return: Correlation value
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = list( combo ),
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = args['embedDims'],
                                predictionHorizon              = args['predictionHorizon'],
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])

	# projection is numpy array: Column 1 is Observations, Column 2 is Predictions
	return ComputeError(projection[:, 1], projection[:, 2], None)

#------------------------------------------------------------------
class Multiview:
    """
	Multiview class : Base class. Contains a Simplex instance

	D represents the number of variables to combine for each
	assessment, if not specified, it is the number of columns.

	E is the embedding dimension of each variable.
	If E = 1, no time delay embedding is done, but the variables
	in the embedding are named X(t-0), Y(t-0)...

	Simplex.Validate() sets knn equal to E+1 if knn not specified,
	so we need to explicitly set knn to D + 1.

	Parameter 'multiview' is the number of top-ranked D-dimensional
	predictions for the final prediction. Corresponds to parameter k
	in Ye & Sugihara with default k = sqrt(m) where m is the number
	of combinations C(n,D) available from the n = D * E columns
	taken D at-a-time.

	Ye H., and G. Sugihara, 2016. Information leverage in
	interconnected ecosystems: Overcoming the curse of dimensionality
	Science 353:922-925.

	Parameter 'trainLib' controls the evaluation strategy for ranking
	column combinations:

	trainLib = True (default):
		Uses in-sample evaluation for ranking. During the Rank() phase,
		predictions are made using test = train (in-sample). This is
		computationally faster but may produce artificially high skill
		scores, as highly accurate in-sample predictions can be made from
		arbitrary non-constant, non-oscillatory vectors. After ranking,
		the final Project() phase uses the specified train and test.

	trainLib = False:
		Uses proper out-of-sample evaluation for ranking. The Rank() phase
		uses the specified train and test parameters to evaluate combinations.
		This is more rigorous but computationally more expensive. Requires
		explicit train and test parameters.

	NOTE: When trainLib = True and no train/test are specified, the data
			is automatically split 50/50 for the final projection phase.
    """

    def __init__( self,
                  data,
                  columns=None,
                  target=None,
                  train=None,
                  test=None,
                  D=0,
                  embedDimensions=0,
                  predictionHorizon=1,
                  knn=0,
                  step=-1,
                  multiview=0,
                  exclusionRadius=0,
                  trainLib=True,
                  excludeTarget=False,
                  ignoreNan=True,
                  verbose=False,
                  numProcess=4,
                  mpMethod=None,
                  chunksize=1):
        """
        Initialize Multiview using plain arguments.

        :param data: 2D numpy array where column 0 is time (unless noTime=True)
        :param columns: Column indices to use (defaults to all except time)
        :param target: Target column index (defaults to column 1)
        :param train: Training set indices [start, end]
        :param test: Test set indices [start, end]
        :param D: State-space dimension (number of variables to combine for each assessment). If 0, defaults to number of columns.
        :param embedDimensions: Embedding dimension (E). If 0, will be set by Validate()
        :param predictionHorizon: Prediction time horizon (Tp)
        :param knn: Number of nearest neighbors. If 0, will be set to E+1 by Validate()
        :param step: Time delay step size (tau). Negative values indicate lag
        :param multiview: Number of top-ranked D-dimensional predictions for final ensemble (parameter k in Ye & Sugihara). If 0, defaults to sqrt(m) where m is the number of combinations C(n,D).
        :param exclusionRadius: Temporal exclusion radius for neighbors
        :param trainLib: Evaluation strategy for ranking column combinations: True uses in-sample evaluation (test=train during Rank phase). False uses proper out-of-sample evaluation with specified train/test.
        :param excludeTarget: Whether to exclude target column from embedding combinations
        :param ignoreNan: Remove NaN values from embedding
        :param verbose: Print diagnostic messages
        :param numProcess: Number of processes for multiprocessing
        :param mpMethod: Multiprocessing context method (ExecutionMode.SPAWN, ExecutionMode.FORK, ExecutionMode.FORKSERVER). If None, uses platform default
        :param chunksize: Chunk size for pool.starmap operations
        """

        # Assign parameters directly
        self.name            = 'Multiview'
        self.Data            = data
        self.columns         = columns
        self.target          = target
        self.embedDimensions = embedDimensions
        self.predictionHorizon = predictionHorizon
        self.knn             = knn
        self.step            = step
        self.D               = D
        self.multiview       = multiview
        self.exclusionRadius = exclusionRadius
        self.trainLib        = trainLib
        self.excludeTarget   = excludeTarget
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Assign split parameters
        self.train = train if train is not None else []
        self.test = test if test is not None else []

        # Assign execution parameters
        self.numProcess = numProcess
        self.mpMethod   = mpMethod
        self.chunksize  = chunksize

        self.Embedding  = None # numpy array
        self.View       = None # numpy array
        self.Projection = None # numpy array

        self.combos             = None # List of column combinations (tuples)
        self.topRankCombos      = None # List of top columns (tuples)
        self.topRankProjections = None # dict of columns : numpy array
        self.topRankStats       = None # dict of columns : dict of stats

        # Setup
        self.Validate() # Multiview Method: set knn default, E if embedded
        self.Setup()    # Embed Data

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run( self ) :
        """
        Execute Multiview prediction and return MultiviewResult.

        :return: Multiview results with ensemble-averaged projection, view rankings, top-ranked projections, and statistics
        """
        self.Rank()
        self.Project()

        # Compute ensemble-averaged prediction
        # M.topRankProjections is dict of combo : numpy array
        # Each array has columns: [Time, Observations, Predictions, Pred_Variance]

        # Get first projection for Time and Observations
        first_proj = next(iter(self.topRankProjections.values()))

        # Collect all predictions (column 2) and average them
        all_predictions = [proj[:, 2] for proj in self.topRankProjections.values()]
        multiviewPredict = mean(all_predictions, axis=0)

        # Create result array: [Time, Observations, Predictions]
        self.Projection = column_stack([first_proj[:, 0], first_proj[:, 1], multiviewPredict])

        # Create View: rankings of column combinations
        colCombos = list(self.topRankProjections.keys())

        topRankStats = {}
        for combo in colCombos :
            proj = self.topRankProjections[combo]
            # proj columns: 0=Time, 1=Observations, 2=Predictions, 3=Variance
            metrics = [ComputeError(proj[:, 1], proj[:, 2], None),
                       ComputeError(proj[:, 1], proj[:, 2], 'MAE'),
                       ComputeError(proj[:, 1], proj[:, 2], 'CAE'),
                       ComputeError(proj[:, 1], proj[:, 2], 'RMSE')]
            topRankStats[combo] = metrics

        self.topRankStats = topRankStats

        # Build View array: each row is [combo_as_str, correlation, MAE, CAE, RMSE]
        view_rows = []
        for combo in colCombos:
            stats = topRankStats[combo]
            view_rows.append([str(combo), stats[0], stats[1], stats[2], stats[3]])

        self.View = view_rows  # List of lists

        return MultiviewResult(
            projection=self.Projection,
            view=self.View,
            topRankProjections=self.topRankProjections,
            topRankStats=self.topRankStats,
            D=self.D,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    #-------------------------------------------------------------------
    def Rank( self ) :
        """
        Multiprocess to rank top multiview vectors
        """

        if self.verbose:
            print( f'{self.name}: Rank()' )

        args = { 'target'          : self.target, 
                 'train'             : self.train,
                 'test'            : self.test,
                 'embedDims'               : self.D,
                 'predictionHorizon'              : self.predictionHorizon,
                 'step'             : self.step,
                 'exclusionRadius' : self.exclusionRadius,
                 'embedded'        : True,
                 'noTime'          : True,
                 'ignoreNan'       : self.ignoreNan }

        if self.trainLib :
            # Set test = train for in-sample training
            args['test'] = self.train

        # Sequential evaluation of all combos
        correlationList = [MultiviewSimplexcorrelation(combo, self.Embedding, args)
                           for combo in self.combos]

        correlationVec    = array( correlationList, dtype = float )
        rank_i    = argsort( correlationVec )[::-1] # Reverse results 
        topRank_i = rank_i[ :self.multiview ]

        self.topRankCombos = [ self.combos[i] for i in topRank_i ]

    #-------------------------------------------------------------------
    # 
    #-------------------------------------------------------------------
    def Project( self ) :
        """
        Projection with top multiview vectors
        """

        if self.verbose:
            print( f'{self.name}: Project()' )

        args = { 'target'          : self.target, 
                 'train'             : self.train,
                 'test'            : self.test,
                 'embedDims'               : self.D,
                 'predictionHorizon'              : self.predictionHorizon,
                 'step'             : self.step,
                 'exclusionRadius' : self.exclusionRadius,
                 'embedded'        : True,
                 'noTime'          : True,
                 'ignoreNan'       : self.ignoreNan }

        # Sequential projection for top-ranked combos
        dfList = [MultiviewSimplexPred(combo, self.Embedding, args)
                  for combo in self.topRankCombos]

        self.topRankProjections = dict( zip( self.topRankCombos, dfList ) )

    #--------------------------------------------------------------------
    def Setup( self ):
    #--------------------------------------------------------------------
        """
        Set D, train, test, combos. Embed Data.
        """
        if self.verbose:
            print( f'{self.name}: Setup()' )

        # Set default train & test if not provided
        if self.trainLib :
            if not len( self.test ) and not len( self.train ) :
                # Set train & test for ranking : train, test = 1/2 data
                self.train  = [ 1, floor( self.Data.shape[0]/2 ) ]
                self.test = [ floor( self.Data.shape[0]/2 ) + 1,
                              self.Data.shape[0]]

        # Establish state-space dimension D
        # default to number of input columns (not embedding columns)
        if self.D == 0 :
            self.D = len( self.columns )

        # Check D is not greater than number of embedding columns
        if self.D > len( self.columns ) * self.embedDimensions :
            newD = len( self.columns ) * self.embedDimensions
            msg = f'Validate() {self.name}: D = {self.D}'      +\
                ' exceeds number of columns in the embedding: {newD}.' +\
                f' D set to {newD}'
            warn( msg )

            self.D = newD

        # Remove target columns from potential combos
        if self.excludeTarget :
            comboCols = [col for col in self.columns if col not in self.target]
        else :
            comboCols = self.columns

        # Embed Data - returns numpy array
        self.Embedding = Embed(self.Data,
                               columns = comboCols,
                               embeddingDimensions = self.embedDimensions,
                               stepSize = self.step,
                               includeTime = False)

        # Map target from original column index to embedded column index
        # Target in embedded space is the first lag (t-0) of the target variable
        # Find which position the target column is in comboCols
        if self.target[0] in comboCols:
            target_pos = comboCols.index(self.target[0])
            # In embedding, this variable's t-0 lag is at index: target_pos * E
            self.target = [target_pos * self.embedDimensions]
        else:
            # Target was excluded, use first embedded column
            self.target = [0]

        # Combinations of possible embedding vectors (column indices), D at-a-time
        # For E=1, columns are 0, 1, 2, ... (one per original column)
        # For E>1, columns are interleaved (col1_t0, col1_t-1, ..., col2_t0, col2_t-1, ...)
        n_embed_cols = self.Embedding.shape[1]
        embed_col_indices = list(range(n_embed_cols))
        self.combos = list( combinations( embed_col_indices, self.D ) )

        # Establish number of ensembles if not specified
        if self.multiview < 1 :
            # Ye & Sugihara suggest sqrt( m ) as number of embeddings to avg
            self.multiview = floor( sqrt( len( self.combos ) ) )

            if self.verbose :
                msg = f'Validate() {self.name}:' +\
                    f' Set view sample size to {self.multiview}'
                print( msg, flush = True )

        if self.multiview > len( self.combos ) :
            msg = f'Validate() {self.name}: multiview ensembles ' +\
                f' {self.multiview} exceeds the number of available' +\
                f' combinations: {len(self.combos)}. Set to {len(self.combos)}.'
            warn( msg )

            self.multiview = len( self.combos )

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        """
        Validate Multiview inputs and parameters

        :raises RuntimeError: if inputs are invalid
        """
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if self.columns is None or not len(self.columns):
            raise RuntimeError( f'Validate() {self.name}: columns required.' )
        if not IsNonStringIterable(self.columns) :
            raise RuntimeError( f'Validate() {self.name}: columns must be a list of integers.' )

        if self.target is None:
            raise RuntimeError( f'Validate() {self.name}: target required.' )
        if not IsNonStringIterable(self.target) :
            self.target = [self.target]

        if not self.trainLib :
            if not len( self.train ) :
                msg = f'{self.name}: Validate(): trainLib False requires' +\
                       ' train specification.'
                raise RuntimeError( msg )

            if not len( self.test ) :
                msg = f'{self.name}: Validate(): trainLib False requires' +\
                       ' test specification.'
                raise RuntimeError( msg )
