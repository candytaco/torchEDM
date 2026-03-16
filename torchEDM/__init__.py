"""Python tools for EDM"""

# provide functional API
from . import Functions
# provide object-based API with train/test splits
from . import Fitters

from .Utils import PlotObsPred, PlotCoeff, ComputeError
from .Hyperparameters import FindOptimalEmbeddingDimensionality, FindOptimalPredictionHorizon, FindSMapNeighborhood
from .FunctionalExamples import FunctionalExamples
from .FitterExamples import FitterExamples
from .Utils import SurrogateData

# Import result objects
from .EDM.Results import (
    SimplexResult,
    SMapResult,
    CCMResult,
    MultiviewResult
)
# Import visualization functions
from .Visualization import (
    plot_prediction,
    plot_smap_coefficients,
    plot_ccm,
    plot_multiview,
    plot_embed_dimension,
    plot_predict_interval,
    plot_predict_nonlinear
)
# Import hyperparameter selector Fitters
from .Fitters.EmbedDimensionFitter import EmbedDimensionFitter
from .Fitters.PredictionHorizonFitter import PredictionHorizonFitter
from .Fitters.SMapNeighborhoodFitter import SMapNeighborhoodFitter

__version__     = "3"
__versionDate__ = "2026-01-02"
