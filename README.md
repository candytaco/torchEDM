## torchEDM

This is forked from the pyEDM repo and refactored to:
- use pyTorch as the backend to vectorize and run on GPUs
- provide a numpy-native API rather than pandas dataframes
- provide OOP objects with sklearn-like semantics that separate train/test and X/Y in the arguments
- some functions have been renamed to be clearer about their functionality

### API comparison with pyEDM for EDM functions

For example usage see:
- `FitterExamples` for OOP objects
- `FunctionalExamples` for the functional API

#### 1. Object-Oriented API (sklearn-like)
The OOP API provides sklearn-like wrappers with explicit train/test separation.

```python
import torchEDM
from torchEDM import ExampleData

# Load data
data = ExampleData.sampleData['TentMap']

# Split data
XTrain = data[0:100, 1]
YTrain = data[0:100, 1]
XTest = data[100:200, 1]
YTest = data[100:200, 1]

# Create fitter
fitter = torchEDM.Fitters.SimplexFitter(
    EmbedDimensions = 3,
    PredictionHorizon = 1,
    KNN = 4,
    Step = -1
)

# Run prediction
result = fitter.Fit(
    XTrain = XTrain,
    YTrain = YTrain,
    XTest = XTest,
    YTest = YTest)
```

#### 2. Functional API (pyEDM-like)
These will be deprecated soon.

The functional API provides simple functions for EDM analysis. 
These functions accept numpy arrays and return predictions directly.
Other than the numpy-nativeness, these functions largely keep the same
argument syntax as the pyEDM package.

```python
import torchEDM
from torchEDM import ExampleData

# Load data
data = ExampleData.sampleData['TentMap']

# Simplex prediction
result = torchEDM.Functions.FitSimplex(
    data = data,
    columns = [1],
    target = 1,
    train = (1, 100),
    test = (100, 200),
    embedDimensions = 3,
    predictionHorizon = 1,
    knn = 4,
    step = -1
)
```

### API comparison with dimx for MDE

If data is loaded as
```
import pandas
flies = pandas.read_csv('<MDE repository that contains dimx>/data/Fly80XY_norm_1061.csv')
```

#### dimx
```
import dimx
Fly_FWD = dimx.MDE( flies,
                    target = 'FWD',
                    removeColumns = ['index','FWD','Left_Right'],
                    D    = 5,
                    lib  = [1,300],
                    pred = [301,600],
                    ccmSlope = 0.01,
                    embedDimRhoMin = 0.65,
                    crossMapRhoMin = 0.5,
                    cores = 72,
                    chunksize = 30,
                    plot = False )
Fly_FWD.Run()
```

#### torchEDM
```
from torchEDM.Fitters.MDEFitter import MDEFitter

featureColumns = [c for c in flies.columns if c not in ['index', 'FWD', 'Left_Right']]
X = flies[featureColumns].values
Y = flies['FWD'].values

XTrain = X[0:301, :]
YTrain = Y[0:301]
XTest = X[301:601, :]
YTest = Y[301:601]

fitter = MDEFitter(MaxD = 5, PredictionHorizon = 1, Step = -1, Convergent = 'pre',
                    CCMLibraryPercentiles = [10, 25, 50, 75, 90],
                    CCMNumSamples = 100,
                    CCMConvergenceThreshold = 0.01,
                    stdThreshold = 0, HalfPrecision = False )
results = fitter.Fit(XTrain, YTrain, XTest, YTest)
```

---
### Data Interface
This package uses a pure NumPy array interface rather than pandas dataframes. 
This package also does not handle I/O.

### Object-Oriented API
- `SimplexFitter`: Simplex projection
- `SMapFitter`: S-Map prediction
- `CCMFitter`: Convergent Cross Mapping
- `MultiviewFitter`: Multiview embedding
- `MDEFitter`: Manifold Dimensional Expansion
- `MDEFitterCV`: MDE with cross-validation
