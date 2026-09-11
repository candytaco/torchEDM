## torchEDM

Empirical dynamic modeling with a pyTorch backend. Forked from pyEDM and rebuilt to:
- run the neighbor search and the least-squares fits as batched tensor operations, on GPU when available
- take and return numpy arrays with the `X_train / Y_train / X_test / Y_test` convention
- batch many candidate variables at once for cross-map screening and greedy variable selection

### The array contract

Every function takes arrays and returns a result record. There is no time column: samples are
assumed evenly spaced, and it is up to the caller to make the data comply. Keyword parameters
follow one style: `embedDimensions, step, predictionHorizon, knn, exclusionRadius, trainRowMask`
come first, then the function's own settings, then `scoringFunction, device, dtype`.

- `X_train [nTrain, nFeatures]`, `Y_train [nTrain, nTargets]` (a 1-D `Y` is one target).
- `X_test [nTest, nFeatures]` is optional. Omitted means in-sample: the training rows predict
  themselves, each row's own state is barred from its neighbors, and `exclusionRadius` bars
  the rows near it.
- `Y_test` is optional; when given, the result carries a score per target.
- A list of arrays in any slot means multiple runs. Each run is processed alone, so history
  and horizon-shifted targets never cross a run boundary, and `Y_pred` comes back as a list.
- The state at row `r` stacks `X[r], X[r + step], ..., X[r + (embedDimensions - 1) * step]`.
  A training pair is the state at row `r` with `Y_train[r + predictionHorizon]`, kept whenever
  the state is complete and the target row lies inside the array.
- `Y_pred[i]` predicts `Y_test[i]` from the state at `X_test[i - predictionHorizon]`, and is
  NaN when that state is incomplete. `Y_pred` always has the shape of `Y_test`. A NaN entry
  of `Y_test` is predicted but never scored, which is how a caller excludes rows from a score.
- `knn = 0` means the state size plus one for the neighbor-averaging predictor and every
  available training state for the locally linear one.

### Functions

```python
import numpy
from torchEDM import SimplexPredict, SMapPredict, SimplexGenerate, MultiviewPredict, ConvergentCrossMap, MDE
from torchEDM import ExampleData

data = ExampleData.sampleData['TentMap']          # column 0 is a time column; never pass it
x = data[:, 1]

# one column stacked to three copies; the test array starts two rows early to carry the history
result = SimplexPredict(x[0:100], x[0:100], x[98:200], x[98:200], embedDimensions = 3, predictionHorizon = 1)
result.Y_pred        # shape (102,), NaN in the first three rows
result.variance
result.score         # array of one correlation

# locally weighted linear map on two columns used as the state
circle = ExampleData.sampleData['circle']
smap = SMapPredict(circle[0:100, [1, 2]], circle[0:100, 1], circle[100:190, [1, 2]], circle[100:190, 1], theta = 4.0)
smap.coefficients    # shape (90, 3): intercept, then one weight per state column

# feed a series forward for 50 steps
future = SimplexGenerate(x[0:200], 50, embedDimensions = 3)       # shape (50, 1)

# cross-map skill of many source columns onto a target across training-subset sizes, in-sample
lorenz = ExampleData.sampleData['Lorenz5D']
screen = ConvergentCrossMap(lorenz[:, 1:5], lorenz[:, 5], embedDimensions = 3, trainSizes = [100, 300, 600, 900],
                            repeats = 20, hasProgressBar = False)
screen.forward_performance   # shape (4 sizes, 4 sources)
# the reverse direction is a second call with X and Y exchanged

# greedy variable selection: candidates are the columns of X_train
selection = MDE(lorenz[0:500, 1:5], lorenz[0:500, 5], lorenz[500:1000, 1:5], lorenz[500:1000, 5], maxVariables = 3, convergenceCheck = False)
selection.selected_variables  # column indices into X_train, padded with -1
selection.Y_pred              # shape (500, 1)
selection.candidate_embed_dimensions  # per (target, candidate) when the convergence check searched them
```

Parameter sweeps live in `torchEDM.Hyperparameters`: `FindOptimalEmbeddingDimensionality`,
`FindOptimalPredictionHorizon`, `FindSMapNeighborhood`, and `FindSelfPredictionEmbeddingDimension`.

### Wrappers

`torchEDM.Fitters` holds parameter-holding classes with a `Fit(X_train, Y_train, X_test = None, Y_test = None)`
method that calls the matching function and keeps the result in `Result`: `SimplexFitter`,
`SMapFitter`, `MultiviewFitter`, `CCMFitter`, `MDEFitter`, and the cross-validated `MDEFitterCV`
and `CCMFitterCV`, which split lists of runs with `Fitters.RunSplitter`. `FitterExamples()` runs
each of them on the sample data.

### Argument names relative to pyEDM

- `E` -> `embedDimensions`, the number of copies of each feature column in the state
- `Tp` -> `predictionHorizon`
- `tau` -> `step`
- `lib`, `pred`, `columns`, `target` -> the `X_train / Y_train / X_test / Y_test` arrays
- `validLib` -> `trainRowMask`
- `theta` and `exclusionRadius` keep their names
