"""
The prediction core: functions that take X_train, Y_train, X_test, Y_test arrays and return
result records.

- Predictors: SimplexPredict, SimplexGenerate, SMapPredict, SMapGenerate
- Multiview: MultiviewPredict
- ConvergentCrossMap: cross-map performance of many source variables across training-subset sizes
- MDE: greedy variable selection with an optional convergence gate
- Setup: turning X and Y arrays into training pairs and test states
- _core: the tensor kernels every predictor shares
- Results: the result records and ResultsIO
"""
