class EDMFitter:
	"""
	Base of the parameter-holding wrappers: the constructor takes the settings, Fit takes
	X_train, Y_train and optionally X_test, Y_test (arrays or lists of runs), and the result
	record is both returned and kept in Result.
	"""

	def __init__(self, progressBar = True):
		""":param progressBar:	show progress bars while fitting; False silences them (stored inverted as hideProgress)"""
		self.Result = None
		self.hideProgress = not progressBar

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		"""
		Run the wrapped function on the arrays and keep its result record in Result.

		:param X_train:	[nTrain, nFeatures] or a list of runs: the columns that form the states
		:param Y_train:	[nTrain, nTargets] (1-D for one target) or a list of runs aligned row by row with X_train
		:param X_test:	[nTest, nFeatures] or a list of runs whose rows are predicted; None predicts the training rows in-sample
		:param Y_test:	truth for the rows of X_test, laid out like Y_train; when given, the result carries a score per target
		:return: the result record of the wrapped function
		"""
		raise NotImplementedError
