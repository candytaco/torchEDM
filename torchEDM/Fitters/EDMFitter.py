class EDMFitter:
	"""
	Base of the parameter-holding wrappers: the constructor takes the settings, Fit takes
	X_train, Y_train and optionally X_test, Y_test (arrays or lists of runs), and the result
	is both returned and kept in Result.
	"""

	def __init__(self, progressBar = True):
		self.Result = None
		self.hideProgress = not progressBar

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		raise NotImplementedError
