
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	counts = {}
	n = len(y)
	for c in y:
		counts[c] = counts.get(c, 0) + 1
	val = 1 - np.sum([(counts[i] / n) ** 2 for i in counts])
	return round(val,3)