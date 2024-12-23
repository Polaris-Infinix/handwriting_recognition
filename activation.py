import math
import numpy as np
def sigmoid(value, steep=1):
    return 1 / (1 + np.exp(-steep * value))

def diff_sigmoid(value):
    return sigmoid(value)*(1-sigmoid(value))

def softmax(arr):
  """
  This function computes the softmax of a 1D array.

  Args:
      arr: A NumPy array of any dimension.

  Returns:
      A NumPy array with the same shape as the input, containing the softmax probabilities.
  """
  # Exponentiate the array and normalize by the sum of exponentials
  exponentiated = np.exp(arr - np.max(arr))
  return exponentiated / np.sum(exponentiated)
