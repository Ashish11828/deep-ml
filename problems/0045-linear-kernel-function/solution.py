import numpy as np

def kernel_function(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    result = np.dot(x1, x2)  # or: result = x1 @ x2
    return result
