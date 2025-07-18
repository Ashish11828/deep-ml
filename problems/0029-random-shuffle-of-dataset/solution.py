import numpy as np

def shuffle_data(X: np.ndarray, y: np.ndarray, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

