import numpy as np

def feature_scaling(X: np.ndarray):
    # Standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_score_scaled = (X - mean) / std

    # Min-Max Scaling
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    min_max_scaled = (X - min_val) / (max_val - min_val)

    # Round both outputs to 4 decimal places
    z_score_scaled = np.round(z_score_scaled, 4)
    min_max_scaled = np.round(min_max_scaled, 4)

    return z_score_scaled.tolist(), min_max_scaled.tolist()
