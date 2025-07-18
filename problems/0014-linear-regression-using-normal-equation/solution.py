import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Convert input to NumPy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Apply the normal equation: theta = (X^T * X)^-1 * X^T * y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Return theta as a flat list
    return theta.flatten().tolist()
