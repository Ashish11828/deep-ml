import numpy as np

def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)

    if predicted_positives == 0:
        return 0.0  # Avoid division by zero

    return true_positives / predicted_positives

