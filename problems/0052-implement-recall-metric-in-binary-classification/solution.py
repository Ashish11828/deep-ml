import numpy as np

def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if true_positive + false_negative == 0:
        return 0.0

    recall_value = true_positive / (true_positive + false_negative)
    return round(recall_value, 3)  # ✅ rounding to 3 decimal places



