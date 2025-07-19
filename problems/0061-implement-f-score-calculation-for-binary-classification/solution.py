import numpy as np

def f_score(y_true, y_pred, beta):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # True Positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # False Negatives
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision and Recall
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # F-score calculation
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0

    f_score_value = (1 + beta_sq) * (precision * recall) / denom
    return round(f_score_value, 3)
