import numpy as np

def to_categorical(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    categorical = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, val in enumerate(y):
        categorical[i][val] = 1.
    return categorical.tolist()


