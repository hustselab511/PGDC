import numpy as np
def min_max_standardization(data, feature_range=(0, 1)):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data_min = np.min(data)
        data_max = np.max(data)
    elif data.ndim == 2:
        data_min = np.min(data, axis=1, keepdims=True)
        data_max = np.max(data, axis=1, keepdims=True)
    else:
        raise ValueError("data must be 1D or 2D")

    scaled_data = (data - data_min) / (data_max - data_min + 1e-8)
    scaled_data = scaled_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return scaled_data
def z_score_standardization(data):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 1:
        mean = np.mean(data)
        std = np.std(data) + 1e-8
    elif data.ndim == 2:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
    else:
        raise ValueError("data must be 1D or 2D")

    standardized_data = (data - mean) / std
    return standardized_data

def resorted_label(labels):
    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    new_labels = np.array(np.searchsorted(unique_labels, labels)).astype(np.int32)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return new_labels, label_map
   