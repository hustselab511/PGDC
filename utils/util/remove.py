import numpy as np
from scipy.stats import iqr


def filter_abnormal_amplitude(amplitudes, threshold_coeff):
    if len(amplitudes) == 0:
        return np.array([], dtype=int)
    q1 = np.percentile(amplitudes, 25)
    q3 = np.percentile(amplitudes, 75)
    iqr_value = iqr(amplitudes)
    low_threshold = q1 - threshold_coeff * iqr_value
    high_threshold = q3 + threshold_coeff * iqr_value
    keep_indices = np.where(
        (amplitudes >= low_threshold) & (amplitudes <= high_threshold)
    )[0]
    return keep_indices


def filter_abnormal_segments(
    segments,
    threshold_coeff=1.5,
):
    amplitudes = np.array([np.max(seg) - np.min(seg) for seg in segments])
    keep_indices = filter_abnormal_amplitude(amplitudes, threshold_coeff)

    return keep_indices


def filter_abnormal_rpeak_segments(
    segments,
    r_pos=75,
    threshold_coeff=1.5,
):

    amplitudes = np.array([seg[r_pos] for seg in segments])
    keep_indices = filter_abnormal_amplitude(amplitudes, threshold_coeff)
    return keep_indices

def filter_correlation(beats, threshold=0.90):
    N = len(beats)
    if N == 0:
        return np.array([]), np.array([])

    template = np.median(beats, axis=0)

    beats_centered = beats - beats.mean(axis=1, keepdims=True)
    template_centered = template - template.mean()
    numerator = np.dot(beats_centered, template_centered)
    beats_norm = np.linalg.norm(beats_centered, axis=1)
    template_norm = np.linalg.norm(template_centered)

    denominator = beats_norm * template_norm
    denominator[denominator == 0] = 1e-8

    pcc_values = numerator / denominator

    keep_mask = pcc_values >= threshold
    keep_indices = np.where(keep_mask)[0]
    return keep_indices

def remove_threshold(segments, threshold_coeff=1):
    segments = np.array(segments)
    mean_segment = np.mean(segments, axis=0)
    distances = np.linalg.norm(segments - mean_segment, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_coeff * std_dist
    keep_indices = np.where(distances <= threshold)[0]
    return keep_indices


def remove_number(segments, select_num=300):
    segments = np.array(segments)
    mean_segment = np.mean(segments, axis=0)
    distances = cosine_distance(segments, mean_segment)
    n_segments = len(segments)
    k = min(n_segments, select_num)
    valid_indices = np.argsort(distances)[:k]
    return valid_indices

def selected_remove(segments,select_num=300,threshold_coeff=1, remove_mode='number'):
    if remove_mode == 'number':
        return remove_number(segments, select_num=select_num)
    else:
        return remove_threshold(segments, threshold_coeff=threshold_coeff)

def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b + 1e-10))  # 加小值避免除零
