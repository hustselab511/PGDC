import os
import sys
from typing import List
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import R_PEAK, FS, SEGMENT_DICT

from utils.util.standardize import z_score_standardization


def segment_by_rpeaks(
    signal_ecg: np.ndarray, qrs_indices, r_pos=R_PEAK, window_size=None, **kwargs
):
    r_peaks = qrs_indices[:, 1].astype(int)
    segments = []
    origin_segments = []
    for peak_idx in r_peaks[1:-1]:
        next = window_size - r_pos
        # 确保窗口不越界
        start_idx = max(0, peak_idx - r_pos)
        end_idx = min(len(signal_ecg), peak_idx + next)
        if end_idx - start_idx != window_size:
            continue
        seg = np.array(signal_ecg[start_idx:end_idx])
        segments.append(z_score_standardization(seg))
        origin_segments.append(z_score_standardization(seg))
        
    return {'segments':np.array(segments), 'origin_segments':origin_segments}


def rri_segment_by_rpeaks(
    qrs_indices, signal_ecg, window_size=None, r_pos=R_PEAK, **kwargs
):
    r_peaks = qrs_indices[:, 1].astype(int)
    segments = []
    origin_segments = []
    length = len(r_peaks)
    for i in range(length)[1:-1]:
        prev = min(int((r_peaks[i] - r_peaks[i - 1]) * r_pos / window_size), r_pos)
        next = min(
            int((r_peaks[i + 1] - r_peaks[i]) * (window_size - r_pos) / window_size),
            window_size - r_pos,
        )

        # 确保窗口不越界
        start_idx = max(0, int(r_peaks[i] - prev))
        end_idx = min(len(signal_ecg), int(r_peaks[i] + next))

        seg1 = np.array(signal_ecg[start_idx:end_idx])
        seg = np.zeros(window_size)
        seg[r_pos - prev : r_pos + next] = seg1
        new_seg = seg
        segments.append(z_score_standardization(new_seg))
        origin_segments.append(z_score_standardization(seg1))

    return {'segments':np.array(segments), 'origin_segments':origin_segments}


def selected_segments_function(qrs_indices, des="rri", signal_ecg=None, fs = FS):
    window_size = int(SEGMENT_DICT[des]["time"] * fs)
    r_pos = int(SEGMENT_DICT[des]["rpeak_time"] * fs)
    if des in ["rri", "center_rri"]:
        segments_dict = rri_segment_by_rpeaks(
            qrs_indices=qrs_indices,
            signal_ecg=signal_ecg,
            r_pos=r_pos,
            window_size=window_size,
        )
        qrs_indices = qrs_indices[1:-1]
    else:
        segments_dict = segment_by_rpeaks(
            qrs_indices=qrs_indices,
            signal_ecg=signal_ecg,
            r_pos=r_pos,
            window_size=window_size,
        )
        qrs_indices = qrs_indices[:]

    return {**segments_dict, "qrs": qrs_indices[1:-1]}