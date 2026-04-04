import os
import sys
import time
from typing import Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

from config import LOW_CUT, HIGH_CUT, R_PEAK, QRS_LENGTH
from utils.util import (
    resample_signal,
    butterworth_bandpass_filter,
    time_split,
)


def resample_time_split(ecg_signal, original_fs=500, target_fs=250, window_time=4,slide_time=1,max_time = 120):
    # 1. 重采样至 250Hz 
    resampled_signal = resample_signal(ecg_signal, original_fs, target_fs)
    resampled_signal = resampled_signal[~np.isnan(resampled_signal)]
    
    filter_signal = butterworth_bandpass_filter(
        resampled_signal, target_fs, 0.5, 40
    )
    
    segments = time_split(filter_signal, fs=target_fs, window_time=window_time,slide_time=slide_time,max_time = max_time)
    return np.array(segments)


def preprocess_ecg(
    ecg_signal, original_fs=500, target_fs=500, method="emrich2023"
) -> Dict:
    # --- 1. 信号预处理 ---
    resampled_signal = resample_signal(ecg_signal, original_fs, target_fs)
    resampled_signal = resampled_signal[~np.isnan(resampled_signal)]
    filter_signal = butterworth_bandpass_filter(
        resampled_signal, target_fs, LOW_CUT, HIGH_CUT
    )
    # --- 2. R 峰检测 ---
    _, info = nk.ecg_process(filter_signal, sampling_rate=target_fs, method=method)
    r_peaks = info["ECG_R_Peaks"]

    r_peaks = r_peaks[~np.isnan(r_peaks)].astype(int)

    pad_samples = int(0.1 * target_fs)  # 100ms 缓冲
    signal_len = len(filter_signal)

    valid_mask = (r_peaks >= pad_samples) & (r_peaks < (signal_len - pad_samples))
    r_peaks = r_peaks[valid_mask]

    _, waves = nk.ecg_delineate(
        filter_signal, r_peaks, sampling_rate=target_fs, method="peak"
    )

    q_peaks_abs = np.array(waves["ECG_Q_Peaks"])
    s_peaks_abs = np.array(waves["ECG_S_Peaks"])

    min_len = min(len(r_peaks), len(q_peaks_abs), len(s_peaks_abs))
    if len(r_peaks) != min_len:
        print(
            f"Warning: Delineation mismatch! Trimming from {len(r_peaks)} to {min_len}"
        )
        r_peaks = r_peaks[:min_len]
        q_peaks_abs = q_peaks_abs[:min_len]
        s_peaks_abs = s_peaks_abs[:min_len]

    # --- 6. 计算相对索引 ---
    q_diffs = q_peaks_abs - r_peaks + R_PEAK
    q_nan = np.isnan(q_diffs)
    q_diffs[q_nan] = R_PEAK - QRS_LENGTH // 2
    s_diffs = s_peaks_abs - r_peaks + R_PEAK
    s_nan = np.isnan(s_diffs)
    s_diffs[s_nan] = R_PEAK - QRS_LENGTH // 2 + QRS_LENGTH

    rr_intervals = np.zeros_like(r_peaks, dtype=float)
    if len(r_peaks) > 1:
        rr_intervals[1:] = np.diff(r_peaks)
    return {
        "filter_signal": filter_signal,
        "resampled_signal": resampled_signal,
        "resampled_fs": target_fs,
        "r_peaks": r_peaks,
        "q_peaks": q_peaks_abs,
        "s_peaks": s_peaks_abs,
        "q_diffs": q_diffs,
        "s_diffs": s_diffs,
        "rr_intervals": rr_intervals,  # 新增：RR 间期（秒）
    }

