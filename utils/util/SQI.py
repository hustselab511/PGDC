import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram

# nk.ecg_quality()
# 论文1
##  https://pubmed.ncbi.nlm.nih.gov/35601886/
### 1.去除平稳段
def remove_flat_segment(data, win_size=0.2, fs=500):
    """
    滑动窗口检测平稳段。
    窗口长度默认为 0.2 秒对应的采样点数。
    若窗口内最大值等于最小值，则认为该段为平稳段，返回 1；否则返回 0。
    """
    n = len(data)
    win_len = int(win_size * fs)   # 确保窗口长度为 win_size 秒
    for i in range(n - win_len + 1):
        seg = data[i:i + win_len]
        if np.max(seg) == np.min(seg):
            return 1
    return 0
### 2.去除心率段 在24-300/min之外的
### 3.根据PSD计算信噪比，阈值筛除
def calculate_snr(data, fs=500): # 假设 fs 为 500Hz 以支持到 250Hz 的分析
    # 1. 计算功率谱密度 (PSD)
    # nperseg 建议取数据长度或 256/512，影响频率分辨率
    f, pxx = periodogram(data, fs=fs, scaling="spectrum")
    # 2. 确定各个频段的掩码 (Mask)
    # 分子：2Hz - 40Hz
    mask_signal = (f >= 2) & (f <= 40)
    # 分母部分1：0Hz - 2Hz
    mask_noise_low = (f >= 0) & (f < 2)
    # 分母部分2：40Hz - 250Hz
    mask_noise_high = (f > 40) & (f <= fs)
    
    # 3. 计算各频段功率总和 (Sum of PSD)
    p_signal = np.sum(pxx[mask_signal])
    p_noise_low = np.sum(pxx[mask_noise_low])
    p_noise_high = np.sum(pxx[mask_noise_high])
    
    denominator = p_noise_low + p_noise_high
    
    if denominator == 0:
        return float('inf') # 避免除以0，返回无穷大表示信号极好
        
    snr = p_signal / denominator
    return snr

def is_high_quality(data, fs=500, threshold=0.5):
    snr_val = calculate_snr(data, fs)
    # 返回 0 表示质量好，1 表示质量差（噪声大）
    return 0 if snr_val >= threshold else 1

# https://iopscience.iop.org/article/10.1088/0967-3334/33/9/1419
## 计算12导联7个SQI指标，共84特征，根据MLP和机器学习方法进行质量评估
# 7个指标计算如下
def calculate_ksqi(signal):
    """
    计算第四阶统计量（峭度）（kSQI）。
    正常参考值大于 5。若数值较低，说明信号过平滑或充满了高斯白噪声。
    """
    return kurtosis(signal, fisher=False)
# sSQI（第三阶统计量/偏度）：正常参考值不等于 0。若数值接近 0，可能意味着信号严重失真或信号中只有噪声。
def calculate_ssqi(signal):
    """
    计算第三阶统计量（偏度）（sSQI）。
    正常参考值不等于 0。若数值接近 0，可能意味着信号严重失真或信号中只有噪声。
    """
    return skew(signal)
# pSQI（频段功率比）：正常参考值大于 0.5。若数值较低，说明能量主要集中在 40Hz 以上的高频段，或者基线漂移非常严重。
def calculate_psqi(signal, fs=500):
    """
    计算频段功率比（pSQI）。
    正常参考值大于 0.5。若数值较低，说明能量主要集中在 40Hz 以上的高频段，或者基线漂移非常严重。
    """
    freqs, psd = welch(signal, fs, nperseg=fs*2)
    p_total = np.sum(psd)
    p_ecg = np.sum(psd[(freqs >= 5) & (freqs < 40)])
    return p_ecg / p_total if p_total != 0 else 0
# basSQI（基线相对质量指数）：衡量1Hz以下能量占比，正常参考值大于 0.9。若数值较低，说明存在严重的基线漂移，通常由呼吸或身体体动引起。
def calculate_bas_sqi(signal, fs=500):
    """
    计算基线相对质量指数（basSQI）。
    衡量1Hz以下能量占比，正常参考值大于 0.9。若数值较低，说明存在严重的基线漂移，通常由呼吸或身体体动引起。
    """
    freqs, psd = welch(signal, fs, nperseg=fs*2)
    p_total = np.sum(psd)
    p_baseline = np.sum(psd[(freqs > 0) & (freqs < 1)])
    return 1 - (p_baseline / p_total) if p_total != 0 else 0
# hSQI（高频能量占比）：正常参考值大于 0.9。若数值较低，说明信号中存在严重的 50Hz 工频干扰或肌电干扰。
def calculate_hsqi(signal, fs=500):
    """
    计算高频能量占比（hSQI）。
    正常参考值大于 0.9。若数值较低，说明信号中存在严重的 50Hz 工频干扰或肌电干扰。
    """
    freqs, psd = welch(signal, fs, nperseg=fs*2)
    p_total = np.sum(psd)
    p_high = np.sum(psd[freqs > 50])
    return 1 - (p_high / p_total) if p_total != 0 else 0
# bSQI（QRS检测器一致性）：核心指标，正常参考值大于 0.8。该指标通过比较两种不同算法的检测结果得出，若两者不一致，说明信号质量肯定较差。
def calculate_bsqi(qrs_annotations_1, qrs_annotations_2, window=0.1):
    """
    bSQI: 比较两种不同QRS检测器（如Pan-Tompkins和Hamilton）的结果一致性。
    window: 匹配窗口（秒），通常为 100ms
    """
    if len(qrs_annotations_1) == 0 or len(qrs_annotations_2) == 0:
        return 0
    
    matches = 0
    for r1 in qrs_annotations_1:
        # 查找第二个检测器是否有在 window 范围内的 R 波
        if np.any(np.abs(qrs_annotations_2 - r1) < window):
            matches += 1
    
    # 计算 F1-score 或简单的匹配率
    precision = matches / len(qrs_annotations_1)
    recall = matches / len(qrs_annotations_2)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# rSQI（相对幅度分布）：正常参考值在 0.1 到 0.3 之间。若数值极高或极低，通常代表信号中存在由于运动引起的剧烈抖动。
# 这是一个简化的变体：通过计算标准化后的标准差来评估能量分布稳定性
def calculate_rsqi(signal):
    return np.std(signal) / (np.max(signal) - np.min(signal))

def calculate_sqi(ecg, fs=500):
    # 计算各指标
    ksqi = calculate_ksqi(ecg)
    ssqi = calculate_ssqi(ecg)
    psqi = calculate_psqi(ecg, fs)
    bas_sqi = calculate_bas_sqi(ecg, fs)
    hsqi = calculate_hsqi(ecg, fs)
    rsqi = calculate_rsqi(ecg)
    snr = calculate_snr(ecg, fs)
    flat_flag = remove_flat_segment(ecg, win_size=0.2, fs=fs)
    quality_flag = is_high_quality(ecg, fs, threshold=0.5)
    
    res = {
            'ksqi': ksqi,
            'ssqi': ssqi,
            'psqi': psqi,
            'bas_sqi': bas_sqi,
            'hsqi': hsqi,
            'rsqi': rsqi,
            'snr': snr,
            'flat_flag': flat_flag,
            'quality_flag': quality_flag
        }
    
    return res