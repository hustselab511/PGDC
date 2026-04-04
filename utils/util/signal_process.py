import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import medfilt, stft

def time_split(ecg_signal, fs=250, window_time=8,slide_time=1,max_time = 120):
    window_size = int(fs * window_time)
    slide_size = int(fs * slide_time)
    segments = []
    length = min(len(ecg_signal),int(max_time*fs))
    for i in range(0, length - window_size + 1, slide_size):
        seg = ecg_signal[i : i + window_size]
        segments.append(seg)
    return np.array(segments)

def resample_signal(ecg: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    """将信号重采样至目标频率（论文中为500Hz）"""
    if original_fs == target_fs:
        return np.array(ecg)
    # 使用scipy重采样，保持信号形状
    new_length = int(len(ecg) * target_fs / original_fs)
    return signal.resample(ecg, new_length)

def powerline_filter(signal, fs, order=2):
    """工频干扰滤波器（50Hz陷波滤波器）"""
    f0 = 50.0  # 工频频率
    # Q = 30.0   # 品质因数
    b, a = signal.butter(order, [f0 - 0.5, f0 + 0.5], btype="bandstop", fs=fs)
    return signal.filtfilt(b, a, signal)


def butterworth_bandpass_filter(
    ecg: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 50, order: int = 4
) -> np.ndarray:
    """
    应用四阶零相位Butterworth带通滤波器（论文中参数）
    """
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    # 零相位滤波避免相位失真
    return signal.filtfilt(b, a, ecg)

def lowpass_filter(data, fs, cutoff, order=5):
    """应用低通滤波（零相位滤波，避免波形偏移）"""
    nyq = 0.5 * fs  # 奈奎斯特频率（采样率的一半）
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    y = signal.filtfilt(b, a, data)  # filtfilt实现零相位滤波（比lfilter更适合ECG）
    return y

def highpass_filter(data, fs, cutoff, order=5):
    """应用高通滤波（零相位滤波）"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def remove_baseline_drift(ecg, fs):
    """
    去除基线漂移
    使用200ms中值滤波器抑制QRS波群和P波，600ms中值滤波器去除T波
    """
    # 计算滤波器窗口大小(采样点数)
    window_200ms = int(0.2 * fs)  # 200ms窗口
    window_600ms = int(0.6 * fs)  # 600ms窗口

    # 确保窗口大小为奇数(中值滤波器要求)
    if window_200ms % 2 == 0:
        window_200ms += 1
    if window_600ms % 2 == 0:
        window_600ms += 1

    # 第一次中值滤波(200ms)抑制QRS和P波
    filtered1 = medfilt(ecg, kernel_size=window_200ms)

    # 第二次中值滤波(600ms)去除T波
    drift_estimate = medfilt(filtered1, kernel_size=window_600ms)

    # 从原始信号中减去估计的漂移
    ecg_no_drift = ecg - drift_estimate

    return ecg_no_drift, drift_estimate


def iirnotch_filter(ecg, b, a):
    """双向滤波（前向+反向）避免相位偏移"""
    return np.flip(signal.lfilter(b, a, np.flip(signal.lfilter(b, a, ecg))))

# 小波变换去噪核心函数
def wavelet_denoise(noisy_ecg, wavelet="sym8", level=5, threshold_method="soft"):
    """
    使用小波变换对ECG信号去噪

    参数:
        noisy_ecg: 含噪ECG信号
        wavelet: 小波基函数
        level: 分解层数
        threshold_method: 阈值类型 ('soft' 或 'hard')
    返回:
        denoised_ecg: 去噪后的ECG信号
    """
    # 1. 小波分解
    coeffs = pywt.wavedec(noisy_ecg, wavelet, level=level)
    # coeffs[0] 是近似系数（低频成分），coeffs[1:] 是细节系数（高频成分）

    # 2. 计算自适应阈值
    # 基于中位数绝对偏差(MAD)估计噪声标准差
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 最顶层细节系数主要是噪声
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_ecg)))  # 通用阈值公式

    # 3. 对细节系数应用阈值处理（噪声主要集中在高频细节）
    denoised_coeffs = [coeffs[0]]  # 保留近似系数（包含主要信号成分）

    for i in range(1, len(coeffs)):
        # 对每个细节系数应用阈值
        if threshold_method == "hard":
            # 硬阈值：将绝对值小于阈值的系数设为0
            denoised_detail = np.where(np.abs(coeffs[i]) < threshold, 0, coeffs[i])
        else:
            # 软阈值：将绝对值小于阈值的系数设为0，其余减去阈值并保留符号
            denoised_detail = pywt.threshold(coeffs[i], threshold, mode="soft")

        denoised_coeffs.append(denoised_detail)

    # 4. 小波重构
    denoised_ecg = pywt.waverec(denoised_coeffs, wavelet)

    # 确保重构信号与原始信号长度一致
    if len(denoised_ecg) != len(noisy_ecg):
        denoised_ecg = denoised_ecg[: len(noisy_ecg)]

    return denoised_ecg