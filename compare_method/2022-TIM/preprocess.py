import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as signal
import neurokit2 as nk
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier

from config import GET_SEGMENT_FS
from utils.util.loader import load2pth,get_data_folder,load2npy

def form_frames(beats, beats_per_frame=2):
    frames = []
    for i in range(len(beats) - beats_per_frame + 1):
        frame = np.concatenate(beats[i : i + beats_per_frame])
        frames.append(frame)
    return frames

class PhaseTransform:
    def __init__(self, alpha=5.0):
        self.alpha = alpha

    def transform(self, frame):
        analytic_signal = hilbert(frame)
        
        x = np.real(analytic_signal)
        x_hat = np.imag(analytic_signal)
        
        pt_signal = np.cos(self.alpha) * x + np.sin(self.alpha) * x_hat
        
        return pt_signal

class FDMDecomposition:
    def __init__(self, m_levels=10,fs = GET_SEGMENT_FS('2022-TIM')):
        self.M = m_levels
        self.fs = fs

    def decompose(self, frame):
        N = len(frame)
        f_coeffs = fft(frame)
        
        fibfs = []
        def get_fdm_indices(N):
            cutoffs_hz = [0.5, 1.95, 3.9, 7.8, 15.6, 31.2, 48.0, 52.0, 125.0]
            
            indices = [int(f * N / self.fs) for f in cutoffs_hz]
            indices = [0] + sorted(list(set(indices))) + [N // 2]
            return indices
        indices = get_fdm_indices(N)
        
        for i in range(self.M):
            mask = np.zeros(N, dtype=complex)
            start, end = indices[i], indices[i+1]
            
            mask[start:end] = f_coeffs[start:end]
            if start == 0:
                mask[N-end+1:] = f_coeffs[N-end+1:]
            else:
                mask[N-end+1:N-start+1] = f_coeffs[N-end+1:N-start+1]
            
            fibf = np.real(ifft(mask))
            fibfs.append(fibf)
            
        return np.array(fibfs)



if __name__ == "__main__":
    data_folder = get_data_folder( dataset_name="ecg_id")
    import os
    data = load2pth(os.path.join(data_folder, 'data_threshold.pth'))
    a = 1