import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import cv2
import librosa
from config import GET_SEGMENT_FS
from utils.util.loader import get_data_folder

import os

class PreMDataset(Dataset):
    def __init__(self, raw_signals, labels, fs=1000, mmap_path='spec_data.mmap', overwrite=False):
        """
        raw_signals: shape (7594, 1000)
        labels: shape (7594,)
        mmap_path: 硬盘缓存文件的路径
        overwrite: 是否覆盖已存在的缓存文件
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_samples = raw_signals.shape[0]
        self.shape = (self.num_samples, 1, 256, 256)
        self.class_num = len(set(labels))
        # 1. 处理时域信号（这个占用内存小，可以直接存）
        self.time_tensors = torch.from_numpy(raw_signals).float()
        
        mmap_path = os.path.join(get_data_folder(dataset_name = '2025-jsen'), mmap_path)
        if not os.path.exists(mmap_path) or overwrite:
            print(f"正在预处理并将频谱图写入磁盘缓存: {mmap_path}...")
            # 创建一个空的内存映射文件
            spec_mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=self.shape)
            
            for i in tqdm(range(self.num_samples)):
                # 假设 prem_spectrogram 是你的转换函数
                spec = prem_spectrogram(raw_signals[i], fs=fs) 
                spec_norm = spec.astype(np.float32) / 255.0
                spec_mmap[i, 0, :, :] = spec_norm
            
            # 确保数据写入磁盘
            spec_mmap.flush()
            del spec_mmap # 释放该对象
            print("预处理完成！")
        else:
            print(f"检测到现有缓存 {mmap_path}，直接加载。")

        # 3. 以只读模式加载内存映射
        # 这一步几乎不占内存，它只是建立了索引
        self.spec_data = np.memmap(mmap_path, dtype='float32', mode='r', shape=self.shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 从磁盘按需读取，OS 会自动管理缓存
        # 使用 .copy() 将数据载入真实内存，防止 PyTorch 多进程冲突
        spec_tensor = torch.from_numpy(self.spec_data[idx].copy()).float()
        
        return self.time_tensors[idx], spec_tensor, self.labels[idx]


def prem_spectrogram(heartbeat_signal, fs=GET_SEGMENT_FS('2025-jsen')):
    """
    实现论文中的 Algorithm 1: PreM Spectrogram 算法
    """
    groups = {
        'group1': {'range': (0.5, 10), 'signal': heartbeat_signal},
        'group2': {'range': (0.5, 3), 'signal': heartbeat_signal},
        'group3': {'range': (0, 50), 'signal': heartbeat_signal}
    }
    mel_spectrograms = []
    n_fft = fs
    hop_length = 2
    n_mels = 64
    for key, config in groups.items():
        fmin, fmax = config['range']
        sig = config['signal']
        
        stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length, window='hann')
        power_spec = np.abs(stft)**2
        
        mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, 
                                        fmin=fmin, fmax=fmax)
        
        mel_spec = np.dot(mel_basis, power_spec)

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spectrograms.append(mel_spec_db)
    
    combined_mel = np.vstack(mel_spectrograms) 
    
    combined_mel_norm = cv2.normalize(combined_mel, None, 0, 255, cv2.NORM_MINMAX)
    
    prem_spectrogram = cv2.resize(combined_mel_norm.astype(np.uint8), (256, 256))
    
    return prem_spectrogram

def save_dataset(segments,labels, save_path):
    labels = torch.tensor(labels, dtype=torch.long)
    num_samples = segments.shape[0]

    print("正在预处理时域信号...")
    time_data = np.pad(segments, ((0, 0), (0, 600)), 'constant')
    time_tensors = torch.from_numpy(time_data.reshape(num_samples, 64, 25)).float()
    
    spec_tensors = torch.zeros((num_samples, 1, 256, 256), dtype=torch.float32)

    for i in tqdm(range(num_samples)):
        spec = prem_spectrogram(segments[i], fs=GET_SEGMENT_FS('2025-jsen')) 
        spec_norm = spec.astype(np.float32) / 255.0
        spec_tensors[i, 0, :, :] = torch.from_numpy(spec_norm)
    
    torch.save({'segments':time_tensors, 'spec':spec_tensors, 'labels':labels}, save_path)
    print(f"数据集已保存到 {save_path}")
    
if __name__ == "__main__":
    heartbeat_signal = np.random.randn(1000)
    prem_spectrogram = prem_spectrogram(heartbeat_signal)
    print(prem_spectrogram.shape)