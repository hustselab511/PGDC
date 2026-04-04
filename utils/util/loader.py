# -*- coding: UTF-8 -*-
"""@Project ：ECG identify 
@File    ：data_process.py
@Author  ：yankangli
@Date    ：2025/10/25 20:39 
"""
import json
import os
import sys
import pickle
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import numpy as np
import torch
import wfdb
from datetime import datetime
from config import DATA_FOLDER, DATASET_FOLDER

def load2pth(filepath, **kwargs):
    if "pth" not in filepath.split(".")[-1]:
        filepath = filepath + ".pth"
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    return data

def save2pth(data, filepath, **kwargs):
    if "pth" not in filepath.split(".")[-1]:
        filepath = filepath + ".pth"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(data, filepath)
    print(f"已保存 {filepath}")

def load2txt(filepath) -> np.ndarray:
    """加载 txt 文件"""
    if "txt" not in filepath.split(".")[-1]:
        filepath = filepath + ".txt"
    return np.loadtxt(filepath)

def save2txt(data, filepath, **kwargs):
    """保存数组为 txt 文件"""
    if "txt" not in filepath.split(".")[-1]:
        filepath = filepath + ".txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, fmt="%.6f")


def save2pkl(data, filepath, **kwargs):
    data = {k: np.array(v) for k, v in data.items()}
    if "pkl" not in filepath.split(".")[-1]:
        filepath = filepath + ".pkl"
    with open(filepath, "wb") as f:  # 二进制模式
        pickle.dump(data, f)


def load2pkl(filepath: str) -> dict:
    if "pkl" not in filepath.split(".")[-1]:
        filepath = filepath + ".pkl"
    with open(filepath, "rb") as f:  # 二进制模式
        return pickle.load(f)


def load2npz(filepath: str, **kwargs):
    kwargs.setdefault("allow_pickle", True)
    loaded_npz = np.load(filepath, **kwargs)
    loaded_dict = {}
    array_keys = [k for k in loaded_npz.files if k != "non_array_data"]
    for key in array_keys:
        loaded_dict[key] = loaded_npz[key]

    if "non_array_data" in loaded_npz.files:
        try:
            non_array_data = pickle.loads(loaded_npz["non_array_data"].item())
            loaded_dict.update(non_array_data)  # 合并到结果字典
        except Exception as e:
            raise RuntimeError(f"反序列化非数组数据失败：{str(e)}")

    loaded_npz.close()

    return loaded_dict


def save2npy(data, filepath):
    """保存单个信号为 npy 文件"""
    if "npy" not in filepath.split(".")[-1]:
        filepath = filepath + ".npy"
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, data)


def load2npy(filepath) -> np.ndarray:
    """加载 npy 文件"""
    if "npy" not in filepath.split(".")[-1]:
        filepath = filepath + ".npy"
    return np.load(filepath)


def load2dat(filepath, **kwargs):
    record = wfdb.rdrecord(filepath)
    return record


def load2json(filepath) -> dict:
    """加载 JSON 文件"""
    if "json" not in filepath.split(".")[-1]:
        filepath = filepath + ".json"
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save2json(data, filepath):
    """保存字典为 JSON 文件"""
    if "json" not in filepath.split(".")[-1]:
        filepath = filepath + ".json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def save2txt(data, filepath):
    """保存列表为 TXT 文件，每个元素占一行"""
    if "txt" not in filepath.split(".")[-1]:
        filepath = filepath + ".txt"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")

def load2txt(filepath) -> list:
    """加载 TXT 文件，返回包含所有行的列表"""
    if "txt" not in filepath.split(".")[-1]:
        filepath = filepath + ".txt"
    return np.loadtxt(filepath)

def natural_sort_key(s):
    """
    将字符串拆解为 [文本, 数字, 文本, 数字...] 的列表
    例如: "person1_rec7" -> ['person', 1, '_rec', 7, '']
    """
    s = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_data_folder(path="", dataset_name="autonomic"):
    path = os.path.join(DATA_FOLDER, dataset_name, path)
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset_folder(path="", dataset_path="autonomic-1.0.0"):
    path = os.path.join(DATASET_FOLDER, dataset_path, path)
    os.makedirs(path, exist_ok=True)
    return path


def re_index(labels):
    """将非连续标签重新映射为0到n-1"""
    unique_labels = np.unique(labels)
    label_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_labels)}
    return np.array([label_mapping[label] for label in labels]), label_mapping


def person2key(all_records, data_folder):
    # 生成人员ID映射
    person_map = {key: idx for idx, key in enumerate(all_records.keys())}
    record_map = {
        key: len(all_records[key]) for idx, key in enumerate(all_records.keys())
    }
    # 保存映射到JSON文件
    with open(os.path.join(data_folder, "person.json"), "w") as f:
        json.dump(person_map, f, indent=4)
    with open(os.path.join(data_folder, "record.json"), "w") as f:
        json.dump(record_map, f, indent=4)
    print("人员ID映射已成功保存")
    return person_map

def smart_parse_time(raw_date,split_char = ''):
    if raw_date is None or raw_date.isspace():
        return None
    formats = [f"%y{split_char}%m{split_char}%d",f"%Y{split_char}%m{split_char}%d",
               f"%d{split_char}%m{split_char}%y",f"%d{split_char}%m{split_char}%Y",]
    
    for fmt in formats:
        try:
            # 尝试解析日期
            dt = datetime.strptime(raw_date, fmt)
            return dt
        except ValueError:
            continue
            
    return None
if __name__ == "__main__":
    func_param = {
        "ptb": {
            "dataset_name": "ptb",
            "dataset_path": "ptb-diagnostic-ecg-database-1.0.0",
        },
        "CYBHi_first": {
            "dataset_name": "CYBHi",
            "dataset_path": "CYBHi",
            "session": "first",
        },
        "CYBHi_second": {
            "dataset_name": "CYBHi",
            "dataset_path": "CYBHi",
            "session": "second",
        },
        "autonomic": {"dataset_name": "autonomic", "dataset_path": "autonomic-1.0.0"},
    }
    # data = load2dat('../../../static/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/10000/10000_hr',)
    data = load2dat(
        "../../../static/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/10000/10000_hr",
    )
    print(data)
