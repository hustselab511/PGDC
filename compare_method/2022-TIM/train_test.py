import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from scipy.stats import skew, kurtosis
from spectrum import pburg
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from utils.util.metrics import get_metrics
import torch

from config import GET_SEGMENT_FS,TRAIN_FOLD
from utils.dataset.segment_dataset import SegmentDatasetBuilder,filter_dict_by_similarity,find_worst_st_dict_by_similarity
from preprocess import form_frames,PhaseTransform,FDMDecomposition

def extract_features(frame, pt_frame, fibfs):
    # 1. PT特征 (从相位变换后的信号提取)
    f_skew = skew(pt_frame)
    f_kurt = kurtosis(pt_frame)
    
    # 2. EDS特征 (从10个FIBFs提取)
    total_energy = np.sum(frame**2)
    eds = [np.sum(f**2) / total_energy for f in fibfs]
    
    ar_model = pburg(frame, 5)
    ar_model.run()
    ar_coeffs = ar_model.ar

    feature_vector = np.concatenate([[f_skew, f_kurt], eds, ar_coeffs])
    return feature_vector

def extract_frames(segments):
    frames = form_frames(segments, beats_per_frame=2)
    feature_vectors = []
    
    for frame in frames:
        pt_frame = PhaseTransform().transform(frame)
        fibfs = FDMDecomposition(fs=GET_SEGMENT_FS('2022-TIM')).decompose(frame)
        feature_vectors.append(extract_features(frame, pt_frame, fibfs))
    
    return np.array(feature_vectors).real

def build_dataset(data_dict):
    X = []
    y = []
    for label, rec in data_dict.items():
        segments = rec['segments']
        feature_vectors = extract_frames(segments)
        X.extend(feature_vectors)
        y.extend([label] * len(feature_vectors))   
    return np.array(X), np.array(y)

class Trainer:
    def __init__(self):
        self.clf = RandomForestClassifier(random_state=42)
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        训练模型并可选执行十折交叉验证
        """
        self.clf.fit(X_train, y_train)
        self.is_trained = True
        print("模型训练完成。")

    def evaluate(self, test_data_dict,test_info = None,dataset_name = None,k = None):
        all_final_ground_truth = []
        
        total_windows = 0
        rejected_windows = 0
        
        logit_list = []
        label_list = []
        
        for label, rec in test_data_dict.items():
            segments = rec['segments']
            X_sub = extract_frames(segments) 
            
            raw_preds = self.clf.predict(X_sub)
            raw_probs = self.clf.predict_proba(X_sub)
            window_size = 7 
            threshold = 4   
            
            for i in range(0, len(raw_preds) - window_size + 1):
                total_windows += 1
                window = raw_preds[i : i + window_size]
                window_probs = raw_probs[i : i + window_size]
                
                mode_result = stats.mode(window, keepdims=True)
                most_frequent_count = mode_result.count[0]
                
                if most_frequent_count >= threshold:
                    all_final_ground_truth.append(label)
                    logit_list.append(np.mean(window_probs,axis=0))
                    label_list.append(label)
                else:
                    rejected_windows += 1
        # 计算统计指标
        
        rejection_rate = (rejected_windows / total_windows * 100) if total_windows > 0 else 0
        
        print(f"--- 评估报告 ---")
        print(f"决策窗口总数: {total_windows}")
        print(f"丢弃窗口数量: {rejected_windows}")
        print(f"丢弃率 (Rejection Rate): {rejection_rate:.2f}%")

        logit_list = torch.tensor(logit_list)
        label_list = torch.tensor(label_list)
        get_metrics(logit_list,label_list,test_info = test_info,save_folder=f"metrics/{test_info}/{dataset_name}/{k}/")

if __name__ == "__main__":
    info = '2022-TIM'
    # TRAIN_FOLD = {'AF':{'k':[None]},'PAC':{'k':[None]},'PVC':{'k':[None]}}
    dataset_name = 'SRRSH'
    TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name]}
    for dataset_name,fold_dict in TRAIN_FOLD.items():
        for k in fold_dict['k']:
            builder = SegmentDatasetBuilder(dataset_name=dataset_name,process_des=info)
            builder.reconstruction()
            first_data = builder.get_session_data_by_fold(session=0, k=k)['train']
            second_origin_data = builder.get_session_data_by_fold(session=1, k=k)['train']
            ids = sorted(list(first_data.keys()))
            second_ids = sorted(list(second_origin_data.keys()))
            unique_ids = list(set(ids) & set(second_ids))
            print(len(unique_ids))
            old2new_map = {id: idx for idx, id in enumerate(unique_ids)}
            new_first_data = {old2new_map[old_id]: value for old_id, value in first_data.items() if old_id in unique_ids}
            new_second_data = {old2new_map[old_id]: value for old_id, value in second_origin_data.items() if old_id in unique_ids}
            X_train,y_train = build_dataset(new_first_data)
            trainer = Trainer()
            trainer.train(X_train, y_train)
            for mode in ['Pearson','MAE']:
                for selection in ['top','middle','bottom']:
                    test_second_data = filter_dict_by_similarity(new_first_data,new_second_data, mode=mode, selection=selection)
                    trainer.evaluate(test_second_data,test_info = info,dataset_name =dataset_name,k = k)
            for mode in ['Pearson','MAE']:
                pqrs_dict = find_worst_st_dict_by_similarity(new_first_data,new_second_data,mode=mode,info=info)
                for selection,data_dict in pqrs_dict.items():
                    test_second_data = data_dict
                    trainer.evaluate(test_second_data,test_info = info,dataset_name =dataset_name,k = k)
            trainer.evaluate(new_second_data,test_info = info,dataset_name =dataset_name,k = k)
