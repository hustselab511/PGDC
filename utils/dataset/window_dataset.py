import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from collections import defaultdict

from scipy.stats import pearsonr
from config import DATASET_DICT, FS, SEGMENT_DES, SESSION_CONFIG, SESSION2ID,SETUP_SEED,SEGMENT_DICT,GET_SEGMENT_FS
from utils import selected_segments_function,filter_correlation,calculate_ksqi, calculate_psqi
from utils.util.loader import  load2pth,save2txt,load2txt,natural_sort_key
from utils.dataset import PersonMap
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class WindowsDataset(Dataset):
    def __init__(self, data: List[np.ndarray], labels: List[int],session:List[int] = None,k_index:List[int] = None):
        self.data = data
        self.labels = labels
        self.session = session if session is not None else np.zeros(len(self.labels), dtype=np.int32)
        self.k_index = k_index if k_index is not None else np.zeros(len(self.labels), dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seg": torch.from_numpy(self.data[idx]).float(),
            "label": torch.tensor(self.labels[idx]).long(),
            "session": torch.tensor(self.session[idx]).long(),
            "k_index": torch.tensor(self.k_index[idx]).long(),
        }
    
    def new_index_dataset(self,indices:List[int] = None):
        data = [self.data[idx] for idx in indices]
        labels = [self.labels[idx] for idx in indices]
        session = [self.session[idx] for idx in indices]
        k_index = [self.k_index[idx] for idx in indices]
        return WindowsDataset(data,labels,session,k_index)


class PersonOriginData:
    def __init__(self, dataset_name: str,k = 0):
        self.dataset_name = dataset_name
        self.person_map = PersonMap(dataset_name)
        self.path = get_data_folder(dataset_name=dataset_name, path="")
        self.session_list = SESSION_CONFIG.get(dataset_name, ['first','second'])
        filenames = [f"{session}_session.pth" for session in self.session_list] if dataset_name in SESSION_CONFIG.keys() else ["all_data.pth"]
        self.data_dict = {session: load2pth(filepath = os.path.join(self.path,'processed_all',SEGMENT_DES, filename)) for session, filename in zip(self.session_list, filenames)}
        self.data_dict = self.reconstruction()
        self.multi_ids_set = self.muti_session_person()
        
    def reconstruction_contiguous(self):
        data_dict = self.data_dict
        new_data_dict = {}
        new_session_list = self.session_list[:2]
        for session,data in data_dict.items():
            record_keys = data.keys()
            records = defaultdict(list)
            for record_key in record_keys:
                person = record_key.split("_")[0]
                records[person].append(record_key)
            records =dict(records)
            for k,v in records.items():
                records[k] = sorted(v,key =natural_sort_key)
        session_num = (max(len(v) for v in records.values()))
        new_session_list = range(session_num)
        for session in new_session_list:
            new_data_dict[session] = defaultdict(list)
            
        for session, data in data_dict.items():
            for k,v in data.items():
                if len(v) == 0:
                    continue
                split_str = k.split("_")
                record_list = records[split_str[0]]
                new_session = record_list.index(k)
                new_data_dict[new_session][self.person_map.get_id(split_str[0])].append(v)
        self.session_list = new_session_list
        return new_data_dict
    
    def reconstruction_interval(self):
        data_dict = self.data_dict
        new_data_dict = {}
        new_session_list = self.session_list
        for session in new_session_list:
            new_data_dict[session] = defaultdict(list)
        for session, data in data_dict.items():

            for k,v in data.items():
                if len(v) == 0:
                    continue
                new_data_dict[session][self.person_map.get_id(k)] = v if isinstance(v, list) else [v]
        self.session_list = new_session_list
        return new_data_dict
    
    def reconstruction(self) -> List[Dict]:
        if self.dataset_name in ['ecg_id','ptb']:
            return self.reconstruction_contiguous()
        else:
            return self.reconstruction_interval()
    
    def muti_session_person(self,data_dict = None):
        if data_dict is None:
            data_dict = self.data_dict
        s1_key = data_dict[self.session_list[0]].keys() 
        s2_key = data_dict[self.session_list[1]].keys()
        multi_ids_set = s1_key & s2_key
        return multi_ids_set
    
    def extract_subset_by_ids(self, target_ids: List[int]) -> Dict:
        target_set = set(target_ids)
        subset_dict = {}
        data_dict = self.data_dict
        
        for session, user_data in data_dict.items():
            subset_dict[session] = defaultdict(list)
            for user_id, records in user_data.items():
                if user_id in target_set:
                    subset_dict[session][user_id] = records
                    
        return subset_dict
    

class RecodeSegmenter:
    @staticmethod
    def extract_windows_from_record(rec: Dict, window_size: int, stride: int = FS, mode: str="enroll",start_k: int = 0) -> List[np.ndarray]:
        segments = []
        k_index = []
        # 1. 基础校验
        ecg_signal = np.array(rec["filter_signal"])
        origin_signal = np.array(rec["resampled_signal"])
        if len(ecg_signal) < window_size:
            return {'segments':[],'k_index':[]}
        r_peaks = np.asarray(rec["r_peaks"], dtype=int)
        # 2. 滑动窗口计算
        num_windows = (len(ecg_signal) - window_size-start_k) // stride + 1
        for k in range(num_windows):
            start = k * stride+start_k
            end = start + window_size
            seg = RecodeSegmenter._extract_beats_from_window(ecg_signal,origin_signal, r_peaks, rec, start, end)
            if seg is not None:
                segments.append(seg)
                k_index.append(k)
                if mode == "enroll":
                    break
        return {'segments':segments,'k_index':k_index,'num_windows':num_windows,'valid_windows':len(segments)}

    @staticmethod
    def _extract_beats_from_window(ecg_signal,origin_signal, r_peaks, meta, start, end):
        win_sig = ecg_signal[start:end]
        if np.std(win_sig) < 0.001 or not np.isfinite(win_sig).all():
            return None

        mask = (r_peaks >= start) & (r_peaks < end)
        valid_r = r_peaks[mask]
        
        win_ori = origin_signal[start:end]
        ksqi = calculate_ksqi(win_ori)
        psqi = calculate_psqi(win_ori,FS)
        if psqi<0.2 or psqi>0.8 or ksqi<5 or ksqi>20:
            return None
        if len(valid_r) == 0: return None

        qrs_indices = np.stack([
            meta["q_diffs"][mask], valid_r,
            meta["s_diffs"][mask], meta["rr_intervals"][mask]
        ], axis=1)

        res = selected_segments_function(
            des=SEGMENT_DES, signal_ecg=ecg_signal, qrs_indices=qrs_indices, fs=meta.get("resampled_fs", FS)
        )
        
        clean_segs = res.get("segments", [])
        if len(clean_segs) > 0:
            mask_outlier = filter_correlation(clean_segs, threshold=0.95)
            # mask_outlier = filter_correlation_dist(clean_segs, threshold=0.9)
            clean_segs = clean_segs[mask_outlier]
        
        return clean_segs if len(clean_segs) >= 2 else None

class WindowsDatasetBuilder:
    def __init__(self, dataset_name: str,enroll_time = 20,test_time = 10):
        self.dataset_name = dataset_name
        self.origin_data = PersonOriginData(dataset_name)
        self.person_map = self.origin_data.person_map
        self.test_ids = []
        
        self.enroll_window_size = int(enroll_time * FS)
        self.test_window_size = int(test_time * FS)
        if dataset_name == 'heartprint':
            self.enroll_window_size = min(self.enroll_window_size, 10 * FS)
            self.test_window_size = min(self.test_window_size, 10 * FS)
        if dataset_name in ['ecg_id']:
            self.test_window_size = min(self.test_window_size, 10 * FS)
            self.enroll_window_size = min(self.enroll_window_size, 20 * FS)
    def get_test_ids(self,k: int = None):
        test_file_path = os.path.join(self.origin_data.path, f"ids_test_fold_{k}.txt")
        if os.path.exists(test_file_path):
            ids_test = set(map(int, load2txt(test_file_path)))
            return ids_test
        val_file_path = os.path.join(self.origin_data.path, f"ids_val_muti.txt")
        if os.path.exists(val_file_path):
            ids_test = set(map(int, load2txt(val_file_path)))
            return ids_test
        ids_test = set().union(*(map(int, session_data.keys()) 
                         for session_data in self.origin_data.data_dict.values()))
        return ids_test
    def get_train_ids(self,k: int = None):  
        ids_train = set().union(*(map(int, session_data.keys()) 
                         for session_data in self.origin_data.data_dict.values()))
        ids_test = set(self.get_test_ids(k))
        if ids_test  == ids_train:
            return ids_train
        ids_train = set(ids_train)-set(ids_test)
        return ids_train
    def build(self,k: int = None,session = None,mode = 'test',ids = None):
        if ids is None:
            ids = self.get_test_ids(k)
        if session is None:
            session = self.origin_data.session_list[0]
        if isinstance(session,int):
            session = self.origin_data.session_list[session]
        raw_data = self.origin_data.extract_subset_by_ids(ids)
        data_dict = defaultdict(list)
        session_data = raw_data[session]
        for person_id,person_data_list in session_data.items():
            segments = []
            for person_data in person_data_list:
                window_result = RecodeSegmenter.extract_windows_from_record(person_data, self.enroll_window_size, mode = mode)
                batches = window_result['segments']
                if batches and len(batches) > 0:
                    segments.extend(batches)
            if len(segments) > 0:
                data_dict[person_id] = {'seg':segments,'k_index':window_result['k_index']}
        
        return data_dict
    
    def build_all_dict(self,k: int = None,ids = None):
        if ids is None:
            ids = self.get_test_ids(k)
        raw_data = self.origin_data.extract_subset_by_ids(ids)
        enroll_ids={}
        enroll_data_dict = defaultdict(list)
        test_data_dict = defaultdict(list)
        for _,session_data in raw_data.items():
            for person_id,person_data_list in session_data.items():
                segments = []
                is_enroll = True
                for person_data in person_data_list:
                    if person_id not in enroll_ids:
                        is_enroll = False
                        window_result = RecodeSegmenter.extract_windows_from_record(person_data, self.enroll_window_size, mode = 'test')
                        batches = window_result['segments']
                        if batches and len(batches) > 0:
                            enroll_ids[person_id] = 1
                            enroll_data_dict[person_id] = {'seg':batches,'k_index':window_result['k_index']}
                            break
                    else:
                        window_result = RecodeSegmenter.extract_windows_from_record(person_data, self.test_window_size, mode = 'test')
                        batches = window_result['segments']
                        if batches and len(batches) > 0:
                            segments.extend(batches)
                if is_enroll and len(segments) > 0 and person_id not in test_data_dict:
                    test_data_dict[person_id] = {'seg':segments}
                elif is_enroll and len(segments) > 0:
                    test_data_dict[person_id]['seg'].extend(segments)
                    test_data_dict[person_id]['k_index'].extend(window_result['k_index'])
        
        return enroll_data_dict,test_data_dict
    
    def k_fold(self, k: int = 5,ids_test = None):
        if ids_test is None:
            ids_test = list(set(self.test_ids)-set(DATASET_DICT[self.dataset_name]['exclude']))
        random.shuffle(ids_test)
        total_len = len(ids_test)
        q, r = divmod(total_len, k)
        
        save2txt(ids_test, os.path.join(self.origin_data.path, f"ids_val_muti.txt"))
        # 3. 开始 K 折循环
        for i in range(k):
            if self.dataset_name in ['ecg_id','ptb']:
                print(f"Fold {i}: Test size={len(ids_test)}, Train size={total_len - len(ids_test)}")
                save2txt(ids_test, os.path.join(self.origin_data.path, f"ids_test_fold_{i}.txt"))
                continue
            # 计算切片的起始和结束索引
            start = i * q + min(i, r)
            end = (i + 1) * q + min(i + 1, r)
            # 切分数据
            test_ids_list = ids_test[start:end]
            print(f"Fold {i}: Test size={len(test_ids_list)}, Train size={total_len - len(test_ids_list)}")
            save2txt(test_ids_list, os.path.join(self.origin_data.path, f"ids_test_fold_{i}.txt"))

    # 跨会话
    def enroll_test(self,k = None,test_first = False,ids = None):
        if ids is None:
            ids = self.get_test_ids(k)
        raw_data = self.origin_data.extract_subset_by_ids(ids)
        enroll_ids={}
        test_ids=[]
        enroll_X = []
        enroll_y = []
        enroll_session = []
        enroll_k_index = []
        first_X = []
        first_y = []
        first_session = []
        first_k_index = []
        test_X = []
        test_y = []
        test_session = []
        test_k_index = []
        enroll_batches = {}
        all_window_num = 0
        all_window_valid = 0
        for session,session_data in raw_data.items():
            session_id = SESSION2ID.get(session, 0)
            for person_id,person_data_list in session_data.items():
                for person_data in person_data_list:
                    if person_id not in enroll_ids:
                        window_result = RecodeSegmenter.extract_windows_from_record(person_data, self.enroll_window_size, mode = 'test')
                        batches = window_result['segments']
                        k_index = window_result['k_index']
                        if batches:
                            enroll_ids[person_id] = 1
                            enroll_batches[person_id] = batches[0]
                            enroll_X.append(batches[0])
                            enroll_y.append(person_id)
                            enroll_session.append(session_id)
                            enroll_k_index.append(k_index[0])
                            first_X.extend(batches)
                            first_y.extend([person_id] * len(batches))
                            first_session.extend([session_id] * len(batches))
                            first_k_index.extend(k_index)
                            break
                    else:
                        if person_id in test_ids and test_first: 
                            continue
                        window_result = RecodeSegmenter.extract_windows_from_record(person_data, self.test_window_size, mode = 'test')
                        batches = window_result['segments']
                        k_index = window_result['k_index']
                        if not (window_result['num_windows'] == 0 or window_result['valid_windows'] == 0):
                            all_window_num += window_result['num_windows']
                            all_window_valid += window_result['valid_windows']
                        if batches:
                            test_ids.append(person_id)
                            test_X.extend(batches)
                            test_y.extend([person_id] * len(batches))
                            test_session.extend([session_id] * len(batches))
                            test_k_index.extend(k_index)
        test_ids = set(test_ids)
        self.test_ids = test_ids
        all_ids = self.origin_data.muti_session_person(raw_data)
        fail_ids = all_ids - test_ids
        print(f"{self.dataset_name} enroll and test fail_ids: {fail_ids}\t\t test_len: {len(test_ids)}")
        valid_ratio = all_window_valid / all_window_num
        print(f"{self.dataset_name} enroll and test all_window_num: {all_window_num}\t\t all_window_valid: {all_window_valid}\t\t valid_ratio: {valid_ratio}")
        new_enroll_X = []
        new_enroll_y = []
        new_enroll_session = []
        new_enroll_k_index = []
        for X,y,session_id,k_index in zip(enroll_X,enroll_y,enroll_session,enroll_k_index):
            if y in test_ids:
                new_enroll_X.append(X)
                new_enroll_y.append(y)
                new_enroll_session.append(session_id)
                new_enroll_k_index.append(k_index)
        return {'enroll_data':{'data':new_enroll_X, 'labels':new_enroll_y,'session':new_enroll_session,'k_index':new_enroll_k_index},
                'first_data':{'data':first_X, 'labels':first_y,'session':first_session,'k_index':first_k_index},
                'test_data':{'data':test_X, 'labels':test_y,'session':test_session,'k_index':test_k_index}}

def build_templates(reg_X, reg_y):
    """
    reg_X: list/array，每个元素是 (n, len)
    reg_y: (N,)
    返回每个人的平均模板 (len,)
    """
    reg_y = np.asarray(reg_y)
    templates = {}

    for label in np.unique(reg_y):
        person_samples = [np.asarray(reg_X[i]).mean(axis=0) for i in range(len(reg_X)) if reg_y[i] == label]
        templates[label] = np.mean(np.stack(person_samples, axis=0), axis=0)

    return templates


def calc_score(x, template, mode='MAE'):
    x = np.asarray(x).ravel()
    template = np.asarray(template).ravel()

    if x.shape != template.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {template.shape}")

    if mode == 'MAE':
        return np.mean(np.abs(x - template))
    elif mode == 'Pearson':
        if np.std(x) == 0 or np.std(template) == 0:
            return np.nan
        return pearsonr(x, template)[0]
    else:
        raise ValueError("mode must be 'MAE' or 'Pearson'")


def find_indices(reg_X, reg_y, auth_X, auth_y, mode='MAE',ratio=0.25, selection='top',roi=None):
    """
    reg_X: 每个元素 shape = (n, len)
    reg_y: (N_reg,)
    auth_X: 每个元素 shape = (n, len)
    auth_y: (N_auth,)
    """
    reg_y = np.asarray(reg_y)
    auth_y = np.asarray(auth_y)

    templates = build_templates(reg_X, reg_y)
    indices = []

    for label in np.unique(auth_y):
        if label not in templates:
            continue

        idxs = np.where(auth_y == label)[0]   # 该 label 在认证集中的原始索引
        template = templates[label]

        scores = []
        valid_idxs = []

        for i in idxs:
            x = np.asarray(auth_X[i])
            x_mean = x.mean(axis=0)
            if roi is not None:
                s = calc_score(x_mean[roi[0]:roi[1]], template[roi[0]:roi[1]], mode=mode)
            else:
                s = calc_score(x_mean, template, mode=mode)
            scores.append(s)
            valid_idxs.append(i)

        scores = np.array(scores)
        valid_idxs = np.array(valid_idxs)

        # 排序
        if mode == 'MAE':
            # 越小越相似
            order = np.argsort(scores)
        else:
            # Pearson 越大越相似
            order = np.argsort(scores)[::-1]

        num_select = max(1, int(np.ceil(len(order) * ratio)))

        if selection == 'top':
            selected = order[:num_select]
        elif selection == 'bottom':
            selected = order[-num_select:]
        elif selection == 'middle':
            selected = order[num_select:-num_select]

        indices.extend(valid_idxs[selected].tolist())

    return indices

def find_worst_st_dict(reg_X, reg_y, auth_X, auth_y, mode='MAE',ratio=0.25, info=SEGMENT_DES):
    r_peak_time = SEGMENT_DICT[info]['rpeak_time']
    st_start_index = int((r_peak_time + 0.04)*GET_SEGMENT_FS(info))
    pqrs_end_index = int((r_peak_time + 0.06)*GET_SEGMENT_FS(info))
    print(st_start_index,pqrs_end_index)
    st_worst_indices = find_indices(reg_X, reg_y, auth_X, auth_y, mode=mode, ratio=0.25, selection='bottom',roi=(st_start_index,-1))
    st_worst_X = [auth_X[i] for i in st_worst_indices]
    st_worst_y = [auth_y[i] for i in st_worst_indices]
    pqrs_dict = {'all':{'X':st_worst_X,'y':st_worst_y}}
    for selection in ['top','middle','bottom']:
        data_dict = {}
        indices = find_indices(reg_X, reg_y, st_worst_X, st_worst_y, mode=mode, ratio=0.25, selection=selection,roi=(0,pqrs_end_index))
        data_dict['X'] = [st_worst_X[i] for i in indices]
        data_dict['y'] = [st_worst_y[i] for i in indices]
        pqrs_dict[selection] = data_dict
    return pqrs_dict

if __name__ == "__main__":
    # 实例化管理器
    SETUP_SEED(seed=42)
    dataset_list = ['CYBHi','heartprint','ecg_id','SRRSH']      
    for dataset_name in dataset_list:
        builder = WindowsDatasetBuilder(dataset_name=dataset_name,enroll_time = 10,test_time = 5)
        builder.enroll_test()
