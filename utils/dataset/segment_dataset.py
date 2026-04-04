import os
import sys
from typing import List, Dict
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config import (
    LOW_CUT, HIGH_CUT, SESSION_CONFIG,SINGLE_SESSION_DATASET,MULTI_SESSION_DATASET,SEGMENT_DES,SESSION2TRAINID,SEGMENT_DICT,GET_SEGMENT_FS
)
from utils.util.loader import get_data_folder, load2pth, load2txt, save2pth
from .personmap import PersonMap
import numpy as np
from scipy.stats import f, pearsonr


class SegmentDataset(Dataset):
    def __init__(self, segments: np.ndarray, labels: np.ndarray, 
                 session: int = 0, transform=None):
        self.segments = segments
        self.labels = labels
        # 统一处理 session
        if isinstance(session, str):
            session = SESSION2TRAINID.get(session, 0)
        self.session = np.full(len(labels), session, dtype=np.int32) if isinstance(session, int) else session
            
        self.transform = transform
        self.num_classes = len(set(labels)) if labels is not None else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, (list, slice, np.ndarray, torch.Tensor)):
            indices = range(*idx.indices(len(self))) if isinstance(idx, slice) else idx
            items = [self[i] for i in indices]
            return torch.utils.data.default_collate(items)

        seg = self.segments[idx].copy() # 避免修改原引用
        
        if self.transform:
            seg_tensor = self.transform(seg)
        else:
            seg_tensor = torch.from_numpy(seg).float()

        return {
            "seg": seg_tensor,
            "label": torch.as_tensor(self.labels[idx], dtype=torch.long),
        }

    def _merge(self, other, offset=0):
        """内部辅助函数，用于合并数据集"""
        self.segments = np.concatenate([self.segments, other.segments], axis=0)
        self.labels = np.concatenate([self.labels, other.labels + offset], axis=0)
        self.num_classes = len(set(self.labels))
        return self

    def __add__(self, other):
        return self._merge(other, offset=0)

    def add_with_offset(self, other):
        offset = np.max(self.labels) + 1 if len(self.labels) > 0 else 0
        return self._merge(other, offset=offset)

def merge_ecg_dicts(dict_list):
    collected = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            collected[key].append(value)
    
    merged_dict = {}
    for key, arrays in collected.items():
        merged_dict[key] = np.concatenate(arrays, axis=0)
        
    return merged_dict

class RawSegmentData:
    def __init__(self, dataset_name: str,process_des = SEGMENT_DES):
        self.dataset_name = dataset_name
        self.person_map = PersonMap(dataset_name)
        path = get_data_folder(dataset_name=dataset_name, path=f"{LOW_CUT}_{HIGH_CUT}/{process_des}/")
        self.session_list = SESSION_CONFIG.get(dataset_name, ['first'])
        self.process_des = process_des
        filenames = [f"{session}_data.pth" for session in self.session_list] if dataset_name in SESSION_CONFIG.keys() else [f"all_segment_data.pth"]
        if dataset_name == 'autonomic':
            filenames = ['data.pth']
        self.data_dict = {session: load2pth(filepath = os.path.join(path, filename)) for session, filename in zip(self.session_list, filenames)}
        if dataset_name not in MULTI_SESSION_DATASET:
            return
        self.data_dict = self.select_reconstruction()
    def select_reconstruction(self) -> Dict:
        if self.dataset_name in ['ptb','ecg_id']:
            return self.reconstruction_contiguous()
        else:
            return self.reconstruction()

    def reconstruction_contiguous(self):
        data_dict = self.data_dict
        new_data_dict = {}
        session_list = ['first','second']
        for session in session_list:
            new_data_dict[session] = defaultdict(list)
        
        for _,data in data_dict.items():
            for k,v in data.items():
                if len(v)>1:
                    new_data_dict[session_list[0]][self.person_map.get_id(k)]=copy.deepcopy(v[0])
                    new_data_dict[session_list[1]][self.person_map.get_id(k)]=merge_ecg_dicts(v[1:])
                    # new_data_dict[session_list[1]][self.person_map.get_id(k)]=v[1]
        self.session_list = session_list
        return new_data_dict
    
    def reconstruction(self) -> Dict:
        data_dict = self.data_dict
        new_data_dict = {}
        session_list = self.session_list
        for session in session_list:
            new_data_dict[session] = defaultdict(list)
        for session, data in data_dict.items():
            for k,v in data.items():
                new_data_dict[session][self.person_map.get_id(k)] = v
        self.session_list = session_list
        return new_data_dict

    def extract_subset_by_ids(self, target_ids: List[int],id_dict: Dict[int,int] = None) -> Dict:
        target_set = set(target_ids)
        subset_dict = {}
        data_dict = self.data_dict
        
        for session, user_data in data_dict.items():
            subset_dict[session] = defaultdict(list)
            for user_id, records in user_data.items():
                if user_id in target_set:
                    if id_dict is not None:
                        user_id = id_dict[user_id]
                    subset_dict[session][user_id] = records
                    
        return subset_dict


def compute_templates(reg_dict):
    templates = {}
    for label, content in reg_dict.items():
        data_list = content.get('segments', [])
        if len(data_list) > 0:
            templates[label] = np.mean(np.array(data_list), axis=0)
    return templates

def calculate_scores(samples, template, mode='MAE'):
    if mode == 'MAE':
        return np.mean(np.abs(samples - template), axis=1)
    elif mode == 'Pearson':
        scores = []
        for s in samples:
            corr, _ = pearsonr(s, template)
            scores.append(corr if not np.isnan(corr) else -1.0)
        return np.array(scores)
    
    else:
        raise ValueError("Mode must be 'MAE' or 'Pearson'")

def filter_dict_by_similarity(reg_dict, auth_dict, ratio=0.25, mode='MAE', selection='top',roi=None):
    templates = compute_templates(reg_dict)
    filtered_dict = {}

    for label, content in auth_dict.items():
        if label not in templates:
            continue
        
        X_auth = np.array(content['segments'])
        template = templates[label]
        
        # 1. 计算得分并排序
        if roi is not None:
            scores = calculate_scores(X_auth[:,roi[0]:roi[1]], template[roi[0]:roi[1]], mode=mode)
        else:
            scores = calculate_scores(X_auth, template, mode=mode)
        if mode == 'MAE':
            # MAE 越小越相似 -> 升序
            sorted_indices = np.argsort(scores)
        else:
            # Pearson 越大越相似 -> 降序
            sorted_indices = np.argsort(scores)[::-1]
            
        # 2. 确定筛选数量
        total_len = len(sorted_indices)
        num_select = max(1, int(np.ceil(total_len * ratio)))

        if selection == 'top':
            selected_indices = sorted_indices[:num_select]
            
        elif selection == 'bottom':
            selected_indices = sorted_indices[-num_select:]
            
        elif selection == 'middle':
            selected_indices = sorted_indices[num_select:-num_select]
            
        else:
            raise ValueError("selection 必须是 'top', 'bottom' 或 'middle'")
            
        # 4. 构建新字典
        filtered_dict[label] = {
            'segments': np.array([content['segments'][i] for i in selected_indices])
        }

    return filtered_dict

def find_worst_st_dict_by_similarity(reg_dict, auth_dict, mode='MAE',info = SEGMENT_DES):
    r_peak_time = SEGMENT_DICT[info]['rpeak_time']
    st_start_index = int((r_peak_time + 0.04)*GET_SEGMENT_FS(info))
    pqrs_end_index = int((r_peak_time + 0.06)*GET_SEGMENT_FS(info))
    print(st_start_index,pqrs_end_index)
    st_worst_dict = filter_dict_by_similarity(reg_dict, auth_dict, mode=mode, ratio=0.25, selection='bottom',roi=(st_start_index,-1))
    pqrs_dict = {'all':st_worst_dict}
    for selection in ['top','middle','bottom']:
        data_dict = filter_dict_by_similarity(reg_dict, st_worst_dict, mode=mode, ratio=0.25, selection=selection,roi=(0,pqrs_end_index))
        pqrs_dict[selection] = data_dict
    return pqrs_dict

class SegmentDatasetBuilder:
    def __init__(self, dataset_name: str,process_des = SEGMENT_DES):
        self.rawData = RawSegmentData(dataset_name,process_des)
        self.dataset_name = dataset_name
        self.person_map = self.rawData.person_map 
    
    def reconstruction(self):
        if self.dataset_name in ['ecg_id','CYBHi','exercise']:
            return
        rawData_dict = self.rawData.data_dict
        session_list = ['first','second']
        new_data_dict = {session:{} for session in session_list}
        for i,session in enumerate(self.rawData.session_list):
            if i < 1:
                new_data_dict[session_list[0]] = rawData_dict[session]
                continue
            for person_id, data in rawData_dict[session].items():
                if person_id not in new_data_dict[session_list[0]]:
                    new_data_dict[session_list[0]][person_id] = data
                elif person_id not in new_data_dict[session_list[-1]]:
                    new_data_dict[session_list[-1]][person_id] = data
                else:
                    new_data_dict[session_list[-1]][person_id] = merge_ecg_dicts([new_data_dict[session_list[-1]][person_id],data])
        self.rawData.session_list = session_list
        self.rawData.data_dict = new_data_dict
    
    def _compile_session_data(self, rawData_dict: Dict):
        compiled_X = []
        compiled_y = []
        for person_id, data in rawData_dict.items():
            compiled_X.extend(data["segments"])
            compiled_y.extend([person_id]*len(data["segments"]))
        return compiled_X, compiled_y
    def get_session_data_by_ids(self,session = None, rawData = None,ids = None):
        if rawData is None:
            rawData = self.rawData
        if session is None:
            session = rawData.session_list[0]
        if isinstance(session,int):
            session = rawData.session_list[session]

        self.train_person2id = self.person_map.get_new_person2id(ids)
        train_person_ids = list(ids)
        train_id2newids = {self.person_map.get_id(person): new_id 
                        for person, new_id in self.train_person2id.items()}
        return self.rawData.extract_subset_by_ids(train_person_ids, train_id2newids)[session]
    def get_session_data_by_fold(self, session = None, k = None, rawData = None):
        """
        根据 fold 参数 k 和预存的 ID 文件，过滤并返回对应的 session 训练数据。
        """
        if rawData is None:
            rawData = self.rawData
        if session is None:
            session = rawData.session_list[0]
        if isinstance(session,int):
            session = rawData.session_list[session]
        test_file_path = os.path.join(get_data_folder(dataset_name=self.dataset_name, path=""), f"ids_test_fold_{k}.txt")
        val_file_path = os.path.join(get_data_folder(dataset_name=self.dataset_name, path=""), f"ids_val_muti.txt")

        if k is None and not os.path.exists(test_file_path) and not os.path.exists(val_file_path):
            return {'train':rawData.data_dict[session],'test':rawData.data_dict[session]}
        if not os.path.exists(val_file_path):
            val_ids = self.person_map.id2person.keys()
        else:
            val_ids = set(map(int, load2txt(val_file_path)))
        if not os.path.exists(test_file_path):
            test_ids = set()
        else:
            test_ids = set(map(int, load2txt(test_file_path)))
        train_ids = val_ids - test_ids

        self.train_person2id = self.person_map.get_new_person2id(train_ids)
        train_person_ids = list(train_ids)
        train_id2newids = {self.person_map.get_id(person): new_id 
                        for person, new_id in self.train_person2id.items()}
        data_dict = {'train':self.rawData.extract_subset_by_ids(train_person_ids, train_id2newids)[session],
                     'test':self.rawData.extract_subset_by_ids(test_ids)[session]}
        return data_dict

    def build(self,rawData = None,k = None,session = None): 
        if rawData is None:
            rawData = self.rawData
        if session is None:
            session = rawData.session_list[0]
        if isinstance(session,int):
            session = rawData.session_list[session]
        session_data = self.get_session_data_by_fold(session=session, k=k, rawData=rawData)
        train_data = session_data['train']
        
        X, y = self._compile_session_data(train_data)
        session = np.full(len(y),SESSION2TRAINID.get(session,SESSION2TRAINID['default']),dtype=np.int32)
        return {'X':X,'y':y,'session':session}
    
    def build_all(self,k = None):
        data_list = []
        raw_sessions = self.rawData.session_list
        for session in range(len(raw_sessions)):
            data_dict = self.build(session=session,k=k)
            data_list.append(data_dict)
        new_data_dict = {k: np.concatenate([d[k] for d in data_list], axis=0) for k in data_list[0].keys()}
        return new_data_dict

def merge_data(dataset_list: List[str],k=0,process_des = SEGMENT_DES):
    all_data_dict = {}
    for dataset_name in dataset_list:
        builder = SegmentDatasetBuilder(dataset_name=dataset_name,process_des=process_des)
        data_list = builder.build_all(k=k)
        all_data_dict.update({dataset_name:data_list})
    return all_data_dict

def pretrain_dataset(dataset_list: List[str],k = 0,process_des = SEGMENT_DES):
    data_dict = merge_data(dataset_list,k=k,process_des=process_des)

    single_set = set(SINGLE_SESSION_DATASET) & set(dataset_list)
    
    if len(single_set) == 0:
        single_len = 1
        single_label_len = 0
    else:
        single_len = sum(len(data_dict[dataset_name]['y']) for dataset_name in single_set)
        single_label_len = sum(len(set(data_dict[dataset_name]['y'])) for dataset_name in single_set)
    
    muti_set = set(MULTI_SESSION_DATASET) & set(dataset_list)
    if len(muti_set) == 0:
        multi_len = 1
    else:
        multi_len = sum(len(data_dict[dataset_name]['y']) for dataset_name in muti_set)
    
    all_data = data_dict[dataset_list[0]]
    index = len(set(all_data['y']))
    
    for k in dataset_list[1:]:
        v = data_dict[k]
        v['y'] = v['y'] + index
        for k in all_data.keys():
            all_data[k] = np.concatenate([all_data[k],v[k]], axis=0)
        index += len(set(v['y']))
    session_count = np.bincount(all_data['session'])
    label_count = np.bincount(all_data['y'])
    return {'all_data':all_data,'single_len':single_len,'multi_len':multi_len,'single_label_len':single_label_len,'session_count':session_count,'label_count':label_count}

def build_templates(X, y):
    """
    计算注册集模板
    X: (N_reg, length) 
    y: (N_reg,)
    返回: {label: template_vector_of_length}
    """
    X = np.asarray(X)
    y = np.asarray(y)
    templates = {}
    for label in np.unique(y):
        # 提取该类别所有样本并计算均值向量
        samples = X[y == label]
        templates[label] = np.mean(samples, axis=0)
    return templates

def calc_score(a, b, mode='MAE'):
    """
    计算两个向量之间的相似度得分
    """
    if mode == 'MAE':
        # 越小越相似
        return np.mean(np.abs(a - b))
    elif mode == 'Pearson':
        # 越大越相似 (-1 到 1)
        corr, _ = pearsonr(a, b)
        return corr if not np.isnan(corr) else -1.0
    else:
        raise ValueError("Mode must be 'MAE' or 'Pearson'")

def find_indices(reg_X, reg_y, auth_X, auth_y, mode='MAE', ratio=0.25, selection='top',roi=None):
    reg_X = np.asarray(reg_X)
    reg_y = np.asarray(reg_y)
    auth_X = np.asarray(auth_X)
    auth_y = np.asarray(auth_y)
    # 1. 建立模板
    templates = build_templates(reg_X, reg_y)
    final_selected_indices = []
    # 2. 按类别进行筛选
    for label in np.unique(auth_y):
        if label not in templates:
            continue
        current_label_indices = np.where(auth_y == label)[0]
        template = templates[label]
        scores = []
        for i in current_label_indices:
            sample = auth_X[i]
            if roi is not None:
                score = calc_score(sample[roi[0]:roi[1]], template[roi[0]:roi[1]], mode=mode)
            else:
                score = calc_score(sample, template, mode=mode)
            scores.append(score)
        scores = np.array(scores)
        if mode == 'MAE':
            sorted_meta_idx = np.argsort(scores)
        else:
            sorted_meta_idx = np.argsort(scores)[::-1]
        num_select = max(1, int(np.ceil(len(sorted_meta_idx) * ratio)))
        if selection == 'top':
            selected_meta_idx = sorted_meta_idx[:num_select]
        elif selection == 'middle':
            selected_meta_idx = sorted_meta_idx[num_select:-num_select]
        else:
            selected_meta_idx = sorted_meta_idx[-num_select:]
        final_selected_indices.extend(current_label_indices[selected_meta_idx].tolist())
    return final_selected_indices

def find_worst_st_dict(reg_X, reg_y, auth_X, auth_y, mode='MAE',info = SEGMENT_DES):
    r_peak_time = SEGMENT_DICT[info]['rpeak_time']
    st_start_index = int((r_peak_time + 0.04)*GET_SEGMENT_FS(info))
    pqrs_end_index = int((r_peak_time + 0.06)*GET_SEGMENT_FS(info))
    print(st_start_index,pqrs_end_index)
    st_worst_indices = find_indices(reg_X, reg_y, auth_X, auth_y, mode=mode, ratio=0.25, selection='bottom',roi=(st_start_index,-1))
    st_worst_X = auth_X[st_worst_indices]
    st_worst_y = auth_y[st_worst_indices]
    pqrs_dict = {'all':{'X':st_worst_X,'y':st_worst_y}}
    for selection in ['top','middle','bottom']:
        data_dict = {}
        indices = find_indices(reg_X, reg_y, st_worst_X, st_worst_y, mode=mode, ratio=0.25, selection=selection,roi=(0,pqrs_end_index))
        data_dict['X'] = st_worst_X[indices]
        data_dict['y'] = st_worst_y[indices]
        pqrs_dict[selection] = data_dict
    return pqrs_dict
if __name__ == "__main__":
    # 测试代码
    dataset_list = ['ecg_id','CYBHi','heartprint']
    builder = SegmentDatasetBuilder(dataset_name='SRRSH')
    all_data = builder.build_all(k = None)
    
    save2pth(all_data,f'{get_data_folder(dataset_name="SRRSH")}/SRRSH_seg_data.pth')
    for dataset_name in dataset_list:
        data = SegmentDatasetBuilder(dataset_name=dataset_name)
        # data.reconstruction()
        # datasets = data.build_all(k = 1)
        # print(dataset_name,datasets)
    # print(1)