import numpy as np
import torch
from torch.utils.data import Dataset
from utils import min_max_standardization,resorted_label
from config import SESSION_CONFIG,MULTI_SESSION_DATASET,SESSION2TRAINID,SINGLE_SESSION_DATASET
from utils.data import get_data_folder,load2pth,load2txt
from utils.dataset import PersonMap
import os

from typing import Dict,List
from collections import defaultdict

class TimeDataset(Dataset):
    def __init__(self,segments: np.ndarray,labels: np.ndarray,):
        self.segments = min_max_standardization(segments)
        self.labels, self.label_map = resorted_label(labels)
        self.num_classes = len(np.unique(self.labels)) if labels is not None else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seg = torch.from_numpy(self.segments[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seg,label
    
class RawTimeData:
    def __init__(self, dataset_name: str,process_des = '2025-ArXiv'):
        self.dataset_name = dataset_name
        self.person_map = PersonMap(dataset_name)
        path = get_data_folder(dataset_name=dataset_name, path=f"time_split_all/{process_des}/")
        self.session_list = SESSION_CONFIG.get(dataset_name, ['first'])
        self.process_des = process_des
        filenames = [f"{session}_session.pth" for session in self.session_list] if dataset_name in SESSION_CONFIG.keys() else [f"all_time_split_data.pth"]
        if dataset_name == 'autonomic':
            filenames = ['all_data.pth']
        self.data_dict = {session: load2pth(filepath = os.path.join(path, filename)) for session, filename in zip(self.session_list, filenames)}
        if dataset_name not in MULTI_SESSION_DATASET:
            return
        self.data_dict = self.select_reconstruction()
    def select_reconstruction(self) -> Dict:
        # int(dt.timestamp() / 86400)
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
                    new_data_dict[session_list[0]][self.person_map.get_id(k)]=v[0]
                    new_data_dict[session_list[1]][self.person_map.get_id(k)]=np.concatenate(v[1:], axis=0)
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

class TimeDatasetBuilder:
    def __init__(self, dataset_name: str,process_des = '2025-ArXiv'):
        self.rawData = RawTimeData(dataset_name,process_des)
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
                new_data_dict[session_list[i]] = rawData_dict[session]
                continue
            for person_id, data in rawData_dict[session].items():
                if person_id not in new_data_dict[session_list[-1]]:
                    new_data_dict[session_list[-1]][person_id] = data
                else:
                    new_data_dict[session_list[-1]][person_id] = np.concatenate([new_data_dict[session_list[-1]][person_id],data], axis=0)
        self.rawData.session_list = session_list
        self.rawData.data_dict = new_data_dict
    
    def _compile_session_data(self, rawData_dict: Dict):
        compiled_X = []
        compiled_y = []
        for person_id, data in rawData_dict.items():
            compiled_X.extend(data)
            compiled_y.extend([person_id]*len(data))
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

def merge_data(dataset_list: List[str],k=0,process_des = '2025-ArXiv'):
    all_data_dict = {}
    for dataset_name in dataset_list:
        builder = TimeDatasetBuilder(dataset_name=dataset_name,process_des=process_des)
        data_list = builder.build_all(k=k)
        all_data_dict.update({dataset_name:data_list})
    return all_data_dict

def pretrain_dataset(dataset_list: List[str],k = 0,process_des = '2025-ArXiv'):
    data_dict = merge_data(dataset_list,k=k,process_des=process_des)
    all_data = data_dict[dataset_list[0]]
    index = len(set(all_data['y']))
    for data_name in dataset_list[1:]:
        data = data_dict[data_name]
        data['y'] = data['y'] + index
        for k in all_data.keys():
            all_data[k] = np.concatenate([all_data[k],data[k]], axis=0)
        index += max(set(data['y'])) + 1
    return {'all_data':all_data}


def get_meta(dataset_name, k = None,session = 'first',process_des='2025-ArXiv',ids = None):
    builder = TimeDatasetBuilder(dataset_name=dataset_name,process_des=process_des)
    builder.reconstruction()
    if ids is not None:
        data_dict = builder.get_session_data_by_ids(session=session, ids = ids)
    elif k is None:
        data_dict = builder.get_session_data_by_fold(session=session, k=k)['train']
    else:
        data_dict = builder.get_session_data_by_fold(session=session, k=k)['test']
    segments = []
    labels = []
    ids = sorted(list(data_dict.keys()))
    old2new_map = {id: idx for idx, id in enumerate(ids)}
    new_data_dict = {old2new_map[old_id]: value for old_id, value in data_dict.items()}
    for label, seg in new_data_dict.items():
        segments.extend(seg)
        labels.extend([label] * len(seg))
    return np.array(segments) , np.array(labels)
if __name__ == "__main__":
    dataset_list = ['ecg_id','CYBHi','heartprint','SRRSH','autonomic']
    for dataset_name in dataset_list:
        segments, labels = get_meta(dataset_name)
        print(segments.shape, labels.shape)
    pretrain_dataset(dataset_list)
    # dataset_list = ['ecg_id','CYBHi','heartprint','SRRSH','autonomic']
    # for dataset_name in ['autonomic']:
    #     raw_data = RawTimeData(dataset_name=dataset_name)
    #     time_dataset_builder = TimeDatasetBuilder(dataset_name=dataset_name)
    #     new_data_dict = time_dataset_builder.build_all(k=0)
        
    #     print(new_data_dict.keys())
