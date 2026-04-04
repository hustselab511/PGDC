from utils.dataset.segment_dataset import SegmentDatasetBuilder
import numpy as np
from config import SEGMENT_DES

def get_meta(dataset_name, k = None,session = 'first',process_des=SEGMENT_DES,ids = None,**kwargs):
    builder = SegmentDatasetBuilder(dataset_name=dataset_name,process_des=process_des)
    builder.reconstruction()
    if ids is not None:
        data_dict = builder.get_session_data_by_ids(session=session, ids = ids)
    elif k is None:
        data_dict = builder.get_session_data_by_fold(session=session, k=k)['train']
    else:
        data_dict = builder.get_session_data_by_fold(session=session, k=k)['train']
    segments = []
    labels = []
    ids = sorted(list(data_dict.keys()))
    old2new_map = {id: idx for idx, id in enumerate(ids)}
    new_data_dict = {old2new_map[old_id]: value for old_id, value in data_dict.items()}
    for label, rec in new_data_dict.items():
        seg = rec['segments']
        segments.extend(seg)
        labels.extend([label] * len(seg))
    return np.array(segments) , np.array(labels)
