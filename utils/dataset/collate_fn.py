import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from torch.nn.utils.rnn import pad_sequence
import torch

def padding_collate_fn(batch):
    data_list = [item["seg"] for item in batch]
    label_list = [item["label"] for item in batch]
    session_list = [item["session"] for item in batch]
    time_list = [item["time"] for item in batch]
    k_index_list = [item["k_index"] for item in batch]
    
    lengths = torch.tensor([t.size(0) for t in data_list])
    padded_data = pad_sequence(data_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(label_list)
    sessions = torch.stack(session_list)
    times = torch.stack(time_list)
    k_indexs = torch.stack(k_index_list)
    
    max_len = padded_data.shape[1]
    padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    return {"seg": padded_data, "label": labels, "session": sessions, "time": times, "k_index": k_indexs, "lengths": lengths, "padding_mask": padding_mask}
