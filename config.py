from sklearn import metrics
import torch
import os
from pathlib import Path

SINGLE_SESSION_DATASET = ['autonomic']
TRAIN_DATASET_LIST = SINGLE_SESSION_DATASET+['CYBHi','heartprint']
TEST_DATASET_LIST = ['CYBHi','heartprint', 'ecg_id','SRRSH']
MULTI_SESSION_DATASET = ['CYBHi','heartprint', 'ecg_id','SRRSH']
TEST_DATASET_NAME = TEST_DATASET_LIST[0]

DATASET_FOLD = {     
                    'ecg_id':{'k':[None],},
                    'SRRSH':{'k':[None]},
                    'heartprint':{'k':range(5)},
                    'CYBHi':{'k':range(5)}
                }

TEST_FOLD = TRAIN_FOLD = {dataset_name: DATASET_FOLD[dataset_name] for dataset_name in TEST_DATASET_LIST}
# model
HIDDEN_SIZE = 256
MODEL_NAME = 'vit'
MOE_NAME = 'cross_attention'

# FOLDER
PROJECT_PATH = os.path.dirname(__file__)
TRAIN_FOLDER = 'metrics/train'
MODEL_FOLDER = 'basic'
# 需要替换为自己的数据集路径
BASE_MNT = Path("/mnt/165d5ea0-c4bf-423a-a83b-e5fb727cf65f/ecgIdentify/")
DATA_FOLDER = str(BASE_MNT / "data")
DATASET_FOLDER = str(BASE_MNT / "dataset")

DATASET_DICT = {
    'CYBHi': {'path': 'CYBHi/data/long-term/', 'num_classes': 63, 'fs': 1000, 'exclude': []},
    'CYBHi-short': {'path': 'CYBHi/data/short-term/85/', 'num_classes': 65, 'fs': 1000, 'exclude': []},
    'ptb': {'path': 'ptb/', 'num_classes': 290, 'fs': 500, 'exclude': []},
    'autonomic': {'path': 'autonomic-1.0.0/', 'num_classes': 1045, 'fs': 1000, 'exclude': []},
    'MITDB': {'path': 'mit-bih-arrhythmia-database-1.0.0/', 'num_classes': 47, 'fs': 360, 'exclude': []},
    'MITST': {'path': 'mit-bih-st-change-database-1.0.0/', 'num_classes': 28, 'fs': 360, 'exclude': []},
    'heartprint': {'path': 'heartprint/', 'num_classes': 199, 'fs': 250, 'exclude': []},
    'ecg_id': {'path': 'ecg_id/', 'num_classes': 90, 'fs': 500, 'exclude': []},
    'SRRSH': {'path': 'SRRSH/', 'num_classes': 66, 'fs': 200, 'exclude': []},
    'AF': {'path': 'SRRSH_abnormal/AF/', 'num_classes': 9, 'fs': 200, 'exclude': []},
    'PAC': {'path': 'SRRSH_abnormal/PAC/', 'num_classes': 10, 'fs': 200, 'exclude': []},
    'PVC': {'path': 'SRRSH_abnormal/PVC/', 'num_classes': 8, 'fs': 200, 'exclude': []},
    'exercise': {'path': 'exercise ECGID database/', 'num_classes': 45, 'fs': 300, 'exclude': []},
}
SESSION2TRAINID = {'first':0,'second':1,'Session-1':0,'Session-2':1,'Session-3L':2,'Session-3R':3,'default':0,'even':0,'odd':1,
                   'first1':0,'first2':1,'first3':1,'second1':1,'second2':1,'second3':1,'third1':1,'third2':1,'third3':1,'thrid':1}
SESSION2ID = {'first':0,'second':1,'Session-1':0,'Session-2':1,'Session-3L':2,'Session-3R':3,'default':0,
              'third':2,'fourth':3,}
SESSION_CONFIG = {
    'heartprint': ['Session-1', 'Session-2','Session-3L','Session-3R'],
    'CYBHi': ['first', 'second'],
    'SRRSH': ['first','second','third','fourth'],
    'AF': ['first', 'second','third'],
    'PAC': ['first', 'second','third'],
    'PVC': ['first', 'second','third'],
    'default': ['first']
}
# data
LOW_CUT = 0.5
HIGH_CUT = 50
FS = 200

SEGMENT_DICT = {'rri': {'time': 0.7, 'rpeak_time': 0.3},
                'center_rri': {'time': 0.8, 'rpeak_time': 0.4},
                'short_long': {'time': 0.6, 'rpeak_time': 0.2},#used
                '2022-TIM': {'time': 0.65, 'rpeak_time': 0.25},
                '2023-TOMM': {'time': 0.512, 'rpeak_time': 0.256},
                '2022-TETCI': {'time': 0.512, 'rpeak_time': 0.128},
                '2025-jsen': {'time': 1.0, 'rpeak_time': 0.375,'fs':1000},
                '2024-jsen': {'time': 0.8, 'rpeak_time': 0.4,'fs':500},
                '2026-ArXiv': {'time': 1.2, 'rpeak_time': 0.4,'fs':200},
                '2025-ArXiv': {'fs':250}}

def GET_SEGMENT_FS(segment_des):
    return SEGMENT_DICT[segment_des].get('fs',FS)

SEGMENT_DES = 'short_long'
PROCESS_LIST = [SEGMENT_DES,'2022-TIM','2023-TOMM','2022-TETCI','2025-jsen','2024-jsen']
PROCESS_DICT = {SEGMENT_DES:{'remove_mode':'number','select_num':100},
                '2022-TIM':{'remove_mode':'threshold','threshold_coeff':1},
                '2023-TOMM':{'remove_mode':'threshold','threshold_coeff':1},
                '2022-TETCI':{'remove_mode':'threshold','threshold_coeff':1},
                '2025-jsen':{'remove_mode':'threshold','threshold_coeff':1},
                '2024-jsen':{'remove_mode':'threshold','threshold_coeff':1.5},
                '2026-ArXiv':{'remove_mode':'threshold','select_num':100}}

# interp
R_PEAK = int(SEGMENT_DICT[SEGMENT_DES]['rpeak_time'] * FS)
DATA_LENGTH = int(SEGMENT_DICT[SEGMENT_DES]['time'] * FS)

# default
QRS_LENGTH = int(0.12 * FS)
QRS_START = R_PEAK - QRS_LENGTH // 2
QRS_END = R_PEAK + QRS_LENGTH - QRS_LENGTH // 2

S_OFFSET_TIME = 0.06
T_OFFSET_TIME = 0.04
S_END = R_PEAK + int(S_OFFSET_TIME*FS)
T_START = R_PEAK + int(T_OFFSET_TIME*FS)

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GET_SEG_LEN(seg='seg',segment_des = SEGMENT_DES):
    fs = GET_SEGMENT_FS(segment_des)
    data_len = int(SEGMENT_DICT[segment_des]['time'] * fs)
    if seg == 'seg':
        return data_len
    r_peak = int(SEGMENT_DICT[segment_des]['rpeak_time'] * fs)
    if seg == 'pqrs':
        return data_len-r_peak+int(T_OFFSET_TIME*fs)
    return data_len-r_peak-int(T_OFFSET_TIME*fs)

# METHOD
def GET_EXPERT_NAME(num):
    return f'{num} experts'

def GET_FOLDER(model_folder=MODEL_FOLDER, model_name=MODEL_NAME,**kwargs):
    return f'{PROJECT_PATH}/metrics/{model_name}/{model_folder}/'

def SETUP_SEED(seed=42):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False