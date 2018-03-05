import os
import numpy as np
from scipy.io import savemat
from config import ECG_eHEALTH_DATA_DIR

SOURCE_DIR = 'trainECG'
TARGET_DIR = 'trainECG_mat'

source_data_path = os.path.join(ECG_eHEALTH_DATA_DIR, SOURCE_DIR)
target_data_path = os.path.join(ECG_eHEALTH_DATA_DIR, TARGET_DIR)
if not os.path.exists(target_data_path):
    os.makedirs(target_data_path)

participant_list = os.listdir(source_data_path)
for participant in participant_list:
    source_participant_path = os.path.join(source_data_path, participant)
    records_list = os.listdir(source_participant_path)
    target_participant_dir = os.path.join(target_data_path, participant)
    print target_participant_dir
    if not os.path.exists(target_participant_dir):
        os.makedirs(target_participant_dir)
    for record in records_list:
        source_record_path = os.path.join(source_data_path, participant, record)
        record = record.replace('.npy', '.mat')  # update file extension
        converted_record_path = os.path.join(target_data_path, participant, record)
        ecg_features = np.load(source_record_path)
        savemat(converted_record_path, {'ecg_data': ecg_features})
