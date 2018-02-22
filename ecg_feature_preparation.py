import os
import numpy as np
from matplotlib import pyplot as plt
import ecg_tools as ecg
from config import BASIC_DIR, ECG_eHEALTH_DATA_DIR

SEGM_START_IND = 0
SEGM_END_IND = 270
ECG_SEGMENT_LENGTH = SEGM_END_IND - SEGM_START_IND
RECORD_VISUALIZE = False
FEATURES_VISUALIZE = False


def feature_preparation(BASIC_DIR, ECG_FEATURES_PATH, participant_list=None, AVG_SEGMENT_NUM="ALL"):
    if participant_list is None:
        participant_list = os.listdir(BASIC_DIR)
    participant_list = sorted(participant_list)
    print participant_list
    data_set_dict = dict()
    for participant_name in participant_list:
        print participant_name
        participant_directory = os.path.join(BASIC_DIR, participant_name)
        participant_records = os.listdir(participant_directory)
        records_num = len(participant_records)
        ecg_features = np.empty([0, ECG_SEGMENT_LENGTH])
        for i in xrange(records_num):
            # reading data from file
            record_name = participant_records[i]
            record_path = os.path.join(participant_directory, record_name)
            # print file_path
            ecg_raw = np.load(record_path)

            if RECORD_VISUALIZE:
                plt.figure(participant_name)
                plt.plot(ecg_raw)
                plt.show()

            # ecg waveform processing
            ecg_process = ecg.preprocessing(ecg_raw)
            r_peaks_val, r_peaks_ind = ecg.r_peak_detection(ecg_process)
            ecg_segments = ecg.segmentation(ecg_process, r_peaks_ind)
            ecg_features_vec = ecg.statistical_features_calculation(ecg_segments[:, SEGM_START_IND:SEGM_END_IND],
                                                                    AVG_SEGMENT_NUM)
            if FEATURES_VISUALIZE:
                plt.figure(record_name)
                plt.title(participant_name)
                for j in xrange(ecg_features_vec.shape[0]):
                    plt.plot(ecg_features_vec[j, :])
                plt.show()

            ecg_features = np.vstack((ecg_features, ecg_features_vec))

        # add to dictionary features from new record
        data_set_dict[participant_name] = ecg_features
    np.savez(ECG_FEATURES_PATH, **data_set_dict)
    plt.show()


if __name__ == '__main__':
    ecgDataDir = ECG_eHEALTH_DATA_DIR
    ecgFeaturesPath = os.path.join(BASIC_DIR, 'eHealth_features.npz')
    feature_preparation(ecgDataDir, ecgFeaturesPath)
