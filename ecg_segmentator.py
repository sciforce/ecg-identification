import os
import numpy as np
from matplotlib import pyplot as plt
from config import BASIC_DIR, SEGMENTATOR_DIR, ECG_eHEALTH_DATA_DIR
import ecg_tools as ecg
import uuid

ECG_RECORDS_DIR = ECG_eHEALTH_DATA_DIR
EXTENSION = '.npy'
PARTICIPANT_LIST = ['user1', 'user2', 'user3']
ECG_SEGMENT_LENGTH = 270
P_DIR = 'P'
P_SEGM_START = 0
P_SEGM_END = 60
QRS_DIR = 'QRS'
QRS_SEGM_START = 60
QRS_SEGM_END = 100
T_DIR = 'T'
T_SEGM_START = 100
T_SEGM_END = ECG_SEGMENT_LENGTH
VISUALIZE = False


def retrieve_part(data, part_start, part_end):
    ecg_part = np.zeros([data.shape[0], part_end-part_start])
    for j in xrange(data.shape[0]):
        ecg_part[j, 0:part_end-part_start] = data[j, part_start:part_end]
    return ecg_part


def save_each_row_of_list_as_file(records_directory, ecg_data_parts, participant_name):
    base_directory = os.path.join(SEGMENTATOR_DIR, records_directory)
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    for j in xrange(ecg_data_parts.shape[0]):
        filename = participant_name + '_' + str(uuid.uuid4().hex) + EXTENSION
        file_path = os.path.join(base_directory, filename)
        np.save(file_path, ecg_data_parts[j, :])


def plot_list_rows(figure_name, data):
    plt.figure(figure_name)
    for j in xrange(data.shape[0]):
        plt.plot(data[j, :])
        plt.hold(True)


def retrieve_records(participant_directory):
    file_list = os.listdir(participant_directory)
    file_path = os.path.join(participant_directory, file_list[0])
    ecg_raw = np.load(file_path)
    ecg_process = ecg.preprocessing(ecg_raw)
    r_peaks_val, r_peaks_ind = ecg.r_peak_detection(ecg_process)
    ecg_segments = ecg.segmentation(ecg_process, r_peaks_ind, ECG_SEGMENT_LENGTH)
    return ecg_segments


def main():
    print PARTICIPANT_LIST

    for participant_name in PARTICIPANT_LIST:
        print 'Working with ' + participant_name
        participant_directory = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, participant_name)
        ecg_segments = retrieve_records(participant_directory)
        plot_list_rows('Segments', ecg_segments)

        # retrieve P parts
        ecg_p_parts = retrieve_part(ecg_segments, P_SEGM_START, P_SEGM_END)
        if VISUALIZE:
            plot_list_rows('P-parts', ecg_p_parts)
        save_each_row_of_list_as_file(P_DIR, ecg_p_parts, participant_name)

        # retrieve QRS parts
        ecg_qrs_parts = retrieve_part(ecg_segments, QRS_SEGM_START, QRS_SEGM_END)
        if VISUALIZE:
            plot_list_rows('QRS-parts', ecg_qrs_parts)
        save_each_row_of_list_as_file(QRS_DIR, ecg_qrs_parts, participant_name)

        # retrieve T parts
        ecg_t_parts = retrieve_part(ecg_segments, T_SEGM_START, T_SEGM_END)
        if VISUALIZE:
            plot_list_rows('T-parts', ecg_t_parts)
        save_each_row_of_list_as_file(T_DIR, ecg_t_parts, participant_name)

    plt.show()

if __name__ == '__main__':
    main()