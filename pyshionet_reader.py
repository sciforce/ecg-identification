import os
import pandas
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from config import BASIC_DIR

ECG_RECORDS_DIR = 'ECG-ID'
DOWNLOADS_DIR = 'source'
RESAMPLED_DIR = 'resampled'
records_metadata_file = 'ECG-ID_RECORDS.txt'
# command template for download csv file from ECG-ID data base at physionet.org
download_command = 'rdsamp -r ecgiddb/%s -c -H -f 0 -t 20 -v -pS >%s'
VISUALIZE = False

records_metadata_path = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, records_metadata_file)
f = open(records_metadata_path, 'r')
records_list = f.read().splitlines()
f.close()

for record in records_list:
    participant_id, record_id = os.path.split(record)
    participant_dir = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, DOWNLOADS_DIR, participant_id)
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
    downloading_path = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, DOWNLOADS_DIR, participant_id, record_id + '.csv')

    print 'downloading record %s' % record
    os.system(download_command % (record, downloading_path))  # executing bash command with given arguments

    dataFrame = pandas.read_csv(downloading_path)
    dataFrame = dataFrame[1:]  # removing first row (measurement units)
    dataFrame = dataFrame[dataFrame.columns[1]]  # selecting column with non-filtered ECG signal
    ecg_I = dataFrame.as_matrix()  # converting dataFrame to numpy

    source_rate = 500  # sampling rate of original signal (in Hz)
    target_rate = 270  # sampling rate of resampled signal (in Hz)
    resampling_factor = float(target_rate) / source_rate
    n_samples_source = len(ecg_I)  # number of samples in source signal
    n_samples_target = int(round(n_samples_source * resampling_factor))
    ecg_I_resampled = signal.resample(ecg_I, n_samples_target)

    participant_dir = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, RESAMPLED_DIR, participant_id)
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
    file_path = os.path.join(participant_dir, record_id)
    np.save(file_path, ecg_I_resampled)

    if VISUALIZE:
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(ecg_I)
        plt.subplot(2, 1, 2)
        plt.plot(ecg_I)
        plt.show()
