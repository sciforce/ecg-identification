__author__ = 'ykhoma'
import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from config import BASIC_DIR, ECG_eHEALTH_DATA_DIR
import ecg_tools as ecg

ECG_RECORDS_DIR = ECG_eHEALTH_DATA_DIR

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLES_NUM = 3000  # number of samples to record

ECG_SEGMENT_LENGTH = 270


def main():
    if len(sys.argv) == 1:
        participant_name = 'Undefined'
    elif len(sys.argv) == 2:
        participant_name = sys.argv[1]
    else:
        print "Wrong number of input parameters"
        sys.exit()

    time.sleep(1)
    print 'recording started'
    data = ecg.serial_reader(SERIAL_PORT, BAUD_RATE, SAMPLES_NUM)
    print 'recording finished'

    file_name = participant_name + time.strftime("_%m_%d_%H_%M_%S")
    participant_dir = os.path.join(BASIC_DIR, ECG_RECORDS_DIR, participant_name)
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
    file_path = os.path.join(participant_dir, file_name)
    np.save(file_path, data)

    # data preprocessing
    filt_data = ecg.preprocessing(data)
    r_peaks_val, r_peaks_ind = ecg.r_peak_detection(filt_data)
    ecg_segments = ecg.segmentation(filt_data, r_peaks_ind, ECG_SEGMENT_LENGTH)

    # data visualization
    plt.figure(1)
    plt.plot(filt_data)
    plt.figure(2)
    for i in xrange(len(ecg_segments)):
        plt.plot(ecg_segments[i, :])

    # data visualization
    plt.figure(3)
    plt.plot(data)

    plt.show()


if __name__ == '__main__':
    main()
