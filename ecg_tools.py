__author__ = 'ykhoma'
import serial
import struct
import numpy as np
from scipy import signal
from biosppy.signals.ecg import hamilton_segmenter


def preprocessing(data):
    # create bandpass filter
    fs = 277.0  # sampling rate (in Hz)
    freq_pass = np.array([4.0, 35.0]) / (fs / 2.0)
    freq_stop = np.array([1.0, 50.0]) / (fs / 2.0)
    gain_pass = 1
    gain_stop = 20
    filt_order, cut_freq = signal.buttord(freq_pass, freq_stop, gain_pass, gain_stop)
    b, a = signal.butter(filt_order, cut_freq, 'bandpass')

    # input signal filtering
    filt_data = signal.filtfilt(b, a, data)

    # input signal normalization
    norm_gain = (max(filt_data) - min(filt_data)) / 2
    norm_data = filt_data / norm_gain

    return norm_data


def r_peak_detection(data, type='hamilton'):
    if type == 'hamilton':
        r_peaks_tuple = hamilton_segmenter(signal=data, sampling_rate=277)
        r_peaks_ind = r_peaks_tuple['rpeaks']
        r_peak_neighbourhood = 7;  # expected range of R peak (in samples)
        for i in xrange(len(r_peaks_ind)):
            start = np.maximum(r_peaks_ind[i] - r_peak_neighbourhood, 1);
            stop = np.minimum(r_peaks_ind[i] + r_peak_neighbourhood, len(data))
            ind = np.argmax(data[start: stop])
            r_peaks_ind[i] = start + ind;
        r_peaks_val = data[r_peaks_ind]
    elif type == 'threshold':
        # R-peak detection
        r_threshold = 0.4
        temp_data = data * 1.0  # temporary data copy
        temp_data[temp_data < r_threshold] = 0  # setting samples below threshold to zero

        r_segm_start = np.array([])
        r_segm_stop = np.array([])
        for i in xrange(len(temp_data) - 1):
            if temp_data[i] == 0 and temp_data[i + 1] != 0:
                r_segm_start = np.append(r_segm_start, i)  # start index for each segment above threshold
            if temp_data[i] != 0 and temp_data[i + 1] == 0:
                r_segm_stop = np.append(r_segm_stop, i)  # end index for each segment above threshold

        nPeaks = len(r_segm_start)
        nSamples = len(data)

        r_peaks_val = np.zeros(nPeaks)
        r_peaks_ind = np.zeros(nPeaks)

        # local maximums search
        for i in xrange(nPeaks):
            ind_start = int(r_segm_start[i])
            ind_stop = int(r_segm_stop[i])
            segm_mask = np.zeros(nSamples)
            segm_mask[ind_start:ind_stop] = 1
            temp_data = data * segm_mask
            val = temp_data.max()
            ind = int(temp_data.argmax())
            r_peaks_val[i] = val  # r-peak value
            r_peaks_ind[i] = int(ind)  # r-peak index

    return r_peaks_val, r_peaks_ind


def segmentation(data, r_peaks_ind, segm_lem=270):
    n_peak = len(r_peaks_ind)
    ecg_segm = np.zeros([n_peak, segm_lem])
    r_peak_dist = np.diff(r_peaks_ind)
    iSegm = 0  # segment counter
    for i in xrange(n_peak - 1):
        start = int(r_peaks_ind[i] - 80)
        stop = int(start + np.minimum(r_peak_dist[i], segm_lem))
        if start < 0:
            ecg_segm = np.delete(ecg_segm, 0, 0)
            continue
        ecg_segm[iSegm, 0:stop - start] = data[start:stop]
        iSegm += 1  # increment segment counter
    return ecg_segm


def statistical_features_calculation(ecg_segm, order):
    if order == "ALL":
        order = ecg_segm.shape[0]
    n_segm, len_segm = ecg_segm.shape
    ecg_features = np.zeros([n_segm - order + 1, len_segm])
    mean_segm = ecg_segm.mean(axis=0)
    norm_gain = (max(mean_segm) - min(mean_segm)) / 2
    ecg_segm /= norm_gain
    for row_ind in xrange(n_segm - order + 1):
        start_ind = row_ind
        end_ind = start_ind + order
        ecg_features[row_ind, :] = ecg_segm[start_ind:end_ind, :].mean(axis=0)
    return ecg_features
