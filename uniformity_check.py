
import numpy as np

def easy_uniformity_check(beats):
    ecg_mean = beats.sum(axis=0)/beats.shape[0]
    mse = ((beats - ecg_mean) ** 2).mean(axis=1)
    mse_thr = 0.0025
    if np.max(mse) > mse_thr:
        beats = np.delete(beats, np.argmax(mse), 0)
        beats = easy_uniformity_check(beats)
    return beats


def windows_uniformity_check(beats):
    thr = 0.003  # empirically selected
    windows_size = 3
    (n_beats, len_beats) = beats.shape

    # expansion beats signal in order to fit the whole windows number
    if (len_beats % windows_size) != 0:
        beats = np.hstack((beats, np.zeros((n_beats, windows_size - (len_beats % windows_size)))))

    # w - arrays with windows coordinates in the beats
    w = np.arange(0, len_beats, windows_size)
    w = np.vstack((w, w + windows_size))

    # calculating the average std's threshold
    beats_mean = np.mean(beats, axis=0)
    i_thr = 0
    for i in xrange(n_beats):
        i_thr = i_thr + np.sum((beats[i, :]-beats_mean) ** 2)
    i_thr = i_thr / (len_beats * n_beats)

    # if the error is exceeds the threshold within the window,
    # then replace the relevant part on the averaged part
    for k in xrange(n_beats):
        e = ((beats[k, :]-beats_mean) ** 2 / len_beats) <= thr * i_thr
        for p in xrange(w.shape[1]):
            stw = w[0, p]
            enw = w[1, p]
            if np.all(e[stw: enw]):
                continue
            else:
                beats[k, stw:enw] = beats_mean[stw:enw]
    return beats
