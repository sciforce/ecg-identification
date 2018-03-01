import os
import numpy as np
import random
from matplotlib import pyplot as plt
import uuid
from config import SEGMENTATOR_DIR, COMBINATOR_DIR

EXTENSION = '.npy'
P_DIR = 'P'
QRS_DIR = 'QRS'
T_DIR = 'T'
VISUALIZE = False


# smooth edge effects for QRS and P-wave segments
def correct_p_part(p_part, qrs_part):
    p_qrs_diff = p_part[-1] - qrs_part[0]
    return [x - p_qrs_diff for x in p_part]


# smooth edge effects for QRS and T-wave segments
def correct_t_part(t_part, qrs_part):
    t_qrs_diff = t_part[0] - qrs_part[-1]
    return [x - t_qrs_diff for x in t_part]


def save_to_file(data):
    if not os.path.exists(COMBINATOR_DIR):
        os.makedirs(COMBINATOR_DIR)
    # generating random and unique filename
    filename = str(uuid.uuid4().hex) + EXTENSION
    file_path = os.path.join(COMBINATOR_DIR, filename)
    np.save(file_path, data)


def main():
    p_base_dir = os.path.join(SEGMENTATOR_DIR, P_DIR)
    p_file_list = os.listdir(p_base_dir)

    qrs_base_dir = os.path.join(SEGMENTATOR_DIR, QRS_DIR)
    qrs_file_list = os.listdir(qrs_base_dir)

    t_base_dir = os.path.join(SEGMENTATOR_DIR, T_DIR)
    t_file_list = os.listdir(t_base_dir)

    if not p_file_list or not qrs_file_list or not t_file_list:
        print 'No data of P, QRS or T parts are found in the file path. ' \
              'Please generate the input data first using Segmentator.'
    else:
        for i in xrange(len(p_file_list)):
            p_file_name = random.choice(p_file_list)
            qrs_file_name = random.choice(qrs_file_list)
            t_file_name = random.choice(t_file_list)
            print 'Creating new beat from: ' + p_file_name + ', ' + qrs_file_name + ', ' + t_file_name

            file_names = [p_file_name.rsplit('_')[0], qrs_file_name.rsplit('_')[0], t_file_name.rsplit('_')[0]]

            # if not all segments belong to the same user
            if not all(el == file_names[0] for el in file_names):
                p_file_path = os.path.join(SEGMENTATOR_DIR, P_DIR, p_file_name)
                p_part = np.load(p_file_path)
                qrs_file_path = os.path.join(SEGMENTATOR_DIR, QRS_DIR, qrs_file_name)
                qrs_part = np.load(qrs_file_path)
                t_file_path = os.path.join(SEGMENTATOR_DIR, T_DIR, t_file_name)
                t_part = np.load(t_file_path)

                p_part[:] = correct_p_part(p_part, qrs_part)
                t_part[:] = correct_t_part(t_part, qrs_part)

                rand_beat = np.concatenate((p_part, qrs_part, t_part))
                # save generated beats
                save_to_file(rand_beat)

                if VISUALIZE:
                    plt.figure(i)
                    plt.plot(rand_beat)
                    plt.show()
            else:
                print 'File names matched.'


if __name__ == '__main__':
    main()