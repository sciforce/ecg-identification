import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn.python.learn as skflow


def load_ecg_featuers(file_path):
    ecg_features = np.load(file_path)
    labels = sorted(ecg_features.files)
    num_class = len(labels)
    ecg_segment_length = ecg_features[labels[0]].shape[1]
    ecg_targets = np.zeros(0)
    ecg_inputs = np.zeros([0, ecg_segment_length])
    for i in xrange(num_class):
        participant_name = labels[i]
        class_inputs = ecg_features[participant_name]
        num_records = class_inputs.shape[0]
        class_targets = np.zeros([num_records])
        class_targets[:] = i  # set true index for current class
        ecg_inputs = np.vstack([ecg_inputs, class_inputs])
        ecg_targets = np.append(ecg_targets, class_targets)

    return ecg_targets.astype(np.int64), ecg_inputs, labels


def randomize_features(targets, inputs):
    num_records = targets.shape[0]
    rand_ind = np.random.permutation(num_records)
    targets_rand = targets[rand_ind]
    inputs_rand = inputs[rand_ind]
    return targets_rand, inputs_rand


def split_features(targets, inputs, gain=0.7):
    class_items = np.unique(targets)
    train_targets = np.empty([0, 1], dtype=int)
    test_targets = np.empty([0, 1], dtype=int)
    train_inputs = np.zeros([0, inputs.shape[1]])
    test_inputs = np.zeros([0, inputs.shape[1]])
    for item in class_items:
        item_indexes = np.where(targets == item)[0]
        indexes_num = len(item_indexes)
        split_index = int(np.floor(indexes_num * gain))
        train_indexes = item_indexes[0:split_index]
        test_indexes = item_indexes[split_index:]
        train_targets = np.append(train_targets, targets[train_indexes])
        test_targets = np.append(test_targets, targets[test_indexes])
        train_inputs = np.vstack((train_inputs, inputs[train_indexes]))
        test_inputs = np.vstack((test_inputs, inputs[test_indexes]))

    return train_targets, train_inputs, test_targets, test_inputs


def split_features_random(targets, inputs, gain=0.7):
    class_items = np.unique(targets)
    train_targets = np.empty([0, 1], dtype=int)
    test_targets = np.empty([0, 1], dtype=int)
    train_inputs = np.zeros([0, inputs.shape[1]])
    test_inputs = np.zeros([0, inputs.shape[1]])
    for item in class_items:
        item_indexes = np.where(targets == item)[0]
        indexes_num = len(item_indexes)
        rand_item_ind = np.random.permutation(item_indexes)
        split_index = int(np.floor(indexes_num * gain))
        train_indexes = rand_item_ind[0:split_index]
        test_indexes = rand_item_ind[split_index:]
        train_targets = np.append(train_targets, targets[train_indexes])
        test_targets = np.append(test_targets, targets[test_indexes])
        train_inputs = np.vstack((train_inputs, inputs[train_indexes]))
        test_inputs = np.vstack((test_inputs, inputs[test_indexes]))

    return train_targets, train_inputs, test_targets, test_inputs


def save_classifer(classifier, lables):
    hidden_units = str(classifier._hidden_units)[1:-1]  # list to string ignoring brackets
    inputs_dimension = str(classifier._feature_columns[0].dimension)
    model_dir = str(classifier._model_dir)
    n_classes = str(classifier._n_classes)

    os.makedirs(model_dir)
    labels_file = os.path.join(model_dir, 'labels_list.txt')
    f = open(labels_file, 'w')
    for label in lables:
        f.write(label + "\n")
    f.close()

    classifier_config_file = os.path.join(model_dir, 'classifier_config.txt')
    f = open(classifier_config_file, 'w')
    f.write(hidden_units + "\n")
    f.write(inputs_dimension + "\n")
    f.write(model_dir + "\n")
    f.write(n_classes + "\n")
    f.close()


def load_classifer(model_path):
    labels_file = os.path.join(model_path, 'labels_list.txt')
    f = open(labels_file, 'r')
    labels = f.read().splitlines()
    f.close()

    classifier_config_file = os.path.join(model_path, 'classifier_config.txt')
    f = open(classifier_config_file, 'r')
    classifier_config_data = f.read().splitlines()
    f.close()

    hidden_units = map(int, classifier_config_data[0].split(","))  # split string and convert to numeric list
    inputs_dimension = int(classifier_config_data[1])
    model_dir = classifier_config_data[2]
    n_classes = int(classifier_config_data[3])

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=inputs_dimension)]
    classifier = skflow.DNNClassifier(hidden_units=hidden_units, n_classes=n_classes, feature_columns=feature_columns,
                                      model_dir=model_dir)

    return classifier, labels