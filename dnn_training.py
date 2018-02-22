import os
import tensorflow as tf
import tensorflow.contrib.learn.python.learn as skflow
import time
import skflow_tools as sktools
import ecg_feature_preparation as fp
from sklearn import metrics
from config import BASIC_DIR, ECG_eHEALTH_TRAIN_DATA_DIR, ECG_eHEALTH_TEST_DATA_DIR, ECG_ID_DATA_DIR, ECG_eHEALTH_DATA_DIR
from sklearn.model_selection import train_test_split

PARTICIPANT_LIST = ['user1', 'user2', 'user3']

ecgTrainDataDir = ECG_eHEALTH_TRAIN_DATA_DIR
ecgTestDataDir = ECG_eHEALTH_TEST_DATA_DIR
ecgID_DataDir = ECG_ID_DATA_DIR
eHealth_DataDir = ECG_eHEALTH_DATA_DIR

trainFeaturesPath = os.path.join(BASIC_DIR, 'train_ecg_features.npz')
testFeaturesPath = os.path.join(BASIC_DIR, 'test_ecg_features.npz')
eHealth_FeaturesPath = os.path.join(BASIC_DIR, 'eHealth_features.npz')
ecgID_FeaturesPath = os.path.join(BASIC_DIR, 'ecg_ID_features.npz')

MODELS_DIR = os.path.join(BASIC_DIR, 'dnn_models')
model_name = 'dnn' + time.strftime("_%m_%d_%H_%M_%S")

DISPLAY_TRAINING_DETAILS = True
DISPLAY_PREDICTIONS = True
FEATURES_RECALCULATION = False
USE_ECG_ID = False
USE_eHEALTH_ECG = True
RANDOM_SPLIT = True


def main():
    if USE_ECG_ID:
        if FEATURES_RECALCULATION:
            fp.feature_preparation(ecgID_DataDir, ecgID_FeaturesPath)
        # prepare train and test sets
        targets, inputs, labels = sktools.load_ecg_featuers(ecgID_FeaturesPath)
        train_targets, train_inputs, test_targets, test_inputs = sktools.split_features(targets, inputs)
    elif USE_eHEALTH_ECG:
        if RANDOM_SPLIT:
            if FEATURES_RECALCULATION:
                fp.feature_preparation(eHealth_DataDir, eHealth_FeaturesPath)
            # prepare train and test sets
            targets, inputs, labels = sktools.load_ecg_featuers(eHealth_FeaturesPath)
            train_targets, train_inputs, test_targets, test_inputs = sktools.split_features_random(targets, inputs)
        else:
            if FEATURES_RECALCULATION:
                fp.feature_preparation(ecgTrainDataDir, trainFeaturesPath, PARTICIPANT_LIST)
                fp.feature_preparation(ecgTestDataDir, testFeaturesPath, PARTICIPANT_LIST)
            # prepare train and test sets
            train_targets, train_inputs, labels = sktools.load_ecg_featuers(trainFeaturesPath)
            test_targets, test_inputs, _ = sktools.load_ecg_featuers(testFeaturesPath)

    train_targets, train_inputs = sktools.randomize_features(train_targets, train_inputs)

    # defining model path
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, model_name)

    # create DNN model
    tf.logging.set_verbosity(tf.logging.ERROR)  # suspend Tensorflow Warnings
    num_classes = len(labels)
    inputs_dimension = train_inputs.shape[1]
    layers = [70, 50, 30]
    num_epochs = 3000
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=inputs_dimension)]
    classifier = skflow.DNNClassifier(hidden_units=layers, n_classes=num_classes, feature_columns=feature_columns,
                                      model_dir=model_path)
    sktools.save_classifer(classifier, labels)
    # fit and save model
    print "\nDisplaying DNN training flow:"
    print "training started"
    classifier.fit(train_inputs, train_targets, steps=num_epochs, batch_size=1)
    print "training finished"

    # accuracy evaluation
    train_outputs = classifier.predict(train_inputs)
    test_outputs = classifier.predict(test_inputs)
    train_score = metrics.accuracy_score(train_targets, train_outputs)
    test_score = metrics.accuracy_score(test_targets, test_outputs)

    if DISPLAY_TRAINING_DETAILS:
        print ("\nDisplaying training details")
        print('Accuracy on train set: {0:f}'.format(train_score))
        print('Accuracy on test set: {0:f}'.format(test_score))

    # displaying predictions (for debug and details estimation)
    test_outputs = classifier.predict(test_inputs)
    if DISPLAY_PREDICTIONS:
        print "\nDisplaying DNN prediction"
        print "predicted:  expected:"
        for i in xrange(len(test_outputs)):
            print labels[test_outputs[i].astype(int)], labels[test_targets[i].astype(int)]


if __name__ == '__main__':
    main()
