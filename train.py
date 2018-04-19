from keras.models import load_model
from models import mlp
import ipynb.fs

from keras.utils import to_categorical

from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import csv, os

import pickle as pkl

import numpy as np
import pandas as pd

from ipynb.fs.defs.Make_test_table import create_test_table

workdir = '/home/users/adoyle/tadpole/data/'

in_df = pd.read_csv(workdir + 'KNN_50_cleaned.csv', index_col=0)
ref_df = pd.read_csv(workdir + 'KNN_50_cleaned_ref.csv', index_col=0)

def plot_graphs(hist, results_dir, fold_num):
    epoch_num = range(len(hist.history['future_diagnosis_acc']))

    plt.clf()
    plt.plot(epoch_num, hist.history['ventricle_volume_mean_squared_error'], label='Ventricle Vol. MSE')
    plt.plot(epoch_num, hist.history['val_ventricle_volume_mean_squared_error'], label='Validation Ventricle Vol. MSE')
    plt.plot(epoch_num, hist.history['future_diagnosis_acc'], label='Future Diagnosis Accuracy')
    plt.plot(epoch_num, hist.history['val_future_diagnosis_acc'], label='Validation Future Diagnosis Accuracy')

    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Metric Value")
    plt.savefig(results_dir + 'training_metrics_fold' + str(fold_num) + '.png', bbox_inches='tight')
    plt.close()


def test_d2(model, all_rids, results_dir):
    (x_t, y_t, delta_t), (dx_next, adas_next, ventricle_next) = load_data_samples(all_rids, mode='train')

    predictions = model.predict([x_t, y_t, delta_t])

    dx_probabilities = predictions[0:3, :]
    adas_predictions = predictions[3, :]
    ventricle_predictions = predictions[4, :]

    dx_predictions = np.argmax(dx_probabilities, axis=-1)

    confusion = confusion_matrix(np.asarray(dx_predictions, dtype='uint8'), np.asarray(dx_next, dtype='uint8'))
    print('Confusion matrix for D2 diagnosis predictions:')
    print(confusion)

    # un-normalize regression predictions

    with open(results_dir + 'd2_predictions.csv', 'w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, lineterminator='\n')

        prediction_writer.writerow(['ID', 'Months', 'P(Control)', 'P(MCI)', 'P(ALZ)', 'ADAS-13', 'Ventricular Volume'])

        for i, (adni_id, dx, adas, ventricle, m) in enumerate(zip(ids, predictions[0], predictions[1], predictions[2], month)):
            prediction_writer.writerow([adni_id, m, dx[0], dx[1], dx[2], adas[0], ventricle[0]])

def test_future(rids, results_dir):
    model = load_model(results_dir + 'best_tadpole_model0.hdf5')

    x_t, y_t, delta_t = create_data_table(rids) # delta t here is the time from the last available timepoint for that subject

    with open(results_dir + 'future_predictions.csv', 'w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, lineterminator='\n')

        prediction_writer.writerow(['ID', 'Months', 'P(Control)', 'P(MCI)', 'P(ALZ)', 'ADAS-13', 'Ventricular Volume'])

        n_months = 60
        for rid in rids:
            x = x_t[x_t[0] == rid]
            y = y_t[x_t[0] == rid]
            delta_t = delta_t[x_t[0] == rid]

            n_timepoints = x.shape[0]
            n_outputs = y.shape[1] + 2

            for t, months_forward in enumerate(range(1, n_months)):
                predictions = np.zeros((n_timepoints, n_months, n_outputs), dtype='float32')

                time_forward = delta_t + months_forward
                dx, adas, vent = model.predict([x, y, time_forward])

                predictions[:, t, 0:3] = dx
                predictions[:, t, 3] = adas
                predictions[:, t, 4] = vent

                prediction_writer.writerow([rid, months_forward, np.mean(predictions[:, t, 0]), np.mean(predictions[:, t, 1]), np.mean(predictions[:, t, 2]), np.mean(predictions[:, t, 3]), np.mean(predictions[:, t, 4])])

    return

def load_data_samples(rids, mode='train'):
    table = create_test_table(in_df, ref_df, rids, mode=mode).dropna()

    # headers = table.columns.values
    #
    # print('Features:', headers[:-7])
    # print('Outputs:', headers[-7:-4])
    # print('Prediction targets:', headers[-3:])
    # print('Timestep:', headers[-4])

    rids = table.iloc[:, 0]
    x_t = table.iloc[:, 1:-7]
    y_t = table.iloc[:, -7:-4]
    y_t_next = table.iloc[:, -3:]
    delta_t = table.iloc[:, -4]

    # current diagnosis is an input feature
    dx = to_categorical(y_t.iloc[:, 0] - 1, num_classes=3)

    # prediction targets
    dx_next = to_categorical(y_t_next.iloc[:, 0] - 1, num_classes=3)
    adas_next = y_t_next.iloc[:, 1]
    ventricle_next = y_t_next.iloc[:, 2]

    y_t_categorical = np.hstack((dx, y_t.iloc[:, 1:]))

    # determine what timepoints have a change in diagnosis
    dx_change = np.zeros(dx.shape[0])
    dx_change[dx != dx_next] = 1

    return (rids), (x_t, y_t_categorical, delta_t), (dx_next, adas_next, ventricle_next), (dx_change)

if __name__ == "__main__":
    print('It\'s not the size that counts, it\'s the connections')

    all_rids = ref_df['RID'].unique()

    print('There are', len(all_rids), 'subjects total in the ADNI dataset')

    try:
        experiment_number = pkl.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pkl.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    kf = KFold(n_splits=5, shuffle=True)

    for k, (train_rids, test_rids) in enumerate(kf.split(all_rids)):

        testing_table = create_test_table(in_df, ref_df, test_rids, mode='train').dropna()

        (rids_train),\
        (x_t_train, y_t_train, delta_t_train),\
        (dx_next_train, adas_next_train, ventricle_next_train),\
        (dx_change_train) = load_data_samples(train_rids, mode='train')

        (rids_test),\
        (x_t_test, y_t_test, delta_t_test),\
        (dx_next_test, adas_next_test, ventricle_next_test),\
        (dx_change_test) = load_data_samples(test_rids, mode='train')

        n_healthy = np.sum(dx_next_train[:, 0])
        n_mci = np.sum(dx_next_train[:, 1])
        n_alzheimers = np.sum(dx_next_train[:, 2])
        n_total = dx_next_train.shape[0]

        print('Distribution of prediction targets this fold (training):')
        print(n_healthy, 'healthy')
        print(n_mci, 'mild cognitive impairment')
        print(n_alzheimers, 'Alzheimer\'s')
        print(n_total, 'total')

        healthy_weight = (n_total - n_healthy) / n_total
        mci_weight = (n_total - n_mci) / n_total
        alzheimers_weight = (n_total - n_alzheimers) / n_total

        print('Class weight (healthy, MCI, Alzheimer\'s):', healthy_weight, mci_weight, alzheimers_weight)

        diagnosis_changes =  np.sum(dx_change_train)
        print('Proportion of samples where diagnosis changes between timepoints:', diagnosis_changes / n_total)

        # initialize model and train it
        model = mlp(x_t_train.shape[-1])
        model.summary()

        model_checkpoint = ModelCheckpoint(results_dir + "best_weights_fold_" + str(k) + ".hdf5",
                                           monitor="val_future_diagnosis_acc",
                                           save_best_only=True)

        model.compile(optimizer='adam',
                      loss={'future_diagnosis': 'categorical_crossentropy',
                            'ventricle_volume': 'mean_squared_error',
                            'as_cog': 'mean_squared_error'},
                      loss_weights={'future_diagnosis': 0.00001, 'ventricle_volume': 1, 'as_cog': 0.0001},
                      metrics={'future_diagnosis': 'accuracy',
                               'ventricle_volume': 'mean_squared_error',
                               'as_cog': 'mean_squared_error'}
                      )

        print(model.metrics_names)
        print(model.metrics)

        hist = model.fit([x_t_train, y_t_train, delta_t_train], [dx_next_train, adas_next_train, ventricle_next_train], epochs=50, validation_data=([x_t_test, y_t_test, delta_t_test], [dx_next_test, adas_next_test, ventricle_next_test]), callbacks=[model_checkpoint])

        model.load_weights(results_dir + "best_weights_fold_" + str(k) + ".hdf5")
        model.save(results_dir + 'best_tadpole_model' + str(k) + '.hdf5')

        plot_graphs(hist, results_dir, k)

    model = load_model(results_dir + 'best_tadpole_model0.hdf5')

    test_d2(model, all_rids, results_dir)
    test_future(model, all_rids, results_dir)
