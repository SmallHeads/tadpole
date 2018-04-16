from keras.models import load_model
from models import mlp
import ipynb.fs

from keras.utils import to_categorical

from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint

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


def test_d2(results_dir):
    model = load_model(results_dir + 'best_tadpole_model0.hdf5')

    feature_list, output_list, rids = compute_data_table(for_predict=True)

    (x, month), (dx, adas, ventricle), (ids) = parse_data(feature_list, output_list, rids)

    predictions = model.predict([x, month])

    # for i, prediction in enumerate(predictions):
    #     print('prediction', i, prediction)

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


if __name__ == "__main__":
    # test_future('/home/users/adoyle/tadpole/data/experiment-5/')

    print('It\'s not the size that counts, it\'s the connections')

    print(in_df.columns.values)

    all_rids = in_df['RIDS'].unique()

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

        training_table = create_test_table(in_df, ref_df, train_rids, mode='train')

        print('Column names')
        print(training_table.columns.values.tolist())

        print(training_table.shape)

        # x_t_train = training_table.iloc[1:-4]
        # y_t_train = training_table.iloc[]
        #
        # x_t_train, y_t_train, delta_t_train, y_t_next_train = training_sample(train_rids)
        # x_t_test, y_t_test, delta_t_test, y_t_next_test = training_sample(train_rids)
        #
        # # prediction targets
        # dx_train, dx_test = y_t_train[:, 0], y_t_test[:, 0]
        # adas_train, adas_test = y_t_train[:, 1], y_t_test[:, 1]
        # ventricle_train, ventricle_test = y_t_train[:, 2], y_t_test[:, 2]
        #
        #
        # model = mlp(feature_inputs)
        # model.summary()
        #
        # model_checkpoint = ModelCheckpoint(results_dir + "best_weights_fold_" + str(k) + ".hdf5",
        #                                    monitor="val_future_diagnosis_acc",
        #                                    save_best_only=True)
        #
        # model.compile(optimizer='adam',
        #               loss={'future_diagnosis': 'categorical_crossentropy',
        #                     'ventricle_volume': 'mean_squared_error',
        #                     'as_cog': 'mean_squared_error'},
        #               loss_weights={'future_diagnosis': 0.00001, 'ventricle_volume': 1, 'as_cog': 0.0001},
        #               metrics={'future_diagnosis': 'accuracy',
        #                        'ventricle_volume': 'mean_squared_error',
        #                        'as_cog': 'mean_squared_error'}
        #               )
        #
        # print(model.metrics_names)
        # print(model.metrics)
        #
        # hist = model.fit([x_t_train, y_t_train, delta_t_train], [dx_train, adas_train, ventricle_train], epochs=50, validation_data=([x_t_train, y_t_train, delta_t_train], [dx_test, adas_test, ventricle_test]), callbacks=[model_checkpoint])
        #
        # model.load_weights(results_dir + "best_weights_fold_" + str(k) + ".hdf5")
        # model.save(results_dir + 'best_tadpole_model' + str(k) + '.hdf5')
        #
        # plot_graphs(hist, results_dir, k)
        #
        # test_d2(results_dir)
        # test_future(results_dir)
