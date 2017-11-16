import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model, load_model

from keras.utils import to_categorical

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from keras.callbacks import ModelCheckpoint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import csv

import os
import pickle as pkl

import numpy as np

from data_formatting import compute_data_table

workdir = '/home/users/adoyle/tadpole/data/'

def mlp(input_variables):
    diagnostic_states = 3

    features = Input(shape=(input_variables,))
    months = Input(shape=(1,))

    h1 = Dense(512)(features)
    l1 = LeakyReLU()(h1)
    d1 = Dropout(0.2)(l1)
    n1 = BatchNormalization()(d1)
    h1m = concatenate([n1, months])

    h2 = Dense(1024)(h1m)
    l2 = LeakyReLU()(h2)
    d2 = Dropout(0.3)(l2)
    n2 = BatchNormalization()(d2)
    h2m = concatenate([n2, months])

    h3 = Dense(1024)(h2m)
    l3 = LeakyReLU()(h3)
    d3 = Dropout(0.4)(l3)
    n3 = BatchNormalization()(d3)
    h3m = concatenate([n3, months])

    h4 = Dense(512)(h3m)
    l4 = LeakyReLU()(h4)
    d4 = Dropout(0.5)(l4)
    n4 = BatchNormalization()(d4)
    h4m = concatenate([n4, months])

    h5 = Dense(512)(h4m)
    l5 = LeakyReLU()(h5)
    d5 = Dropout(0.5)(l5)
    n5 = BatchNormalization()(d5)
    h5m = concatenate([n5, months])

    future_diagnosis = Dense(diagnostic_states, activation='softmax', name='future_diagnosis')(h5m)

    ventricle_volume = Dense(1, activation='linear', name='ventricle_volume')(h5m)
    as_cog = Dense(1, activation='linear', name='as_cog')(h5m)

    model = Model(inputs=[features, months], outputs=[future_diagnosis, ventricle_volume, as_cog])

    return model

def encoder(input_variables, latent_space_dims):
    features = Input(shape=(input_variables,))
    months = Input(shape=(1,))

    denoise = Dropout(0.3)(features)
    conditioned_denoised = concatenate([features, months])(denoise)
    encoded = Dense(latent_space_dims, activation='relu', name='encoded')(conditioned_denoised)

    model = Model(inputs=[features, months], outputs=[encoded])

    return model

def decoder(input_variables, latent_space_dims):
    encoded = Input(shape=(latent_space_dims,))
    months = Input(shape=(1,))

    decoded = Dense(input_variables, activation='relu', name='decoded')(encoded)

    model = Model(inputs=[encoded], outputs=[decoded])

    return model

def autoencoder(input_variables, latent_space_dims):

    features = Input(shape=(input_variables,))
    months = Input(shape=(1,))

    enc = encoder(input_variables, latent_space_dims)
    dec = decoder(input_variables, latent_space_dims)



    return model


def train_stacked_autoencoder(x_train, y_train, x_test, y_test):
    diagnostic_states = 3



    layer1 = autoencoder(25, 20)
    layer1.compile(optimizer='adam', loss='mean_squared_error')
    layer1.fit(x_train, x_train, epochs=10)
    # layer1.fit([x_train, month_train], [dx_train, adas_train, ventricle_train], epochs=20,
    #                  validation_data=([x_test, month_test], [dx_test, adas_test, ventricle_test]),
    #                  callbacks=[model_checkpoint])

    x_train_encoded = layer1.predict(x_train)

    layer2 = autoencoder(20, 10)
    layer2.compile(optimizer='adam', loss='mean_squared_error')
    layer2.fit(x_train_encoded, x_train_encoded, epochs=10)

    input = Input(shape=25,)
    months = Input(shape=1,)

    first_encoding = layer1.get_layer(name='encoding')(input)
    second_encoding = layer2.get_layer(name='encoding')(first_encoding)

    h1i = concatenate([second_encoding, months])

    h1 = Dense(64, activation='relu')(h1i)
    d1 = Dropout(0.5)(h1)
    h1o = concatenate([d1, months])

    future_diagnosis = Dense(diagnostic_states, activation='softmax', name='future_diagnosis')(h1o)
    ventricle_volume = Dense(1, activation='linear', name='ventricle_volume')(h1o)
    as_cog = Dense(1, activation='linear', name='as_cog')(h1o)

    model = Model(inputs=[input], outputs=[future_diagnosis, ventricle_volume, as_cog])

    return model


def parse_data(feature_list, output_list, ids):
    # inputs
    x_ = []
    month = []

    # outputs
    dx = []
    adas = []
    ventricle = []

    rids = []

    for i, (x, y, adni_id) in enumerate(zip(feature_list, output_list, ids)):
        for y_timepoint in y:
            x_.append(x)
            rids.append(adni_id)

            month.append(y_timepoint[0])

            dx.append(y_timepoint[1])
            adas.append(y_timepoint[2])
            ventricle.append(y_timepoint[3])

    return (np.asarray(x_, dtype='float32'), np.asarray(month, dtype='float32')), (to_categorical(np.asarray((dx), dtype='uint8')), np.asarray(adas, dtype='float32'), np.asarray(ventricle, dtype='float32')), (np.asarray(rids, dtype='uint8'))

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

        prediction_writer.writerow(['ID', 'Months', 'P(Control)', 'P(MCI)', 'P(ALS)', 'ADAS', 'Ventricular Volume'])

        for i, (adni_id, dx, adas, ventricle, m) in enumerate(zip(ids, predictions[0], predictions[1], predictions[2], month)):
            prediction_writer.writerow([adni_id, m, dx[0], dx[1], dx[2], adas[0], ventricle[0]])

def test_future(results_dir):
    model = load_model(results_dir + 'best_tadpole_model0.hdf5')

    feature_file = workdir + 'ADAS_prediction_matrix.csv'

    with open(results_dir + 'future_predictions.csv', 'w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, lineterminator='\n')

        prediction_writer.writerow(['ID', 'Months', 'P(Control)', 'P(MCI)', 'P(ALS)', 'ADAS', 'Ventricular Volume'])

        with open(feature_file, 'r') as features:
            feature_reader = csv.reader(features)
            next(feature_reader)

            for m, feature_line in enumerate(feature_reader):

                if len(feature_line[-3]) > 0:
                    feature_line[-3] = int(float(feature_line[-3]) - 1)
                else:
                    feature_line[-3] = int(0)

                month = (m+1)%60
                # all_features = feature_line[1:-1]
                rid = feature_line[:-5]
                del feature_line[:-5]

                features = np.asarray(feature_line[:-1], dtype='float32')

                predictions = model.predict([features[np.newaxis, ...], np.asarray(month, dtype='float32')])

                prediction_writer.writerow([rid, month, predictions[0][0], predictions[0][1], predictions[0][2], predictions[1][0], predictions[2][0]])


if __name__ == "__main__":
    test_future('/home/users/adoyle/tadpole/data/experiment-5/')
    #
    # print('It\'s not the size that counts, it\'s the connections')
    #
    # feature_list, output_list, rids = compute_data_table()
    #
    # (x, month), (dx, adas, ventricle), (ids) = parse_data(feature_list, output_list, rids)
    #
    # print('x shape:', x.shape, month.shape)
    # print('y shape:', dx.shape, adas.shape, ventricle.shape)
    #
    # try:
    #     experiment_number = pkl.load(open(workdir + 'experiment_number.pkl', 'rb'))
    #     experiment_number += 1
    # except:
    #     print('Couldnt find the file to load experiment number')
    #     experiment_number = 0
    #
    # print('This is experiment number:', experiment_number)
    #
    # results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    # os.makedirs(results_dir)
    #
    # pkl.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))
    #
    # n_samples = x.shape[0]
    # feature_inputs = x.shape[1]
    #
    # # skf = StratifiedKFold(n_splits=5)
    #
    # ss = ShuffleSplit(n_splits=1, test_size=0.1)
    #
    # for k, (train_indices, test_indices) in enumerate(ss.split(range(n_samples))):
    #     model = mlp(feature_inputs)
    #     model.summary()
    #
    #     model_checkpoint = ModelCheckpoint(results_dir + "best_weights_fold_" + str(k) + ".hdf5",
    #                                        monitor="val_future_diagnosis_acc",
    #                                        save_best_only=True)
    #
    #     model.compile(optimizer='adam',
    #                   loss={'future_diagnosis': 'categorical_crossentropy',
    #                         'ventricle_volume': 'mean_squared_error',
    #                         'as_cog': 'mean_squared_error'},
    #                   loss_weights={'future_diagnosis': 1.0, 'ventricle_volume': 0.00001, 'as_cog': 0.0001},
    #                   metrics={'future_diagnosis': 'accuracy',
    #                            'ventricle_volume': 'mean_squared_error',
    #                            'as_cog': 'mean_squared_error'}
    #                   )
    #
    #     print(model.metrics_names)
    #     print(model.metrics)
    #
    #     #inputs
    #     x_train, x_test = x[train_indices], x[test_indices]
    #     month_train, month_test = month[train_indices], month[test_indices]
    #
    #     #outputs
    #     dx_train, dx_test = dx[train_indices], dx[test_indices]
    #     adas_train, adas_test = adas[train_indices], adas[test_indices]
    #     ventricle_train, ventricle_test = ventricle[train_indices], ventricle[test_indices]
    #
    #     hist = model.fit([x_train, month_train], [dx_train, adas_train, ventricle_train], epochs=200, validation_data=([x_test, month_test], [dx_test, adas_test, ventricle_test]), callbacks=[model_checkpoint])
    #
    #     model.load_weights(results_dir + "best_weights_fold_" + str(k) + ".hdf5")
    #     model.save(results_dir + 'best_tadpole_model' + str(k) + '.hdf5')
    #
    #     plot_graphs(hist, results_dir, k)
    #
    #     test_d2(results_dir)
    #     test_future(results_dir)
