import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model

from keras.utils import to_categorical

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import os
import pickle as pkl

import numpy as np

from data_formatting import compute_data_table

workdir = 'E:/tadpole/'

def mlp(input_variables):
    diagnostic_states = 3

    features = Input(shape=(input_variables,))
    months = Input(shape=(1,))

    h1 = Dense(256)(features)
    l1 = LeakyReLU()(h1)
    d1 = Dropout(0.5)(l1)
    n1 = BatchNormalization()(d1)
    h1m = concatenate([n1, months])
    h2 = Dense(256)(h1m)
    l2 = LeakyReLU()(h2)
    d2 = Dropout(0.5)(l2)
    n2 = BatchNormalization()(d2)
    h2m = concatenate([n2, months])
    h3 = Dense(256)(h2m)
    l3 = LeakyReLU()(h3)
    d3 = Dropout(0.5)(l3)
    n3 = BatchNormalization()(d3)
    h3m = concatenate([n3, months])

    future_diagnosis = Dense(diagnostic_states, activation='softmax', name='future_diagnosis')(h3m)

    ventricle_volume = Dense(1, activation='linear', name='ventricle_volume')(h3m)
    as_cog = Dense(1, activation='linear', name='as_cog')(h3m)

    model = Model(inputs=[features, months], outputs=[future_diagnosis, ventricle_volume, as_cog])

    return model


def autoencoder(input_dims, latent_space_dims):

    input = Input(shape=input_dims,)

    denoise = Dropout(0.2)(input)
    encoded = Dense(latent_space_dims, activation='relu', name='encoding')(denoise)
    decoded = Dense(input_dims, activation='linear')(encoded)

    model = Model(input=[input], output=[decoded])

    return model


def train_stacked_autoencoder(x_train, y_train, x_test, y_test):
    diagnostic_states = 3

    layer1 = autoencoder(25, 20)
    layer1.compile(optimizer='adam', loss='mean_squared_error')
    layer1.fit(x_train, x_train, epochs=10)

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


def parse_data(feature_list, output_list):
    # inputs
    x_ = []
    month = []

    # outputs
    dx = []
    adas = []
    ventricle = []

    for i, (x, y) in enumerate(zip(feature_list, output_list)):
        for y_timepoint in y:
            x_.append(x)

            month.append(y_timepoint[0])

            dx.append(y_timepoint[1])
            adas.append(y_timepoint[2])
            ventricle.append(y_timepoint[3])

    return (np.asarray(x_, dtype='float32'), np.asarray(month, dtype='float32')), (to_categorical(np.asarray((dx), dtype='float32')), np.asarray(adas, dtype='float32'), np.asarray(ventricle, dtype='float32'))


if __name__ == "__main__":
    print('It\'s not the size that counts, it\'s the connections')

    feature_list, output_list = compute_data_table(range(1000))
    (x, month), (dx, adas, ventricle) = parse_data(feature_list, output_list)

    print('x shape:', x.shape, month.shape)
    print('y shape:', dx.shape, adas.shape, ventricle.shape)

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

    n_samples = x.shape[0]
    feature_inputs = x.shape[1]



    # skf = StratifiedKFold(n_splits=5)

    ss = ShuffleSplit(n_splits=10)

    for k, (train_indices, test_indices) in enumerate(ss.split(range(n_samples))):
        model = mlp(feature_inputs)
        model.summary()

        model_checkpoint = ModelCheckpoint(results_dir + 'best_qc_model.hdf5',
                                           monitor="val_acc",
                                           save_best_only=True)

        model.compile(optimizer='adam',
                      loss={'future_diagnosis': 'categorical_crossentropy',
                            'ventricle_volume': 'mean_squared_error',
                            'as_cog': 'mean_squared_error'},
                      loss_weights={'future_diagnosis': 0.5, 'ventricle_volume': 0.25, 'as_cog': 0.25},
                      callbacks=[model_checkpoint]
                      )



        #inputs
        x_train, x_test = x[train_indices], x[test_indices]
        month_train, month_test = month[train_indices], month[test_indices]

        #outputs
        dx_train, dx_test = dx[train_indices], dx[test_indices]
        adas_train, adas_test = adas[train_indices], adas[test_indices]
        ventricle_train, ventricle_test = ventricle[train_indices], ventricle[test_indices]

        hist = model.fit([x_train, month_train], [dx_train, adas_train, ventricle_train], len(), validation_data=[x_test, month_test], [dx_test, adas_test, ventricle_test])

