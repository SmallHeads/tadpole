import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model

from keras.utils import to_categorical

from sklearn.model_selection import ShuffleSplit

import numpy as np

from data_formatting import compute_data_table


workdir = 'E:/tadpole/'

def mlp():
    diagnostic_states = 3

    features = Input(shape=(25,))
    months = Input(shape=(1,))

    h1 = Dense(64, activation='relu')(features)
    d1 = Dropout(0.5)(h1)
    n1 = BatchNormalization()(d1)
    h1m = concatenate([n1, months])
    h2 = Dense(64, activation='relu')(h1m)
    d2 = Dropout(0.5)(h2)
    n2 = BatchNormalization()(d2)
    h2m = concatenate([n2, months])
    h3 = Dense(64, activation='relu')(h2m)
    d3 = Dropout(0.5)(h3)
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
    x_ = []
    y_ = []

    month = []

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

    # print('x:', feature_list.shape)
    # print('y:', output_list.shape)

    (x, month), (dx, adas, ventricle) = parse_data(feature_list, output_list)

    print('x shape:', x.shape, month.shape)
    print('y shape:', dx.shape, adas.shape, ventricle.shape)

    model = mlp()

    model.compile(optimizer='adam',
                  loss={'future_diagnosis': 'categorical_crossentropy',
                        'ventricle_volume': 'mean_squared_error',
                        'as_cog': 'mean_squared_error'},
                  loss_weights={'future_diagnosis': 0.5, 'ventricle_volume': 0.25, 'as_cog': 0.25}
                  )

    model.fit([x, month], [dx, adas, ventricle])