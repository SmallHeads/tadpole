import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model

from sklearn.model_selection import ShuffleSplit

import numpy as np


workdir = 'E:/tadpole/'

def mlp():
    diagnostic_states = 3

    features = Input(shape=(300,))
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

    layer1 = autoencoder(300, 200)
    layer1.compile(optimizer='adam', loss='mean_squared_error')
    layer1.fit(x_train, x_train, epochs=10)

    x_train_encoded = layer1.predict(x_train)

    layer2 = autoencoder(200, 125)
    layer2.compile(optimizer='adam', loss='mean_squared_error')
    layer2.fit(x_train_encoded, x_train_encoded, epochs=10)

    input = Input(shape=300,)
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

    x_train = []
    y_train = []

    for i, (x, y) in enumerate(zip(feature_list, output_list)):
        for y_timepoint in y:
            x_train.append(x)
            y_train.append(y_timepoint)

    return np.asarray(x_train, dtype='float32'), np.asarray(y_train, dtype='float32')

def get_data():
    features = []
    outputs = []
    return features, outputs

if __name__ == "__main__":
    print('It\'s not the size that counts, it\'s the connections')

    feature_list, output_list = get_data()

    x, y = parse_data(feature_list, output_list)

    model = mlp()

    model.compile(optimizer='adam',
                  loss={'future_diagnosis': 'categorical_crossentropy',
                        'ventricle_volume': 'mean_squared_error',
                        'as_cog': 'mean_squared_error'},
                  loss_weights={'future_diagnosis': 0.5, 'ventricle_volume': 0.25, 'as_cog': 0.25}
                  )

    model.fit_generator()
