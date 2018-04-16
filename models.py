import keras

from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from keras.layers.merge import concatenate

from keras.models import Model

def mlp(x_t_length):
    diagnostic_states = 3

    x_t = Input(shape=(x_t_length,))
    y_t = Input(shape=(3,))
    delta_t = Input(shape=(1,))

    h1 = Dense(512)(x_t)
    l1 = LeakyReLU()(h1)
    d1 = Dropout(0.2)(l1)
    n1 = BatchNormalization()(d1)
    h1m = concatenate([n1, delta_t])

    h2 = Dense(1024)(h1m)
    l2 = LeakyReLU()(h2)
    d2 = Dropout(0.3)(l2)
    n2 = BatchNormalization()(d2)
    h2m = concatenate([n2, delta_t])

    h3 = Dense(1024)(h2m)
    l3 = LeakyReLU()(h3)
    d3 = Dropout(0.4)(l3)
    n3 = BatchNormalization()(d3)
    h3m = concatenate([n3, delta_t])

    h4 = Dense(512)(h3m)
    l4 = LeakyReLU()(h4)
    d4 = Dropout(0.5)(l4)
    n4 = BatchNormalization()(d4)
    h4m = concatenate([n4, delta_t])

    h5 = Dense(512)(h4m)
    l5 = LeakyReLU()(h5)
    d5 = Dropout(0.5)(l5)
    n5 = BatchNormalization()(d5)
    h5m = concatenate([n5, delta_t])

    future_diagnosis = Dense(diagnostic_states, activation='softmax', name='future_diagnosis')(h5m)

    ventricle_volume = Dense(1, activation='linear', name='ventricle_volume')(h5m)
    as_cog = Dense(1, activation='linear', name='as_cog')(h5m)

    model = Model(inputs=[x_t, y_t, delta_t], outputs=[future_diagnosis, ventricle_volume, as_cog])

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