from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


def get_dense_network(input_shape, output_dimension, learning_rate):
    model = Sequential([
        Dense(units=512, input_shape=input_shape, activation='relu'),
        Dense(units=output_dimension)
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    return model
