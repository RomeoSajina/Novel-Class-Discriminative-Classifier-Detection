from config import Config
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Softmax, Dense, Flatten, AveragePooling2D, AveragePooling1D, Conv2D, Dropout, \
                                           MaxPooling2D, MaxPooling1D, LeakyReLU, Conv1D, MaxPooling1D, BatchNormalization, \
                                           ReLU, LayerNormalization, Reshape, LSTM, Input, Dense, Dropout, Activation, \
                                           Concatenate, GlobalAveragePooling2D


class StreamModel:

    def __init__(self, config: Config, x_shape, y_shape):
        self.config = config
        self.x_shape = x_shape
        self.y_shape = y_shape

        self._build()

    def _build(self):

        input_shape = self.x_shape[1:]
        out_shape = self.y_shape[1]

        padding = "valid"

        model = Sequential()
        #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding=padding, input_shape=input_shape))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding=padding, input_shape=input_shape))
        #model.add(Conv2D(filters=2, kernel_size=(3, 3), activation="relu", padding=padding, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(padding=padding))
        #model.add(MaxPooling2D(padding=padding))
        #model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding=padding))
        model.add(Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding=padding))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(padding=padding))
        #model.add(MaxPooling2D(padding=padding))
        model.add(Flatten())
        model.add(Dropout(0.1))

        #model.add(Dense(1024, activation="relu"))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.1))


        #model.add(Dense(512, activation=None))

        #model.add(Dense(512, activation="relu"))
        #model.add(Dense(512, activation=None))
        #model.add(ReLU())

        #model.add(Dropout(0.1))

        model.add(Dense(units=out_shape, activation="softmax"))

        #model.add(Dense(units=out_shape))
        #model.add(Softmax())

        #model.summary()

        model.compile(loss="categorical_crossentropy", optimizer=Adam(0.005), metrics=["accuracy"])

        self.model = model

    def get_layer_output(self, data, output_layer_index=-2):

        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers]
        functors = [K.function(inp, out) for out in outputs]

        out = None
        batch_size = 10000

        for step in range(0, int(len(data) / batch_size) + 1):

            start = step * batch_size
            end = (step + 1) * batch_size
            end = end if end <= len(data) - 1 else len(data)

            step_out = functors[output_layer_index](data[start:end])

            if self.config.include_softmax_output:
                sftmx_step_out = functors[-1](data[start:end])
                step_out = np.append(step_out, sftmx_step_out, axis=1)

            out = np.concatenate((out, step_out)) if out is not None else step_out

        return out

    def predict(self, x):
        return self.model.predict(x)

    def predict_classes(self, x):
        # p = self.model.predict(x)
        # return np.argmax(p, axis=1)
        return self.model.predict_classes(x)

    def train(self, x, y):
        self.model.fit(x, y, verbose=self.config.verbose, epochs=self.config.train_epochs, shuffle=True)#, batch_size=128)
