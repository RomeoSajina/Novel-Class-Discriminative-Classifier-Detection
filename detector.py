from config import Config
from dataset import *
from model import *
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Softmax, Dense, Flatten, AveragePooling2D, AveragePooling1D, Conv2D, Dropout, MaxPooling2D, MaxPooling1D, LeakyReLU, Conv1D, MaxPooling1D, BatchNormalization, ReLU, LayerNormalization, Reshape, LSTM
from tensorflow.python.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.python.keras.utils import np_utils
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class NovelClassDetector(ABC):

    def __init__(self, config: Config, nn_model: StreamModel, dataset: StreamDataset):
        assert nn_model is not None
        assert dataset is not None
        self.config = config
        self.dataset = dataset
        self.nn_model = nn_model
        self.name = "detector"

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError()

    def after_model_trained(self):
        pass


class SoftmaxStatsDetector(NovelClassDetector):

    def __init__(self, config: Config, nn_model: StreamModel, dataset: StreamDataset):
        super().__init__(config, nn_model, dataset)
        self.name = "softmax_stats"
        self.mean = 0.
        self.std = 0.

    def after_model_trained(self):

        probs = self.nn_model.predict(self.dataset.x_train)
        m_probs = np.max(probs, axis=1)

        self.mean, self.std = np.mean(m_probs), np.std(m_probs)

    def predict(self, x):

        probs = self.nn_model.predict(x)
        m_probs = np.max(probs, axis=1)

        print("Predict: ", np.mean(m_probs), np.std(m_probs))
        splt = int(len(x)/2)
        print("Predict_known: ", np.mean(m_probs[:splt]), np.std(m_probs[:splt]))
        print("Predict_novel: ", np.mean(m_probs[splt:]), np.std(m_probs[splt:]))

        idx = np.where(m_probs < self.mean - self.std)[0]

        out = np.zeros((len(x), 1))
        out[idx] = 1

        return out


class DiscriminatorDetector(NovelClassDetector):

    def __init__(self, config: Config, nn_model: StreamModel, dataset: StreamDataset):
        super().__init__(config, nn_model, dataset)
        self.output_layer_index = -2
        self.name = "discriminator"
        self.input_shape = None

        self.build()

    def build(self):
        self.input_shape = [self.nn_model.model.layers[self.output_layer_index].output_shape[1] + (10 if self.config.include_softmax_output else 0)]
        self._build()

    def _build(self):

        model = Sequential()

        """
        model.add(Reshape((self.input_shape[0], 1), input_shape=self.input_shape))
        model.add(Conv1D(filters=16, kernel_size=3, activation="relu", padding="valid"))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation="softmax"))
        """

        #model.add(Dense(2, activation="softmax", input_shape=self.input_shape))

        """
        model.add(Dense(1024, activation="relu", input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation="softmax"))
        """

        model.add(Dense(512, activation="relu", input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.1))
        #model.add(Dense(256, activation="relu"))
        #model.add(Dropout(0.1))
        #model.add(Dense(64, activation="relu"))
        #model.add(Dropout(0.1))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(2, activation="softmax"))

        #model.add(Dense(1, activation="sigmoid"))

        #model.summary()
        #model.compile(loss="binary_crossentropy", optimizer=Adam(0.005), metrics=["accuracy"])
        model.compile(loss="categorical_crossentropy", optimizer=Adam(0.005), metrics=["accuracy"])

        self.model = model

    def predict(self, x):
        out = self.nn_model.get_layer_output(x, self.output_layer_index)

        if self.config.treshold is None:
            out = self.model.predict_classes(out, batch_size=10000)
            return out
            #return np.concatenate((np.zeros(int(len(x)/2)), np.ones(int(len(x)/2))))

        out = self.model.predict(out, batch_size=10000)

        #out = np.argmax(out, axis=1)
        #return out
        #print("D_p: ", out.shape)
        #print(out)

        treshold = self.config.treshold
        real_out = []
        for i in range(len(out)):

            if out[i][0] < 1. - treshold:
                real_out.append(1)
            else:
                real_out.append(np.argmax(out[i]))

        return np.array(real_out)

    def after_model_trained(self):

        noise, _ = self.dataset.get_augmented()

        train_out = self.nn_model.get_layer_output(self.dataset.x_train, self.output_layer_index)
        noise_out = self.nn_model.get_layer_output(noise, self.output_layer_index)

        des_x = np.concatenate((train_out, noise_out))
        des_y = np.concatenate((np.zeros((len(train_out), 1)), np.ones((len(noise_out), 1))))

        """
        # <
        self.config.augmented_scale = 0.1
        self.config.augmented_rotation_scale = 0.05
        a_k, _ = self.dataset.get_augmented()

        self.config.augmented_scale = 1.
        self.config.augmented_rotation_scale = 1.
        a_n, _ = self.dataset.get_augmented()

        k_out = self.nn_model.get_layer_output(a_k, self.output_layer_index)
        n_out = self.nn_model.get_layer_output(a_n, self.output_layer_index)

        des_x = np.concatenate((des_x, k_out, n_out))
        des_y = np.concatenate((des_y, np.zeros((len(k_out), 1)), np.ones((len(n_out), 1))))
        # >
        """
        #print("Noise: Max: ", max(noise.flatten()), "Min: ", min(noise.flatten()))
        #print("Train: Max: ", max(self.dataset.x_train.flatten()), "Min: ", min(self.dataset.x_train.flatten()))
        #print("Known: ", (train_out.shape, des_y.flatten().tolist().count(0)), "Novel: ", (train_out.shape, des_y.flatten().tolist().count(1)))

        des_y = np_utils.to_categorical(des_y, 2)

        # Shuffle
        #idx = np.arange(len(des_y))
        #np.random.shuffle(idx)
        #des_x = des_x[idx]
        #des_y = des_y[idx]

        self.model.fit(des_x, des_y, verbose=self.config.verbose, epochs=self.config.train_epochs, shuffle=True)#, batch_size=128)

        # <train_eval>
        #cm0 = confusion_matrix(np.argmax(des_y, axis=1), self.predict(np.concatenate((self.dataset.x_train, noise))))
        #t_known_acc = cm0[0][0] / (cm0[0][0] + cm0[0][1])
        #t_novel_acc = cm0[1][1] / (cm0[1][0] + cm0[1][1])
        #print("cm0 known_acc: {0:.2f}, novel_acc: {1:.2f}, avg_acc: {2:.2f}\n".format(t_known_acc, t_novel_acc, (t_known_acc + t_novel_acc) / 2), cm0, "\n")
        # </train_eval>
