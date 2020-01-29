from config import Config
from dataset import *
from model import *
from detector import *
from stats_collector import StatsCollector
import pickle
import os
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils import np_utils
from datetime import datetime
from logger import logger


class StreamSimulator:

    def __init__(self, config: Config):
        self.config = config

        self.model = None
        self.detector = None
        self.dataset = None
        self.stats = None
        self._reset()

    def _reset(self):
        self.dataset = globals()[self.config.dataset_name + "Dataset"](config=self.config)
        self.model = StreamModel(config=self.config, x_shape=self.dataset.x_train.shape, y_shape=self.dataset.y_train.shape)
        self.detector = globals()[self.config.detector_name + "Detector"](config=self.config, nn_model=self.model, dataset=self.dataset)

    def train(self):

        print("Train with classes: ", np.unique(np.argmax(self.dataset.y_train, axis=1)))

        for i in np.unique(np.argmax(self.dataset.y_train, axis=1)):

            print("i: " + str(i) + " " + str(np.where(np.argmax(self.dataset.y_train, axis=1) == i)[0].shape))

        if self.config.do_split_training:
            x_train, y_train, x_unseen, y_unseen = StreamDataset.prepare_data(x=self.dataset.x_train,
                                                                              y=np.argmax(self.dataset.y_train, axis=1),
                                                                              percentage=.7)
            y_train = np_utils.to_categorical(y_train, self.dataset.num_of_outputs)
            self.model.train(x_train, y_train)

        else:
            self.model.train(self.dataset.x_train, self.dataset.y_train)

        #self.model.train(self.dataset.x_train, self.dataset.y_train)

        self.detector.after_model_trained()

    def stream(self):

        self._on_start()
        self.stats = StatsCollector()

        try:
            randomly_generated = FileManager.load_from_cache("generated_train_classes")
        except:
            randomly_generated = np.array([np.random.choice(np.arange(10), 5, replace=False) for _ in range(10)])
            FileManager.cache_object(randomly_generated, "generated_train_classes")

        for i in range(self.config.repeat):

            print("Repeat: " + str(i+1))

            train_classes = randomly_generated[i]
            test_classes = np.delete(np.arange(10), train_classes)

            self._reset()

            #self.stats.add(key=str(train_classes), entry=self.run_stream(train_classes=train_classes, test_classes=test_classes))
            self.stats.add_sim_stats(train_classes, self.run_stream(train_classes=train_classes, test_classes=test_classes))

        FileManager.cache_object(self.stats, self.config.dataset_name + "_" + self.config.detector_name)

        self.stats.print(logger=logger)

        return self.stats

    """
    def run_stream2(self, train_classes, test_classes):

        stats = {}

        idxs = np.where([q in train_classes for q in np.argmax(self.dataset.y_train, axis=1)])[0]

        self.dataset.info()

        self.dataset.x_unseen = np.concatenate((self.dataset.x_unseen, np.delete(self.dataset.x_train, idxs, axis=0)))
        self.dataset.y_unseen = np.concatenate((self.dataset.y_unseen, np.delete(self.dataset.y_train, idxs, axis=0)))

        self.dataset.x_train = self.dataset.x_train[idxs]
        self.dataset.y_train = self.dataset.y_train[idxs]

        def train_detector(k_out, n_out, config):
            des_x = np.concatenate((k_out, n_out))
            des_y = np.concatenate((np.zeros((len(k_out), 1)), np.ones((len(n_out), 1))))

            n_o = self.model.get_layer_output(self.dataset.get_augmented())
            k_o = self.model.get_layer_output(self.dataset.x_train)

            des_x = np.concatenate((des_x, k_o, n_o))
            des_y = np.concatenate((des_y, np.zeros((len(k_o), 1)), np.ones((len(n_o), 1))))

            des_y = np_utils.to_categorical(des_y, 2)
            print("train detector: " + str(des_x.shape), str(des_y.shape))
            self.detector.model.fit(des_x, des_y, verbose=self.config.verbose, epochs=self.config.train_epochs*2, shuffle=True)

        x, y, k_outputs, n_outputs = None, None, None, None
        m_trained = False
        for tc in train_classes:

            ix = np.where(np.argmax(self.dataset.y_train, axis=1) == tc)[0]
            x_ = self.dataset.x_train[ix]
            y_ = self.dataset.y_train[ix]

            x = x_ if x is None else np.concatenate((x, x_))
            y = y_ if y is None else np.concatenate((y, y_))

            print(np.unique(np.argmax(y, axis=1)))

            if m_trained:
                n_o = self.model.get_layer_output(x)
                n_outputs = np.concatenate((n_outputs, n_o)) if n_outputs is not None else n_o

            self.model.train(x, y)
            m_trained = True

            k_o = self.model.get_layer_output(x)
            k_outputs = np.concatenate((k_outputs, k_o)) if k_outputs is not None else k_o

        train_detector(k_outputs, n_outputs, self.config)

        # Take one class from seen classes and one novel class and evaluate
        # Than train with predicted classes
        for k in range(len(train_classes)):
            t, n = train_classes[k], test_classes[k]
            key = str(t) + "-" + str(n)
            print(key)

            novel_idxs = np.where(np.argmax(self.dataset.y_unseen, axis=1) == n)[0]
            n_x = self.dataset.x_unseen[novel_idxs][:self.config.num_of_class_instances] #TODO: TMP
            n_y = self.dataset.y_unseen[novel_idxs][:self.config.num_of_class_instances] #TODO: TMP

            n_o = self.model.get_layer_output(n_x)
            n_outputs = np.concatenate((n_outputs, n_o))

            known_idxs = np.where(np.argmax(self.dataset.y_unseen, axis=1) == t)[0]
            k_x = self.dataset.x_unseen[known_idxs][:self.config.num_of_class_instances] #TODO: TMP
            k_y = self.dataset.y_unseen[known_idxs][:self.config.num_of_class_instances] #TODO: TMP

            k_o = self.model.get_layer_output(k_x)
            k_outputs = np.concatenate((k_outputs, k_o))

            # Evaluate
            _x, _y, score = self.eval(k_x, k_y, n_x, n_y, n)

            print("Ha ha ha")
            train_detector(k_outputs, n_outputs, self.config)

            x = np.concatenate((x, _x))
            y = np.concatenate((y, _y))

            self.model.train(self.dataset.x_train, self.dataset.y_train)
            # TODO train detector on predicted newly

            self.dataset.x_train = x
            self.dataset.y_train = y

            indexes = np.concatenate((novel_idxs, known_idxs))
            self.dataset.x_unseen = np.delete(self.dataset.x_unseen, indexes, axis=0)
            self.dataset.y_unseen = np.delete(self.dataset.y_unseen, indexes, axis=0)

            stats[key] = score

        return stats
    """

    def run_stream(self, train_classes, test_classes):

        stats = {}

        idxs = np.where([q in train_classes for q in np.argmax(self.dataset.y_train, axis=1)])[0]

        self.dataset.info()

        self.dataset.x_unseen = np.concatenate((self.dataset.x_unseen, np.delete(self.dataset.x_train, idxs, axis=0)))
        self.dataset.y_unseen = np.concatenate((self.dataset.y_unseen, np.delete(self.dataset.y_train, idxs, axis=0)))

        self.dataset.x_train = self.dataset.x_train[idxs]
        self.dataset.y_train = self.dataset.y_train[idxs]

        # Take one class from seen classes and one novel class and evaluate
        # Than train with predicted classes
        for k in range(len(train_classes)):
            t, n = train_classes[k], test_classes[k]

            self.dataset.info()
            self.train()

            key = str(t) + "-" + str(n)
            print(key)

            novel_idxs = np.where(np.argmax(self.dataset.y_unseen, axis=1) == n)[0]
            novel_x = self.dataset.x_unseen[novel_idxs][:self.config.num_of_class_instances] #TODO: TMP
            novel_y = self.dataset.y_unseen[novel_idxs][:self.config.num_of_class_instances] #TODO: TMP

            known_idxs = np.where(np.argmax(self.dataset.y_unseen, axis=1) == t)[0]
            known_x = self.dataset.x_unseen[known_idxs][:self.config.num_of_class_instances] #TODO: TMP
            known_y = self.dataset.y_unseen[known_idxs][:self.config.num_of_class_instances] #TODO: TMP

            _x, _y, score = self.eval(known_x, known_y, novel_x, novel_y, n)

            self.dataset.x_train = np.concatenate((self.dataset.x_train, _x))
            self.dataset.y_train = np.concatenate((self.dataset.y_train, _y))

            indexes = np.concatenate((novel_idxs, known_idxs))
            self.dataset.x_unseen = np.delete(self.dataset.x_unseen, indexes, axis=0)
            self.dataset.y_unseen = np.delete(self.dataset.y_unseen, indexes, axis=0)

            stats[key] = score

        return stats

    def eval(self, x_known, y_known, x_novel, y_novel, novel_class_index):

        d_y_real = np.concatenate((np.zeros((len(x_known), 1)), np.ones((len(x_novel), 1))))
        d_x = np.concatenate((x_known, x_novel))

        p_cls = self.detector.predict(d_x)

        cm = confusion_matrix(d_y_real, p_cls)

        known_acc = cm[0][0] / len(x_known)
        novel_acc = cm[1][1] / len(x_novel)
        s = {"known_acc": known_acc, "novel_acc": novel_acc, "avg_acc": (known_acc + novel_acc) / 2}
        print("eval known_acc: {0:.2f}, novel_acc: {1:.2f}, avg_acc: {2:.2f}\n".format(s["known_acc"], s["novel_acc"], s["avg_acc"]), cm, "\n")

        # Predicted
        novel_idx = np.where(p_cls == 1)[0]
        novel_x = d_x[novel_idx]
        known_x = np.delete(d_x, novel_idx, axis=0)

        print("New class index: ", novel_class_index)
        novel_y = np_utils.to_categorical(np.repeat([novel_class_index], len(novel_x)), self.model.y_shape[1])
        known_y = np_utils.to_categorical(self.model.predict_classes(known_x), self.model.y_shape[1])

        p_x = np.concatenate((known_x, novel_x))
        p_y = np.concatenate((known_y, novel_y))

        if self.config.dev_use_true_labels: # Return real values (dev only)
            p_y = np.concatenate((y_known, y_novel))

        return p_x, p_y, (d_y_real, p_cls) #s

    def _on_start(self):
        logger.info("\n".ljust(100, "-"))
        logger.info("".rjust(40) + datetime.now().strftime("%m.%d.%Y. %H:%M:%S"))
        logger.info("".rjust(100, "-"))

        logger.info(str(self.config.dump()).replace(", ", "\n").replace("{", "").replace("}", ""))


class FileManager:

    @staticmethod
    def create_folder_if_needed(filename):

        folder = filename.rsplit("/", 1)[0]

        if not os.path.exists(folder):
            os.makedirs(folder)

    @staticmethod
    def cache_object(o, name):

        filename = "./.cache/" + name + ".pickle"
        FileManager.create_folder_if_needed(filename)

        with open(filename, "wb") as handle:
            pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_cache(name):
        with open("./.cache/" + name + ".pickle", "rb") as handle:
            return pickle.load(handle)
