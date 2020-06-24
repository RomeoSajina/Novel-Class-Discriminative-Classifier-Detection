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

            if i > 0: # Print stats for info
                self.stats.print(logger=None)

            print("Repeat: " + str(i+1))

            train_classes = randomly_generated[i]
            test_classes = np.delete(np.arange(10), train_classes)

            self._reset()

            self.stats.add_sim_stats(train_classes, self.run_stream(train_classes=train_classes, test_classes=test_classes))

        FileManager.cache_object(self.stats, self.config.dataset_name + "_" + self.config.detector_name + "_" + datetime.now().strftime("%m_%d_%Y-%H_%M"))

        self.stats.print(logger=logger)

        return self.stats

    def run_stream(self, train_classes, test_classes):

        stats = {}

        # Move all test instances into unseen dataset
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

        cm = confusion_matrix(d_y_real, p_cls, labels=[0, 1])

        known_acc = cm[0][0] / len(x_known)
        novel_acc = cm[1][1] / len(x_novel)
        s = {"known_acc": known_acc, "novel_acc": novel_acc, "avg_acc": (known_acc + novel_acc) / 2}
        print("eval known_acc: {0:.2f}, novel_acc: {1:.2f}, avg_acc: {2:.2f}\n".format(s["known_acc"], s["novel_acc"], s["avg_acc"]), cm, "\n")

        # Predicted
        novel_idx = np.where(p_cls == 1)[0]
        novel_x = d_x[novel_idx]
        known_x = np.delete(d_x, novel_idx, axis=0)

        print("New class index: ", novel_class_index)
        novel_y = np_utils.to_categorical(np.repeat([novel_class_index], len(novel_x)), self.dataset.y_train.shape[1])
        known_y = np_utils.to_categorical(self.model.predict_classes(known_x), self.dataset.y_train.shape[1])

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

    def load_stats(self, date_str):
        self.stats = FileManager.load_from_cache(self.config.dataset_name + "_" + self.config.detector_name + "_" + date_str)


class DistanceStreamSimulator(StreamSimulator):

    def __init__(self, config: Config):
        super().__init__(config)

    def generate_streaming_data(self, train_classes, test_classes):

        self.dataset.info()

        train_init_lenghts = [1000, 2000, 3000, 4000, 5000]
        test_init_lenghts = [1000, 2000, 3000, 4000, 5000]
        min_sample, max_sample = 200, 500

        stream_datasets = []
        classes_in_stream = np.array(train_classes)

        x_train = np.array(self.dataset.x_train)
        y_train = np.array(self.dataset.y_train)
        x_unseen = np.array(self.dataset.x_unseen)
        y_unseen = np.array(self.dataset.y_unseen)

        x_t = np.array([]).reshape(np.concatenate([[0], x_train.shape[1:]]))
        y_t = np.array([]).reshape([0, y_train.shape[1]])
        x_u = np.array([]).reshape(np.concatenate([[0], x_train.shape[1:]]))
        y_u = np.array([]).reshape([0, y_train.shape[1]])


        def random_data(f_class, x_unseen, y_unseen):

            c_idx = np.where(np.argmax(y_unseen, axis=1) == f_class)[0]

            if f_class in classes_in_stream:
               r_idx = np.random.choice(c_idx, np.random.randint(min_sample, max_sample), replace=False)
            else:
               r_idx = np.random.choice(c_idx, np.random.randint(min_sample * 2, max_sample), replace=False)

            xu = x_unseen[r_idx]
            yu = y_unseen[r_idx]

            x_unseen = np.delete(x_unseen, r_idx, axis=0)
            y_unseen = np.delete(y_unseen, r_idx, axis=0)

            return xu, yu, x_unseen, y_unseen

        for i in range(16):
        #for i in range(18):

            x_t = np.concatenate((x_t, x_u))
            y_t = np.concatenate((y_t, y_u))

            x_u = x_u[0:0]
            y_u = y_u[0:0]

            for k in range(5):

                if train_init_lenghts[k] < 0 and np.random.randint(0, 100) < 30:
                    continue

                xu, yu, x_unseen, y_unseen = random_data(train_classes[k], x_unseen, y_unseen)

                x_u = np.concatenate((x_u, xu))
                y_u = np.concatenate((y_u, yu))

                train_init_lenghts[k] -= len(xu)

            # Novel/Unseen classes
            for k in range(5):

                if train_init_lenghts[k] > 0 or np.random.randint(0, 100) < 30:
                    continue
                else:
                    classes_in_stream = np.unique(np.concatenate((classes_in_stream, [test_classes[k]])))

                xu, yu, x_unseen, y_unseen = random_data(test_classes[k], x_unseen, y_unseen)

                x_u = np.concatenate((x_u, xu))
                y_u = np.concatenate((y_u, yu))

                test_init_lenghts[k] -= len(xu)

            unique, counts = np.unique(np.concatenate((np.argmax(y_u, axis=1), np.arange(0, 10))), return_counts=True)
            print("{0}/18".format(i), dict(zip(unique, counts - 1)))

            stream_datasets.append([np.array(x_t), np.array(y_t), np.array(x_u), np.array(y_u)])

        stream_datasets = stream_datasets[1:]

        return stream_datasets

    def run_stream(self, train_classes, test_classes):

        stats = {}
        known_classes = []

        self.dataset.x_unseen = np.concatenate((self.dataset.x_unseen, self.dataset.x_train))
        self.dataset.y_unseen = np.concatenate((self.dataset.y_unseen, self.dataset.y_train))
        self.dataset.x_train = np.array([]).reshape(np.concatenate([[0], self.dataset.x_unseen.shape[1:]]))
        self.dataset.y_train = np.array([]).reshape([0, self.dataset.y_unseen.shape[1]])
        self.dataset.info()

        stream_datasets = self.generate_streaming_data(train_classes, test_classes)

        for i, sd in enumerate(stream_datasets):

            key = str(i)
            self.dataset.x_train, self.dataset.y_train, self.dataset.x_unseen, self.dataset.y_unseen = sd

            if i % 3 == 0:
                self.dataset.info()
                self.train()
                known_classes = np.unique(np.argmax(self.dataset.y_train, axis=1))

            k_idx, n_idx = [], []
            for i, c in enumerate(np.argmax(self.dataset.y_unseen, axis=1)):
                if c in known_classes:
                    k_idx.append(i)
                else:
                    n_idx.append(i)

            x_known = self.dataset.x_unseen[k_idx]
            y_known = self.dataset.y_unseen[k_idx]
            x_novel = self.dataset.x_unseen[n_idx]
            y_novel = self.dataset.y_unseen[n_idx]

            _x, _y, score = self.eval(x_known, y_known, x_novel, y_novel, 0)

            stats[key] = score

        return stats


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
