import os
from config import Config
from abc import ABC, abstractmethod
from tensorflow.python.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.python.keras.utils import np_utils
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class StreamDataset(ABC):

    def __init__(self, config: Config):
        self.config = config
        self.num_of_outputs = 100
        self.dataset_name = "dataset_name"

        self.x_train = None
        self.y_train = None
        self.x_unseen = None
        self.y_unseen = None

        self.load()

    @staticmethod
    def prepare_data(x, y, num=1000, percentage=None):

        if len(x.shape) < 4:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        y = y.flatten()

        x_train, y_train = np.array([]).reshape(0, x.shape[1], x.shape[2], x.shape[3]), np.array([])

        #m = max(y) + 1
        #print("Prepare_data: max: " + str(m))
        indicies = np.unique(y)
        print("Prepare_data: indicies: " + str(indicies))

        selected_indicies = []

        #for i in range(0, m):
        for i in indicies:
            idx = np.where(y == i)[0]

            if percentage is not None:
                idx = np.random.choice(idx, int(len(idx) * percentage), replace=False)
            else:
                idx = np.random.choice(idx, num, replace=False)

            assert max(y[idx]) == min(y[idx]) == i

            x_train = np.concatenate((x_train, x[idx]))
            y_train = np.concatenate((y_train, y[idx]))

            selected_indicies.extend(idx)

            assert len(selected_indicies) == len(np.unique(selected_indicies))

        x_unseen = np.delete(x, selected_indicies, axis=0)
        y_unseen = np.delete(y, selected_indicies, axis=0)

        return x_train, y_train, x_unseen, y_unseen

    def load(self):
        (x_train, y_train), (x_test, y_test) = self._load()

        x_all = np.concatenate((x_train, x_test))
        y_all = np.concatenate((y_train, y_test))

        x_train, y_train, x_unseen, y_unseen = StreamDataset.prepare_data(x=x_all, y=y_all, num=self.config.num_of_class_instances)

        # Set numeric type to float32 from uint8
        x_train = x_train.astype("float32")
        x_unseen = x_unseen.astype("float32")

        # Normalize value to [0, 1]
        x_train /= 255
        x_unseen /= 255

        # Transform lables to one-hot encoding
        y_train = np_utils.to_categorical(y_train, self.num_of_outputs)
        y_unseen = np_utils.to_categorical(y_unseen, self.num_of_outputs)

        self.x_train = x_train
        self.y_train = y_train
        self.x_unseen = x_unseen
        self.y_unseen = y_unseen

    @abstractmethod
    def _load(self):
        raise NotImplementedError()

    def build_img_data_gen(self):

        scale = self.config.augmented_scale

        def apply_scale_for_brightness_range(scale):
            start = np.array([0.8, 1., 1.])
            m = np.array([[-scale, 0, 0], [0, 1 + scale, 0], [1, 0, 0]])
            return list(start.dot(m)[:2])

        MAX = 1.

        datagen = ImageDataGenerator(
            rotation_range=180 * self.config.augmented_rotation_scale,
            width_shift_range=MAX * scale,
            height_shift_range=MAX * scale,
            brightness_range=apply_scale_for_brightness_range(scale),
            #brightness_range=(0.1, 1.9),
            shear_range=MAX * scale,
            zoom_range=.2) #MAX * scale)

        return datagen

    def get_augmented(self):

        x, y = self.x_train, self.y_train

        datagen = self.build_img_data_gen()

        train_generator = datagen.flow(x, y)

        x_, y_ = np.array([]).reshape(0, x.shape[1], x.shape[2], x.shape[3]), np.array([]).reshape(0, y.shape[1])

        for i, (_x, _y) in enumerate(train_generator):
            x_ = np.concatenate((x_, _x))
            y_ = np.concatenate((y_, _y))

            if len(x_) >= len(x):
                break

        x_ = x_.astype("float32")
        x_ /= 255

        return x_[:len(x)], y_[:len(x)]

    def info(self):
        print("Train: ", (self.x_train.shape, self.y_train.shape), "Unseen: ", (self.x_unseen.shape, self.y_unseen.shape))


class MnistDataset(StreamDataset):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "mnist"

    def _load(self):
        return mnist.load_data()


class FashionMnistDataset(StreamDataset):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "fashion_mnist"

    def _load(self):
        return fashion_mnist.load_data()


class NotMnistDataset(StreamDataset):

    url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
    export_path = ".cache/datasets/notmnist"

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "not_mnist"

    def _download_if_needed(self):
        filename = ".cache/datasets/notmnist.tar.gz"

        if not os.path.exists(".cache/datasets"):
            os.mkdir(".cache/datasets")

        if not os.path.exists(self.export_path):
            from six.moves.urllib.request import urlretrieve
            from distutils.dir_util import copy_tree, remove_tree
            import tarfile

            print("Attempting to download from {0}".format(self.url))
            urlretrieve(self.url, filename)

            tar = tarfile.open(filename)
            tar.extractall()
            tar.close()

            extracted = "./notMNIST_small"

            if not os.path.exists(self.export_path):
                os.mkdir(self.export_path)

            copy_tree(extracted, self.export_path)

            remove_tree(extracted)
            os.remove(filename)

    def __load_from_exported(self):
        from PIL import Image

        X = []
        labels = []
        for directory in os.listdir(self.export_path):
            for image in os.listdir(self.export_path + "/" + directory):
                try:
                    file_path = self.export_path + "/" + directory + "/" + image
                    img = Image.open(file_path)
                    img.load()
                    img_data = np.asarray(img, dtype=np.int16)
                    X.append(img_data)
                    labels.append(directory)
                except:
                    print("error loading image " + file_path)

        N = len(X)
        img_size = len(X[0])
        X = np.asarray(X).reshape(N, img_size, img_size)
        labels = np.asarray(list(map(lambda x: ord(x) - ord("A"), labels)))

        return (X[:-1000], labels[:-1000]), (X[-1000:], labels[-1000:])

    def _load(self):
        self._download_if_needed()
        return self.__load_from_exported()


class KuzushijiMnistDataset(StreamDataset):

    urls = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz",
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz",
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz",
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz"]

    root = ".cache/datasets/kmnist/"

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "kuzushiji_mnist"

    def _download(self):
        import requests

        tqdm = lambda x, total, unit: x

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        for url in self.urls:
            path = self.root + url.split("/")[-1]
            r = requests.get(url, stream=True)
            with open(path, "wb") as f:
                total_length = int(r.headers.get("content-length"))
                print("Downloading {} - {:.1f} MB".format(path, (total_length / 1024000)))

                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)

        print("All dataset files downloaded!")

    def _load(self):
        try:
            np.load(self.root + "kmnist-train-imgs.npz")["arr_0"]
        except:
            self._download()

        x_train = np.load(self.root + "kmnist-train-imgs.npz")["arr_0"]
        y_train = np.load(self.root + "kmnist-train-labels.npz")["arr_0"]
        x_test = np.load(self.root + "kmnist-test-imgs.npz")["arr_0"]
        y_test = np.load(self.root + "kmnist-test-labels.npz")["arr_0"]
        return (x_train, y_train), (x_test, y_test)


class Cifar10BWDataset(StreamDataset):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "cifar10bw"

    def _load(self):
        from PIL import Image

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = np.array([np.array(Image.fromarray(x, mode="RGB").convert('L')) for x in x_train])
        x_test = np.array([np.array(Image.fromarray(x, mode="RGB").convert('L')) for x in x_test])

        return (x_train, y_train), (x_test, y_test)


class Cifar10Dataset(StreamDataset):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.dataset_name = "cifar10"

    def _load(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)
