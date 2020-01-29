from stream_simulator import StreamSimulator
from config import Config

datasets = ["Mnist", "FashionMnist", "KuzushijiMnist", "NotMnist", "Cifar10BW"]
detectors = ["SoftmaxStats", "Discriminator"]

def demo():
    config = Config()
    #config.dataset_name = "Cifar10"
    #config.dataset_name = "FashionMnist"
    #config.dataset_name = "Mnist"
    #config.dataset_name = "KuzushijiMnist"
    #config.dataset_name = "NotMnist"
    config.dataset_name = "Cifar10BW"

    #config.detector_name = "SoftmaxStats"
    config.detector_name = "Discriminator"

    config.dev_use_true_labels = False

    config.train_epochs = 10
    config.num_of_class_instances = 100
    config.repeat = 1

    config.augmented_scale = 1.
    config.augmented_rotation_scale = 1.
    #config.treshold = None
    config.treshold = .1
    #config.augmented_scale = 1.
    #config.augmented_rotation_scale = 1.

    from logger import logger
    logger.disabled = True

    sim = StreamSimulator(config=config)
    s = sim.stream()
    s.print()


def run():

    for dt in detectors:
        for ds in datasets:

            config = Config()
            config.dataset_name = ds
            config.detector_name = dt

            config.train_epochs = 100
            config.num_of_class_instances = 1000
            config.repeat = 10

            sim = StreamSimulator(config=config)
            sim.stream()


if __name__ == "__main__":
    run()
