from stream_simulator import StreamSimulator, DistanceStreamSimulator
from config import Config

datasets = ["Mnist", "FashionMnist", "KuzushijiMnist", "NotMnist", "Cifar10BW", "Cifar10"]
detectors = ["SoftmaxStats", "Discriminator"]


def demo():
    config = Config()
    #config.dataset_name = "FashionMnist"
    config.dataset_name = "Mnist"
    #config.dataset_name = "KuzushijiMnist"
    #config.dataset_name = "NotMnist"
    #config.dataset_name = "Cifar10BW"

    #config.detector_name = "SoftmaxStats"
    config.detector_name = "Discriminator"

    config.dev_use_true_labels = False

    config.train_epochs = 100
    config.num_of_class_instances = 100
    config.repeat = 1

    config.augmented_scale = 1.
    config.augmented_rotation_scale = 1.
    config.treshold = None
    #config.treshold = .1
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


def run_distance_sim():

    for dt in detectors[0:1]:
        for ds in datasets[0:1]:

            config = Config()
            config.dataset_name = ds
            config.detector_name = dt

            config.train_epochs = 100
            #config.num_of_class_instances = 1000 -- not necessary anymore
            config.repeat = 10
            
            config.treshold = None
            config.augmented_rotation_scale = .5
            config.augmented_scale = .3
            config.verbose = 2

            sim = DistanceStreamSimulator(config=config)
            sim.stream()


#if __name__ == "__main__":
#    run_distance_sim()
