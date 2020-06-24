

class Config():

    def __init__(self):

        self.dataset_name = "Mnist"
        self.detector_name = "Discriminator"
        self.repeat = 1
        self.num_of_class_instances = 1000
        self.train_epochs = 20
        self.verbose = 2

        self.augmented_scale = 1.
        self.augmented_rotation_scale = 1.

        self.include_softmax_output = False
        self.do_split_training = False

        self.dev_use_true_labels = False
        self.treshold = None

    def dump(self):
        return vars(self)
