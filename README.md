# Novel-Class-Discriminative-Classifier-Detection

**This project support stream simulation with a few datasets and few detectors**

### Get started

```python

from stream_simulator import StreamSimulator
from config import Config

supported_datasets = ["Mnist", "FashionMnist", "KuzushijiMnist", "NotMnist", "Cifar10BW"]
supported_detectors = ["SoftmaxStats", "Discriminator"]

config = Config()
config.dataset_name = supported_datasets[0]
config.detector_name = supported_detectors[0]

config.train_epochs = 100
config.num_of_class_instances = 1000
config.repeat = 1

sim = StreamSimulator(config=config)
s = sim.stream()
s.print()

```
