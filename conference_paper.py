import matplotlib.pyplot as plt
from dataset import MnistDataset
from config import Config
import numpy as np
import matplotlib.cm as cm


def plot_imgs():

    ds = MnistDataset(Config())
    ds.x_train = ds.x_train[:1]
    ds.y_train = ds.y_train[:1]
    x_a, _ = ds.get_augmented()


    plt.subplot(1, 2, 1)
    plt.imshow(ds.x_train[0].squeeze(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(x_a[0].squeeze(), cmap="gray")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(6, 3)

    plt.savefig("./img/mnist_augmented.svg")
    plt.close("all")

    num_of_instances = 1000
    #colors = cm.rainbow(np.linspace(0, 1, 10))
    #colors = [[0, 0, 0, 1]] + [list(x[:3]) + [0.6] for x in colors]

    colors = ["#212121", "#3E2723", "#BF360C", "#E65100", "#FF6F00", "#33691E", "#1B5E20", "#004D40", "#01579B", "#311B92"]
    train_classes = np.arange(0, 5)
    stream_classes = np.arange(5, 10)

    for t in train_classes:
        plt.hlines(y=t, xmin=0, xmax=num_of_instances, linewidth=10, color=colors[t])

    for i, s in enumerate(stream_classes):
        step = i
        plt.hlines(y=s, xmin=step*num_of_instances, xmax=(step+1)*num_of_instances, linewidth=10, color=colors[s])

        t = train_classes[i]
        plt.hlines(y=t, xmin=step*num_of_instances, xmax=(step+1)*num_of_instances, linewidth=10, color=colors[t])

    classes = np.random.choice(np.arange(10), 10, replace=False)
    plt.legend(classes, loc="best", title="Class")

    leg = plt.gca().get_legend()
    for i, h in enumerate(reversed(leg.legendHandles)):
        h.set_color(colors[i])

    plt.xlim(0, num_of_instances*6)
    plt.yticks(np.arange(10), labels=reversed(classes))
    plt.ylabel("Class")
    plt.xlabel("Number of instances")
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.savefig("./img/instances_over_time.svg")
    plt.close("all")


if __name__ == "__main__":
    plot_imgs()