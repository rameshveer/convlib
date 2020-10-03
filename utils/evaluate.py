import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def class_wise_accuracies(model, loader, device, classes):
    """Print test accuracy for each class in dataset.
    Args:
        model: Model instance.
        loader: Data loader.
        device: Device where data will be loaded.
        classes: List of classes in the dataset.
    """

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def plot_graph(values, metric):
    fig = plt.figure(figsize=(7, 5))

    plt.plot(values)
    plt.title(f'Validation{metric}')

    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend location
    location = 'upper' if metric == 'Loss' else 'lower'

    fig.savefig(f'{metric.lower()}_change.png')
