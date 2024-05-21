import numpy as np


def bmu_to_string(tensors):
    return "".join([str(tensor.item()) for tensor in tensors])


def convert_bmu_labels(bmu_labels):
    som_labels = list(set(bmu_labels.values()))
    som_labels_dict = dict(zip(som_labels, range(len(som_labels))))
    return np.array([som_labels_dict[bmu_labels[n]] for n in bmu_labels.keys()])
