import os

import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from helpers.utils import KneeInjuryDataset, KneeInjuryUnlabeledDataset, byol_transforms, data_transforms


def get_datasets(data_root, train_ratio, val_ratio, test_ratio, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_samples = []
    for folder in os.listdir(data_root):
        path = os.path.join(data_root, folder)
        if os.path.isdir(path):
            label = int(folder.split('_')[-1])  # Parse class from folder name
            all_samples.append((path, label))

    # Shuffle and stratified split
    random.shuffle(all_samples)
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    labels = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(1 - train_ratio),
        stratify=labels, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(
        test_ratio / (val_ratio + test_ratio)), stratify=y_temp, random_state=seed)

    train_samples = list(zip(X_train, y_train))
    val_samples = list(zip(X_val, y_val))
    test_samples = list(zip(X_test, y_test))

    train_dataset = KneeInjuryDataset(train_samples, transform=data_transforms())
    val_dataset = KneeInjuryDataset(val_samples, transform=data_transforms())
    test_dataset = KneeInjuryDataset(test_samples, transform=data_transforms())

    return train_dataset, val_dataset, test_dataset


def get_unl_dataset(data_root_unl):

    all_unl_samples = []
    for folder in os.listdir(data_root_unl):
        path = os.path.join(data_root_unl, folder)
        all_unl_samples.append(path)

    unl_dataset = KneeInjuryUnlabeledDataset(all_unl_samples, transform=byol_transforms())

    return unl_dataset


def get_weighted_sampler(dataset, num_classes):
    labels = [label for _, label in dataset]
    class_counts = [labels.count(c) for c in range(num_classes)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler
