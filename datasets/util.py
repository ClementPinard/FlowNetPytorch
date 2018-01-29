import numpy as np


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples
