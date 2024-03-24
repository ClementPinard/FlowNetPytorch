import numpy as np


def split2list(images, split, split_save_path, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == "1" for x in f.readlines()]
        assert len(images) == len(split_values)
    elif split is None:
        split_values = np.random.uniform(0, 1, len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0, 1, len(images)) < split
    if split_save_path is not None:
        with open(split_save_path, "w") as f:
            f.write("\n".join(map(lambda x: str(int(x)), split_values)))
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples
