import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torch
from scipy.ndimage import imread
import os
import os.path
import random
import glob
import math
import numpy as np

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def make_dataset(dir,split = 0):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in glob.iglob(os.path.join(dir,'*_flow.flo')):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-9]
        img1 = root_filename+'_img1.ppm'
        img2 = root_filename+'_img2.ppm'
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([img1,img2,flow_map])

    assert(len(images) > 0)
    random.shuffle(images)
    split_index = math.floor(len(images)*split/100)
    assert(split_index >= 0 and split_index <= len(images))
    return images[:split_index], images[split_index+1:]


def default_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    return [imread(img) for img in imgs],load_flo(flo)

def flying_chairs(root, transform=None, target_transform=None,
                 co_transform=None, split = 80)
    train_list, test_list = make_dataset(root,split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list)

    return train_dataset, test_dataset

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None :
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)

class RandomBalancedSampler(Sampler):
    """Samples elements randomly, with an arbitrary size, independant from dataset length.
    this is a balanced sampling that will sample the whole dataset with a random permutation.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, epoch_size):
        self.data_size = len(data_source)
        self.epoch_size = epoch_size
        self.index = 0

    def __next__(self):
        if self.index == 0:
            #re-shuffle the sampler
            self.indices = torch.randperm(len(self.data_source))
        self.index = (self.index+1)%len(self.data_source)
        return self.indices[self.index]

    def __iter__(self):
        return self

    def __len__(self):
        return min(self.data_size),self.epoch_size if self.epoch_size>0 else self.data_size

class SequentialBalancedSampler(Sampler):
    """Samples elements dequentially, with an arbitrary size, independant from dataset length.
    this is a balanced sampling that will sample the whole dataset before resetting it.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, epoch_size):
        self.data_size = len(data_source)
        self.epoch_size = epoch_size
        self.index = 0

    def __next__(self):
        self.index = (self.index+1)%len(self.data_source)
        return self.index

    def __iter__(self):
        return self

    def __len__(self):
        return min(self.data_size,self.epoch_size) if self.epoch_size>0 else self.data_size