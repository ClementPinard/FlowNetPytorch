import os.path
import random
import glob
import math
from .listdataset import ListDataset

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

        images.append([[img1,img2],flow_map])

    assert(len(images) > 0)
    random.shuffle(images)
    split_index = int(math.floor(len(images)*split/100))
    assert(split_index >= 0 and split_index <= len(images))
    return (images[:split_index], images[split_index:]) if split_index < len(images) else (images, [])



def flying_chairs(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    train_list, test_list = make_dataset(root,split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform)

    return train_dataset, test_dataset

