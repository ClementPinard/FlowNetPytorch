import os.path
import glob
from .listdataset import ListDataset
from .util import split2list
import flow_transforms

'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The dataset is not very big, you might want to only pretrain on it for flownet
'''


def make_dataset(dataset_dir, split, dataset_type='clean'):
    flow_dir = 'flow'
    assert(os.path.isdir(os.path.join(dataset_dir,flow_dir)))
    img_dir = dataset_type
    assert(os.path.isdir(os.path.join(dataset_dir,img_dir)))

    images = []
    for flow_map in sorted(glob.glob(os.path.join(dataset_dir,flow_dir,'*','*.flo'))):
        flow_map = os.path.relpath(flow_map,os.path.join(dataset_dir,flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        frame_nb = int(frame_nb)
        img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
        img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        flow_map = os.path.join(flow_dir,flow_map)
        if not (os.path.isfile(os.path.join(dataset_dir,img1)) and os.path.isfile(os.path.join(dataset_dir,img2))):
            continue
        images.append([[img1,img2],flow_map])

    return split2list(images, split, default_split=0.87)


def mpi_sintel_clean(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split, 'clean')
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset


def mpi_sintel_final(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split, 'final')
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset


def mpi_sintel_both(root, transform=None, target_transform=None,
                    co_transform=None, split=None):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    assert(isinstance(split, str)), 'To avoid data snooping, you must provide a static list of train/val when dealing with both clean and final.'
    ' Look at Sintel_train_val.txt for an example'
    train_list1, test_list1 = make_dataset(root, split, 'clean')
    train_list2, test_list2 = make_dataset(root, split, 'final')
    train_dataset = ListDataset(root, train_list1 + train_list2, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list1 + test_list2, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset
