import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from main import flow2rgb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--max_flow', default=None,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')

best_EPE = -1
n_iter = 0


def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    save_path = data_dir/'flow'
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*0.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + '1.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).cuda()
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']

    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.autograd.Variable(torch.cat([img1, img2],0).cuda(), volatile=True).unsqueeze(0)

        # compute output
        output = model(input_var)
        rgb_flow = flow2rgb(args.div_flow * output.data[0].cpu().numpy(), max_value=args.max_flow)
        to_save = (rgb_flow * 255).astype(np.uint8)
        imsave(save_path/(img1_file.namebase[:-2] + '_flow.png'), to_save)


if __name__ == '__main__':
    main()
