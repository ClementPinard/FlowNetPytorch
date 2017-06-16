import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import models
import datasets
from multiscaleloss import multiscaleloss
import balancedsampler
import csv
import os
import datetime

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))

dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
                    choices=dataset_names,
                    help='dataset type : ' +
                        ' | '.join(dataset_names) +
                        ' (default: flying_chairs)')
parser.add_argument('-s', '--split', default=80, type=float, metavar='%',
                    help='split percentage of train samples vs test (default: 80)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: flownets)')
parser.add_argument('--solver', default = 'adam',choices=['adam','sgd'],
                    help='solvers: adam | sgd')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run (default: 90')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay (default: 4e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default = None,
                    help='path to pre-trained model')
parser.add_argument('--log-summary', default = 'progress_log_summary.csv',
                    help='csv where to save per-epoch train and test stats')
parser.add_argument('--log-full', default = 'progress_log_full.csv',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--loss', default='L1', help='loss function to apply to multiScaleCriterion : L1 (default)| SmoothL1| MSE')
parser.add_argument('--div-flow', default = 20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')

best_EPE = -1

def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
                flow_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
                normalize
        ])
    target_transform = transforms.Compose([
                flow_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
        ])

    if 'KITTI' in args.dataset:
        co_transform=flow_transforms.Compose([
            flow_transforms.RandomCrop((320,448)),
            #random flips are not supported yet for tensor conversion, but will be
            #flow_transforms.RandomVerticalFlip(),
            #flow_transforms.RandomHorizontalFlip()
        ])
    else:
        co_transform=flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10,5),
            flow_transforms.RandomCrop((320,448)),
            #random flips are not supported yet for tensor conversion, but will be
            #flow_transforms.RandomVerticalFlip(),
            #flow_transforms.RandomHorizontalFlip()
        ])


    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=balancedsampler.RandomBalancedSampler(train_set,args.epoch_size),
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](args.pretrained).cuda()

    model = torch.nn.DataParallel(model).cuda()
    criterion = multiscaleloss(sparse = 'KITTI' in args.dataset, loss=args.loss).cuda()
    high_res_EPE = multiscaleloss(scales=1, downscale=4, weights=(1), loss='L1', sparse = 'KITTI' in args.dataset).cuda()
    cudnn.benchmark = True


    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                betas = (args.momentum, args.beta),
                                weight_decay=args.weight_decay)
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        best_EPE = validate(val_loader, model, criterion, high_res_EPE)
        return

    with open(os.path.join(save_path,args.log_summary), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss','train_EPE','EPE'])
    
    with open(os.path.join(save_path,args.log_full), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss','train_EPE'])

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, criterion, high_res_EPE, optimizer, epoch)

        # evaluate o validation set

        EPE = validate(val_loader, model, criterion, high_res_EPE)
        if best_EPE<0:
            best_EPE = EPE

        # remember best prec@1 and save checkpoint
        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
        }, is_best)


        with open(os.path.join(save_path,args.log_summary), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss,train_EPE,EPE])


def train(train_loader, model, criterion, EPE, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input = [j.cuda(0) for j in input]
        input_var = torch.autograd.Variable(torch.cat(input,1))
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        flow2_EPE = args.div_flow*EPE(output[0], target_var)
        # record loss and EPE
        losses.update(loss.data[0], target.size(0))
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(os.path.join(save_path,args.log_full), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.data[0],flow2_EPE.data[0]])

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs))


    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, criterion, EPE):
    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(torch.cat(input,1).cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        flow2_EPE = args.div_flow*EPE(output, target_var)
        # record EPE
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time,
                   flow2_EPE=flow2_EPEs))

    print(' * EPE {flow2_EPE.avg:.3f}'
          .format(flow2_EPE=flow2_EPEs))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 after 300K iterations, 400K and 500K"""
    if epoch == 100 or epoch == 150 or epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2

if __name__ == '__main__':
    main()
