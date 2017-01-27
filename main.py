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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on FlyingChairs')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-s', '--split', default=80, type=float, metavar='%',
                    help='split percentage of train samples vs test (default: 80)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='FlowNetS',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: flownets)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run (default: 90')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default = None,
                    help='path to pre-trained model')

best_EPE = -1


def main():
    global args, best_EPE
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()
    criterion = multiscaleloss().cuda()
    high_res_EPE = multiscaleloss(scales=1, downscale=4, weights=(1), loss='L1').cuda()
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.FlyingChairs(
        args.data,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]),
        target_transform=None,
        co_transform=flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomCropRotate(10,360,5),
            flow_transforms.RandomCrop((320,448))
        ]),
        split=args.split
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True)
    dataset.eval()
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True)


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        best_EPE = validate(val_loader, model, criterion, high_res_EPE)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, high_res_EPE, optimizer, epoch)

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
            'state_dict': model.state_dict(),
            'best_EPE': best_EPE,
        }, is_best)


def train(train_loader, model, criterion, EPE, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to train mode
    train_loader.dataset.train()
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(torch.cat(input,1))
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        flow2_EPE = EPE(output[0], target_var)
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

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs))


def validate(val_loader, model, criterion, EPE):
    batch_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    val_loader.dataset.eval()
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(torch.cat(input,1), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        flow2_EPE = EPE(output[0], target_var)
        # record loss and EPE
        losses.update(loss.data[0], target.size(0))
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   flow2_EPE=flow2_EPEs))

    print(' * EPE {flow2_EPE.avg:.3f}'
          .format(flow2_EPE=flow2_EPEs))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    """Sets the learning rate to the initial LR decayed by 3 every 15 epochs"""
    lr = args.lr * (0.3 ** (epoch // 15))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
