import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os, sys, random
import shutil
import argparse
import time
import logging
import math

from resnet import *
from data import *

import torchvision.models.utils as utils
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Exit-Ensemble Distillation Training CIFAR100')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--data-dir', default='data', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--arch', metavar='ARCH', default='EED_ResNet18',
                        help='model architecture')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')

    parser.add_argument('--use_EED', default='True',
                        help='use exit-ensemble distillation')
    parser.add_argument('--loss_output', choices=['KL', 'MSE'], default='MSE',
                        help='loss function for output distillation')
    parser.add_argument('--use_feature_dist', default=True,
                        help='use feature distillation')

    #kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(args.arch, args.resume))
        run_test(args)

def run_test(args):
    model = ResNet()
    model = model.cuda()

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
            exit()

    cudnn.benchmark = True

    test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()
    validate(args, test_loader, model, criterion)

def run_training(args):
    model = ResNet()
    model = model.cuda()
    best_prec1 = 0

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True
    train_loader = prepare_cifar100_train_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    MSEloss = nn.MSELoss(reduction='mean').cuda()

    end = time.time()
    model.train()
    step = 0
    for current_epoch in range(args.start_epoch, args.epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        total_losses = AverageMeter()
        middle1_top1 = AverageMeter()
        middle2_top1 = AverageMeter()
        middle3_top1 = AverageMeter()

        adjust_learning_rate(args, optimizer, current_epoch)
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            target = target.squeeze().long().cuda()
            input = Variable(input).cuda()

            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            middle1_loss = criterion(middle_output1, target)
            middle2_loss = criterion(middle_output2, target)
            middle3_loss = criterion(middle_output3, target)
            L_C = loss + middle1_loss + middle2_loss + middle3_loss

            if args.use_EED:
                target_output = (middle_output1/4 + middle_output2/4 + middle_output3/4 + output/4).detach()
                target_fea = (middle1_fea/4 + middle2_fea/4 + middle3_fea/4 + final_fea/4).detach()
            else:
                target_output = output.detach()
                target_fea = final_fea.detach()

            if args.loss_output == 'KL':
                temp = target_output / args.temperature
                temp = torch.softmax(temp, dim=1)
                loss1by4 = kd_loss_function(middle_output1, temp, args) * (args.temperature**2)
                loss2by4 = kd_loss_function(middle_output2, temp, args) * (args.temperature**2)
                loss3by4 = kd_loss_function(middle_output3, temp, args) * (args.temperature**2)
                L_O = 0.1 * (loss1by4 + loss2by4 + loss3by4)
                if args.use_EED:
                    loss4by4 = kd_loss_function(output, temp, args) * (args.temperature**2)
                    L_O += 0.1 * loss4by4

            elif args.loss_output == 'MSE':
                loss_mse_1 = MSEloss(middle_output1, target_output)
                loss_mse_2 = MSEloss(middle_output2, target_output)
                loss_mse_3 = MSEloss(middle_output3, target_output)
                L_O = loss_mse_1 + loss_mse_2 + loss_mse_3
                if args.use_EED:
                    loss_mse_4 = MSEloss(output, target_output)
                    L_O += loss_mse_4
            total_loss = L_C + L_O

            if args.use_feature_dist:
                feature_loss_1 = feature_loss_function(middle1_fea, target_fea)
                feature_loss_2 = feature_loss_function(middle2_fea, target_fea)
                feature_loss_3 = feature_loss_function(middle3_fea, target_fea)
                L_F = feature_loss_1 + feature_loss_2 + feature_loss_3
                if args.use_EED:
                    feature_loss_4 = feature_loss_function(final_fea, target_fea)
                    L_F += feature_loss_4
                total_loss += L_F

            total_losses.update(total_loss.item(), input.size(0))

            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))

            middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
            middle3_top1.update(middle3_prec1[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                current_epoch,
                                i,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=total_losses,
                                top1=top1)
                )
        prec1 = validate(args, test_loader, model, criterion, current_epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print("best: ", best_prec1)
        if is_best:
            checkpoint_path = os.path.join(args.save_path, 'model_best.path.tar'.format(current_epoch))
            save_checkpoint({
                'epoch': current_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best, filename=checkpoint_path)
        torch.cuda.empty_cache()

def validate(args, test_loader, model, criterion, current_epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        target = target.squeeze().long().cuda()
        input = Variable(input).cuda()

        output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info("Loss {loss.avg:.3f}\t"
                 "Prec@1 {top1.avg:.3f}\t"
                 "Middle1@1 {middle1_top1.avg:.3f}\t"
                 "Middle2@1 {middle2_top1.avg:.3f}\t"
                 "Middle3@1 {middle3_top1.avg:.3f}\t".format(
                    loss=losses,
                    top1=top1,
                    middle1_top1=middle1_top1,
                    middle2_top1=middle2_top1,
                    middle3_top1=middle3_top1))

    model.train()
    return top1.avg

def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).mean()

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

def adjust_learning_rate(args, optimizer, epoch):
    if 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
    elif epoch >=180:
        lr = args.lr * (args.step_ratio ** 3)
    else:
        lr = args.lr

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)

if __name__ == '__main__':
    main()
