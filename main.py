from dataset import ExpWDataset
from model import Expression

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from apex import amp

import torch.nn as nn
import numpy as np
import shutil
import torch
import yaml
import time
import sys
import os


WEIGHTS = torch.Tensor([3670, 3994, 1087, 30536, 10558, 7059])

best_acc = float('-inf')
global_step = 0
num_steps = 0


def load_data(args):
    dataset = ExpWDataset(args['DATASET'], args['ANNOTATION'], args['INPUT_SIZE'], True)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    val_size = int(args['VAL_RATIO'] * len(dataset))
    val_idx, train_idx = indices[: val_size], indices[val_size:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=args['BS'], sampler=train_sampler, num_workers=args['NW'], pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args['BS'], sampler=val_sampler, num_workers=args['NW'], pin_memory=True)
    data_loaders = {'train': train_loader, 'val': val_loader}
    return data_loaders


def show_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr += param_group['lr']
    return lr


def train(model, criterion, data_loader, epoch, optimizer, apex):
    global global_step, num_steps
    num_steps = 0

    print('\n' + '-' * 10)
    print('Epoch: {}'.format(epoch))
    print('Current Learning rate: {}'.format(show_lr(optimizer)))

    model.train()

    timer = time.time()
    dataset_size = len(data_loader.dataset)
    train_loss, train_acc, processed_size = 0, 0, 0

    for inputs, labels in data_loader:
        # Forward
        inputs, labels = inputs.cuda(), labels.cuda()
        logits = model(inputs)
        # Compute loss
        softmax = criterion(logits)
        _, preds = torch.max(softmax.data, 1)
        loss = torch.zeros(1).cuda()
        for i in range(inputs.size(0)):
            a = torch.zeros(6).cuda()
            a[labels[i]] = 1
            loss -= torch.sum(softmax[i] * a) * WEIGHTS[preds[i]] / (torch.sum(WEIGHTS ** 2) / torch.sum(WEIGHTS)) / inputs.size(0)
        # loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Record and display data
        processing_size = len(inputs)
        processed_size += processing_size

        batch_loss = loss.item()
        train_loss += batch_loss * processing_size

        batch_acc = (preds == labels.data).sum().item() / processing_size
        train_acc += batch_acc * processing_size

        train_writer.add_scalar('loss', batch_loss, global_step)
        train_writer.add_scalar('acc', batch_acc, global_step)
        sys.stdout.write('\rProcess: [{:5.0f}/{:5.0f} ({:2.2%})] '
                         'Batch Loss: {:.4f} '
                         'Train Loss: {:.4f} '
                         'Batch Acc: {:.2%} '
                         'Train Acc {:.2%} '
                         'Estimated time: {:.2f}s'.format(
                            processed_size, dataset_size, processed_size / dataset_size,
                            float(batch_loss),
                            float(train_loss) / processed_size,
                            float(batch_acc),
                            float(train_acc) / processed_size,
                            (time.time() - timer))),
        sys.stdout.flush()
        global_step += 1
        num_steps += 1
        timer = time.time()

    # Record and display data
    print('\nTrain Loss: {:.4f} Train Acc: {:.2%}'.format(train_loss / processed_size, train_acc / processed_size))


def val(model, criterion, data_loader, epoch, save):
    global best_acc

    model.eval()

    with torch.no_grad():
        val_loss, val_acc, processed_size = 0, 0, 0
        for inputs, labels in data_loader:
            # Forward
            inputs, labels = inputs.cuda(), labels.cuda()
            logits = model(inputs)

            # Record and display data
            processing_size = len(inputs)
            processed_size += processing_size
            # Compute loss
            softmax = criterion(logits)
            _, preds = torch.max(softmax.data, 1)
            loss = torch.zeros(1).cuda()
            for i in range(inputs.size(0)):
                a = torch.zeros(6).cuda()
                a[labels[i]] = 1
                loss -= torch.sum(softmax[i] * a) * WEIGHTS[preds[i]] / (torch.sum(WEIGHTS ** 2) / torch.sum(WEIGHTS)) / inputs.size(0)
            # loss = criterion(logits, labels)
            val_loss += loss.item() * processing_size
            _, preds = torch.max(logits.data, 1)
            val_acc += (preds == labels.data).sum().item()

        # Record and display data
        val_loss /= processed_size
        val_acc /= processed_size
        val_writer.add_scalar('loss', val_loss, epoch * num_steps)
        val_writer.add_scalar('acc', val_acc, epoch * num_steps)
        print('Val Loss: {:.4f} Val Acc:{:.2%}'.format(val_loss, val_acc))

        # Save model
        if save and val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, 'best-epoch{}-{:.4f}.pt'.format(epoch, best_acc)))


def main(args):
    model = Expression(args['NC'], args['LAYERS'])
    if args['MODEL']:
        model = torch.load(args['MODEL'])
    else:
        print('train from scratch')
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['LR'], weight_decay=5e-5)
    if args['APEX']:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['STEP_SIZE'], gamma=args['GAMMA'])
    criterion = nn.LogSoftmax(dim=1)

    data_loaders = load_data(args)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:{:,} .'.format(total_params))
    print('total_trainable_parameters:{:,} .'.format(total_trainable_params))

    val(model, criterion, data_loaders['val'], 0, False)
    for epoch in range(args['NE']):
        torch.cuda.empty_cache()
        train(model, criterion, data_loaders['train'], epoch + 1, optimizer, args['APEX'])
        val(model, criterion, data_loaders['val'], epoch + 1, True)
        scheduler.step()


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    train_writer = SummaryWriter(log_dir=os.path.join('logs-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime()), 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join('logs-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime()), 'val'))

    save_path = './models-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    main(config)
