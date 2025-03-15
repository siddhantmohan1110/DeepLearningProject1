'''Train CIFAR10 with PyTorch.'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from all_transforms import *

import os
import argparse
from torchinfo import summary
import sys
import io

from datasets import *
from models import *
from utils import progress_bar

'''
Training hyperparameters
modresnet_type :
    Type 1 has 4.1M parameters with the four residual blocks having channels 64, 128, 192, 256
    Type 2 has 4.9M parameters with the four residual blocks having channels 64, 128, 208, 288
dropout : whether to add dropout layers in the ResNet
epochs : total epochs for training
weight_decay : weight decay for SGD
momentum : momentum for SGD
lr : learning rate for SGD
label_smoothing : label smoothing for the cross-entropy loss
'''

params = {
        'modresnet_type' : 1,
        'dropout' : 0,
        'epochs' : 2,
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.1,
        'label_smoothing' : 0.0,
    }

# Function for training the model
def train(model, epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch_data in enumerate(trainloader):
        inputs = batch_data[0].to(device)
        targets = batch_data[1].to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs) 
        loss = criterion(outputs, targets)
        
        #backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        #computing number of correct train predictions to get the train data accuracy
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # train data accuracy after each epoch
    train_acc = 100.*correct/total
    train_loss = train_loss/len(trainloader)
    return train_acc, train_loss


def test_during_training(model, epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs = batch_data[0].to(device)
            targets = batch_data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            #computing number of correct test predictions to get the train data accuracy
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    test_acc = 100.*correct/total
    test_loss = test_loss/len(testloader)

    if epoch%10==0 or test_acc>best_acc:
        print('Saving checkpoint')
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if epoch%10==0:
            torch.save(state, os.path.join(os.path.join(args.exp_name, args.exp_name+'_checkpoints'), 
                                           args.exp_name+'_ep{}.pth'.format(epoch)))
        
        if test_acc>best_acc:
            torch.save(state, os.path.join(os.path.join(args.exp_name, args.exp_name+'_checkpoints'), 
                                           args.exp_name+'_best.pth'.format(epoch)))
            best_acc = test_acc
    
    return test_acc, test_loss


def basic_plotter(tr_acc, ts_acc, tr_loss, ts_loss, save_path):

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].plot(tr_acc, color='tab:blue', label='Train')
    axes[0].plot(ts_acc, color='tab:orange', label='Test')
    axes[0].set_title('Accuracies v Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracies')
    axes[0].legend()

    axes[1].plot(tr_loss, color='tab:blue', label='Train')
    axes[1].plot(ts_loss, color='tab:orange', label='Test')
    axes[1].set_title('Losses v Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Losses')
    axes[1].legend()

    fig.savefig(os.path.join(save_path, args.exp_name+'_accs_losses.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--exp_name', default='exp9', type=str, help='Experiment name')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    directory = os.path.join(args.exp_name, args.exp_name+'_checkpoints')
    os.makedirs(directory, exist_ok=True)

    # Model
    print('==> Building model..')
    net = ModResNet18(type=params['modresnet_type'], dropout=params['dropout'],)
    net = net.to(device)

    # Get its summary
    with open(os.path.join(args.exp_name, args.exp_name+'_model_summary.txt'), 'w') as f, io.StringIO() as buf:
        sys.stdout = buf
        summary(net, input_size=(1, 3, 32, 32), col_names=('output_size', 'num_params', 'params_percent'), device=device)
        sys.stdout = sys.__stdout__
        f.write(buf.getvalue())
    print('Summary generated!')

    if device=='cuda' or device=='mps':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(os.path.join(args.exp_name, args.exp_name+'_checkpoints'), 
                                           args.exp_name+'_best.pth'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
    optimizer = optim.SGD(net.parameters(), lr=params['lr'],
                        momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch+1, params['epochs']+1):
        train_acc, train_loss = train(net, epoch)
        test_acc, test_loss = test_during_training(net, epoch)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss) 
        scheduler.step()
    
    basic_plotter(train_accuracies, test_accuracies, train_losses, test_losses, args.exp_name)
    
