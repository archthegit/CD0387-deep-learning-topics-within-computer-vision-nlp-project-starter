#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_corrects=0
    for (inputs, labels) in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
    total_acc = running_corrects / len(test_loader.dataset)
    print(f"Test set: Accuracy: {100*total_acc}")
    
def train(model, train_loader, epochs, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    count = 0
    for e in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            if count > 400:
                break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133))

    return model

def create_data_loaders(data, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
    trainset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    validset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    testset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=False),
        torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False))

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    
    train_loader, valid_loader, test_loader = create_data_loaders(args.train, args.batch_size, args.test_batch_size)
    
    model=train(model, train_loader, args.epochs, loss_criterion, optimizer)
    
    test(model, test_loader)
    
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=64,metavar="N",help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size",type=int,default=1000,metavar="N",help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs",type=int,default=14,metavar="N",help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    
    main(args)
