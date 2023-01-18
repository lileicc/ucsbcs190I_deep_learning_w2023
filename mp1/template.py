#!/usr/bin/env python
# coding: utf-8


import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


# set the random seed for reproduction 
SEED=190
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# checking if GPU is available or not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# set directory 
main_folder = 'celeba-dataset/'
images_folder = main_folder + 'img_align_celeba/img_align_celeba/'
comp_path = 'cs190-winter23-deep-learning-mp1'

# set hyperparameter
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
NUM_EPOCH = 20
LEARNING_RATE = 0.001

########################### Dataset ###########################

class CelebADataset(Dataset):
    """
        A customized dataset to load the CelebA image dataset.
    """
    def __init__(self, img_path, imgs, labels, resize=None, inference=False):
        """
            img_path: str, the directory of celeba dataset 
            imgs: List[str], the image file names
            labels: List[int], the 0/1 label for each image
            resize: None or int, whether downsample or upsample the image to certain size
            inference: bool, True for the data without the label
        """
        self.img_path = img_path
        self.resize = resize
        self.imgs = imgs
        self.labels = labels
        self.inference = inference
        
        # Center crop the alingned celeb dataset to 178x178 to include the face area 
        # and then downsample to 128x128 .
        self.pre_process = transforms.Compose([
                                            transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128,128)),
                                            ])

                          
        # first transform the images to tensor format, then normalize the pixel values
        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
        
        if resize is not None:
            self.resampling = transforms.Resize((resize, resize))
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, self.imgs[index])
        img = Image.open(image_path).convert('RGB')
        img = self.pre_process(img)
        img_tensor = self.totensor(img)
        if self.resize is not None:
            img_tensor = self.resampling(img_tensor)
        if not self.inference:
            label = self.labels[index]
            return img_tensor, label
        else:
            return img_tensor
        
    def __len__(self):
        return len(self.imgs)



class TestDataset(Dataset):
    """
        The test dataset module
    """
    def __init__(self, imgs, resize=None):
        """
            Similar with CelebADataset 
                imgs: open images
                resize: None or int, whether downsample or upsample the image to certain size
        """
        self.imgs = imgs
        self.resize = resize
        
        # Center crop the alingned dataset to 178x178 to include the face area 
        # and then downsample to 128x128 .
        self.pre_process = transforms.Compose([
                                            transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128,128)),
                                            ])

                          
        # first transform the images to tensor format, then normalize the pixel values
        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
        
        if resize is not None:
            self.resampling = transforms.Resize((resize, resize))
    
    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.pre_process(img)
        img_tensor = self.totensor(img)
        if self.resize is not None:
            img_tensor = self.resampling(img_tensor)
        return img_tensor
        
    def __len__(self):
        return len(self.imgs)


########################### Trainer ###########################


class Trainer(object):
    """
        A learning pipeline to train and validate the model.
    """
    def __init__(self, model, criterion, optimizer, max_epoch):
        """
            model: nn model
            criterion: loss function
            optimizer: optimizer
            max_epoch: maximum training epoch
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        
    def run(self,train_loader, valid_loader):
        """
            Main entry
                train_loader: training dataset, each item is (img, label)
                valid_loader: validation dataset, each item is (img, label)
        """
        # calculate the inital loss and accu on validation set
        valid_best_loss = self.validate(-1, valid_loader, best_loss=None)
        for epoch in range(self.max_epoch):
            self.train(epoch, train_loader)
            # save the checkpoint with the lowest validation loss
            valid_best_loss = self.validate(epoch, valid_loader, valid_best_loss)
        
    def train(self, epoch, loader):
        """
            Single training loop
                epoch: int, current epoch index
                loader: training loader
        """
        # switch to the evaluation mode, do not calculate the gradient
        self.model.train()
        running_loss, total, correct = 0.0, 0, 0
        with tqdm(enumerate(loader, 0), mininterval=10) as tepoch:
            for i, data in tepoch:
                # get the inputs; data is a list of [inputs, labels]
                # inputs: tensor, (batch_size, image_size, image_size)
                # labels: tensor, (batch_size, 1)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                ########################################################
                # TODO: replace the outputs and loss and update optimizer
                # 1. zero the parameter gradients
                # 2. forward + backward
                # 3. update the parameters
                outputs = None
                loss = None
                ########################################################
                
                # calculate the metric
                match, number = self.cal_metric(outputs.data, labels)
                
                # gather statistics
                total += number
                correct += match
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        running_loss /= len(loader)

        print('Training | Epoch: {}| Loss: {:.3f} | Accuracy on train images: {:.1f}'.format(epoch+1, running_loss, 100 * correct / total))
        
    def validate(self, epoch, loader, best_loss=None):
        """
            Single evaluation loop
                epoch: int, current epoch index
                loader: validation loader
                best_loss: float, current best loss
        """
        # switch to the evaluation mode, do not need to calculate the gradient
        self.model.eval()
        running_loss, total, correct = 0.0, 0, 0
        for i, data in tqdm(enumerate(loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            ########################################################
            # TODO: replace the outputs and loss  
            outputs = None
            loss = None
            ########################################################

            # calculate the metric
            match, number = self.cal_metric(outputs.data, labels)
            
            # gather statistics
            total += number
            correct += match
            running_loss += loss.item()

        running_loss /= len(loader)

        if best_loss is None or running_loss < best_loss:
            # if a better loss appears, save the checkpoint
            save_file = 'best_epoch{}_loss{:.2f}_accu{:.2f}.pt'.format(epoch+1, running_loss, 100 * correct / total)
            print('Save to file: ', save_file)
            torch.save(self.model, save_file)
            
            # overwrite the best_checkpoint.pt file
            torch.save(self.model, 'best_checkpoint.pt')
            
            best_loss = running_loss

        print('Validation | Epoch: {}| Loss: {:.3f} | Accuracy on val images: {:.1f}'.format(epoch+1, running_loss,100 * correct / total))

        return best_loss

                
    def cal_metric(self, outputs, labels):
        """
            Calculate the accuracy
                outputs: tensor (batch_size, number_class), the output of the model
                labels: tensor (batch_size, 1), the ground truth
        """
        # compare predictions to ground truth
        _, predicted = torch.max(outputs, 1)
        number = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct, number


########################### Model ###########################


class MLP(nn.Module):
    """
        Multilayer perceptron network
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*128*128, 2)
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        # F.log_softmax returns the log probabilities of each class
        # of shape (num_samples, num_classes)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    """
        LeNet architecture
    """
    def __init__(self):
        super().__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size = 5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        ########################################################
        # TODO: replace the input_size
        # figure out the input dimension of the first linear layer
        self.fc1 = torch.nn.Linear(input_size, 120)  
        ########################################################
        
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        
        ########################################################
        # TODO: organize the forward pass
        # Hint: 
        #     1. check the LeNet link above if you are not familiar with it
        #     2. do not forget the activation function F.relu() 
        #     3. you may want to use torch.flatten() before the full connection layer
        #     4. be careful with the dimension
        ########################################################
        
        return x


########################### utils ###########################


def predict(model_path, test_file):
    """
        Load the model and use it to predict test file 
    """
    test_data = torch.load(test_file)
    test_dataset = TestDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=TEST_BATCH_SIZE
                )
    
    model = torch.load(model_path)
    model.eval()
    preds = []
    with torch.no_grad():
        # labels are not available for the actual test set
        for feature in tqdm(test_loader):
            # calculate outputs by running images through the network
            outputs = model(feature.to(device))
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())

    return preds







if __name__ == "__main__":

    # create the dataset
    train_df = pd.read_csv(os.path.join(comp_path, 'celeba_train.csv'))
    valid_df = pd.read_csv(os.path.join(comp_path, 'celeba_valid.csv'))
    train_ds = CelebADataset(images_folder, train_df['id'], train_df['label'])
    valid_ds = CelebADataset(images_folder, valid_df['id'], valid_df['label'])

    # build the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=TEST_BATCH_SIZE
    )


    # define the model
    model = MLP()
    print(model)
    print('Model Parameters ', sum(p.numel() for p in model.parameters()))
    print('Trainable Parameters ', sum(p.numel() for p in model.parameters() if p.requires_grad))


    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    trainer = Trainer(model, criterion, optimizer, max_epoch=NUM_EPOCH)
    trainer.run(train_loader, valid_loader)


    # load and predict
    model_path = "best_checkpoint.pt"
    test_file = os.path.join(comp_path, "test_img.pt")
    preds = predict(model_path,test_file)
    df = pd.DataFrame({'id': list(range(len(preds))),'label': preds})
    df.to_csv('submission.csv', index=False)

