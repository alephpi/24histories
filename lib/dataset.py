# from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms 
import pytorch_lightning as L
import numpy as np
import cv2

from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import train_test_split



def stratified_split(dataset, ratio):
    X = np.array([i for i in range(len(dataset))])
    y = np.array(dataset.targets)
    train_idx, val_idx, _, _ = train_test_split(X,y, test_size=ratio, stratify=y)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset

class CharDataset(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", train_batch_size=256, val_test_batch_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage):
        train_val_set = datasets.ImageFolder(os.path.join(self.data_dir,'train'), 
                                  loader=lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), 
                                  transform=transforms.ToTensor())
        self.test_set = datasets.ImageFolder(os.path.join(self.data_dir,'test'),
                                  loader=lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), 
                                  transform=transforms.ToTensor())

        # train_dataset, val_dataset = random_split(charset, [0.9, 0.1])
        self.train_set, self.val_set = stratified_split(train_val_set, 0.1)

        # self.mnist_train = MNIST(self.data_dir, train=False, download=True)
        # self.mnist_train.targets = self.mnist_train.targets.to("cuda")
        # self.mnist_train.data = self.mnist_train.data.to("cuda")

    def train_dataloader(self, load_at_once=True):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
        # if load_at_once:
        #   for data, target in train_loader:
        #       data = data.to('cuda')
        #       target = target.to('cuda')
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, batch_size=self.val_test_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.val_test_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return test_loader
