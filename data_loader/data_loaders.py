from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
import pandas as pd
import os

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            # transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            # transforms.RandomRotation(10),     # Rotates the image to a specified angel
            # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ContrailsDataset(torch.utils.data.Dataset):
    """
    Google Research contrils dataset 
    """
    def __init__(self, df, image_size=384, train=True):

        self.df = df
        self.trn = train
        self.normalize_image = transforms.Normalize((0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225))
        self.image_size = image_size
        if image_size != 256:
            self.resize_image = transforms.transforms.Resize(image_size)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        con_path = row.path
        con = np.load(str(con_path))

        img = con[..., :-1]
        label = con[..., -1]

        label = torch.tensor(label)

        img = torch.tensor(np.reshape(img, (256, 256, 3))).to(torch.float32).permute(2, 0, 1)

        if self.image_size != 256:
            img = self.resize_image(img)

        img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self): 
        return len(self.df)

class ContrailsDataLoader(BaseDataLoader):
    """
    Google Research Contrils data loading using BaseDataLoader
    """ 
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        path = 'data/GR_ICRGW_Dataset/archive'
        contrails = os.path.join(path, "contrails/")
        train_path = os.path.join(path, "train_df.csv")
        train_df = pd.read_csv(train_path)
        train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
        # self.dataset = Subset(ContrailsDataset(train_df, 384, train=True),np.arange(1,101))
        self.dataset = ContrailsDataset(train_df, 384, train=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def split_validation(self):
        path = 'data/GR_ICRGW_Dataset/archive'
        contrails = os.path.join(path, "contrails/")
        valid_path = os.path.join(path, "valid_df.csv")
        valid_df = pd.read_csv(valid_path)
        valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"
        #return Subset(ContrailsDataset(valid_df, 384, train=False),np.arange(1,101))
        return ContrailsDataset(valid_df, 256, train=False)