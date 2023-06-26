from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os 
import pandas as pd
from itertools import islice

class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=256, train=True):

        self.df = df
        self.trn = train
        #self.normalize_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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

        #img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self):
        return len(self.df)

path = 'data/GR_ICRGW_Dataset/archive'
contrails = os.path.join(path, "contrails/")
train_path = os.path.join(path, "train_df.csv")
valid_path = os.path.join(path, "valid_df.csv")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"

dataset_train = ContrailsDataset(train_df, 384, train=True)
dataset_validation = ContrailsDataset(valid_df, 384, train=False)

data_loader_train = DataLoader(
    dataset_train,
    batch_size=48,
    shuffle=True,
    num_workers=2,
)
data_loader_validation = DataLoader(
    dataset_validation,
    batch_size=128,
    shuffle=False,
    num_workers=2,
)

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     print(npimg)
#     plt.imshow(np.transpose(npimg, (1, 2, 0, 1)))

def main():
    # get some random training images
    dataiter = iter(data_loader_train)
    images, labels = dataiter.next()

    # show images
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(labels[0].numpy(), (0, 1)))

    plt.show()

if __name__ == "__main__":
    main()
