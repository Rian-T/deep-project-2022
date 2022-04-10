from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder, MNIST

from .FrameSequence import FrameSequenceDataset

class MKFrameActionDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/MarioKartFrameAction", batch_size: int = 32
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform =  transforms.Compose([
            transforms.Grayscale(), 
            transforms.ToTensor(),
            transforms.Normalize(0.4505, 0.1786)
            ])

    def setup(self, stage: Optional[str] = None):
        self.mk_train = ImageFolder(self.data_dir + "/train", transform=self.transform)
        self.mk_test = ImageFolder(self.data_dir + "/test", transform=self.transform)

        self.mk_train, self.mk_val = random_split(
            self.mk_train, [len(self.mk_train) - 3000, 3000]
        )

    def train_dataloader(self):
        return DataLoader(self.mk_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mk_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mk_test, batch_size=self.batch_size, shuffle=False)

class MKFrameSequenceActionDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/MarioKartFrameSequence16", batch_size: int = 32
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform =  transforms.Compose([
            #transforms.Grayscale(), 
            transforms.ToTensor(),
            transforms.Normalize(0.4505, 0.1786)
            ])

    def setup(self, stage: Optional[str] = None):
        self.mk_train = FrameSequenceDataset(self.data_dir, train=True, transform=self.transform)
        self.mk_test = FrameSequenceDataset(self.data_dir, train=False, transform=self.transform)

        self.mk_train, self.mk_val = random_split(
            self.mk_train, [len(self.mk_train) - 3000, 3000]
        )

    def train_dataloader(self):
        return DataLoader(self.mk_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mk_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mk_test, batch_size=self.batch_size, shuffle=False)

# Sanity check:
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.mnist_train = MNIST(
            "datasets/MNIST", train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_test = MNIST(
            "datasets/MNIST", train=False, download=True, transform=transforms.ToTensor()
        )

        self.mnist_train, self.mnist_val = random_split(
            self.mnist_train, [len(self.mnist_train) - 3000, 3000]
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)