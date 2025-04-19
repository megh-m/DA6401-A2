#!pip install pytorch_lightning torchvision #Uncomment if needed
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import pandas as pd
import wandb

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"
wandb.init(project="DA6401_A2", name="Fine-Tuning")

# Dataset Setup
class DataManager(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder('/kaggle/input/inaturalist/inaturalist_12K/train', transform=self.train_transform)
        
        # Manual 80-20 split without sklearn
        size = len(full_dataset)
        idxs = list(range(size))
        np.random.shuffle(idxs)
        split = int(np.floor(0.2 * size)) #20% as val
        
        self.train_dataset = torch.utils.data.Subset(full_dataset, idxs[split:])
        self.val_dataset = torch.utils.data.Subset(full_dataset, idxs[:split])
        
        self.test_dataset = datasets.ImageFolder('/kaggle/input/inaturalist/inaturalist_12K/val', transform=self.val_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Model Definition
class FineTunedResNet(pl.LightningModule):
    def __init__(self, lr_backbone=1e-4, lr_classifier=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Loading pretrained ResNet50
        self.model = torchvision.models.resnet50(pretrained=True)
        
        # Freezing all layers except last block and classifier
        for name, param in self.model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
                
        # Replace final layer with series of dense 
        num_ftrs = self.model.fc.in_features 
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x): #Pre-defined name 
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Different learning rates for backbone and classifier
        optimizer = torch.optim.Adam([
            {'params': self.model.layer4.parameters(), 'lr': self.hparams.lr_backbone},
            {'params': self.model.fc.parameters(), 'lr': self.hparams.lr_classifier}
        ])
        return optimizer

# Training
#if __name__ == "__main__":
#wandb.login(key="eb9574fa5b11da36782604ea27df8bf1989ddefd")

dm = DataManager(batch_size=64)
model = FineTunedResNet()

trainer = pl.Trainer(
    max_epochs=15,
    logger=WandbLogger(project='fine-tuning'),
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='val_acc', mode='max')
    ],
    accelerator='auto'
)

trainer.fit(model, dm)

# Test best model
#trainer.test(dataloaders=dm.test_dataloader(), ckpt_path='best')
