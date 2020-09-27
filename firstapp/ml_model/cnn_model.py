import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer

class TrainNet(pl.LightningModule):

  def training_step(self, batch, batch_idx):
    x, t = batch
    y = self.forward(x)
    loss = self.lossfun(y, t)
    results = {'loss': loss}
    return results

class ValidationNet(pl.LightningModule):

  def validation_step(self, batch, batch_idx):
    x, t = batch
    y = self.forward(x)
    loss = self.lossfun(y, t)
    y_label = torch.argmax(y, dim=1)
    acc = torch.sum(t == y_label) * 1.0 / len(t)
    results = {'val_loss': loss, 'val_acc': acc}
    return results

  def validation_end(self, outputs):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    results = {'val_loss': avg_loss, 'val_acc': avg_acc}
    return results

class Net(TrainNet, ValidationNet):

  def __init__(self):
    super().__init__()

    # # バッチサイズ
    # self.batch_size = batch_size

    # 画像解像度　torch.Size([3, 128, 128])
    self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1) # (128 + 2 - 3 / 1) + 1 = 128
    self.pool = nn.MaxPool2d(2, 2) # (128 - 2 / 2) + 1 = 64
    self.bn = nn.BatchNorm1d(3 * 64 * 64)  # 64 * 64 ピクセルの画像、特徴マップ3枚
    self.fc1 = nn.Linear(3 * 64 * 64, 100)
    self.fc2 = nn.Linear(100, 4)


  def lossfun(self, y, t):
    return F.cross_entropy(y, t)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.01,  weight_decay=0.001)

  def forward(self, x):
    x = self.conv(x)
    x = F.relu(x)
    x = self.pool(x)

    x = x.view(x.size(0), -1)
    x = self.bn(x)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x