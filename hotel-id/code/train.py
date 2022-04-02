import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset

torch.cuda.is_available()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

### Path to folder with input images
root_dir=r"D:\Kaggle\Hotel-ID\hotel-id-to-combat-human-trafficking-2022-fgvc9\train_images"



### Dataset

class FirstDataset(Dataset):
    def __init__(self,)

### Utilities


### CNN

class FirstNet(nn.Module):
    del __init__(self):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        # nn.Dropout2d(),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3),

    )


### Hyperparameters

### Training