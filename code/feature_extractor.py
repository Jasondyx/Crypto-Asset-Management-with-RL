
import numpy as np
import pandas as pd
import os
import copy
import torch
from torch import nn
from torch.optim import Adam
from DNN import DNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, n_asset, filepath=None):
        super(FeatureExtractor, self).__init__()

        model = DNN(n_asset=n_asset)
        if filepath:
            model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
        model_list = list(model.children())
        self.conv1 = model_list[0]
        self.conv2 = model_list[1]
        self.LSTM = model_list[2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous().squeeze(3)
        _, (_, output) = self.LSTM(x)
        return output


