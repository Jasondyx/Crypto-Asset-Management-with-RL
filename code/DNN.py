
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNN(nn.Module):
    def __init__(self, n_asset=2):
        super(DNN, self).__init__()
        # n_asset, n_lookback = 2, 3000
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=16,
                kernel_size=(15, 1),
                stride=(3, 1),
                dilation=(2, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(10, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(10, n_asset),
                stride=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.LSTM = nn.LSTM(
            input_size=32,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.Linear(100, n_asset)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1,2).contiguous().squeeze(3)
        _, (_, x) = self.LSTM(x)
        x = x.squeeze(0)  # ori: [1, 64, 256]
        # x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self, n_asset=14, filepath=None, device='cpu'):
        super(FeatureExtractor, self).__init__()
        self.device = device
        model = DNN(n_asset=n_asset).to(device)
        if filepath:
            model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
            print(f'load pretrained model: {filepath.split("/")[-1]} on device: {self.device}.')
        model_list = list(model.children())
        self.conv1 = model_list[0]
        self.conv2 = model_list[1]
        self.LSTM = model_list[2]

    def forward(self, x):
        x = x.to(self.device).float()
        # print(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous().squeeze(3)
        _, (_, output) = self.LSTM(x)
        output = output.squeeze(0)
        # output = output.flatten()
        return output


if __name__ == '__main__':
    # data
    asset_id = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    # asset_id = (1, 6)
    data = pd.read_csv('./data/supplemental_train.csv',
                       dtype={'Asset_ID': 'int8', 'Count': 'int32',
                              'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64',
                              'Volume': 'float64', 'VWAP': 'float64'
                              },
                       nrows=100000)
    data.set_index('timestamp', inplace=True)
    data_shape0, data_shape1 = data.shape
    names = locals()
    min_ind, max_ind = 0, np.inf
    for i in asset_id:
        names['data' + str(i)] = data[data['Asset_ID'] == i]
        names['data' + str(i)] = eval('data' + str(i)).reindex(
            range(eval('data' + str(i)).index[0], eval('data' + str(i)).index[-1] + 60, 60), method='pad')
        min_ind = max(min_ind, eval('data' + str(i)).index[0])
        max_ind = min(max_ind, eval('data' + str(i)).index[-1])
    X = np.zeros((data_shape1 - 2, int((max_ind - min_ind) / 60) + 1, len(asset_id)))
    y = np.zeros((int((max_ind - min_ind) / 60) + 1, len(asset_id)))
    for j, i in enumerate(asset_id):
        X[:, :, j] = eval('data' + str(i)).loc[min_ind:max_ind].iloc[:, 1:-1].T
        y[:, j] = eval('data' + str(i)).loc[min_ind:max_ind].iloc[:, -1]
    y[np.isnan(y)] = 0

    model = DNN(n_asset=len(asset_id))
    print(model)

    feature_extractor = FeatureExtractor(n_asset=len(asset_id))
    print(feature_extractor)

    X_ = X[:, :3000, :]
    X_ = torch.from_numpy(X_).float().unsqueeze(0)
    model(X_)
    feature_extractor(X_)

    print(0)