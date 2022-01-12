import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import time
from datetime import datetime


# test start date setting
totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))
test_start_date = totimestamp('01/06/2021')


class MyDataset(Dataset):

    def __init__(self,
                 filepath='./data/train.csv',
                 asset_id=(1,6),
                 device='cpu',
                 lookback_period=3000,
                 target_scaler=100):
        data = pd.read_csv(filepath,
                           dtype={'Asset_ID': 'int8', 'Count': 'int32',
                                  'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64',
                                  'Volume': 'float64', 'VWAP': 'float64'
                                  })
        data.set_index('timestamp', inplace=True)
        data_shape0, data_shape1 = data.shape
        names = locals()
        min_ind, max_ind = 0, np.inf
        for i in asset_id:
            names['data'+str(i)] = data[data['Asset_ID'] == i]
            min_ind = max(min_ind, eval('data' + str(i)).dropna().index[0])
            max_ind = min(max_ind, eval('data' + str(i)).dropna().index[-1])
            names['data'+str(i)] = eval('data'+str(i)).reindex(range(eval('data'+str(i)).index[0],eval('data'+str(i)).index[-1]+60,60),method='pad').replace([np.inf, -np.inf], np.nan).fillna(method='ffill', axis=0)
            assert not np.any(np.isnan(eval('data'+str(i)).loc[min_ind:])) and not np.any(np.isinf(eval('data'+str(i)).loc[min_ind:]))  # no exceptional value

        # train-test-split
        max_ind = min(max_ind, test_start_date)

        self.X = np.zeros((data_shape1 - 2, int((max_ind - min_ind) / 60) + 1, len(asset_id)))
        self.y = np.zeros((int((max_ind - min_ind) / 60) + 1, len(asset_id)))
        for j, i in enumerate(asset_id):
            self.X[:, :, j] = eval('data' + str(i)).loc[min_ind:max_ind].iloc[:, 1:-1].T
            self.y[:, j] = eval('data' + str(i)).loc[min_ind:max_ind].iloc[:, -1]
        self.y[np.isnan(self.y)] = 0
        # to torch
        self.X = torch.from_numpy(self.X).float().to(device)
        self.y = torch.from_numpy(self.y).float().to(device)

        self.y = self.y * target_scaler
        self.n = lookback_period
        print(f'Data preprocess finished.')

    def __len__(self):
        return self.y.shape[0] - self.n + 1

    def __getitem__(self, item):

        X_ = self.X[:, item:item+self.n, :]
        y_ = self.y[item+self.n-1, :]
        return X_, y_


# class MyDatasetChunk(Dataset)


def customize_data(data, asset_id, device=torch.device('cpu')):
    data_shape0, data_shape1 = data.shape
    names = locals()
    # min_ind, max_ind = 0, np.inf
    # for i in asset_id:
    #     names['data' + str(i)] = data[data['Asset_ID'] == i]
    #     names['data' + str(i)] = eval('data' + str(i)).reindex(
    #         range(eval('data' + str(i)).index[0], eval('data' + str(i)).index[-1] + 60, 60), method='pad')
    #     min_ind = max(min_ind, eval('data' + str(i)).index[0])
    #     max_ind = min(max_ind, eval('data' + str(i)).index[-1])
    min_ind, max_ind = 0, np.inf
    for i in asset_id:
        names['data' + str(i)] = data[data['Asset_ID'] == i]
        min_ind = max(min_ind, eval('data' + str(i)).dropna().index[0])
        max_ind = min(max_ind, eval('data' + str(i)).dropna().index[-1])
        names['data' + str(i)] = eval('data' + str(i)).reindex(
            range(eval('data' + str(i)).index[0], eval('data' + str(i)).index[-1] + 60, 60), method='pad').replace(
            [np.inf, -np.inf], np.nan).fillna(method='ffill', axis=0)
        assert not np.any(np.isnan(eval('data' + str(i)).loc[min_ind:])) and not np.any(np.isinf(eval('data' + str(i)).loc[min_ind:]))  # no exceptional value

    X = np.zeros((data_shape1 - 2, int((max_ind - min_ind) / 60) + 1, len(asset_id)))
    for j, i in enumerate(asset_id):
        X[:, :, j] = eval('data' + str(i)).loc[min_ind:max_ind].iloc[:, 1:-1].T
    timestamp = [i for i in range(min_ind, max_ind+60, 60)]
    return torch.from_numpy(X).to(device), timestamp

if __name__ == '__main__':
    mydata = MyDataset()
    X, y = mydata[len(mydata)-1]
    mydataloader = DataLoader(mydata, batch_size=64, drop_last=True)


