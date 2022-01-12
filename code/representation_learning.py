
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from data_preprocess import MyDataset
from DNN import DNN
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
asset_list = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
mydata = MyDataset(filepath='./data/train.csv', asset_id=asset_list, device=device, lookback_period=3000)
print(f'Load data with length of {len(mydata)}, n_batch of {len(mydata)/64}, and asset list of {asset_list}.')
mydata_loader = DataLoader(mydata, batch_size=64)

model = DNN(n_asset=len(asset_list))
model.to(device)
# load trained model
model.load_state_dict(torch.load('./result/representation_learning/model_state_i_batch_9000_time_2021-12-17 07_55', map_location=torch.device(device)))

optimizer = Adam(model.parameters(), lr=0.0001)
loss_func = nn.MSELoss(reduction='sum')

i_batch_t = 'last_batch'
for i_batch, (X_batch, y_batch) in enumerate(mydata_loader):
    # for test
    # if not i_batch>9610:  # 9618
    #     print(f'i_batch={i_batch}.') if i_batch%1000==0 else None
    #     continue

    assert not torch.any(torch.isnan(X_batch)) and not torch.any(torch.isnan(y_batch))

    y_pre_batch = model(X_batch)
    loss = loss_func(y_pre_batch, y_batch)
    if i_batch % 1 == 0:
        print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i_batch % 1000 == 0:
        print(f'i_batch={i_batch}. Save model.')
        torch.save(model.state_dict(), f'./result/representation_learning/model_state_time_{str(datetime.datetime.now())[:16].replace(":", "_")}_i_batch_{i_batch}')
        # model.load_state_dict(torch.load(f'./result/representation_learning/model_state_{str(datetime.datetime.now())[:16].replace(":", "_")}'))

    if torch.isnan(loss) or loss < 1e-10:
        print(f'Loss = {loss}.')
        i_batch_t = i_batch
        break

torch.save(model.state_dict(), f'./result/representation_learning/model_state_time_{str(datetime.datetime.now())[:16].replace(":","_")}_i_batch_{i_batch_t}')
print(0)






