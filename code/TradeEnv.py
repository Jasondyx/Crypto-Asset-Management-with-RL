import random
import json
import gym
import pandas as pd
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from data_preprocess import MyDataset, customize_data
import torch


class TradeEnv(gym.Env):
    """A market environment for OpenAI gym"""

    def __init__(self,
                 rnd_seed=0,
                 test=False,
                 datafile='./data/train.csv',
                 asset_list=list(range(0, 14)),
                 lookback_period=3000,
                 lookforward_period=60*5,
                 rebalance_period=60*24,
                 device=torch.device('cpu'),
                 outdir=None):
        super(TradeEnv, self).__init__()
        # self.action_space = spaces
        # self.observation_space = spaces.Box(low=-0.5, high=0.5, shape=(ASSET_NUM, TRACE_BACK_DAYS), dtype=np.float16)
        np.random.seed(rnd_seed)
        self.test = test
        self.asset_list = asset_list
        self.lookback_period = lookback_period
        self.lookforward_period = lookforward_period  # for reward calculation
        self.rebalance_period = rebalance_period
        self.device = device
        self._load_data(datafile)
        self.position = np.zeros(len(asset_list))
        self.curr_ind, self.start_ind = 0, 0
        self.ave_return_list = []  # for test
        self.outdir = outdir

    def _load_data(self, datafile):
        totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))
        data = pd.read_csv(datafile).set_index('timestamp')
        test_start_date = totimestamp('01/06/2021')
        if self.test:
            data = data.loc[test_start_date:]
        else:
            data = data.loc[:test_start_date]
        self.data, self.timestamp = customize_data(data, self.asset_list, self.device)
        self.data_len = self.data.shape[1]
        print(f'TradeEnv has loaded data length of {self.data_len}.')

    def _get_state(self):
        return self.data[:, self.curr_ind-self.lookback_period:self.curr_ind, :]

    def _get_reward(self):
        price_list = self.data[4, self.curr_ind:self.curr_ind+self.lookforward_period, :].numpy()
        return_list = np.diff(price_list, n=1, axis=0) / price_list[:-1, :] - 1
        ave_return_list = (self.position * return_list).sum(axis=1)
        # semi_std = ave_return_list[ave_return_list < 0].std()  # future downside deviation
        semi_std = (ave_return_list * (ave_return_list < 0)).std() * np.sqrt(360)  # future downside deviation
        done = self.curr_ind+self.lookforward_period > self.data_len
        return -semi_std, done

    def _store_performance(self):
        ave_return = ((self.data[4, self.curr_ind, :] / self.data[4, self.curr_ind-1, :]).numpy() * self.position).sum() - 1
        self.ave_return_list.append(ave_return)

    def step(self, action):
        # action = torch.from_numpy(softmax(action)).to(self.device)
        action = softmax(action)
        if (self.curr_ind-self.start_ind) % self.rebalance_period == 0:
            self.position = action
        if self.test:
            self._store_performance()
            print(f'Testing {self.curr_ind}/{self.data_len} data points.') if self.curr_ind % 100 == 0 else None
            self.render() if self.curr_ind % 100000 == 0 else None
        self.curr_ind = self.curr_ind + 1
        obs = self._get_state()
        reward, done = self._get_reward()
        if self.test and done:
            self.render()
        return obs, reward, done, {}

    def reset(self, min_episode_step=100):
        self.position[:] = 0
        self.curr_ind = np.random.randint(self.lookback_period, self.data_len-self.lookforward_period-min_episode_step) if not self.test else self.lookback_period
        self.start_ind = self.curr_ind
        return self._get_state()

    def render(self):
        ave_return_list = np.array(self.ave_return_list)
        value_list = np.cumprod(1+ave_return_list)
        semi_std = (ave_return_list*(ave_return_list < 0)).std() * np.sqrt(360)
        print(f'Downside deviation of testing period={semi_std*100:.2f}%.')
        plt.clf()
        plt.plot(value_list)
        plt.savefig(f'{self.outdir}/value_list.png')
        np.save(f'{self.outdir}/value_list', value_list)


if __name__ == '__main__':
    env = TradeEnv()
    print(0)