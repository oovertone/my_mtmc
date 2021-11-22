# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/11/10 9:19
@Description: 训练数据集
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('../')
import utils

# 读取配置文件
aic_configs = utils.get_aic_configs(Path(__file__).parents[1])
BATCH_SIZE = aic_configs['train_configs']['BATCH_SIZE']  # 训练 batch_size
TEST_BATCH_SIZE = aic_configs['train_configs']['TEST_BATCH_SIZE']  # 测试 batch_size
EPOCHS = aic_configs['train_configs']['EPOCHS']  # 训练次数
LEARNING_RATE = aic_configs['train_configs']['LEARNING_RATE']  # 学习率
GAMMA = aic_configs['train_configs']['GAMMA']  # 学习率衰减系数
NO_CUDA = aic_configs['train_configs']['NO_CUDA']  # 关闭 CUDA
RANDOM_SEED = aic_configs['train_configs']['RANDOM_SEED']  # 随机种子
SAVE_MODEL = aic_configs['train_configs']['SAVE_MODEL']  # 保存模型
TRAIN_DIR = aic_configs['train_configs']['TRAIN_DIR']  # 训练集目录
TEST_DIR = aic_configs['train_configs']['TEST_DIR']  # 测试集目录


class My_Dataset(Dataset):
    """
    数据集类
    """

    def __init__(self, data_path_list, train_test, rate=1.0):
        """
        初始化
        """
        # 遍历组合数据
        self.data = []
        for data_path in tqdm(data_path_list):
            data = np.load(data_path).tolist()
            random.shuffle(data)
            data = data[0:int(len(data) * rate)]
            self.data += data
        self.data = np.array(self.data)

        if train_test == 'train':
            data_1 = list(self.data[self.data[:, -1] == 1, :])
            data_0 = list(self.data[self.data[:, -1] == 0, :])

            # 平衡正负样本
            if len(data_1) >= len(data_0):
                data_m, data_l = data_1, data_0
            else:
                data_m, data_l = data_0, data_1

            random.shuffle(data_m)
            data_m = data_m[0:len(data_l)]
            self.data = data_l + data_m
            random.shuffle(self.data)
            self.data = np.array(self.data)

    def __getitem__(self, index):
        """
        获取单个数据
        """
        feat = torch.from_numpy(self.data[index][0:-2]).float()  # x,y,t,od,img 特征
        # feat = torch.from_numpy(self.data[index][4:-2]).float()  # img 特征
        label = torch.from_numpy(self.data[index][-1:]).long()

        return feat, label

    def __len__(self):
        """
        数据长度
        """

        return len(self.data)


class Net(nn.Module):
    """
    网络
    """

    def __init__(self):
        """
        初始化
        """

        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12292, 1024)  # x,y,t,od,img 特征
        # self.fc1 = nn.Linear(12288, 1024)  # img 特征
        self.fc2 = nn.Linear(1024, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def my_mse_loss(x, y, device):
    """
    自定义 mse loss
    """

    k = torch.ones([len(y), 1], device=device)
    k[y == 1] = 9

    return torch.mean(torch.pow((x - y) * k, 2))


def train(model, device, train_loader, optimizer):
    """
    训练
    """

    model.train()
    correct_num = 0
    loss_mean = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.flatten())
        loss.backward()
        loss_mean += loss.item()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct_num += pred.eq(target.view_as(pred)).sum().item()
    loss_mean = np.round(loss_mean / len(train_loader), 4)
    accuracy = int(100 * correct_num / len(train_loader.dataset))
    print(f'训练集：平均loss：{loss_mean}，准确率：{accuracy}% [{correct_num}/{len(train_loader.dataset)}]', end=' / ')


def test(model, device, test_loader):
    """
    测试
    """

    model.eval()
    loss_mean = 0
    correct_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_mean += F.nll_loss(output, target.flatten()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_num += pred.eq(target.view_as(pred)).sum().item()
    loss_mean = np.round(loss_mean / len(test_loader), 4)
    accuracy = int(100 * correct_num / len(test_loader.dataset))
    print(f'测试集：平均loss：{loss_mean}，准确率：{accuracy}% [{correct_num}/{len(test_loader.dataset)}]')


def main():
    """
    主程序
    """

    # 检查是否使用 cuda
    use_cuda = not NO_CUDA and torch.cuda.is_available()

    # 随机种子
    torch.manual_seed(RANDOM_SEED)

    # 设备
    device = torch.device("cuda" if use_cuda else "cpu")

    # 训练和测试的 batch_size
    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}

    # 使用 cuda
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 8,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 导入数据集
    # 测试集
    test_path_list = list(map(lambda x: os.path.join(TEST_DIR, x), os.listdir(TEST_DIR)))
    dataset_test = My_Dataset(test_path_list, 'test', 0.1)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    del dataset_test

    # 训练集
    train_path_list = list(map(lambda x: os.path.join(TRAIN_DIR, x), os.listdir(TRAIN_DIR)))
    train_path_list_2 = utils.chunks(train_path_list, 30)

    model = Net().to(device)  # 实例化 model

    # 优化器与调度器
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    # 训练
    epoch_mini = 10
    epoch_num = int(EPOCHS / epoch_mini)
    for i in range(epoch_num):
        print(f'\n[{i + 1}/{epoch_num}]')
        for j, train_path_list in enumerate(train_path_list_2):
            # 导入训练集
            print(f'\n[{j + 1}/{len(train_path_list_2)}]')
            dataset_train = My_Dataset(train_path_list, 'train')
            train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
            del dataset_train
            for k in range(epoch_mini):
                train(model, device, train_loader, optimizer)
                test(model, device, test_loader)
        scheduler.step()

    # 保存模型
    if SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(Path(TRAIN_DIR).parents[1], 'mtmc.pt'))


if __name__ == '__main__':
    main()
