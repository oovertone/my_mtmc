# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/11/10 9:19
@Description: 训练数据集
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
TEST_SAMPLE_RATE = aic_configs['train_configs']['TEST_SAMPLE_RATE']  # 测试集下采样率
ONLY_IMG_FEAT = aic_configs['train_configs']['ONLY_IMG_FEAT']  # 仅 img 特征
TRAIN_FILE_BATCH_SIZE = aic_configs['train_configs']['TRAIN_FILE_BATCH_SIZE']  # 训练集文件 batch_size，每次读 batch_size 个文件进内存


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
        self.test_data = []
        for data_path in tqdm(data_path_list):
            data = np.load(data_path).tolist()
            random.shuffle(data)
            data = data[0:int(len(data) * rate)]
            self.data += data
        self.data = np.array(self.data)

        if train_test == 'train':
            # 新增异同相机，异同车辆分类
            car_1_flag = self.data[:, -1] == 1  # 两个样本是同一车辆
            car_0_flag = self.data[:, -1] == 0  # 两个样本是不同车辆
            same_diff = self.data[:, -2] == 1  # 两个样本是同一相机
            diff_cam_flag = self.data[:, -2] == 0  # 两个样本是不同相机
            data_same_1 = list(self.data[same_diff & car_1_flag, :])
            data_same_0 = list(self.data[same_diff & car_0_flag, :])
            data_diff_1 = list(self.data[diff_cam_flag & car_1_flag, :])
            data_diff_0 = list(self.data[diff_cam_flag & car_0_flag, :])
            # data_1 = list(self.data[self.data[:, -1] == 1, :])
            # data_0 = list(self.data[self.data[:, -1] == 0, :])
            # data_same = list(self.data[self.data[:, -2] == 1, :])
            # data_diff = list(self.data[self.data[:, -2] == 0, :])

            # 平衡所有类别样本
            len_data_min = np.min([len(data_same_1), len(data_same_0), len(data_diff_1), len(data_diff_0)])

            random.shuffle(data_same_1)
            data_same_1 = data_same_1[0:len_data_min]

            random.shuffle(data_same_0)
            # data_same_0 = data_same_0[0:len_data_min]  # 减少负样本的训练数据
            data_same_0 = data_same_0[0:int(np.floor(len_data_min / 2))]  # 减少负样本的训练数据

            random.shuffle(data_diff_1)
            data_diff_1 = data_diff_1[0:len_data_min]

            random.shuffle(data_diff_0)
            # data_diff_0 = data_diff_0[0:len_data_min]  # 减少负样本的训练数据
            data_diff_0 = data_diff_0[0:int(np.floor(len_data_min / 2))]  # 减少负样本的训练数据

            self.data = data_same_1 + data_same_0 + data_diff_1 + data_diff_0
            random.shuffle(self.data)
            self.data = np.array(self.data)

    def __getitem__(self, index):
        """
        获取单个数据
        """
        feat_index_start = 4 if ONLY_IMG_FEAT else 0
        feat = torch.from_numpy(self.data[index][feat_index_start:-2]).float()
        same_diff = torch.from_numpy(self.data[index][-2:-1]).long()
        label = torch.from_numpy(self.data[index][-1:]).long()

        return feat, same_diff, label

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
        feat_len = 12288 if ONLY_IMG_FEAT else 12292
        self.fc1 = nn.Linear(feat_len, 1024)
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


def my_loss():
    """
    自定义 mse loss
    """

    pass


def cal_loss_num(label, same_diff, output, label_assign=-1, same_diff_assign=-1):
    """
    计算 loss num
    """

    pred = output.argmax(dim=1, keepdim=True)  # 计算预测值
    idx_label, idx_same_diff = (_ := torch.ones(len(pred)) > 0), _.clone()  # 初始化布尔索引

    # 筛选
    if (label_assign > -1) and (same_diff_assign > -1):
        idx_label = label == label_assign
        idx_same_diff = same_diff == same_diff_assign
    idx = idx_label & idx_same_diff
    pred_sel = pred[idx]
    label_sel = label[idx]
    output_sel = output[idx.flatten()]

    loss = F.nll_loss(output_sel, label_sel.flatten())
    num_total = len(pred_sel)
    num_correct = pred_sel.eq(label_sel.view_as(pred_sel)).sum().item()

    loss_num_dict = {
        'label_assign': label_assign,
        'same_diff_assign': same_diff_assign,
        'loss': loss,
        'num_total': num_total,
        'num_correct': num_correct
    }

    return loss_num_dict


def print_loss_acc(loss_num_list, train_test):
    """
    打印 loss acc
    """

    def pla(df, label_assign=-1, same_diff_assign=-1):
        """
        打印结果
        """

        loss_mean = np.round(np.mean(df.loss), 4)
        num_correct = np.sum(df.num_correct)
        num_total = np.sum(df.num_total)
        acc = np.round(num_correct / num_total * 100, 4)

        if (label_assign > -1) and (same_diff_assign > -1):
            print(f'{train_test}：同一辆车：{label_assign}， 同一相机:{same_diff_assign}， 平均loss：{loss_mean}，准确率：{acc} % [{num_correct} / {num_total}]')
        else:
            print(f'{train_test}：总体： 平均loss：{loss_mean}，准确率：{acc} % [{num_correct} / {num_total}]')

    print('-' * 100)
    # 总体结果
    loss_num_list = list(filter(lambda x: ~np.isnan(x['loss'].item()), loss_num_list))  # 剔除 nan
    loss_num_df = pd.DataFrame.from_records(loss_num_list)
    loss_num_df['loss'] = loss_num_df['loss'].apply(lambda x: x.item())
    pla(loss_num_df)

    # 分类结果
    assign = [0, 1]
    for label_assign in assign:
        for same_diff_assign in assign:
            loss_num_df_sel = loss_num_df.loc[
                (loss_num_df.label_assign == label_assign) & (loss_num_df.same_diff_assign == same_diff_assign), :
                ]
            pla(loss_num_df_sel, label_assign, same_diff_assign)
    print('-' * 100)


def train(model, device, train_loader, optimizer):
    """
    训练
    """

    model.train()

    loss_num_list = []
    for batch_idx, (feat, same_diff, label) in enumerate(train_loader):
        feat, same_diff, label = feat.to(device), same_diff.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feat)

        # 分类计算
        assign = [0, 1]
        for label_assign in assign:
            for same_diff_assign in assign:
                loss_num_dict = cal_loss_num(label, same_diff, output, label_assign, same_diff_assign)
                loss_num_list.append(loss_num_dict)

        # 总体计算
        loss_num_dict = cal_loss_num(label, same_diff, output)
        loss = loss_num_dict['loss']
        loss.backward()
        optimizer.step()
    print_loss_acc(loss_num_list, '训练集')


def test(model, device, test_loader):
    """
    测试
    """

    model.eval()
    loss_num_list = []
    with torch.no_grad():
        for feat, same_diff, label in test_loader:
            feat, same_diff, label = feat.to(device), same_diff.to(device), label.to(device)
            output = model(feat)
            # 分类计算
            assign = [0, 1]
            for label_assign in assign:
                for same_diff_assign in assign:
                    loss_num_dict = cal_loss_num(label, same_diff, output, label_assign, same_diff_assign)
                    loss_num_list.append(loss_num_dict)
        print_loss_acc(loss_num_list, '测试集')


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
    dataset_test = My_Dataset(test_path_list, 'test', TEST_SAMPLE_RATE)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    del dataset_test

    # 训练集
    train_path_list = list(map(lambda x: os.path.join(TRAIN_DIR, x), os.listdir(TRAIN_DIR)))
    train_path_list_2 = utils.chunks(train_path_list, TRAIN_FILE_BATCH_SIZE)

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
                print('=' * 100)
                train(model, device, train_loader, optimizer)
                test(model, device, test_loader)
                print('=' * 100)
        scheduler.step()

    # 保存模型
    if SAVE_MODEL:
        model_name = 'mtmc_no_ts.pt' if ONLY_IMG_FEAT else 'mtmc.pt'
        torch.save(model.state_dict(), os.path.join(Path(TRAIN_DIR).parents[1], model_name))


if __name__ == '__main__':
    main()
