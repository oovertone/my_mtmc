# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/11/10 9:19
@Description: 训练数据集
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset


class My_Dataset(Dataset):
    """
    数据集类
    """

    def __init__(self, data_path_list, train_test):
        """
        初始化
        """
        # 遍历组合数据
        self.data = []
        for data_path in data_path_list:
            data = np.load(data_path).tolist()
            self.data += data
        self.data = np.array(self.data)

        if train_test == 'train':
            data_1 = list(self.data[self.data[:, -1] == 1, :])
            data_0 = list(self.data[self.data[:, -1] == 0, :])
            random.shuffle(data_0)
            data_0 = data_0[0:len(data_1)]
            self.data = data_0 + data_1
            random.shuffle(self.data)
            self.data = np.array(self.data)

        print(f'sum: {sum(self.data[:, -1])}')

    def __getitem__(self, index):
        """
        获取单个数据
        """
        feat = torch.from_numpy(self.data[index][0:-2]).float()
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
        self.fc1 = nn.Linear(12292, 1024)
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


def parse_args():
    """
    解析命令行参数
    """

    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='训练 batch_size（默认： 64）')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N', help='测试 batch_size（默认： 64）')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='训练次数（默认： 10）')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='学习率（默认： 1e-4）')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='学习率阶跃（默认： 0.7）')
    parser.add_argument('--no_cuda', action='store_true', default=False, help=' 关闭 CUDA')
    parser.add_argument('--dry_run', action='store_true', default=False, help='快速检查单通')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='随机种子（默认： 1）')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='打印训练状态批次间隔（默认： 1）')
    parser.add_argument('--save_model', action='store_true', default=False, help='保存当前模型')
    args = parser.parse_args()

    return args


def my_mse_loss(x, y, device):
    """
    自定义 mse loss
    """

    k = torch.ones([len(y), 1], device=device)
    k[y == 1] = 9

    return torch.mean(torch.pow((x - y) * k, 2))


def train(args, model, device, train_loader, optimizer, epoch):
    """
    训练数据
    """

    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.flatten())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), correct,
                len(train_loader.dataset), 100. * correct / len(train_loader.dataset)
            ))
            if args.dry_run:
                break


def test(model, device, test_loader):
    """
    测试数据
    """

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.flatten(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    """
    主程序
    """

    # 训练设置，命令行参数
    args = parse_args()

    # 检查是否使用 cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 随机种子
    torch.manual_seed(args.seed)

    # 设备
    device = torch.device("cuda" if use_cuda else "cpu")

    # 训练和测试的 batch_size
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # 使用 cuda
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 导入数据集
    # 测试集
    test_dir = './dataset/feat_label/test/'
    test_path_list = os.listdir(test_dir)
    test_path_list = [os.path.join(test_dir, i) for i in test_path_list]
    dataset_test = My_Dataset(test_path_list, 'test')
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    # 训练集
    train_dir = './dataset/feat_label/train/'
    train_path_list = os.listdir(train_dir)

    model = Net().to(device)  # 实例化 model

    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # 优化器
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 训练
    for epoch in range(1, args.epochs + 1):
        for train_path in train_path_list:
            dataset_train = My_Dataset([os.path.join(train_dir, train_path)], 'train')
            train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        scheduler.step()

    # 保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
