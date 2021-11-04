# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/11/3 10:46
@Description: 数据集类
"""

from torch.utils.data import Dataset


class MyDataset(Dataset):  # 需要继承Dataset
    def __init__(self):
        # TODO

        # 1. 初始化文件路径或文件名列表。
        # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        pass

    def __getitem__(self, index):
        # TODO

        # 1.
        # 从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        # 2.
        # 预处理数据（例如torchvision.Transform）。
        # 3.
        # 返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data
        pass

    def __len__(self):
        # TODO

        # 应该将0更改为数据集的总大小。
        pass
