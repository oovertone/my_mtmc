# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/31 15:23
@Description: 配置文件
"""

aic_configs = {
    'global_configs': {
        'SCENE_DIR': './dataset/AIC21_Track3_MTMC_Tracking/test/S06/',  # 场景目录
        'CPU_WORKER_NUM': 8,  # 处理器核数
        'GPU_ID': 0,  # GPU ID
    },
    'show_save_video_configs': {
        'CAM_NUM_MAX': 6,  # 最大显示视频数
        'NUM_COLS': 3  # 子窗口列数
    },
    'prepare_dataset_configs': {
        'CAM_ID_LIST': ['c041', 'c042'],  # 数据集中包含的相机列表
        'DOWN_SAMPLING_RATE': [0.01, 0.1],  # 下采样率 [0]：同相机车辆下采样率，[1]：异相机下采样率
        'TRAIN_VALI_TEST_RATE': [0.5, 0.25, 0.25],  # 训练集比例
        'SAVE_DIR': './dataset',  # 保存目录
        'BATCH_SIZE': [64, 1024]  # 批次大小：[0]：计算 reid 特征  [1]：保存数据集
    },
    'train_configs': {
        'BATCH_SIZE': 64,  # 训练 batch_size
        'TEST_BATCH_SIZE': 1024,  # 测试 batch_size
        'EPOCHS': 150,  # 训练次数
        'LEARNING_RATE': 1e-1,  # 学习率
        'GAMMA': 0.7,  # 学习率衰减系数
        'NO_CUDA': False,  # 关闭 CUDA
        'RANDOM_SEED': 1,  # 随机种子
        'SAVE_MODEL': True,  # 保存模型
        'TRAIN_DIR': './dataset/c041_c042_[0.01, 0.1]/feat_label/train/',  # 训练集目录
        'TEST_DIR': './dataset/c041_c042_[0.01, 0.1]/feat_label/test/',  # 测试集目录
        'TEST_SAMPLE_RATE': 1.0,  # 测试集采样率
        'ONLY_IMG_FEAT': False,  # 仅 img 特征
        'TRAIN_FILE_BATCH_SIZE': 30,  # 训练集文件 batch_size，每次读 batch_size 个文件进内存
    }
}

