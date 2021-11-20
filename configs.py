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
        'DOWN_SAMPLING_RATE': 0.1,  # 下采样率
        'TRAIN_VALI_TEST_RATE': [0.5, 0.25, 0.25],  # 训练集比例
        'SAVE_DIR': './dataset',  # 保存目录
        'BATCH_SIZE': [64, 1024]  # 批次大小
    }
}
