# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/15 11:06
@Description: 
"""
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../')
import utils


def get_gt_df_sc(cam_dir):
    """
    计算单个摄像机的轨迹
    """

    # 读取真值
    gt_df = utils.get_tracking_df(os.path.join(cam_dir, 'gt', 'gt.txt'))

    # 读取相机标定参数
    hm, ipm, dc = utils.get_calibration_parms(os.path.join(cam_dir, 'calibration.txt'))

    # 定义像素坐标
    gt_df = utils.define_img_coordinates(gt_df)

    # 畸变矫正
    gt_df = utils.distortion_correction(gt_df, ipm, dc)

    # 坐标变换
    gt_df = utils.calculate_world_coordinates(gt_df, hm)

    return gt_df


def main():
    """
    主程序
    """

    # 读取配置文件
    aic_configs = utils.get_aic_configs(Path(__file__).parents[2])

    # 读取场景内全部相机参数
    scene_dir = aic_configs['SCENE_DIR']
    cam_id_list = list(filter(lambda x: x.startswith('c'), os.listdir(scene_dir)))
    # cam_id_list = ['c018']

    gt_df_all = pd.DataFrame()
    for cam_id in cam_id_list:
        print(cam_id)

        # 相机目录
        cam_dir = aic_configs['SCENE_DIR'] + cam_id

        # 读取结果
        gt_df = get_gt_df_sc(cam_dir)
        gt_df['cam_id'] = cam_id
        gt_df = gt_df.rename(columns={'ID': 'car_id'})

        # 合并
        gt_df_all = gt_df_all.append(gt_df)

    # 多摄像头筛选
    gt_df_all = utils.multi_cam_filter(gt_df_all, 2)
    print(np.unique(gt_df_all.cam_id))
    car_id_list = np.unique(gt_df_all.car_id)

    # 画图
    plt.figure()
    plt.axis('equal')
    plt.scatter(gt_df_all['x_wrd'], gt_df_all['y_wrd'], s=1)
    plt.show()


if __name__ == '__main__':
    main()
