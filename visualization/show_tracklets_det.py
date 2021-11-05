# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/15 11:06
@Description: 展示检测的车辆轨迹
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../')
import utils


def get_ret_df_sc(cam_dir):
    """
    计算单个摄像机的轨迹
    """

    # 读取真值
    ret_df = pd.read_csv(os.path.join(cam_dir, 'mtsc', 'mtsc_tnt_mask_rcnn.txt'), header=None)
    ret_df.columns = ['frame_id', 'car_id', 'left', 'top', 'width', 'height', '1', '-1', '-1', '-1']

    # 读取相机标定参数
    hm, ipm, dc = utils.get_calibration_parms(os.path.join(cam_dir, 'calibration.txt'))

    # 定义像素坐标
    ret_df = utils.define_img_coordinates(ret_df)

    # 畸变矫正
    ret_df = utils.distortion_correction(ret_df, ipm, dc)

    # 坐标变换
    ret_df = utils.calculate_world_coordinates(ret_df, hm)

    return ret_df


def main():
    """
    主程序
    """

    # 读取配置文件
    aic_configs = utils.get_aic_configs(Path(__file__).parents[1])

    # 读取场景内全部相机参数
    scene_dir = aic_configs['global_configs']['SCENE_DIR']
    cam_id_list = list(filter(lambda x: x.startswith('c'), os.listdir(scene_dir)))

    ret_df_all = pd.DataFrame()
    for cam_id in cam_id_list:
        print(f'cam_id: {cam_id}')

        # 相机目录
        cam_dir = os.path.join(aic_configs['global_configs']['SCENE_DIR'], cam_id)

        # 读取结果
        ret_df = get_ret_df_sc(cam_dir)
        ret_df['cam_id'] = cam_id
        ret_df = ret_df.rename(columns={'ID': 'car_id'})

        # 合并
        ret_df_all = ret_df_all.append(ret_df)

    # 多摄像头筛选
    ret_df_all = utils.multi_cam_filter(ret_df_all, 6)

    # 画图
    plt.figure()
    plt.axis('equal')
    plt.scatter(ret_df_all['x_wrd'], ret_df_all['y_wrd'], s=1)
    plt.show()


if __name__ == '__main__':
    main()
