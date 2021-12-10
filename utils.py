#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/15 11:08
@Description: 
"""

import os
import random

import numpy as np
import pandas as pd
import cv2

import configs


def create_dir(dir_list):
    """
    创建目录
    """

    for dir_ in dir_list:
        if not os.path.isdir(dir_):
            print(f'创建目录：{dir_}')
            os.makedirs(dir_)


def get_aic_configs(prj_path):
    """
    读取配置文件
    """

    # 导入配置文件
    aic_configs = configs.aic_configs

    # 拼接路径
    aic_configs['global_configs']['SCENE_DIR'] = os.path.join(prj_path, aic_configs['global_configs']['SCENE_DIR'])

    return aic_configs


def get_calibration_parms(calibration_path):
    """
    读取相机标定参数
    """

    with open(calibration_path) as f:
        line_list = f.readlines()
        # hm: 单应性矩阵 homography matrix
        hm_str_list = line_list[0].split('\n')[0].split('Homography matrix: ')[1].split(';')
        hm = np.array(list(map(lambda x: np.array(x.split(' ')).astype(np.float64), hm_str_list)))

        if len(line_list) == 4:
            # ipm: 相机内参 intrinsic parameter matrix
            ipm_str_list = line_list[1].split('\n')[0].split('Intrinsic parameter matrix: ')[1].split(';')
            ipm = np.array(list(map(lambda x: np.array(x.split(' ')).astype(np.float64), ipm_str_list)))

            # dc: 畸变系数 distortion coefficients
            dc = np.array(
                line_list[2].split('\n')[0].split('Distortion coefficients: ')[1].split(' ')
            ).astype(np.float64)
        else:
            ipm = np.array([])
            dc = np.zeros(5)
    return hm, ipm, dc


def define_img_coordinates(df, method='center'):
    """
    定义像素坐标
    """

    if method == 'center':
        # 以bbox中心点作为目标像素坐标
        df['x_img'] = df.apply(lambda x: int(x.left + 0.5 * x.width), axis=1)
        df['y_img'] = df.apply(lambda x: int(x.top + 0.5 * x.height), axis=1)
    return df


def distortion_correction(df, ipm, dc):
    """
    畸变矫正
    """

    if len(ipm):
        df.loc[:, ['x_img', 'y_img']] = cv2.undistortPoints(
            src=np.array(df.loc[:, ['x_img', 'y_img']]).reshape(-1, 1, 2).astype(np.float64),
            cameraMatrix=ipm,
            distCoeffs=dc,
            P=ipm
        ).reshape(-1, 2)
    return df


def coordinate_transformation(x_input, y_input, H):
    """
    根据单应性矩阵进行坐标变换
    """

    A_1 = H[2, 0] * x_input - H[0, 0]
    B_1 = H[0, 1] - H[2, 1] * x_input
    C_1 = H[0, 2] - H[2, 2] * x_input

    A_2 = H[2, 0] * y_input - H[1, 0]
    B_2 = H[1, 1] - H[2, 1] * y_input
    C_2 = H[1, 2] - H[2, 2] * y_input

    y_output = (B_2 * C_1 - B_1 * C_2) / (A_1 * B_2 - A_2 * B_1)
    x_output = (A_1 * C_2 - A_2 * C_1) / (A_2 * B_1 - A_1 * B_2)

    return [x_output, y_output]


def calculate_world_coordinates(df, hm):
    """
    根据单应性矩阵计算坐标
    """

    df['xy_wrd'] = df.apply(lambda x: coordinate_transformation(x.x_img, x.y_img, hm), axis=1)
    df['x_wrd'] = df.xy_wrd.apply(lambda x: x[0])
    df['y_wrd'] = df.xy_wrd.apply(lambda x: x[1])
    return df


def cal_vector_similarity(v_1, v_2):
    """
    计算向量相似度
    s_1: 模长之比（小 / 大）
    s_2: 向量角余弦值（缩放至 0-1 ）
    """

    # 计算模长比
    l_1 = np.linalg.norm(v_1)
    l_2 = np.linalg.norm(v_2)
    if max([l_1, l_2]) > 0:
        s_1 = min([l_1, l_2]) / max([l_1, l_2])
    else:
        s_1 = 1

    # 计算向量角余弦值
    if l_1 * l_2 > 0:
        s_2 = np.dot(v_1, v_2) / (l_1 * l_2)
        s_2 = s_2 / 2 + 0.5
    else:
        if max([l_1, l_2]) == 0:
            s_2 = 1
        else:
            s_2 = 0

    return [s_1, s_2]


def multi_cam_filter(in_df, n):
    """
    筛选至少经过 n 个摄像头的数据
    df 须包含 cam_id 和 car_id
    """

    car_id_list = np.unique(in_df.car_id)
    out_df = pd.DataFrame()
    for car_id in car_id_list:
        img_sel_df = in_df.loc[in_df.car_id == car_id, :].reset_index(drop=True)
        cam_id_list = np.unique(img_sel_df.cam_id)
        if len(cam_id_list) >= n:
            out_df = out_df.append(img_sel_df)
            print(car_id)
    out_df = out_df.reset_index(drop=True)
    return out_df


def chunks(lst, batch_size):
    """
    按照 BATCH_SIZE 分批
    """

    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def normalization(lst_in):
    """
    数据归一化
    """
    lst_in_max = np.max(lst_in)
    lst_in_min = np.min(lst_in)
    lst_out = list(map(lambda x: (x - lst_in_min) / (lst_in_max - lst_in_min), lst_in))

    return lst_out


def false_2_true(x, thr):
    """
    如果 False，一定概率变成 True
    """

    if not x:
        x = x or (random.random() < thr)
    return x


def whether_same_cam(i, j):

    return 1


def whether_same_car(i, j):

    return 1