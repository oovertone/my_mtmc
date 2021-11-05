# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/29 14:18
@Description: 划分数据集
"""

import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('../')
import utils
from config import cfg
from reid.reid import ReidFeature

# 读取配置文件
aic_configs = utils.get_aic_configs(Path(__file__).parents[2])

CPU_WORKER_NUM = aic_configs['global_configs']['CPU_WORKER_NUM']  # 处理器核数
DOWN_SAMPLING_RATE = aic_configs['prepare_dataset_configs']['DOWN_SAMPLING_RATE']  # 下采样率
TRAIN_VALI_TEST_RATE = aic_configs['prepare_dataset_configs']['TRAIN_VALI_TEST_RATE']  # 训练集比例
SAVE_DIR = aic_configs['prepare_dataset_configs']['SAVE_DIR']  # 保存路径
GPU_ID = aic_configs['global_configs']['GPU_ID']  # 保存路径
BATCH_SIZE = aic_configs['prepare_dataset_configs']['BATCH_SIZE']  # 批次大小


def get_cam_parms_dict(cam_dir):
    """
    读取单个相机参数
    """

    # 相机编号
    cam_id = os.path.split(cam_dir)[-1]

    # 视频帧率
    video = cv2.VideoCapture(os.path.join(cam_dir, 'vdo.avi'))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    # 起始时间
    scene = os.path.split(Path(cam_dir).parents[0])[-1]
    cam_timestamp_path = os.path.join(Path(cam_dir).parents[2], "cam_timestamp", f'{scene}.txt')
    cam_timestamp_df = pd.read_csv(cam_timestamp_path, sep=' ', header=None)
    cam_timestamp_df.columns = ['cam_id', 'cam_timestamp']
    cam_timestamp = np.array(cam_timestamp_df.cam_timestamp)[cam_timestamp_df.cam_id == cam_id][0]

    # 相机标定参数
    hm, ipm, dc = utils.get_calibration_parms(os.path.join(cam_dir, 'calibration.txt'))

    # 合并相机参数
    cam_parms_dict = {
        'cam_id': cam_id,
        'cam_timestamp': cam_timestamp,
        "fps": fps,
        'frame_count': frame_count,
        'hm': hm,
        'ipm': ipm,
        'dc': dc,
        'mean': [0.5, 0.5, 0.5],
        'std': [0, 0, 0]
    }
    return cam_parms_dict


def get_lane_color_df(lane_color_path):
    """
    获取车道颜色表
    """

    # 读取车道颜色表
    lane_color_df = pd.read_csv(lane_color_path, header=None)
    lane_color_df.columns = ['cam_id', 'lane_id', 'r', 'g', 'b']

    # 重新计算车道 id
    cam_id_list = np.unique(lane_color_df.cam_id)
    max_lane_id = -1
    for cam_id in cam_id_list:
        lane_color_df.loc[lane_color_df.cam_id == cam_id, 'lane_id'] = \
            np.array(lane_color_df.loc[lane_color_df.cam_id == cam_id, 'lane_id']) + max_lane_id + 1
        max_lane_id = max(np.array(lane_color_df.loc[lane_color_df.cam_id == cam_id, 'lane_id']))

    # 计算 [b, g, r]
    lane_color_df['bgr'] = lane_color_df.apply(lambda x: [x.b, x.g, x.r], axis=1)

    # 删除 r, g, b 列
    lane_color_df.drop(['r', 'g', 'b'], axis=1, inplace=True)

    return lane_color_df


def get_mtmc_df(aic_configs):
    """
    获取 mtmc_df 结果
    """

    mtmc_df = pd.read_csv(os.path.join(aic_configs['SCENE_DIR'], 'track3.txt'), sep=' ', header=None)
    mtmc_df.columns = ['cam_id', 'car_id', 'frame_id', 'left', 'top', 'width', 'height', '-1', '-1']
    mtmc_df['cam_id'] = mtmc_df.cam_id.apply(lambda x: 'c0' + str(x))
    mtmc_df.frame_id = mtmc_df.frame_id - 1
    mtmc_df = mtmc_df.sort_values(by='frame_id').reset_index(drop=True)

    return mtmc_df


def cal_lane_id(lane_color_df, cam_id, pixel):
    """
    计算车辆所在车道
    """

    lane_color_sel_df = lane_color_df.loc[lane_color_df.cam_id == cam_id, :].reset_index(drop=True)

    # 计算向量相似度
    lane_color_sel_df['s_1_2'] = lane_color_df.bgr.apply(lambda x: utils.cal_vector_similarity(x, pixel))
    lane_color_sel_df['s'] = lane_color_sel_df.s_1_2.apply(lambda x: x[0] * x[1])

    # 相似度筛选
    thr = 0.8
    lane_color_sel_df = lane_color_sel_df.loc[lane_color_sel_df.s > thr, :].reset_index(drop=True)
    if len(lane_color_sel_df):
        lane_color_sel_df.sort_values(by='s', ascending=False, inplace=True)
        lane_id = lane_color_sel_df.lane_id[0]
    else:
        lane_id = -1

    return lane_id


def get_img_df(args):
    """
    获取单个相机数据集
    """

    # 解析参数
    cam_dir = args[0]
    lane_color_df = args[1]
    mtmc_df = args[2]
    cam_parms_df = args[3]

    # 相机编号
    cam_id = os.path.split(cam_dir)[-1]

    # 车道颜色图
    lane_color_img = cv2.imread(os.path.join('./lane', f'{cam_id}.jpg'))

    # 保存路径
    save_dir = os.path.join(Path(__file__).parents[0], './dataset/img', cam_id)

    # 真值
    mtmc_df = mtmc_df.loc[mtmc_df.cam_id == cam_id, :].reset_index(drop=True)

    # 视频
    video = cv2.VideoCapture(os.path.join(cam_dir, 'vdo.avi'))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_img_dict(lane_color_df, save_dir, cam_id, lane_color_img, frame_id, frame, x):
        """
        提取单张目标图片信息
        """

        # 图片保存路径
        save_path = os.path.join(save_dir, f'{cam_id}_{frame_id}_{x.car_id}.jpg')

        # 提取并保存图片
        if not os.path.isfile(save_path):
            img = frame[x.top:x.top + x.height, x.left:x.left + x.width, :]
            cv2.imwrite(save_path, img)

        # 计算所在车道
        img_coor = [int(x.left + 0.5 * x.width), int(x.top + 0.5 * x.height)]

        pixel = lane_color_img[img_coor[1], img_coor[0], :]
        lane_id = cal_lane_id(lane_color_df, cam_id, pixel)

        img_dict = {
            'cam_id': cam_id,
            'frame_id': frame_id,
            'car_id': x.car_id,
            'bbox': [[x.left, x.top], [x.left + x.width, x.top + x.height]],
            'lane_id': lane_id,
            'img_path': save_path,
        }
        return img_dict

    # 每一帧提取数据
    img_dict_list = []
    frame_all = np.zeros([height, width, 3])
    for frame_id in tqdm(range(frame_count)):
        _, frame = video.read()  # 读取单帧图像
        frame_all += frame
        mtmc_sel_df = mtmc_df.loc[mtmc_df.frame_id == frame_id, :].reset_index(drop=True)
        if len(mtmc_sel_df):
            if not os.path.isdir(save_dir):  # 目录不存在则创建
                os.makedirs(save_dir)
            img_dict_list += list(mtmc_sel_df.apply(lambda x: get_img_dict(
                lane_color_df, save_dir, cam_id, lane_color_img, frame_id, frame, x
            ), axis=1))
    img_df = pd.DataFrame.from_records(img_dict_list)
    img_df['timestamp'] = img_df.frame_id.apply(
        lambda x: x * 1 / np.array(cam_parms_df.fps)[cam_parms_df.cam_id == cam_id][0] +
                  np.array(cam_parms_df.cam_timestamp)[cam_parms_df.cam_id == cam_id][0]
    )

    # 计算相机画面均值和标准差
    frame_all /= (frame_count * 255)
    mean = [np.mean(frame_all[0]), np.mean(frame_all[1]), np.mean(frame_all[2])]
    std = [np.std(frame_all[0]), np.std(frame_all[1]), np.std(frame_all[2])]
    mean_std = [mean, std]

    # 去黑
    img_df = img_df.loc[img_df.lane_id > -1, :].reset_index(drop=True)

    # 定义像素坐标
    img_df['x_img'] = img_df.bbox.apply(lambda x: 0.5 * (x[0][0] + x[1][0]))
    img_df['y_img'] = img_df.bbox.apply(lambda x: 0.5 * (x[0][1] + x[1][1]))

    # 畸变矫正
    ipm = np.array(cam_parms_df.ipm)[cam_parms_df.cam_id == cam_id][0]
    dc = np.array(cam_parms_df.dc)[cam_parms_df.cam_id == cam_id][0]
    img_df = utils.distortion_correction(img_df, ipm, dc)

    # 计算 gps 坐标
    hm = np.array(cam_parms_df.hm)[cam_parms_df.cam_id == cam_id][0]
    img_df = utils.calculate_world_coordinates(img_df, hm)

    return [img_df, mean_std]


def cal_od_matrix(lane_color_df, df):
    """
    计算 od 矩阵
    """

    # 初始化 od 矩阵
    lane_id_list = np.unique(lane_color_df.lane_id)
    od_matrix = np.zeros([len(lane_id_list), len(lane_id_list)])

    # 计算每辆车的 od 链
    car_id_list = np.unique(df.car_id)
    for car_id in tqdm(car_id_list):
        img_sel_df = df.loc[df.car_id == car_id, :].sort_values(by='timestamp').reset_index(drop=True)

        for i in range(len(img_sel_df)):
            lane_id_i = np.array(img_sel_df.lane_id)[i]
            for j in range(i + 1, len(img_sel_df)):
                lane_id_j = np.array(img_sel_df.lane_id)[j]
                od_matrix[lane_id_i, lane_id_j] += 1

    # 归一化
    for i in range(len(od_matrix)):
        if sum(od_matrix[i, :]):
            od_matrix[i, :] /= sum(od_matrix[i, :])
        else:
            od_matrix[i, :] = 0
    od_matrix[np.isnan(od_matrix)] = 0

    return od_matrix


def down_sampling(img_df):
    """
    下采样
    """

    ds_df = pd.DataFrame()
    cam_id_list = np.unique(img_df.cam_id)
    for cam_id in cam_id_list:
        img_sel_df = img_df.loc[img_df.cam_id == cam_id, :]
        car_id_list = np.unique(img_sel_df.car_id)
        for car_id in car_id_list:
            img_sel_sel_df = img_sel_df.loc[img_sel_df.car_id == car_id, :]
            if len(img_sel_sel_df) * DOWN_SAMPLING_RATE < 1:
                rate = 1.1 / len(img_sel_sel_df)
            else:
                rate = DOWN_SAMPLING_RATE
            sel_df, _ = train_test_split(img_sel_sel_df, test_size=1 - rate)
            ds_df = ds_df.append(sel_df)
    ds_df = ds_df.reset_index(drop=True)

    return ds_df


def cal_span_dict(ds_df, lane_color_df):
    """
    计算平均区域最大跨度
    """

    ts_feat_dict = dict()
    cam_id_list = np.unique(ds_df.cam_id)
    for cam_id_i in cam_id_list:
        ds_df_i = ds_df.loc[ds_df.cam_id == cam_id_i, :]
        for cam_id_j in cam_id_list:
            key = cam_id_i + cam_id_j

            ds_df_j = ds_df.loc[ds_df.cam_id == cam_id_j, :]
            ds_df_ij = ds_df_i.append(ds_df_j).reset_index(drop=True)
            # 去重
            ds_df_ij.drop_duplicates(['img_path'], keep='first', inplace=True)

            # x 跨度
            span_x = np.max(ds_df_ij.x_wrd) - np.min(ds_df_ij.x_wrd)

            # y 跨度
            span_y = np.max(ds_df_ij.y_wrd) - np.min(ds_df_ij.y_wrd)

            # 时间跨度
            span_list = []
            car_id_list = np.unique(ds_df_ij.car_id)
            for car_id in car_id_list:
                ds_sel_df = ds_df_ij.loc[ds_df_ij.car_id == car_id, :]
                span_list.append(max(ds_sel_df.timestamp) - min(ds_sel_df.timestamp))
            span_t = int(np.mean(span_list))

            # 速度阈值
            v_thr = geodesic(
                (np.min(ds_df_ij.y_wrd), np.min(ds_df_ij.x_wrd)), (np.max(ds_df_ij.y_wrd), np.max(ds_df_ij.x_wrd))
            ).m / span_t

            # od 矩阵
            print(f'计算 od matrix, {cam_id_i}-{cam_id_j}')
            od_matrix = cal_od_matrix(lane_color_df, ds_df_ij)

            # reid 特征跨度
            reid_feat_array = np.array(list(
                ds_df_ij.apply(lambda x: list(x.reid_feat_1) + list(x.reid_feat_2) + list(x.reid_feat_3), axis=1)
            ))
            reid_feat_min = np.min(reid_feat_array, axis=0)
            reid_feat_max = np.max(reid_feat_array, axis=0)

            ts_feat_dict[key] = {
                'span_x': span_x,
                'span_y': span_y,
                'span_t': span_t,
                'v_thr': v_thr,
                'od_matrix': od_matrix,
                'reid_feat': {
                    'reid_feat_min': reid_feat_min,
                    'reid_feat_max': reid_feat_max,
                    'span_reid_feat': reid_feat_max - reid_feat_min
                }
            }

    return ts_feat_dict


def cal_reid_feat(ds_df, cfg_path_list, cam_parms_df):
    """
    提取图片 reid 特征
    """
    reid_feat_name = ['reid_feat_1', 'reid_feat_2', 'reid_feat_3']

    # 分摄像头计算
    out_df = pd.DataFrame()
    cam_id_list = np.unique(ds_df.cam_id)
    for cam_id in cam_id_list:
        print(cam_id)
        ds_sel_df = ds_df.loc[ds_df.cam_id == cam_id, :]
        # 图像均值和方差
        mean = np.array(cam_parms_df['mean'])[cam_parms_df.cam_id == cam_id][0]
        std = np.array(cam_parms_df['std'])[cam_parms_df.cam_id == cam_id][0]
        for i, cfg_path in enumerate(cfg_path_list):
            # 导入配置文件
            cfg.merge_from_file(cfg_path)
            cfg.freeze()

            # reid 模型
            model = ReidFeature(GPU_ID, cfg, mean, std)

            # 分批
            image_path_list = list(ds_sel_df.img_path)
            image_path_bl_list = utils.chunks(image_path_list, BATCH_SIZE)

            # 提取特征
            reid_feat_list = []
            for image_path_bl in tqdm(image_path_bl_list):
                reid_feat_list += list(model.extract(image_path_bl))
            ds_sel_df[reid_feat_name[i]] = reid_feat_list

        # 拼接
        out_df = out_df.append(ds_sel_df)

    return out_df


def cal_save_dataset(ds_df, batch_size, span_dict):
    """
    两两计算数据集并保存
    """

    # 初始化
    dataset_all_tr_va_te_list_dict = {
        'dataset_all_list': [],
        'dataset_train_list': [],
        'dataset_vali_list': [],
        'dataset_test_list': [],
    }
    num_tr_va_te_dict = {
        'num_train': 0,
        'num_vali': 0,
        'num_test': 0,
    }
    num_all_same_diff_dict = {
        'num_all': 0,
        'num_all_true': 0,
        'num_same': 0,
        'num_same_true': 0,
        'num_diff': 0,
        'num_diff_true': 0
    }
    save_dir_tr_va_te_dict = {
        'save_dir_train': os.path.join(SAVE_DIR, 'feat_label', 'train'),
        'save_dir_vali': os.path.join(SAVE_DIR, 'feat_label', 'vali'),
        'save_dir_test': os.path.join(SAVE_DIR, 'feat_label', 'test'),
    }

    def count_num():
        """
        统计个数
        """

        num_all_same_diff_dict['num_all'] += 1
        num_all_same_diff_dict['num_all_true'] += int(ds_df.car_id[i] == ds_sel_df.car_id[j])
        num_all_same_diff_dict['num_same'] += int(ds_df.cam_id[i] == ds_sel_df.cam_id[j])
        num_all_same_diff_dict['num_same_true'] += int(
            (ds_df.cam_id[i] == ds_sel_df.cam_id[j]) and ds_df.car_id[i] == ds_sel_df.car_id[j]
        )
        num_all_same_diff_dict['num_diff'] += int(ds_df.cam_id[i] != ds_sel_df.cam_id[j])
        num_all_same_diff_dict['num_diff_true'] += int(
            ds_df.cam_id[i] != ds_sel_df.cam_id[j] and ds_df.car_id[i] == ds_sel_df.car_id[j]
        )

    def save_dataset(tr_va_te, batch_size):
        """
        保存数据集
        """

        def s_d(dataset_list, num, save_dir):
            """
            保存数据集
            """

            dataset_list_remain = dataset_list[batch_size:]
            dataset_list = dataset_list[0:batch_size]

            print(f'{i} / {len(ds_df)}')

            # 保存
            print(f'save dataset_{tr_va_te}_list_{num}.pkl')

            # 打印标签
            print(
                f'total:{len(dataset_list)}, \
                1: {sum((np.array(dataset_list)[:, -1] == 1).astype(int))}, \
                0: {sum((np.array(dataset_list)[:, -1] == 0).astype(int))}'
            )

            np.savetxt(
                os.path.join(save_dir, f'dataset_{tr_va_te}_list_{num}.txt'),
                dataset_list,
                delimiter=','
            )

            print('done')
            return dataset_list_remain

        if tr_va_te == 'train':
            while len(dataset_all_tr_va_te_list_dict['dataset_train_list']) >= batch_size:
                dataset_all_tr_va_te_list_dict['dataset_train_list'] = s_d(
                    dataset_all_tr_va_te_list_dict['dataset_train_list'],
                    num_tr_va_te_dict['num_train'],
                    save_dir_tr_va_te_dict['save_dir_train']
                )
                num_tr_va_te_dict['num_train'] += 1
        elif tr_va_te == 'vali':
            while len(dataset_all_tr_va_te_list_dict['dataset_vali_list']) >= batch_size:
                dataset_all_tr_va_te_list_dict['dataset_vali_list'] = s_d(
                    dataset_all_tr_va_te_list_dict['dataset_vali_list'],
                    num_tr_va_te_dict['num_vali'],
                    save_dir_tr_va_te_dict['save_dir_vali']
                )
                num_tr_va_te_dict['num_vali'] += 1
        elif tr_va_te == 'test':
            while len(dataset_all_tr_va_te_list_dict['dataset_test_list']) >= batch_size:
                dataset_all_tr_va_te_list_dict['dataset_test_list'] = s_d(
                    dataset_all_tr_va_te_list_dict['dataset_test_list'],
                    num_tr_va_te_dict['num_test'],
                    save_dir_tr_va_te_dict['save_dir_test']
                )
                num_tr_va_te_dict['num_test'] += 1

    def save_dataset_with_batch_size(batch_size):
        """
        分段保存数据
        """

        if len(dataset_all_tr_va_te_list_dict['dataset_all_list']) >= batch_size:
            # 划分训练集
            train_list, vali_test_list = train_test_split(
                dataset_all_tr_va_te_list_dict['dataset_all_list'],
                test_size=sum(TRAIN_VALI_TEST_RATE[1:])
            )
            # 验证集与测试集
            vali_list, test_list = train_test_split(
                vali_test_list,
                test_size=(TRAIN_VALI_TEST_RATE[2] / sum(TRAIN_VALI_TEST_RATE[1:]))
            )
            print(f'train:{len(train_list)}, vali:{len(vali_list)}, test:{len(test_list)}')

            # 清空
            dataset_all_tr_va_te_list_dict['dataset_all_list'].clear()

            # 训练集
            dataset_all_tr_va_te_list_dict['dataset_train_list'] += train_list
            save_dataset('train', batch_size)

            # 验证集
            dataset_all_tr_va_te_list_dict['dataset_vali_list'] += vali_list
            save_dataset('vali', batch_size)

            # 训练集
            dataset_all_tr_va_te_list_dict['dataset_test_list'] += test_list
            save_dataset('test', batch_size)

    for i in tqdm(range(len(ds_df))):
        ds_sel_df = ds_df.loc[ds_df.timestamp < ds_df.timestamp[i], :].reset_index(drop=True)
        if not len(ds_sel_df):
            continue

        # 各种筛选
        def various_filter(x):
            bool_list = []
            ts_f_dict = span_dict[x.cam_id + ds_df.cam_id[i]]

            # 时间筛选
            bool_list.append(x.timestamp > (ds_df.timestamp[i] - ts_f_dict['span_t']))

            # 速度筛选
            v = geodesic(
                (ds_df.y_wrd[i], ds_df.x_wrd[i]), (x.y_wrd, x.x_wrd)
            ).m / (ds_df.timestamp[i] - x.timestamp)
            bool_list.append(v < ts_f_dict['v_thr'])

            # od 筛选
            bool_list.append(utils.false_2_true(bool(ts_f_dict['od_matrix'][x.lane_id, ds_df.lane_id[i]]), 0.1))

            return np.array(bool_list).all()

        ds_sel_df['bool'] = ds_sel_df.apply(lambda x: various_filter(x), axis=1)
        ds_sel_df = ds_sel_df.loc[ds_sel_df['bool'], :].reset_index(drop=True)
        if not len(ds_sel_df):
            continue

        for j in range(len(ds_sel_df)):
            # 统计个数
            count_num()

            # # 特征跨度
            # sp_dict = span_dict[ds_df.cam_id[i] + ds_sel_df.cam_id[j]]
            #
            # # reid 特征差
            # reid_feat_list = list(
            #     (np.array(
            #         list(ds_df.reid_feat_1[i]) + list(ds_df.reid_feat_2[i]) + list(ds_df.reid_feat_3[i])
            #     ) - sp_dict['reid_feat']['reid_feat_min']) / sp_dict['reid_feat']['span_reid_feat']
            # ) + list(
            #     (np.array(
            #         list(ds_sel_df.reid_feat_1[j]) + list(ds_sel_df.reid_feat_2[j]) + list(ds_sel_df.reid_feat_3[j])
            #     ) - sp_dict['reid_feat']['reid_feat_min']) / sp_dict['reid_feat']['span_reid_feat']
            # )
            #
            # # 特征拼接
            # feat_list = [
            #                 np.abs(ds_df.x_wrd[i] - ds_sel_df.x_wrd[j]) / sp_dict['span_x'],
            #                 np.abs(ds_df.y_wrd[i] - ds_sel_df.y_wrd[j]) / sp_dict['span_y'],
            #                 (ds_df.timestamp[i] - ds_sel_df.timestamp[j]) / sp_dict['span_t'],
            #                 sp_dict['od_matrix'][ds_df.lane_id[i], ds_sel_df.lane_id[j]]
            #             ] + reid_feat_list
            #
            # # l2 归一化
            # feat_list = preprocessing.normalize(np.array(feat_list).reshape(1, -1)).flatten().tolist()
            # feat_list.append(int(ds_df.car_id[i] == ds_sel_df.car_id[j]))
            #
            # # 加入
            # dataset_all_tr_va_te_list_dict['dataset_all_list'].append(feat_list)
            #
            # # 分段保存数据
            # save_dataset_with_batch_size(batch_size)

    # 打印计数
    for k in num_all_same_diff_dict.keys():
        print(f'{k}: {num_all_same_diff_dict[k]}')


def main():
    """
    主程序
    """

    # 创建目录
    dir_list = [
        os.path.join(SAVE_DIR, 'img'),
        os.path.join(SAVE_DIR, 'feat_label', 'train'),
        os.path.join(SAVE_DIR, 'feat_label', 'vali'),
        os.path.join(SAVE_DIR, 'feat_label', 'test'),
    ]
    utils.create_dir(dir_list)

    # 读取场景内全部相机参数
    scene_dir = aic_configs['SCENE_DIR']
    cam_id_list = list(filter(lambda x: x.startswith('c0'), os.listdir(scene_dir)))
    cam_id_list = ['c041', 'c042']
    cam_parms_df = pd.DataFrame.from_records(
        list(map(lambda x: get_cam_parms_dict(os.path.join(scene_dir, x)), cam_id_list))
    )

    # 读取车道颜色表
    lane_color_df = get_lane_color_df('./lane/lane_color.txt')

    # 读取结果
    mtmc_df = get_mtmc_df(aic_configs)

    # 获取场景内全部相机数据集
    if not os.path.isfile('./dataset/img_df_list.pkl'):
        cam_dir_list = list(map(lambda x: os.path.join(scene_dir, x), cam_id_list))
        args_list = [(cam_dir, lane_color_df, mtmc_df, cam_parms_df) for cam_dir in cam_dir_list]

        # 开启多进程
        with Pool(CPU_WORKER_NUM) as p:
            output_list = p.map(get_img_df, args_list)
        img_df_list = [output_list[0][0], output_list[1][0]]

        # 修改cam_parms_df 参数
        cam_parms_df['mean'] = [output_list[0][1][0], output_list[1][1][0]]
        cam_parms_df['std'] = [output_list[0][1][1], output_list[1][1][1]]

        # 保存数据
        pickle.dump(img_df_list, open(os.path.join(SAVE_DIR, 'img_df_list.pkl'), 'wb'))
        pickle.dump(cam_parms_df, open(os.path.join(SAVE_DIR, 'cam_parms_df.pkl'), 'wb'))
    else:
        with open(os.path.join(SAVE_DIR, 'img_df_list.pkl'), 'rb') as f:
            img_df_list = pickle.load(f)
        with open(os.path.join(SAVE_DIR, 'cam_parms_df.pkl'), 'rb') as f:
            cam_parms_df = pickle.load(f)

    # 拼接数据集
    img_df = pd.DataFrame()
    for i_df in img_df_list:
        img_df = img_df.append(i_df)
    img_df = img_df.reset_index(drop=True)

    # 筛选经过至少2个摄像头的数据
    img_df = utils.multi_cam_filter(img_df, 2)

    # 帧下采样
    ds_df = down_sampling(img_df)

    # 提取 reid 特征
    if not os.path.isfile('./dataset/ds_df.pkl'):
        cfg_path_list = [
            '../config/aic_reid1.yml',
            '../config/aic_reid2.yml',
            '../config/aic_reid3.yml',
        ]
        ds_df = cal_reid_feat(ds_df, cfg_path_list, cam_parms_df)
        # 保存数据
        pickle.dump(ds_df, open(os.path.join(SAVE_DIR, 'ds_df.pkl'), 'wb'))
    else:
        with open(os.path.join(SAVE_DIR, 'ds_df.pkl'), 'rb') as f:
            ds_df = pickle.load(f)

    # 计算特征跨度
    if not os.path.isfile('./dataset/span_dict.pkl'):
        span_dict = cal_span_dict(ds_df, lane_color_df)
        # 保存数据
        pickle.dump(span_dict, open(os.path.join(SAVE_DIR, 'span_dict.pkl'), 'wb'))
    else:
        with open(os.path.join(SAVE_DIR, 'span_dict.pkl'), 'rb') as f:
            span_dict = pickle.load(f)

    # 两两计算数据集
    cal_save_dataset(ds_df, 1024, span_dict)

    # # 导入数据
    # with open('./dataset/feat_label/train/dataset_train_list_0.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    # dataset_df = pd.DataFrame(dataset)

    print('Done!')


if __name__ == '__main__':
    main()
