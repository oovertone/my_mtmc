# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/29 14:18
@Description: 划分数据集
"""

import os
import random
import sys
from multiprocessing import Pool
from multiprocessing import Process, JoinableQueue
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('../')
import utils
from config import cfg
from reid.reid import ReidFeature

# 读取配置文件
aic_configs = utils.get_aic_configs(Path(__file__).parents[1])
SCENE_DIR = aic_configs['global_configs']['SCENE_DIR']  # 场景目录
GPU_ID = aic_configs['global_configs']['GPU_ID']  # GPU ID
CPU_WORKER_NUM = aic_configs['global_configs']['CPU_WORKER_NUM']  # 处理器核数
CAM_ID_LIST = aic_configs['prepare_dataset_configs']['CAM_ID_LIST']  # 数据集中包含的相机列表
DOWN_SAMPLING_RATE = aic_configs['prepare_dataset_configs']['DOWN_SAMPLING_RATE']  # 下采样率
TRAIN_VALI_TEST_RATE = aic_configs['prepare_dataset_configs']['TRAIN_VALI_TEST_RATE']  # 训练集比例
SAVE_DIR = aic_configs['prepare_dataset_configs']['SAVE_DIR']  # 保存路径
BATCH_SIZE = aic_configs['prepare_dataset_configs']['BATCH_SIZE']  # 批次大小


class Save_Dataset(object):
    """
    保存 Dataset 类
    """

    def __init__(self, save_dir):
        """
        初始化
        """

        self.dataset_list = []  # 数据列表
        self.file_num = -1  # 当前保存文件数
        self.save_dir = save_dir
        self.save_path = ""

    def update_save_path(self):
        """
        更新保存路径
        """
        pass

    def queue_up(self, q, batch_size, last=False):
        """
        如果数据长度大于 batch_size，放入队列
        """

        while (len(self.dataset_list) >= batch_size) or last:
            if len(self.dataset_list):
                # 更新
                self.file_num += 1
                self.update_save_path()

                data = self.dataset_list.copy()  # 复制一份数据

                save_dict = {
                    'data': data[0:batch_size],
                    'save_path': self.save_path
                }
                self.dataset_list = data[batch_size:]
                q.put(save_dict)
            else:
                break
        return q


class TVT_Dataset(Save_Dataset):
    """
    TVT_Dataset 类
    TVT: train vali test
    """

    def __init__(self, train_vali_test, save_dir):
        """
        初始化
        """

        super().__init__(save_dir)
        self.train_vali_test = train_vali_test  # 是否为同一相机

    def update_save_path(self):
        """
        更新保存路径
        """
        self.save_path = os.path.join(
            self.save_dir, self.train_vali_test, f'{self.train_vali_test}_{self.file_num}.npy'
        )


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


def get_mtmc_df():
    """
    获取 mtmc_df 结果
    """

    mtmc_df = pd.read_csv(os.path.join(SCENE_DIR, 'track3.txt'), sep=' ', header=None)
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


def split_car_id_tvt(img_df):
    """
    根据车辆 id 分割训练集、验证集与测试集（剩余）
    """
    car_id_list = list(np.unique(img_df.car_id))
    random.shuffle(car_id_list)
    car_id_list_train = car_id_list[0:int(len(car_id_list) * TRAIN_VALI_TEST_RATE[0])]
    car_id_list_vali = list(
        filter(lambda x: x not in car_id_list_train, car_id_list)
    )[0:int(len(car_id_list) * TRAIN_VALI_TEST_RATE[1])]
    car_id_list_test = list(
        filter(lambda x: (x not in car_id_list_train) and (x not in car_id_list_vali), car_id_list)
    )
    car_id_list_tvt_dict = {
        'car_id_list_train': car_id_list_train,
        'car_id_list_vali': car_id_list_vali,
        'car_id_list_test': car_id_list_test
    }

    return car_id_list_tvt_dict


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
            image_path_bl_list = utils.chunks(image_path_list, BATCH_SIZE[0])

            # 提取特征
            reid_feat_list = []
            for image_path_bl in tqdm(image_path_bl_list):
                reid_feat_list += list(model.extract(image_path_bl))
            ds_sel_df[reid_feat_name[i]] = reid_feat_list

        # 拼接
        out_df = out_df.append(ds_sel_df)

    return out_df


def split_data(data, rate):
    """
    分割数据集
    """
    idx = list(range(len(data)))
    random.shuffle(idx)
    len_sel = int(len(data) * rate)
    data_1 = data[0:len_sel]
    data_2 = data[len_sel:]

    return data_1, data_2


def consume_dataset(q):
    """
    消费：从队列中取出数据并保存
    """

    while True:
        save_dict = q.get()

        # 保存数据
        save_dir = Path(save_dict['save_path']).parents[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save(save_dict['save_path'], save_dict['data'])
        print(f"已保存： {save_dict['save_path']} ! 队列长度：{q.qsize()}")
        q.task_done()


def produce_dataset(ds_df, car_id_list_tvt_dict, save_dir, span_dict, q):
    """
    生产：两两计算数据集并保存
    区分同一摄像头和不同摄像头
    区分 label 为 true 和 false
    """

    def init_dict():
        """
        初始化
        """

        # 初始化 count_dict
        count_dict = {
            'num': 0,
            'num_0': 0,
            'num_1': 0,
            'num_same_dict': {
                'num': 0,
                'num_0': 0,
                'num_1': 0
            },
            'num_diff_dict': {
                'num': 0,
                'num_0': 0,
                'num_1': 0
            }
        }

        # 初始化 TVT 对象
        tvt_dataset_dict = {
            'train': TVT_Dataset('train', save_dir),
            'vali': TVT_Dataset('vali', save_dir),
            'test': TVT_Dataset('test', save_dir)
        }

        return count_dict, tvt_dataset_dict

    def cal_feat_label_list():
        """
        计算特征与标签
        """

        sp_dict = span_dict[ds_df.cam_id[i] + ds_sel_df.cam_id[j]]  # 特征跨度

        # reid 特征差
        reid_feat_list = list(
            (np.array(
                list(ds_df.reid_feat_1[i]) + list(ds_df.reid_feat_2[i]) + list(ds_df.reid_feat_3[i])
            ) - sp_dict['reid_feat']['reid_feat_min']) / sp_dict['reid_feat']['span_reid_feat']
        ) + list(
            (np.array(
                list(ds_sel_df.reid_feat_1[j]) + list(ds_sel_df.reid_feat_2[j]) + list(ds_sel_df.reid_feat_3[j])
            ) - sp_dict['reid_feat']['reid_feat_min']) / sp_dict['reid_feat']['span_reid_feat']
        )

        # 特征拼接 + label
        feat_label_list = [
                              np.abs(ds_df.x_wrd[i] - ds_sel_df.x_wrd[j]) / sp_dict['span_x'],
                              np.abs(ds_df.y_wrd[i] - ds_sel_df.y_wrd[j]) / sp_dict['span_y'],
                              (ds_df.timestamp[i] - ds_sel_df.timestamp[j]) / sp_dict['span_t'],
                              sp_dict['od_matrix'][ds_df.lane_id[i], ds_sel_df.lane_id[j]]
                          ] + reid_feat_list + \
                          [
                              int(ds_df.cam_id[i] == ds_sel_df.cam_id[j]),  # 是否为同一相机
                              int(ds_df.car_id[i] == ds_sel_df.car_id[j])  # 是否为同一辆车
                          ]

        return feat_label_list

    def get_tvt(car_id_1, car_id_2):
        """
        区分训练集、验证集与测试集
        """
        for tvt in car_id_list_tvt_dict:
            if (car_id_1 in car_id_list_tvt_dict[tvt]) and (car_id_2 in car_id_list_tvt_dict[tvt]):
                return tvt.split('_')[-1]
        return None

    count_dict, tvt_dataset_dict = init_dict()  # 初始化

    # 随机打乱 ds_df
    idx = list(ds_df.index)
    random.shuffle(idx)
    ds_df.index = idx
    ds_df.sort_index(inplace=True)

    for i in tqdm(range(len(ds_df))):
        ds_sel_df = ds_df.loc[ds_df.timestamp < ds_df.timestamp[i], :].reset_index(drop=True)
        if not len(ds_sel_df):
            continue

        # 各种筛选
        def various_filter(x):
            """
            各种筛选
            """

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
            bool_list.append(utils.false_2_true(bool(ts_f_dict['od_matrix'][x.lane_id, ds_df.lane_id[i]]), 0.01))

            return np.array(bool_list).all()

        ds_sel_df['bool'] = ds_sel_df.apply(lambda x: various_filter(x), axis=1)
        ds_sel_df = ds_sel_df.loc[ds_sel_df['bool'], :].reset_index(drop=True)
        if not len(ds_sel_df):
            continue

        for j in range(len(ds_sel_df)):
            tvt = get_tvt(ds_df.car_id[i], ds_sel_df.car_id[j])
            if not tvt:
                continue

            feat_label_list = cal_feat_label_list()  # 计算特征

            # 分配
            if ds_sel_df.cam_id[j] == ds_df.cam_id[i]:  # 同一个相机
                if ds_sel_df.car_id[j] == ds_df.car_id[i]:  # true
                    count_dict['num_same_dict']['num_1'] += 1
                else:  # false
                    count_dict['num_same_dict']['num_0'] += 1
                count_dict['num_same_dict']['num'] = count_dict['num_same_dict']['num_0'] + \
                                                     count_dict['num_same_dict']['num_1']
            else:  # 不同相机
                if ds_sel_df.car_id[j] == ds_df.car_id[i]:  # true
                    count_dict['num_diff_dict']['num_1'] += 1
                else:  # false
                    count_dict['num_diff_dict']['num_0'] += 1
                count_dict['num_diff_dict']['num'] = count_dict['num_diff_dict']['num_0'] + \
                                                     count_dict['num_diff_dict']['num_1']
            count_dict['num_0'] = count_dict['num_same_dict']['num_0'] + count_dict['num_diff_dict']['num_0']
            count_dict['num_1'] = count_dict['num_same_dict']['num_1'] + count_dict['num_diff_dict']['num_1']
            count_dict['num'] = count_dict['num_0'] + count_dict['num_1']

            # 按概率分到 train，vali，test
            if tvt == 'train':
                tvt_dataset_dict['train'].dataset_list.append(feat_label_list)
            elif tvt == 'vali':
                tvt_dataset_dict['vali'].dataset_list.append(feat_label_list)
            else:
                tvt_dataset_dict['test'].dataset_list.append(feat_label_list)
            # 依次向保存队列中生产数据
            for key in tvt_dataset_dict:
                q = tvt_dataset_dict[key].queue_up(q, BATCH_SIZE[1])

    for key in tvt_dataset_dict:
        if len(tvt_dataset_dict[key].dataset_list):
            q = tvt_dataset_dict[key].queue_up(q, BATCH_SIZE[1], last=True)

    q.put({
        'data': count_dict,
        'save_path': os.path.join(save_dir, 'count_dict.npy')
    })
    q.join()
    print(f'count_dict: {count_dict}')


def main():
    """
    主程序
    """

    # 读取场景内全部相机参数
    cam_id_list = list(filter(lambda x: x.startswith('c0'), os.listdir(SCENE_DIR)))
    cam_parms_df = pd.DataFrame.from_records(
        list(map(lambda x: get_cam_parms_dict(os.path.join(SCENE_DIR, x)), cam_id_list))
    )

    # 读取车道颜色表
    lane_color_df = get_lane_color_df('./lane/lane_color.txt')

    # 读取结果
    mtmc_df = get_mtmc_df()

    # 获取场景内全部相机数据集
    if not os.path.isfile(os.path.join(SAVE_DIR, 'img_df.pkl')):
        cam_dir_list = list(map(lambda x: os.path.join(SCENE_DIR, x), cam_id_list))
        args_list = [(cam_dir, lane_color_df, mtmc_df, cam_parms_df) for cam_dir in cam_dir_list]

        # 开启多进程
        with Pool(CPU_WORKER_NUM) as p:
            output_list = p.map(get_img_df, args_list)

        # 取出 img_df
        img_df_list = [output_list[i][0] for i in range(len(output_list))]
        img_df = pd.DataFrame()
        for i_df in img_df_list:
            img_df = img_df.append(i_df)
        img_df = img_df.reset_index(drop=True)
        img_df = utils.multi_cam_filter(img_df, 2)  # 筛选经过至少2个摄像头的数据

        # 修改cam_parms_df 参数
        cam_parms_df['mean'] = [output_list[i][1][0] for i in range(len(output_list))]
        cam_parms_df['std'] = [output_list[i][1][1] for i in range(len(output_list))]

        # 保存数据
        joblib.dump(img_df, open(os.path.join(SAVE_DIR, 'img_df.pkl'), 'wb'))
        joblib.dump(cam_parms_df, open(os.path.join(SAVE_DIR, 'cam_parms_df.pkl'), 'wb'))
    else:
        with open(os.path.join(SAVE_DIR, 'img_df.pkl'), 'rb') as f:
            img_df = joblib.load(f)
        with open(os.path.join(SAVE_DIR, 'cam_parms_df.pkl'), 'rb') as f:
            cam_parms_df = joblib.load(f)

    # 确定训练集、验证集、测试集车辆 id
    if not os.path.isfile(os.path.join(SAVE_DIR, 'car_id_list_tvt_dict.pkl')):
        car_id_list_tvt_dict = split_car_id_tvt(img_df)

        # 保存数据
        joblib.dump(car_id_list_tvt_dict, open(os.path.join(SAVE_DIR, 'car_id_list_tvt_dict.pkl'), 'wb'))
    else:
        with open(os.path.join(SAVE_DIR, 'car_id_list_tvt_dict.pkl'), 'rb') as f:
            car_id_list_tvt_dict = joblib.load(f)

    img_df = img_df.loc[
             img_df.cam_id.apply(lambda x: x in CAM_ID_LIST), :
             ].reset_index(drop=True)  # 筛选在 CAM_ID_LIST 内的 img_df
    dataset_dir = os.path.join(SAVE_DIR, '_'.join(CAM_ID_LIST) + '_' + str(DOWN_SAMPLING_RATE))  # 保存数据集的目录
    utils.create_dir([dataset_dir])  # 创建目录

    # 帧下采样
    ds_df = down_sampling(img_df)

    # 提取 reid 特征
    if not os.path.isfile(os.path.join(dataset_dir, 'ds_df.pkl')):
        cfg_path_list = [
            '../config/aic_reid1.yml',
            '../config/aic_reid2.yml',
            '../config/aic_reid3.yml',
        ]
        ds_df = cal_reid_feat(ds_df, cfg_path_list, cam_parms_df)
        # 保存数据
        joblib.dump(ds_df, open(os.path.join(dataset_dir, 'ds_df.pkl'), 'wb'))
    else:
        with open(os.path.join(dataset_dir, 'ds_df.pkl'), 'rb') as f:
            ds_df = joblib.load(f)

    # 计算特征跨度
    if not os.path.isfile(os.path.join(dataset_dir, 'span_dict.pkl')):
        span_dict = cal_span_dict(ds_df, lane_color_df)
        # 保存数据
        joblib.dump(span_dict, open(os.path.join(dataset_dir, 'span_dict.pkl'), 'wb'))
    else:
        with open(os.path.join(dataset_dir, 'span_dict.pkl'), 'rb') as f:
            span_dict = joblib.load(f)

    # 开启队列并启动消费者
    q = JoinableQueue()  # 创建队列
    c_list = []  # 多消费者
    for i in range(1):
        c = Process(target=consume_dataset, args=(q,))
        c.daemon = True  # 设置为守护进程
        c_list.append(c)
    for c in c_list:
        c.start()

    # 两两计算数据集并放入保存队列
    if not os.path.isfile(os.path.join(dataset_dir, 'feat_label', 'count_dict.pkl')):
        p = Process(
            target=produce_dataset,
            args=(ds_df, car_id_list_tvt_dict, os.path.join(dataset_dir, 'feat_label'), span_dict, q,)
        )  # 生产者
        # 启动进程
        p.start()
        p.join()
    else:
        with open(os.path.join(dataset_dir, 'feat_label', 'count_dict.pkl'), 'rb') as f:
            count_dict = joblib.load(f)
            print(f'count_dict: {count_dict}')

    print('Done!')


if __name__ == '__main__':
    main()
