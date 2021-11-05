# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/14 12:45
@Description: 展示与保存视频
"""

import argparse
import os
import sys
import random
import tkinter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
import utils

# 读取配置文件
aic_configs = utils.get_aic_configs(Path(__file__).parents[1])
CAM_NUM_MAX = aic_configs['show_save_video_configs']['CAM_NUM_MAX']  # 最大显示视频数
NUM_COLS = aic_configs['show_save_video_configs']['NUM_COLS']  # 子窗口列数


class CameraVideo(object):
    """
    相机视频类
    """

    def __init__(self, cam_dir, ret_df):
        """
        初始化
        """

        # 相机 id
        self.cam_id = os.path.split(cam_dir)[-1]

        # 读取 roi
        self.roi = cv2.imread(os.path.join(cam_dir, 'roi.jpg'))

        # 读取视频
        self.video = cv2.VideoCapture(os.path.join(cam_dir, 'vdo.avi'))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frame_current = 0

        # 读取检测文件
        self.ret_df = ret_df.loc[ret_df.cam_id == self.cam_id, :].reset_index(drop=True)

    def add_info(self):
        """
        添加 info
            bbox
            car_id
        """

        ret_df_sel = self.ret_df.loc[self.ret_df.frame_id == self.frame_current + 1, :].reset_index(drop=True)
        for i in range(len(ret_df_sel)):
            # cam_id
            self.frame = cv2.putText(
                img=self.frame,
                text=self.cam_id,
                org=(self.width - 150, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=[0, 0, 255],
                thickness=3
            )

            # bbox
            pt1 = (
                int(ret_df_sel.left[i]),
                int(ret_df_sel.top[i]),
            )
            pt2 = (
                int(ret_df_sel.left[i] + ret_df_sel.width[i]),
                int(ret_df_sel.top[i] + ret_df_sel.height[i]),
            )
            self.frame = cv2.rectangle(
                img=self.frame,
                pt1=pt1,
                pt2=pt2,
                color=ret_df_sel.color[i],
                thickness=3
            )

            # car_id
            self.frame = cv2.putText(
                img=self.frame,
                text=str(ret_df_sel.car_id[i]),
                org=pt1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=ret_df_sel.color[i],
                thickness=3
            )

    def get_frame(self):
        """
        获取视频单帧图像
        """

        # 读取单帧图像，读到最后一帧冻结
        _, self.frame = self.video.read()
        if self.frame_current == self.frame_count - 1:
            self.frame_last = self.frame
        if self.frame_current > self.frame_count - 1:
            self.frame = self.frame_last.copy()
        self.frame_current += 1

        # 添加roi
        self.frame *= (self.roi > 0)

        # 添加 info
        self.add_info()

    def adjust_to_new_size(self, width, height):
        """
        将图片放到新尺寸窗口中
        """

        if self.width >= self.height:
            self.frame = cv2.resize(
                self.frame,
                (width, int(self.height * width / self.width))
            )
            # 上下拼接
            diff = height - self.frame.shape[0]
            black_1 = np.zeros([int(np.floor(diff / 2)), self.frame.shape[1], 3]).astype(np.uint8)
            black_2 = np.zeros([int(np.ceil(diff / 2)), self.frame.shape[1], 3]).astype(np.uint8)
            self.frame = np.concatenate((black_1, self.frame, black_2), axis=0)
        else:
            self.frame = cv2.resize(
                self.frame,
                (int(self.width * height / self.height), height)
            )
            # 左右拼接
            diff = width - self.frame.shape[1]
            black_1 = np.zeros([self.frame.shape[0], int(np.floor(diff / 2)), 3]).astype(np.uint8)
            black_2 = np.zeros([self.frame.shape[0], int(np.ceil(diff / 2)), 3]).astype(np.uint8)
            self.frame = np.concatenate((black_1, self.frame, black_2), axis=1)


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='展示与保存结果视频')
    parser.add_argument('--scene_dir', type=str, required=True, help='数据集场景目录')
    parser.add_argument('--cam_id_list', type=str, default='', nargs='+', help='展示的相机 id')
    parser.add_argument('--ret_df_path', type=str, required=True, help='检测结果路径')
    parser.add_argument('--show', action='store_true', default=False, help='是否展示结果视频')
    parser.add_argument('--save', action='store_true', help='是否保存结果视频')
    parser.add_argument('--save_path', type=str, default='./video_save.avi', help='结果视频存储路径')
    args = parser.parse_args()

    return args


def get_cam_id_list(args):
    """
    获取相机 id 列表
    """

    # 场景 id
    scene_id = os.path.split(args.scene_dir)[-1]

    if len(args.cam_id_list):
        cam_id_list = args.cam_id_list[0:CAM_NUM_MAX]
    else:
        cam_id_list = list(filter(lambda x: x.startswith('c'), os.listdir(args.scene_dir)))
        cam_id_list = cam_id_list[0:CAM_NUM_MAX]
    print(f'场景：{scene_id}，相机列表：{cam_id_list}')

    return cam_id_list


def add_random_color(df):
    """
    为每一个 id 赋予随机的颜色
    """

    def get_random_color():
        """
        获取随机颜色
        """

        color_range = [0, 200]
        color = [
            random.randint(color_range[0], color_range[1]),
            random.randint(color_range[0], color_range[1]),
            random.randint(color_range[0], color_range[1]),
        ]

        return color

    car_id_list = np.unique(df.car_id)
    color_list = list(map(lambda x: get_random_color(), car_id_list))
    color_dict = dict(zip(car_id_list, color_list))

    df['color'] = df.car_id.apply(lambda x: color_dict[x])

    return df


def get_area_dict(cam_id_list):
    """
    获取子窗口尺寸
    """

    screen = tkinter.Tk()
    width = screen.winfo_screenwidth()
    height = screen.winfo_screenheight()
    num_rows = np.ceil(len(cam_id_list) / NUM_COLS)  # 行数
    width_sub = int(np.floor(width / NUM_COLS))
    height_sub = int(np.floor(height / num_rows))
    size_sub = [width_sub, height_sub]
    area_dict = {
        'width': width,
        'height': height,
        'width_sub': width_sub,
        'height_sub': height_sub,
        'num_cols': NUM_COLS,
        'num_rows': num_rows
    }

    return area_dict


def main():
    """
    主程序
    """

    # 解析命令行参数
    args = parse_args()

    # 相机 id 列表
    cam_id_list = get_cam_id_list(args)

    # 相机 id 目录
    cam_dir_list = list(map(lambda x: os.path.join(args.scene_dir, x), cam_id_list))

    # 读取真值
    ret_df = pd.read_csv(args.ret_df_path, sep=' ', header=None)
    ret_df.columns = ['cam_id', 'car_id', 'frame_id', 'left', 'top', 'width', 'height', '-1', '-1']
    ret_df.frame_id += 1
    ret_df['cam_id'] = ret_df.cam_id.apply(lambda x: 'c0' + str(x))

    # 赋予 car_id 颜色
    ret_df = add_random_color(ret_df)

    # 建立相机视频实例列表
    cam_video_list = [CameraVideo(cam_dir, ret_df) for cam_dir in cam_dir_list]

    # 确定新窗口尺寸
    area_dict = get_area_dict(cam_id_list)

    # 展示视频
    frame_count_max = max([cam_video.frame_count for cam_video in cam_video_list])
    # 保存视频
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            args.save_path,
            fourcc,
            10,
            (area_dict['width'], area_dict['height'])
        )

    for frame_id in tqdm(range(frame_count_max)):
        frame = np.zeros([area_dict['height'], area_dict['width'], 3]).astype(np.uint8)
        for i, cam_video in enumerate(cam_video_list):
            cam_video.get_frame()  # 读取一帧
            cam_video.adjust_to_new_size(area_dict['width_sub'], area_dict['height_sub'])  # 调整成子窗口大小

            # 拼接
            row = np.floor(i / area_dict['num_cols'])  # 所在行数
            col = i - row * area_dict['num_cols']  # 所在列数
            h_1 = int(row * area_dict['height_sub'])
            h_2 = int((row + 1) * area_dict['height_sub'])
            w_1 = int(col * area_dict['width_sub'])
            w_2 = int((col + 1) * area_dict['width_sub'])
            frame[h_1:h_2, w_1:w_2, :] = cam_video.frame

            frame = cv2.putText(
                img=frame,
                text=f'frame_id: {frame_id}',
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=[0, 0, 255],
                thickness=2
            )

        # 保存视频
        if args.save:
            out.write(frame)

        # 显示视频
        if args.show:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    print('完成！')


if __name__ == '__main__':
    main()
