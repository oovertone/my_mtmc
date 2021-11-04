# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/14 12:45
@Description: 展示官方给出的真值
"""
import os
import sys

sys.path.append('../')

import cv2

import utils


def add_bbox(frame, frame_current, df):
    """
    添加bbox
    """
    df_sel = df.loc[df.frame == frame_current + 1, :].reset_index(drop=True)
    for i in range(len(df_sel)):
        # 画bbox
        pt1 = (
            int(df_sel.left[i]),
            int(df_sel.top[i]),
        )
        pt2 = (
            int(df_sel.left[i] + df_sel.width[i]),
            int(df_sel.top[i] + df_sel.height[i]),
        )
        frame = cv2.rectangle(
            img=frame,
            pt1=pt1,
            pt2=pt2,
            color=[0, 0, 255]
        )

        if 'ID' in list(df.columns):
            # id
            cv2.putText(
                img=frame,
                text=str(df_sel.ID[i]),
                org=pt1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=[0, 0, 255],
                thickness=2
            )
    return frame


def show_video(cam_dir, df):
    """
    展示视频
    """
    # 读取roi
    roi_path = os.path.join(cam_dir, 'roi.jpg')
    roi = cv2.imread(roi_path)

    # 读取视频
    video_path = os.path.join(cam_dir, 'vdo.avi')
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_current = 0

    # # 保存视频
    # fps = video.get(cv2.CAP_PROP_FPS)
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(
    #     'vdo_mtmc_c001.avi',
    #     fourcc,
    #     fps,
    #     (width, height)
    # )

    while frame_current <= frame_count - 1:
        # 读取单帧图像
        _, frame = video.read()

        # 添加roi
        frame = frame * (roi > 0)

        # 添加bbox
        frame = add_bbox(frame, frame_current, df)

        # 展示视频
        cv2.imshow('video', frame)

        # 保存视频
        # out.write(frame)

        frame_current += 1

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
            break
    video.release()
    cv2.destroyAllWindows()


def main():
    """
    主程序
    """
    # 相机目录
    cam_dir = '../../datasets/AIC21_Track3_MTMC_Tracking/test/S06/c042'
    cam_dir = '../../datasets/AIC21_Track3_MTMC_Tracking/train/S04/c019'

    # 读取真值
    gt_path = os.path.join(cam_dir, 'gt', 'gt.txt')
    gt_df = utils.get_tracking_df(gt_path)

    # 展示视频
    show_video(cam_dir, gt_df)


if __name__ == '__main__':
    main()
