# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/10/19 10:01
@Description: 展示检测的视频
"""
import os
import cv2


def sort_img_name_list(img_name_list):
    """
    对图片名列表排序
    """
    num_list = list(map(lambda x: x.split('.')[0].split('_')[-1], img_name_list))
    n = max(list(map(lambda x: len(x), num_list)))
    num_list = list(map(lambda x: '0' * (n - len(x)) + x, num_list))
    img_name_dict = dict(zip(num_list, img_name_list))
    img_name_list = [img_name_dict[k] for k in sorted(img_name_dict.keys())]
    return img_name_list


def read_show_img(cam_name):
    """
    读取并展示图片
    """
    # 获取图片名列表
    cam_dir = os.path.join('../../datasets/detect_provided/', cam_name)
    img_name_list = os.listdir(os.path.join(cam_dir, 'dets_debug'))
    img_name_list = sort_img_name_list(img_name_list)

    # 保存视频
    video_path = os.path.join('../../datasets/AIC21_Track3_MTMC_Tracking/test/S06/', cam_name, 'vdo.avi')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        'vdo_det.avi',
        fourcc,
        fps,
        (width, height)
    )

    for img_name in img_name_list:
        img = cv2.imread(os.path.join(cam_dir, 'dets_debug', img_name))

        # 保存视频
        out.write(img)

        # 显示图片
        cv2.imshow('video', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
            break


def main():
    """
    主程序
    """
    cam_name = 'c043'
    cam_dir = '../../datasets/detect_provided/c043'
    read_show_img(cam_name)


if __name__ == '__main__':
    main()
