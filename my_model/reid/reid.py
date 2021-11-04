# -*- coding: utf-8 -*-
"""
@Author: Qchail
@Time: 2021/11/1 14:17
@Description: reid 模块
"""

import os

import torch
import torchvision.transforms as T
from PIL import Image

from .reid_inference.reid_model import build_reid_model


class ReidFeature(object):
    """
    提取 reid 特征
    """

    def __init__(self, gpu_id, _mcmt_cfg, mean, std):
        print("初始化 reid 模型")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()
        self.val_transforms = T.Compose(
            [T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3), T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

    def extract(self, img_path_list):
        """
        提取图片特征
        特征尺寸：(2048,) float32
        """

        img_batch = []
        for img_path in img_path_list:
            img = Image.open(img_path).convert('RGB')
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat
