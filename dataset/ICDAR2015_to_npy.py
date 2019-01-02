#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: ICDAR2015_to_npy.py
@time: 2018/12/31 15:29
@desc:
'''
import numpy as np
import cfg
import util
import os
from tqdm import tqdm

def get_mask_and_weight(points, labels):
    """
    Args:
        points: [[x1,y1,x2,y2,x3,y3,x4,y4],[x1,y1,x2,y2,x3,y3,x4,y4]...]
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        0: background
                                                        1: text

    note: shape 180,320,2
    Return:
        pixel_cls_label
        pixel_cls_weight
    """

    h,w = cfg.score_map_shape
    score_map_shape = [int(w/cfg.img_scall),int(h/cfg.img_scall)]

    text_label = cfg.text_label
    background_label = cfg.background_label

    mask = np.zeros(score_map_shape, dtype=np.int32)
    pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)

    bbox_masks = []
    pos_mask = mask.copy()

    points = list((np.asarray(points) / cfg.img_scall).astype('int32'))
    for bbox_idx, bbox_points in enumerate(points):
        if labels[bbox_idx] == background_label:
            continue

        bbox_mask = mask.copy()
        util.img.draw_mask(bbox_mask, bbox_points)
        bbox_masks.append(bbox_mask)
        if labels[bbox_idx] == text_label:
            pos_mask += bbox_mask

    # 去除重复的label
    pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)

    num_positive_bboxes = len(labels)
    num_positive_pixels = np.sum(pos_mask)

    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_positive_pixel_mask = bbox_mask * pos_mask

        num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
        if num_bbox_pixels > 0:
            per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
            per_pixel_weight = per_bbox_weight / num_bbox_pixels
            pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight

    return np.concatenate([np.expand_dims(pos_mask,axis=-1),np.expand_dims(pixel_cls_weight,axis=-1)],axis=-1)

if __name__ == '__main__':
    """
    """
    # label_dir = "../../Data/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt"
    # result_dir = "../../Data/PixelLink_py36/label_train"

    label_dir = '../data/text'
    result_dir = '../data/npy'

    label_list = os.listdir(label_dir)
    for i in tqdm(label_list):
        file_dir = os.path.join(label_dir,i)
        txt = open(file_dir, encoding='utf-8')

        list_points = []
        list_labels = []
        for line in txt:
            line_cols = str(line).strip('\ufeff').split(',')
            list_c = [int(i) for i in line_cols[:8]]
            list_points.append(list_c)
            list_labels.append(1)

        mask = get_mask_and_weight(list_points,list_labels)
        file_path = os.path.join(result_dir, i[:-4] + '.npy')
        np.save(file_path,mask)