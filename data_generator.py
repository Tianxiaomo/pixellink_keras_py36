#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: data_generator.py
@time: 2019/1/3 15:40
@desc:
'''
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

import cfg
import util

# 数据生成器 for pixellink
def gen_data(batch_size=cfg.batch_size, is_val=False):
    # img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    img_h, img_w = [1280, 720]
    x = np.zeros((batch_size, img_w, img_h, cfg.num_channels), dtype=np.float32)
    # pixel_num_h = img_h // cfg.pixel_size
    # pixel_num_w = img_w // cfg.pixel_size
    pixel_num_h, pixel_num_w = [320, 180]
    y = np.zeros((batch_size, pixel_num_w, pixel_num_h, 18), dtype=np.float32)
    if is_val:
        f_list = os.listdir(os.path.join(cfg.data_dir, cfg.train_image_dir_name))
    else:
        f_list = os.listdir(os.path.join(cfg.data_dir, cfg.train_image_dir_name))
    while True:
        for i in range(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split('.')[0]
            # load img and label
            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    random_img)
            img = image.load_img(img_path)
            img = image.img_to_array(img)

            x[i] = preprocess_input(img, mode='tf')
            x[i] = img

            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   'gt_' + img_filename + '.npy')

            label_and_weight = np.load(gt_file)

            link_label = get_pixel_link_label(label_and_weight[:, :, 0])
            link_label_weight = link_label * np.expand_dims(label_and_weight[:, :, -1], axis=-1)

            label = np.concatenate([label_and_weight, link_label, link_label_weight], axis=-1)

            y[i] = label
        yield x, y


def get_pixel_link_label(mask):
    """
    1 2 3
    4 0 5
    6 7 8
    :param mask:
    :return:
    """
    w, h = mask.shape
    mask_extend = np.zeros([w + 2, h + 2], dtype=np.int32)
    mask_extend[1:-1, 1:-1] = mask_extend[1:-1, 1:-1] + mask

    mask_lu = mask_extend[2:, 2:] * mask
    mask_ld = mask_extend[:-2, 2:] * mask
    mask_rd = mask_extend[:-2, :-2] * mask
    mask_ru = mask_extend[2:, :-2] * mask

    mask_ri = mask_extend[1:-1, :-2] * mask
    mask_le = mask_extend[1:-1, 2:] * mask
    mask_up = mask_extend[2:, 1:-1] * mask
    mask_dn = mask_extend[:-2, 1:-1] * mask
    mask_list = [mask_lu, mask_up, mask_ru, mask_le, mask_ri, mask_ld, mask_dn, mask_rd]
    mask_list_1 = []
    for i in mask_list:
        mask_list_1.append(np.expand_dims(i, axis=-1))
    return np.concatenate(mask_list_1, axis=-1)


# def get_pixel_link_label(mask,weight):
#     """
#     1 2 3
#     4 0 5
#     6 7 8
#     :param mask:
#     :return:
#     """
#     return mask * np.expand_dims(weight,axis=-1)


def cal_gt_for_single_image(points, labels):
    """
    Args:
        xs, ys: both in shape of (N, 4),
            and N is the number of bboxes,
            their values are normalized to [0,1]
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        1: text
    Return:
        pixel_cls_label
        pixel_cls_weight
        pixel_link_label
        pixel_link_weight
    """

    score_map_shape = cfg.score_map_shape
    h, w = score_map_shape

    text_label = cfg.text_label
    ignore_label = cfg.ignore_label
    background_label = cfg.background_label
    num_neighbours = cfg.num_neighbours

    assert np.ndim(np.asarray(points)) == 2
    assert np.asarray(points).shape[-1] == 8
    assert np.asarray(points).shape[0] == len(labels)

    num_positive_bboxes = np.sum(np.asarray(labels) == text_label)

    mask = np.zeros(score_map_shape, dtype=np.int32)
    pixel_cls_label = np.zeros(score_map_shape, dtype=np.int32)
    pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)

    pixel_link_weight = np.ones((h, w, num_neighbours), dtype=np.float32)

    bbox_masks = []
    pos_mask = mask.copy()
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
    num_positive_pixels = np.sum(pos_mask)

    sum_mask = np.sum(bbox_masks, axis=0)
    not_overlapped_mask = sum_mask == 1

    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_label = labels[bbox_idx]
        if bbox_label == ignore_label:
            bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
            pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
            continue

        if labels[bbox_idx] == background_label:
            continue

        bbox_positive_pixel_mask = bbox_mask * pos_mask
        # background or text is encoded into cls gt
        pixel_cls_label += bbox_positive_pixel_mask * bbox_label

        num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
        if num_bbox_pixels > 0:
            per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
            per_pixel_weight = per_bbox_weight / num_bbox_pixels
            pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight

    pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
    pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis=-1)

    pixel_link_label = get_pixel_link_label(pixel_cls_label)

    return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight

# =====================Ground Truth Calculation End====================


if __name__ == "__main__":
    """
    """
    from tqdm import tqdm
    import cv2
    import matplotlib.pyplot as plt

    label_dir = 'data'

    file_list = os.listdir(label_dir)
    list_n = []
    for i in tqdm(file_list):
        file_dir = os.path.join(label_dir, i)
        txt = open(file_dir, encoding='utf-8')

        list_points = []
        list_labels = []
        for line in txt:
            mask = np.zeros([1280, 768])
            line_cols = str(line).strip('\ufeff').split(',')
            list_c = [int(i) for i in line_cols[:8]]
            list_points.append(list_c)
            list_labels.append(1)

        pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight = cal_gt_for_single_image(list_points,
                                                                                                         list_labels)
        plt.imshow(pixel_cls_label, cmap=plt.cm.get_cmap('gray'))
        plt.imsave('1.jpg', pixel_cls_label)
        plt.show()
        plt.imshow(pixel_cls_weight, cmap=plt.cm.get_cmap('gray'))
        plt.imsave('2.jpg', pixel_cls_weight)
        plt.show()
        for j in range(4):
            plt.imshow(pixel_link_label[:, :, j], cmap=plt.cm.get_cmap('gray'))
            plt.imsave('3.jpg', pixel_link_label[:, :, j])
            plt.show()
            plt.imshow(pixel_link_weight[:, :, j], cmap=plt.cm.get_cmap('gray'))
            plt.imsave('4.jpg', pixel_link_weight[:, :, j])
            plt.show()