#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: img.py
@time: 2018/12/28 20:36
@desc:
'''
import cv2
import numpy as np

def draw_contours(img, contours, idx=-1, color=1, border_width=1):
    cv2.drawContours(img, contours, idx, color, border_width)
    return img


class pixellink_link_label:
    """

    """
    def __init__(self,mask):
        """

        :param shape:
        """
        w,h = mask.shape
        self.mask_extend = np.zeros([w+2,h+2],type=np.int8)
        self.mask_extend = self.mask_extend[1:-1,1:-1] + mask

        self.mask_lu = self.mask_extend[2:,2:]      * mask
        self.mask_ld = self.mask_extend[:-2,2:]     * mask
        self.mask_rd = self.mask_extend[:-2,:-2]    * mask
        self.mask_ru = self.mask_extend[2:,:-2]     * mask

        self.mask_ri = self.mask_extend[1:-1,:-2]   * mask
        self.mask_le = self.mask_extend[1:-1,2:]    * mask
        self.mask_up = self.mask_extend[2:,1:-1]    * mask
        self.mask_dn = self.mask_extend[:-2,1:-1]   * mask


def draw_mask(img,points):
    """
    有点画mask，
    point的形式
    :param img:
    :param points: list [x1,y1,x2,y2,x3,y3,x4,y4]
    :return:
    """
    points = np.asarray(points).reshape(1,4,1,2)
    cv2.drawContours(img,points,-1,1,-1)
    return img