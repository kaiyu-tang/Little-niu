#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/17 下午2:23  
# @Author  : Kaiyu  
# @Site    :   
# @File    : ocr.py
import sys
import os
sys.path.insert(0, "./src")
sys.path.append(os.path.dirname(__file__))
import cv2
import time
import math


import numpy as np
import tensorflow as tf
import argparse
from locality_aware_nms import nms_locality
# import locality_aware_nms as nms_locality
from lanms import merge_quadrangle_n9

from recognizr import Recognizer
from model import model
import json
from icdar import restore_rectangle

# default parameters
tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', './ocr/src/models/', '')
tf.app.flags.DEFINE_string('output_dir', '../result/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
# default model path
FLAGS = tf.app.flags.FLAGS
ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
model_path_ = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
print(model_path_)

class OCR(object):
    def __init__(self, model_path=model_path_):
        self._test = False
        self.rec = Recognizer()
        with tf.get_default_graph().as_default():
            self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            self._f_score, self._f_geometry = model(self._input_images, is_training=False)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            print('Restore from {}'.format(model_path))
            saver.restore(self._session, model_path)
        pass

    def _resize_image(self, im, max_side_len=768):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def _detect(self, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        if self._test:
            print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start
        # nms part
        start = time.time()
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        boxes = merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        timer['nms'] = time.time() - start

        if boxes.shape[0] == 0:
            return None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, timer

    def _sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def _text_detect(self, image, sess, f_score, f_geometry, input_images):
        im = image
        if im is None:
            return []
        im = im[:, :, ::-1]
        height, width = im.shape[:2]

        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = self._resize_image(im)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = self._detect(score_map=score, geo_map=geometry, timer=timer)
        if self._test:
            print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(""
                                                                             , timer['net'] * 1000,
                                                                             timer['restore'] * 1000,
                                                                             timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        # print('[timing] {}'.format(duration))

        chips = []
        if boxes is None:
            return chips
        for box in boxes:
            box = self._sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue

            min_x, max_x = min(box[0, 0], box[1, 0], box[2, 0], box[3, 0]), max(box[0, 0], box[1, 0], box[2, 0],
                                                                                box[3, 0])
            min_y, max_y = min(box[0, 1], box[1, 1], box[2, 1], box[3, 1]), max(box[0, 1], box[1, 1], box[2, 1],
                                                                                box[3, 1])
            if min_y >= height / 2:  # or min_x >= width / 3:
                continue

            chip = img_gray[min_y:max_y, min_x:max_x]
            chips.append(chip)

        return chips

    def text_detect(self, image=None):
        result = []
        if image is not None:
            chips = self._text_detect(image, self._session, self._f_score, self._f_geometry, self._input_images)

            for i, chip in enumerate(chips):
                _, res = self.rec.infer(chip)
                if res != "":
                    result.append(res)
            print(result)
        return result


if __name__ == '__main__':
    ocr = OCR()
    ocr.text_detect(cv2.imread('/Users/harry/PycharmProjects/toys/cut-video-with-text/ocr/src/sample001.png'))
    ocr.text_detect(cv2.imread('/home/atlab/Workspace/kaiyu/Demo/toys/ocr/src/sample002.png'))
