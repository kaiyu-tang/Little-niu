#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/16 下午6:38  
# @Author  : Kaiyu  
# @Site    :   
# @File    : video-cut.py

import json

import cv2
import os
import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from ocr.src.ocr import OCR
from Text_classification.text_classify.predict import Predictor
from Text_classification.text_classify.data.load_data import DataLoader


class Tcv(object):
    def __init__(self):
        self._cap = None
        self._fps = None
        self._size = None
        self._num_frames = None
        self._fourcc = None
        # self._fourcc = -1
        self._live_texts = None
        self._live_times = None
        self._ocr_model = OCR()
        self._text_predictor = Predictor()
        self._live_predicts = []
        self._live_merged = []
        self._time_threhold = 1
        self._cuda = False
        pass

    def live_text_pred(self, sentences, clean=True):
        self._live_texts = sentences
        labels = [8 for _ in range(len(sentences))]
        data_iter = DataLoader(sentences, labels, clean=clean)
        self._live_predicts, logits = self._text_predictor.predict(data_iter)
        self._live_predicts = list(map(int, self._live_predicts))
        self._merge_label()

    def set_live_times(self, times):
        self._live_times = times

    def _merge_label(self):
        start_index = 0
        next_index = 0
        length = len(self._live_times)
        while next_index < length:
            if self._live_predicts[start_index] != self._live_predicts[next_index]:
                start_time = self._live_times[start_index] - self._time_threhold
                end_time = self._live_times[next_index - 1] + self._time_threhold
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                cur_label = self._live_predicts[start_index]
                self._live_merged.append((start_time, end_time, cur_label))
            else:
                next_index += 1
        start_time = self._live_times[start_index]
        end_time = self._live_times[next_index - 1]
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        self._live_merged.append(
            (start_time, end_time, self._live_predicts[start_index]))

    def get_video_time(self, time_data):
        time = time_data[0].split(':')[0]
        return time

    def cut_video(self, video_path):

        # initialize video capture parameters
        self._cap = cv2.VideoCapture(video_path)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._size = (int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        for item in self._live_merged:
            start_time = item[0]
            end_time = item[1]
            label = item[2]
            cur_time = -1
            while cur_time != start_time:
                cur_ret, cur_frame = self._cap.read()
                time_data = self._ocr_model.text_detect(cur_frame)
                cur_time = self.get_video_time(time_data)
                if cur_time != start_time:
                    for i in range(20):
                        self._cap.read()
            cur_outvideo_name = '/video/{}-{}-{}.mp4'.format(start_time, end_time, label)
            cur_outvideo = cv2.VideoWriter(cur_outvideo_name, self._fourcc, self._fps,
                                           (cur_frame.shape[1], cur_frame.shape[0]))
            while cur_frame != end_time + 1:
                cur_ret, cur_frame = self._cap.read()
                time_data = self._ocr_model.text_detect(cur_frame)
                cur_time = self.get_video_time(time_data)
                cur_outvideo.write(cur_frame)
            cur_outvideo.release()
        self._cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    text_path = 'cut-text.json'
    video_path = 'test.mp4'
    tcv = Tcv()
    sentences, targets, times = [], [], []
    with open(text_path, encoding='utf-8') as f:
        for item in json.load(f):
            sentences.append(item[0])
            times.append(item[1])
            targets.append(8)

    tcv.set_live_times(times)
    tcv.live_text_pred(sentences)
    tcv.cut_video(video_path)
