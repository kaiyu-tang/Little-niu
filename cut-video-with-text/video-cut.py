#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/7/16 下午6:38  
# @Author  : Kaiyu  
# @Site    :   
# @File    : video-cut.py

import json
import re
import gc
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch
from football.tsn_config import Config
from football.tsn_inference_football import football_inference, stream
from ocr.src.ocr import OCR
from Text_classification.text_classify.predict import Predictor
from Text_classification.text_classify.data.load_data import DataLoader



class Tcv(object):
    football_infer_handler_ = football_inference(Config.weights, Config.arch, Config.modality,
                                                 Config.num_class, Config.gpu, Config.scale_size,
                                                 Config.input_size, Config.input_mean, Config.input_std,
                                                 Config.buffer_size)

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
        self._cuda = torch.cuda.is_available()
        self._video_inter = 600
        self.football_infer_handler_ = Tcv.football_infer_handler_
        pass

    def live_text_pred(self, sentences, clean=True):
        self._live_texts = sentences
        labels = [8 for _ in range(len(sentences))]
        data_iter = DataLoader(sentences, labels, clean=clean, cuda=True)
        self._live_predicts, logits = self._text_predictor.predicts(data_iter)
        self._live_predicts = list(map(int, self._live_predicts))
        self._merge_label()

        del self._text_predictor
        del data_iter
        gc.collect()

    def set_live_times(self, times):
        self._live_times = times

    def _merge_label(self):
        """
        merge the clips of same label,and save as (start_time,end_time,label)
        :return:
        """

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
                start_index = next_index
            next_index += 1
        start_time = self._live_times[start_index]
        end_time = self._live_times[next_index - 1]
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        self._live_merged.append(
            (start_time, end_time, self._live_predicts[start_index]))

    def _get_video_time(self, time_data):
        """
        convert the raw ocr result to time
        :param time_data: the ocr result
        :return: time
        """
        time_data = ' '.join(time_data)
        time = re.findall('(\d+):(\d)', time_data)
        if len(time_data) != 0:
            print()
        if len(time) == 0:
            return -1
        else:
            return int(time[0][0])

    def cut_video(self, video_path):
        """
        save according to text time and label
        :param video_path: the abs path of video
        :return: None
        """
        # initialize video capture parameters
        self._cap = cv2.VideoCapture(video_path)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._size = (int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # cut video according to text_time
        for item in self._live_merged:
            start_time = item[0]
            end_time = item[1]
            label = item[2]
            cur_time = -1
            # go to the first start_time
            count = 0
            while cur_time != start_time:
                cur_ret, cur_frame = self._cap.read()
                if not cur_ret:
                    break
                cv2.imshow('video', cur_frame)
                if cv2.waitKey(1) and 0xFF == ord('q'):
                    break
                # ocr time for every 30 frames
                count += 1
                try:
                    if count > self._video_inter:
                        count = 0
                        time_data = self._ocr_model.text_detect(cur_frame)
                        cur_time = self._get_video_time(time_data)
                        print("start_time: {} end_time: {} cur_time: {}".format(start_time, end_time, cur_time))
                        if cur_time > start_time:
                            continue
                except Exception as e:
                    print(e)
                    cv2.imwrite('error.png', cur_frame)

            # find start time and open the video writer
            cur_outvideo_name = os.path.join(os.getcwd(), 'videos/tmp.mp4')
            cur_outvideo = cv2.VideoWriter(cur_outvideo_name, self._fourcc, self._fps,
                                           (cur_frame.shape[1], cur_frame.shape[0]))
            # save video frame
            while cur_time != end_time:
                cur_ret, cur_frame = self._cap.read()
                if not cur_ret:
                    break
                cv2.imshow('video', cur_frame)
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                try:
                    if count > self._video_inter:
                        count = 0
                        time_data = self._ocr_model.text_detect(cur_frame)
                        cur_time = self._get_video_time(time_data)
                        print("start_time: {} end_time: {} cur_time: {}".format(start_time, end_time, cur_time))
                except Exception as e:
                    print(e)
                    continue
                cur_outvideo.write(cur_frame)
            # release current video
            cur_outvideo.release()
            os.system('ffmpeg -i tmp.mp4 {}-{}-{}.mp4'.format(start_time, end_time, label))
            os.remove(cur_outvideo_name)
        # release all resource
        self._cap.release()
        cv2.destroyAllWindows()

    def cut_video_penalty(self, data_path="./videos"):
        video_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(data_path) if os.path.isfile(file)]

        def cut(stream_path, window_size=100):
            stream_handle = stream(Config.test_segments, stream_path)
            cap = cv2.VideoCapture(stream_path)
            cap = cv2.VideoCapture(stream_path)
            # print(cap.isOpened())
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            outVideo = cv2.VideoWriter("./tmp.mp4", fourcc, fps, (size[0], size[1]))
            pre = []
            while stream_handle.is_init:
                stream_is_ok, frames = stream_handle.get_frame()
                if stream_is_ok:
                    probs = football_infer_handler.eval(frames)
                    dianqiu_prob = probs[Config.dianqiu_class_index]
                    dianqiu_label = dianqiu_prob > Config.dianqiu_class_thresh
                    while len(pre) > window_size:
                        pre.pop(0)
                    pre.append(frames)
                    while dianqiu_label:
                        pass



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
    sentences.reverse()
    times.reverse()
    tcv.set_live_times(times)
    tcv.live_text_pred(sentences)
    del DataLoader
    del Predictor
    gc.collect()
    tcv.cut_video(video_path)
