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
import queue
import os
import sys
import shutil

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch
from football.tsn_config import Config
from football.tsn_inference_football import football_inference, stream
from ocr.src.ocr import OCR
from Text_classification.text_classify.predict import Predictor


class Tcv(object):
    football_infer_handler_ = football_inference(Config.weights, Config.arch, Config.modality,
                                                 Config.num_class, Config.gpu, Config.scale_size,
                                                 Config.input_size, Config.input_mean, Config.input_std,
                                                 Config.buffer_size)
    predictor = Predictor()

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
        self._text_predictor = Tcv.predictor
        self._live_predicts = []
        self._live_merged = []
        self._time_threhold = 1
        self._cuda = torch.cuda.is_available()
        self._video_inter = 600
        self.football_infer_handler_ = Tcv.football_infer_handler_
        pass

    def live_text_pred(self, sentences, clean=True):
        self._live_texts = sentences
        # from Text_classification.text_classify.data.load_data import DataLoader
        # data_iter = DataLoader(sentences, [0 for _ in range(len(sentences))], clean=clean, cuda=True)
        # self._live_predicts, logits = self._text_predictor.predicts(data_iter)
        # self._live_predicts = list(map(int, self._live_predicts))
        # self._live_predicts = [i for i in self._live_predicts if i]
        # self._merge_label()
        for sentence_ in sentences:
            self._live_predicts.append(self._text_predictor.predict(sentence_)[0])
        del self._text_predictor

        # del data_iter
        gc.collect()

    def set_live_times(self, times):
        self._live_times = times
        n = [0]
        tmp_p, tmp_t = [], []
        for t, p in zip(self._live_times, self._live_predicts):
            if p not in n:
                tmp_p.append(p)
                tmp_t.append(t)
        self._live_predicts = tmp_p
        self._live_times = tmp_t
        self._merge_label()

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
            cur_outvideo_name = os.path.join(os.getcwd(), 'tmp.mp4')
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
            os.system(
                'ffmpeg -i tmp.mp4 -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high ,/videos/{}-{}-{}.mp4'.format(
                    start_time, end_time, label))
            os.remove(cur_outvideo_name)
        # release all resource
        self._cap.release()
        cv2.destroyAllWindows()

    def cut(self, stream_path, window_size=100, interval=100):
        cv2.namedWindow("Video")
        stream_handle = stream(Config.test_segments, stream_path)
        cap = cv2.VideoCapture(stream_path)
        name = stream_path.split("/")[-1][:-4]
        # print(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.release()
        pre = queue.Queue(maxsize=window_size)
        stream_is_ok = True
        while stream_handle.is_init and stream_is_ok:
            num_of_video = 0
            stream_is_ok, frame = stream_handle.get_frame()
            if stream_is_ok:
                probs = self.football_infer_handler_.eval(frame)
                dianqiu_label = probs[Config.dianqiu_class_index] > Config.dianqiu_class_thresh
                while pre.qsize() >= window_size:
                    pre.get()
                pre.put(frame)
                cv2.imshow("Video", cv2.cvtColor(np.asarray(frame[0]), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frames = []
                if dianqiu_label:
                    index_ = 0
                    while dianqiu_label or index_ < interval:
                        stream_is_ok, frame = stream_handle.get_frame()
                        if not stream_is_ok:
                            print("finished")
                            exit(1)
                        cv2.imshow("Video", cv2.cvtColor(np.asarray(frame[0]), cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        probs = self.football_infer_handler_.eval(frame)
                        dianqiu_label = probs[Config.dianqiu_class_index] > Config.dianqiu_class_thresh
                        if not dianqiu_label:
                            index_ += 1
                        else:
                            index_ = 0
                        frames.append(frame)
                if len(frames):
                    outVideo = cv2.VideoWriter("tmp.mp4", fourcc, fps, (size[0], size[1]))
                    while pre.qsize():
                        outVideo.write(pre.get())
                    for frame_ in frames:
                        outVideo.write(frame_)
                    for i in range(window_size):
                        stream_is_ok, frame = stream_handle.get_frame()
                        if not stream_is_ok:
                            print("finished")
                            exit(1)
                        cv2.imshow("Video", cv2.cvtColor(np.asarray(frame[0]), cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        if stream_is_ok:
                            outVideo.write(frame)
                        else:
                            print("video finished")
                            exit(1)
                    outVideo.release()
                    os.system(
                        'ffmpeg -i tmp.mp4 -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high {}-{}.mp4'.format(
                            name, num_of_video))
                    print("name: {} num: {}".format(name,num_of_video))
                    os.remove(os.path.join(os.getcwd(), "tmp.mp4"))
        stream_handle.destroy()
        return

    def cut_video_penalty(self, data_path="./videos"):
        video_paths = [os.path.join(os.getcwd(), "videos", file) for file in os.listdir(data_path) if
                       not os.path.isdir(file)]
        #cut("tmp.mp4")
        for video_path in video_paths:
            # # encode for nvvl
            # os.system("ffmpeg -i {} -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high tmp.mp4".format(video_path))
            # shutil.copy("tmp.mp4", video_path)
            # os.remove("tmp.mp4")
            print("starting {}".format(video_path))
            self.cut(video_path)


if __name__ == '__main__':
    text_path = 'cut-text.json'
    video_path = 'test.mp4'
    tcv = Tcv()
    sentences, targets, times = [], [], []
    with open(text_path, encoding='utf-8') as f:
        for item in json.load(f):
            sentences.append(item[0])
            times.append(item[1])
            targets.append(0)
    sentences.reverse()
    times.reverse()

    tcv.live_text_pred(sentences)
    tcv.set_live_times(times)
    del Tcv.predictor
    del Predictor
    gc.collect()
    tcv.cut_video(video_path)
    tcv.cut_video_penalty()
