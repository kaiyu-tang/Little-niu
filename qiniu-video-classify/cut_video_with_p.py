import random

import cv2
from sys import argv
import os
import sys
import time


#  cut the video of the probablity larger than threshold
def cut(csv_name, video_name, threshold):
    csv_name = csv_name
    video_name = video_name
    print(csv_name)
    print(video_name)
    frames = []
    data = []
    idx = 0

    with open(csv_name, "r") as f:  # load data
        for line in f:
            if idx == 0:
                idx += 1
                continue
            line = line.strip('\n').split(',')
            frames.append((int(line[0]), int(line[1])))
            end_ = float(line[2])
            data.append(end_)
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = -1
    res = []
    index = 0
    print("load primary data")
    data_length = len(data)
    while index < data_length:  # save the video segments
        start = -1
        end = float('inf')
        while index < data_length and data[index] >= threshold:
            if start == -1:
                start = frames[index][0]
            if frames[index][0] <= end:
                end = frames[index][1]
                index += 1
            else:
                break
        index += 1
        if start != -1:
            res.append((start, end))
    # print(res)
    index_frame = 0
    index = 0
    print("finished load data")
    res_length = len(res)
    video_root, video_name = os.path.split(video_name)
    if not os.path.exists(os.path.join(video_root, "positive")):
        os.mkdir(os.path.join(video_root, "positive"))
    if not os.path.exists(os.path.join(video_root, "negative")):
        os.mkdir(os.path.join(video_root, "negative"))
    ret, image = cap.read()
    size = (image.shape[1], image.shape[0])
    while index_frame < num_frames:  # find and save the segments that probablity larger than
        # threshold
        video_name = str(random.randint(0, 99999999))
        if index < res_length and res[index][0] <= index_frame:
            start, end = res[index]
            file_name = os.path.join(video_root, "positive",
                                     "positive-" + video_name + "-" + str(start) + "-" + str(end) + ".flv")
            file_name.strip()
            outVideo = cv2.VideoWriter(file_name, fourcc, fps, size)

            while index_frame < end:
                ret, image = cap.read()
                outVideo.write(image)
                # time.sleep(0.00001)
                index_frame += 1
            index += 1
            outVideo.release()

        else:

            if index < res_length:
                end_ = res[index][0]
            else:
                end_ = num_frames
            file_name = os.path.join(video_root, "negative",
                                     "negative-" + video_name + "-" + str(index_frame) + "-" + str(end_) + ".flv")
            print(file_name)
            print(index_frame)
            print(end_)
            outVideo = cv2.VideoWriter(file_name, fourcc, fps, size)
            print(file_name)

            while index_frame < end_:
                ret, image = cap.read()
                outVideo.write(image)
                index_frame += 1
                # time.sleep(0.00001)
            outVideo.release()

    cap.release()
    # outVideo.release()
    cv2.destroyAllWindows()


def cut_dir(csv_root, video_root, threshold):
    print("start")
    csv_file_name = [name for name in os.listdir(csv_root)]
    video_file_name = [name for name in os.listdir(video_root)]
    csv_file_name.sort()
    video_file_name.sort()
    csv_nums = len(csv_file_name)
    video_nums = len(video_file_name)
    index_csv = 0
    index_video = 0
    while index_csv < csv_nums and index_video < video_nums:
        while index_video < video_nums:
            if video_file_name[index_video] + ".csv" != csv_file_name[index_csv] and \
                    ("result-" + video_file_name[index_video] + ".csv" != csv_file_name[index_csv]):
                index_video += 1
            else:
                break
        if index_video < video_nums:

            video_name_ = os.path.join(video_root, video_file_name[index_video])
            csv_name_ = os.path.join(csv_root, csv_file_name[index_csv])
            if os.path.isfile(video_name_) and os.path.isfile(csv_name_):
                cut(csv_name_, video_name_, threshold)
                print(video_file_name[index_video])
                print(csv_file_name[index_csv])
            index_csv += 1
    print("end")


if __name__ == '__main__':
    if sys.getdefaultencoding() != 'utf-8':
        print('No!!!your encoding is not utf-8')
    csv_name = argv[1]
    video_name = argv[2]
    threshold = float(argv[3])
    if os.path.isfile(csv_name) and os.path.isfile(video_name):
        cut(csv_name, video_name, threshold)
    if os.path.isdir(csv_name) and os.path.isdir(video_name):
        cut_dir(csv_name, video_name, threshold)
