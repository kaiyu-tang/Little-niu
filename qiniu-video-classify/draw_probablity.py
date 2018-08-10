# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import argv


# plot video and probablity curve
def simple_plot(csv_name, video_name, threhold=0.9):
    """
    simple plot
    """
    frames = []
    data = []
    idx = 0
    with open(csv_name, "r") as f:  # read data from csv
        for line in f:
            if idx == 0:
                idx += 1
                continue
            line = line.strip('\n').split(',')
            frames.append(int(line[0]))
            tmp = float(line[3])
            if line[2] == '0':
                tmp = 1 - tmp
            data.append(tmp)

    max_frames = frames[-1]
    cap = cv2.VideoCapture(video_name)
    plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.subplot()
    index_frame = 0
    windows_size = 50
    index = 0
    tmp_y = [np.NaN for j in range(windows_size)]  # NaN represent show nothing
    threhold_line = np.array([threhold for _ in range(windows_size)])
    while (index_frame < max_frames):
        ret, fram = cap.read()
        cv2.imshow('', fram)
        cv2.waitKey(40)
        index_frame += 1
        # ax.set_ylim(-0.02, 1.02)

        if index_frame == frames[index]:  # if it's time to show the probablity, show it!
            index += 1
            plt.cla()
            if index < windows_size:
                tmp = 0
                for j in range(index):
                    tmp_y[tmp] = data[j]
                    tmp += 1
                tmp_x = np.linspace(0, frames[windows_size], windows_size)
            else:
                tmp_y = data[index - windows_size: index]
                tmp_x = frames[index - windows_size:index]
            ax.set_title("Prediction Rate")
            '''
            segments_x = np.r_[x[0], x[1:-1].repeat(2), x[-1]].reshape(-1,2)
            segments_y = np.r_[tmp_y[0], tmp_y[1:-1].repeat(2), tmp_y[-1]].reshape(-1,2)
            linecolors = ['red' if y_[0]>threhold and y_[1] > threhold else 'blue'
                          for y_ in segments_y]
            segments = [zip(x_, y_) for x_, y_ in zip(segments_x,segments_y)]
            ax.add_collection(LineCollection(segments, colors=linecolors))
            '''
            plt.ylim(-0.02, 1.02)
            tmp_x = list(map(lambda x_: x_ / 25, tmp_x))
            # print(tmp_y)
            plt.plot(tmp_x, tmp_y, color='blue', linewidth='2')
            plt.plot(tmp_x, threhold_line, color='red', linestyle=':')
            plt.pause(0.001)

        if index_frame % 200 == 0:
            plt.clf()
    return


if __name__ == '__main__':
    csv_name = argv[1]
    video_name = argv[2]
    if len(argv) > 3:
        threshold = float(argv[3])
        simple_plot(csv_name, video_name, threshold)
    else:
        simple_plot(csv_name, video_name)
