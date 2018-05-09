# -*- coding: utf-8 -*-


import cv2
from sys import argv


#  cut the video of the probablity larger than threshold
def cut(csv_name, video_name, threshold):
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
            end_ = float(line[3])
            if line[2] == '0':
                end_ = 1 - end_
            data.append(end_)
    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    res = []
    index = 0
    print("load primary data")
    data_length = len(data)
    while index < data_length:  # save the video segments
        start = -1
        end = -1
        while data[index] >= threshold:
            if start == -1:
                start = frames[index][0]
                end = frames[index][1]
            if frames[index][1] <= end:
                end = frames[index][1]
            index += 1
        index += 1
        if start != -1:
            res.append((start, end))
    index_frame = 0
    index = 0
    print("finished load data")

    res_length = len(res)
    max_frame = frames[-1][1]
    while index_frame < max_frame and index < res_length:  # find and save the segments that probablity larger than
        # threshold

        ret, image = cap.read()
        if index_frame == res[index][0]:
            start, end = res[index]
            file_name = video_name[:-4] + "-" + str(start) + "-" + str(end) + ".avi"
            outVideo = cv2.VideoWriter(file_name, fourcc, 25, (image.shape[1], image.shape[0]))
            print(file_name)

            while index_frame < end:
                outVideo.write(image)
                ret, image = cap.read()
                index_frame += 1
            index += 1
        index_frame += 1
    cap.release()
    outVideo.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    csv_name = argv[1]
    video_name = argv[2]
    threshold = float(argv[3])
    cut(csv_name, video_name, threshold)
