#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 上午10:17  
# @Author  : Kaiyu  
# @Site    :   
# @File    : main.py
from PIL import ImageDraw, ImageFont
from face_reco.reco_main import PlayerFaceReco
from football.tsn_config import Config
from football.tsn_inference_football import football_inference, stream
from io import BytesIO
import json
import cv2
import numpy as np
from player_info.player_info import player_info
import matplotlib.pyplot as plt

if __name__ == '__main__':
    stream_path = "./positive-28709-40414-40498.mp4"
    stream_path = "./football/test_data/football_PK_test.mp4"  # '/home/atlab/Workspace/kaiyu/Demo/toys/football-Demo/football/test_data/football_PK_test.mp4'
    player_info_path = './player_info/foxsport.json'
    output_video_name = "./test1.mp4"
    football_infer_handler = football_inference(Config.weights, Config.arch, Config.modality,
                                                Config.num_class, Config.gpu, Config.scale_size,
                                                Config.input_size, Config.input_mean, Config.input_std,
                                                Config.buffer_size)
    player_face_reco = PlayerFaceReco()
    player_inf = player_info()
    stream_handle = stream(Config.test_segments, stream_path)
    test_status = 0  # test_status: 0 do inference, others skip
    player_name = ""
    pkg = ""
    pk = ""
    dianqiu_label = False
    # font = ImageFont.truetype（24)
    print(stream_path)
    cap = cv2.VideoCapture(stream_path)
    print(cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outVideo = cv2.VideoWriter(output_video_name, fourcc, fps, (size[0], size[1]))
    print('size:{} fps:{} open:{}'.format(size, fps, outVideo.isOpened()))
    # print(outVideo.isOpened())
    cv2.namedWindow("Image")
    while stream_handle.is_init:
        stream_is_ok, frames = stream_handle.get_frame()
        if stream_is_ok:
            if test_status == 0:
                a = BytesIO()
                frames[0].save(a, 'png')
                a.getvalue()
                player_cur_name = player_face_reco.reco(a.getvalue(), local=2)
                probs = football_infer_handler.eval(frames)
                dianqiu_prob = probs[Config.dianqiu_class_index]
                dianqiu_label = dianqiu_prob > Config.dianqiu_class_thresh

                img_text = 'Name: {} PKG:{} PK:{}'.format(player_name.replace(" ", ", "), pkg, pk, )
                if player_cur_name != '':
                    player_name = player_cur_name
                    player_basic = player_inf.get_info(player_name)
                    if "penalty_kick_goals" in player_basic:
                        pkg = player_basic["penalty_kick_goals"]
                    if "penalty_kick" in player_basic:
                        pk = player_basic["penalty_kick"]
                    img_text = 'Name: {} PKG:{} PK:{}'.format(player_name.replace(" ", ", "), pkg, pk, )

                print(img_text)
                cv_frame = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_RGB2BGR)
                if dianqiu_label:
                    plt.imshow(cv_frame)
                    plt.text(0, 40, img_text, fontdict={'size': '16', 'color': 'white'})
                    plt.show()


                # cv_frame = cv_frame.copy()
                # draw = ImageDraw.Draw(frames[0])
                # draw.text((0, 40), 'Nicolás', fill=(255, 255, 255))
                # frames[0].show()
                # cv_text_frame = cv2.putText(cv_frame.copy(), img_text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                             (255, 255, 255),
                #                             2)
                outVideo.write(cv_frame)
                cv2.imshow('Image', cv_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            stream_handle.destroy()

    outVideo.release()
    cap.release()
    cv2.destroyAllWindows()
