#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 上午10:17  
# @Author  : Kaiyu  
# @Site    :   
# @File    : main.py
import os

from PIL import ImageDraw, ImageFont
from face_reco.reco_main import PlayerFaceReco
from ..football.tsn_config import Config
from ..football.tsn_inference_football import football_inference, stream
from io import BytesIO
import cv2
import numpy as np
from player_info.player_info import player_info
import sys

if __name__ == '__main__':
    args = sys.argv
    stream_path = args[1]
    output_video_name = args[2]
    # stream_path = "./positive-28709-40414-40498.mp4"
    # stream_path = "./Video/vs Cvs C.mp4"  # '/home/atlab/Workspace/kaiyu/Demo/toys/football-Demo/football/test_data/football_PK_test.mp4'
    # player_info_path = './player_info/foxsport.json'
    # output_video_name = "./test-C.mp4"
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
    # print(stream_path)
    cap = cv2.VideoCapture(stream_path)
    # print(cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outVideo = cv2.VideoWriter("./tmp.mp4", fourcc, fps, (size[0], size[1]))
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
                    if len(player_basic) > 1:
                        player_basic = player_basic['performance']
                        # player_basic = player_basic['performance']
                        if "penalty_kick_goals" in player_basic:
                            pkg = player_basic["penalty_kick_goals"]
                        if "penalty_kick" in player_basic:
                            pk = player_basic["penalty_kick"]
                        img_text = 'Name: {} PKG:{} PK:{}'.format(player_name.replace(" ", ", "), pkg, pk, )

                print(img_text)

                if dianqiu_label:
                    draw = ImageDraw.Draw(frames[0])
                    font = ImageFont.truetype("XHDF.ttf", 35)
                    draw.text((10, 8), img_text, fill=(255, 255, 255), font=font)

                    # cv2.putText(cv_frame, img_text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv_frame = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_RGB2BGR)
                # cv_frame = cv_frame.copy()
                # draw = ImageDraw.Draw(frames[0])
                # draw.text((0, 40), 'Nicolás', fill=(255, 255, 255))
                # frames[0].show()
                # cv_text_frame = cv2.putText(cv_frame.copy(), img_text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                             (255, 255, 255),
                #                             2)
                c = outVideo.write(cv_frame)
                # print(c)
                cv2.imshow('Image', cv_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            stream_handle.destroy()
            # cv2.waitKey(0)

    outVideo.release()
    cap.release()
    cv2.destroyAllWindows()
    os.system('ffmpeg -i tmp.mp4 {}'.format(output_video_name))
    os.remove("./tmp.mp4")
