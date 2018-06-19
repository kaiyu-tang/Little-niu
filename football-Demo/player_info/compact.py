#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午12:02  
# @Author  : Kaiyu  
# @Site    :   
# @File    : compact.py
import json
import numpy as np
import sys
import os

import cv2
from PIL import Image, ImageDraw, ImageFont


def compact_data(dir, output_path):
    files = os.listdir(dir)
    players = {}
    for file in files:
        file_path = os.path.join(dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'players' in data:
                for item in data['players']:
                    players[item['full-name']] = item
    with open(output_path, 'w') as f:
        json.dump(players, f, ensure_ascii=False, indent=4, separators=(',', ': '))


def test_font(font_path, data_path):
    with open(data_path) as f:
        font = ImageFont.truetype(font_path, 60)
        data = json.load(f)
        cv2.namedWindow("Image")
        for name in data:
            im = Image.new('RGB', (800, 400), 'white')
            draw = ImageDraw.Draw(im)
            draw.text((50, 200), "0Ozil, Mesut", fill=(0, 0, 0), font=font)
            cv_frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            cv2.imshow('Image', cv_frame)
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                continue


if __name__ == '__main__':
    # dir = '/Users/harry/PycharmProjects/toys/football-Demo/player_info/foxsports-matches'
    # output_path = 'foxsport.json'
    # compact_data(dir, output_path)

    test_font("../hanyidahei.ttf", "./foxsport.json")
