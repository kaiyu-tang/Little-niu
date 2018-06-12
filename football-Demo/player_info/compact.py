#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/6/6 下午12:02  
# @Author  : Kaiyu  
# @Site    :   
# @File    : compact.py
import json
import sys
import os


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


if __name__ == '__main__':
    dir = '/Users/harry/PycharmProjects/toys/football-Demo/player_info/foxsports-matches'
    output_path = 'foxsport.json'
    compact_data(dir, output_path)
