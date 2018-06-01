#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/29 下午3:05  
# @Author  : Kaiyu  
# @Site    :   
# @File    : TextCNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        Ci = 1
        kernel_num = args.kernel_num
        kernel_size = args.kernel_size
        dropout = args.dropout

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.conv1s = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size)*kernel_num, class_num)


    def forward(self, x):
        x = self.embed(x)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1s]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
