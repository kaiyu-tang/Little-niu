#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/29 下午3:05  
# @Author  : Kaiyu  
# @Site    :   
# @File    : Model.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from Config import Config


class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()

        self.name = "TextCNN"

        embed_num = Config.sequence_length
        embed_dim = Config.embed_dim
        class_num = Config.cls
        inchanel = 1
        kernel_num = Config.options["TextCNN"]["kernel_num"]
        kernel_size = Config.options["TextCNN"]["kernel_size"]
        dropout = Config.options["TextCNN"]["drop_out"]

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.conv1s = nn.ModuleList([nn.Conv2d(inchanel, kernel_num, (K, embed_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def forward(self, x):
        # print()
        # x = self.embed(x)
        # if Config.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1s]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()


class TextRNN(nn.Module):
    def __init__(self, embed_size=Config.embed_dim, hid_size=Config.options["TextRNN"]["hidden_size"], cls=Config.cls,
                 cell=None, dropout=Config.options["TextRNN"]["dropout"]):
        super(TextRNN, self).__init__()
        self.name = "TextRNN"
        self._embed_size = embed_size
        self._hid_size = hid_size
        self._cls = cls
        self._drop = dropout
        if cell == "RNN":
            self._wordRNN = nn.RNN(self._embed_size, hidden_size=hid_size, bidirectional=True, dropout=self._drop)
        elif cell == "GRU":
            self._wordRNN = nn.GRU(self._embed_size, hidden_size=hid_size, bidirectional=True, dropout=self._drop)
        else:
            self._wordRNN = nn.LSTM(self._embed_size, hidden_size=hid_size, bidirectional=True, dropout=self._drop)

        self._relu = nn.ReLU()
        self._tanh = torch.tanh
        # attention
        self._word_atten = nn.Linear(2 * hid_size, 2 * hid_size)
        self._attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)
        # classify
        self._cls_linear = nn.Linear(2 * hid_size, self._cls)

    def _attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = rnn_outputs * att_weights
        return torch.sum(attn_vectors, 1)

    def forward(self, inp, hidden_state):
        out_state, hid_state = self._wordRNN(inp, hidden_state)
        word_attention = self._tanh(self._word_atten(out_state))
        # word_attention = F.tanh(word_attention)
        attn = F.softmax(self._attn_combine(word_attention), dim=1)
        sent = self._attention_mul(out_state, attn)
        cls = self._cls_linear(sent)

        return cls, hid_state


class BasicConvResBlock(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_size, padding=1, stride=1, shortcut=True, downsample=None):
        super(BasicConvResBlock, self).__init__()

        self._input_dim = input_dim
        self._n_filters = n_filters
        self._kernel_szie = kernel_size
        self._padding = padding
        self._stride = stride
        self._short_cut = shortcut
        self._downsample = downsample

        self._conv1 = nn.Conv1d(self._input_dim, self._n_filters, kernel_size=self._kernel_szie, padding=self._padding,
                                stride=self._stride)
        self._bn1 = nn.BatchNorm1d(self._n_filters)
        self._relu = nn.ReLU()
        self._conv2 = nn.Conv1d(self._n_filters, self._n_filters, kernel_size=self._kernel_szie, padding=self._padding,
                                stride=self._stride)
        self._bn2 = nn.BatchNorm1d(self._n_filters)

    def forward(self, x):
        residual = x

        out = self._conv1(x)
        out = self._bn1(out)
        out = self._relu(out)

        out = self._conv2(out)
        out = self._bn2(out)

        if self._short_cut:
            if self._downsample is not None:
                residual = self._downsample(residual)
            try:
                out += residual
            except Exception as e:
                # conv1 = nn.Conv1d(256, 512, kernel_size=1, padding=1, bias=False).cuda()
                # bn = nn.BatchNorm1d(512).cuda()
                print(e)

        return self._relu(out)


class TextVDCNN(nn.Module):
    def __init__(self, embed_dim=Config.embed_dim, n_class=Config.cls, voca_size=None,
                 net_depth=Config.options["TextVDCNN"]["net_depth"],
                 n_fc_neurons=Config.options["TextVDCNN"]["n_fc_neurons"],
                 shortcut=Config.options["TextVDCNN"]["shortcut"]):
        super(TextVDCNN, self).__init__()
        layers, fc_layers = [], []
        self.name = "TextVDCNN"
        #layers.append(torch.nn.Embedding(num_embeddings=voca_size, embedding_dim=embed_dim))
        layers.append(BasicConvResBlock(embed_dim, n_filters=64, kernel_size=3, shortcut=False))

        if net_depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif net_depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif net_depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif net_depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3
        else:
            print("Wrong net depth")
        layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, shortcut=shortcut))
        for _ in range(n_conv_block_64 - 1):
            layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        downsample = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
        layers.append(BasicConvResBlock(64, 128, kernel_size=3, shortcut=shortcut, downsample=downsample))
        for _ in range(n_conv_block_128 - 1):
            layers.append(BasicConvResBlock(128, 128, kernel_size=3, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        downsample = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        layers.append(BasicConvResBlock(128, 256, kernel_size=3, shortcut=shortcut, downsample=downsample))
        for _ in range(n_conv_block_256 - 1):
            layers.append(BasicConvResBlock(256, 256, kernel_size=3, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        downsample = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=False), nn.BatchNorm1d(512))
        layers.append(BasicConvResBlock(256, 512, kernel_size=3, shortcut=shortcut, downsample=downsample))
        for _ in range(n_conv_block_512 - 1):
            layers.append(BasicConvResBlock(512, 512, kernel_size=3, shortcut=shortcut))

        if Config.options[self.name]["last_pool"] == "k-max":
            layers.append(nn.AdaptiveMaxPool1d(8))
            fc_layers.append(nn.Linear(8 * 512, n_fc_neurons))
            fc_layers.append(nn.ReLU())
        elif Config.options[self.name]["last_pool"] == "max":
            layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
            fc_layers.append(nn.Linear(61 * 512, n_fc_neurons))
            fc_layers.append(nn.ReLU())
        else:
            raise NameError

        fc_layers.append(nn.Linear(n_fc_neurons, n_fc_neurons))
        fc_layers.append(nn.ReLU())

        fc_layers.append(nn.Linear(n_fc_neurons, n_class))

        self._layers = nn.Sequential(*layers)
        self._fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):

        out = self._layers(x)
        out = out.transpose(1, 2)
        out = out.contiguous().view(out.shape[0], -1)
        out = self._fc_layers(out)
        return out
