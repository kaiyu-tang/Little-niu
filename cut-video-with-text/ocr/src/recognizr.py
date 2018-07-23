#encoding:utf8
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from utils import keys
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageOps

import torch
from torch.autograd import Variable
import torch.nn as nn

debug = False

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 69)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Recognizer():
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), "models/recognizer_model.v2.pth"), keys=keys):
        self.model = LeNet()

        self.model.load_state_dict(torch.load(model_path))
        self.transform = trans
        self.keys = keys

    def post_process(self, text):
        if text == "":
            return text

        if text[-1] == "-" and len(text) >= 8 and len(text) <= 9:
            text = text[0:-1] + "'"

        colon_index = text.find(":")
        if colon_index - 2 >= 0 and colon_index + 3 <= len(text):
            if "+" not in text and "'" not in text:
                text = text[colon_index - 2:colon_index + 3]
            elif colon_index + 6 <= len(text):
                accent_index = text[colon_index + 1:].find("'") + colon_index + 2

                if accent_index <= len(text):
                    text = text[colon_index - 2:accent_index]

        rail_index = text.find("-")
        if rail_index - 1 > 0 and rail_index + 1 <= len(text) - 1:
            if text[rail_index - 1].isnumeric() and text[rail_index + 1].isnumeric():
                res = ""
                for t in text:
                    if (t >= '0' and t <= '9') or t == "-":
                        res += t
                return res

        if text.find("'") >= 0 and text.find("+") == -1:
            text = text.replace("'", "")
        up = 0
        down = 0
        for t in text:
            if t > 'A' and t < 'Z':
                up += 1
            elif t > 'a' and t < 'z':
                down += 1

        if up > down:
            return text.upper()
        else:
            if text[0].isupper():
                return text.lower().capitalize()
            return text.lower()

    def infer(self, img):
        region_imgs = self._seg_img(img)
        self.model.eval()
        result = ""

        for img in region_imgs:

            img = cv2.bitwise_not(img)
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            output = self.model(img.view(1, 1, 28, 28))

            _, predict = torch.max(output, 1)

            result += self.keys[predict.item()]

        return region_imgs, self.post_process(result)

    def inverse(self, img):

        w, h = img.shape[0:2]

        color = int(img[1, 1] / 4.0 + img[1, h - 2] / 4.0 + img[w - 2, 1] / 4.0 + img[w - 2, h - 2] / 4.0)
        if color < 128:
            return cv2.bitwise_not(img)
        return img

    def _seg_img(self, src_img):
        '''

        :param src_img:  source image opencv 
        :return: character img chips
        '''
        src_img = self.inverse(src_img)
        scale = src_img.shape[1] * 1.0 / 32
        weight = src_img.shape[0] / scale

        retval, img = cv2.threshold(src_img, 105, 255, cv2.THRESH_BINARY_INV)

        w = [0] * img.shape[1]

        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                t = img[y, x]
                if t != 0:
                    w[x] += 1

        # print(w)
        if len(w) > 1 and w[-1] != 0 and w[-2] == 0:
            w[-1] = 0
        if len(w) > 1 and w[0] != 0 and w[1] == 0:
            w[0] = 0
        start = 0
        status = 0
        regions = []
        for i in range(0, len(w)):
            if w[i] != 0 and w[i] != 1 and status == 0:
                status = 1
                start = i - 1
                if start < 0:
                    start = 0

            elif (w[i] == 0 or w[i] == 1) and status == 1:

                regions.append([start, i])
                status = 0

        if status == 1:
            regions.append([start, len(w) - 1])

        # print(regions)
        region_imgs = []

        for region in regions:
            if region[1] - region[0] <= 1:
                if region[0] > 0:
                    region[0] = region[0] - 1
                elif region[1] < len(w) - 1:
                    region[1] = region[1] + 1

            if region[1] - region[0] > 20:
                continue

            crop_img = src_img[:, region[0]:region[1] + 1]
            height, width = crop_img.shape[0:2]

            if width < height:
                padding = int((height - width) / 2)

                color = (
                    crop_img[1, 1] / 4 + crop_img[height - 1, width - 1] / 4 + crop_img[height - 1, 1] / 4 + crop_img[
                        1, width - 1] / 4)

                crop_img = cv2.copyMakeBorder(crop_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=color)
            crop_img = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_CUBIC)

            region_imgs.append(crop_img)

        return region_imgs


if __name__ == "__main__":

    rec = Recognizer()
    img = cv2.imread("labeled/1/3.jpeg", 0)
    rec._seg_img(img)
    # rec._seg_by_contour(img)
    if not os.path.exists("chips1"):
        os.mkdir("chips2")

    files = os.listdir("labeled/1")
    for fname in sorted(files, key=lambda s: s[0]):

        img = cv2.imread("labeled/1/" + fname, 0)
        if img is None:
            print("error img:", fname)
            continue
        chips, text = rec.infer(img)
        print(fname, ":", text)

        for i, chip in enumerate(chips):
            chip = cv2.bitwise_not(chip)
            cv2.imwrite("chips1/" + fname + "-" + str(i) + ".png", chip)
