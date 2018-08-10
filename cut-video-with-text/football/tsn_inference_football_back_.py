# coding=utf-8
from __future__ import print_function

import torch.nn.parallel
import torchvision
from tsn_config import Config
from tsn_model import TSN
from transforms import *
from PIL import Image
#import nvvl
import numpy as np
import base64
import datetime
import hashlib
import json
import requests
import cv2


class stream(object):
    def __init__(self, test_segment, stream_path):
        self.test_segment = test_segment
        self.video_reader = nvvl.VideoReader(0, "warn")
        self.is_init = self.init(stream_path)

    def init(self, stream_path):
        """init video reader
        :param stream_path: input stream address
        :return: init status
        """
        try:
            self.video_reader.read_stream(stream_path)
            self.is_init = True
        except Exception as e:
            print('Read stream error: ', stream_path)
            self.is_init = False
        return self.is_init

    def get_frame(self):
        """get frame from stream
        :return: status, images
        """
        if not self.is_init:
            return False, None

        try:
            #current_time = time.time()
            frame_num, tensor_imgs = self.video_reader.stream_receive(self.test_segment)
            #print("nvvl reader time:", time.time() - current_time)
            if frame_num == -1:
                return False, None
            else:
                images = list()
                for tensor in tensor_imgs:
                    tensor_img = tensor.numpy().astype(np.uint8)
                    img = Image.fromarray(tensor_img)
                    images.append(img)
                return True, images
        except Exception as e:
            return False, None

    def destroy(self):
        """destroy video reader
        """
        self.video_reader.destroy()
        self.is_init = False


class football_inference(object):
    def __init__(self, model_file, model_arch, modality, num_class,
                 gpu, scale_size, input_size, input_mean, input_std,
                 buffer_size):
        """init football inference handler
        """
        self.model = self._init_models(model_file, model_arch,
                                     modality, num_class, gpu)
        self.image_transformer = self._init_transform(scale_size, input_size,
                                                      input_mean, input_std)
        self.buffer_size = buffer_size
        self.confidence_buffer = np.zeros((self.buffer_size, num_class))
        self.buffer_pos = 0 # buffer_pos: 0 ~ (buffer_size-1)

        self.num_class = num_class

    def reset(self):
        self.confidence_buffer = np.zeros((self.buffer_size, self.num_class))
        self.buffer_pos = 0


    def _init_models(self, model_file, model_arch, modality, num_class, gpu):
        """init model
        :param model_file: model weight file
        :param model_arch: model architecture, e.g. resnet152
        :param modality: RGB
        :param num_class: num of classes
        :param gpu: which gpu to use
        :return: inited model object
        """
        net = TSN(num_class, 1, modality, model_arch)
        checkpoint = torch.load(model_file)
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        net.load_state_dict(base_dict)
        net = torch.nn.DataParallel(net.cuda(gpu), device_ids=[gpu])
        net.eval()
        return net

    def _init_transform(self, scale_size, input_size, input_mean, input_std):
        """init image transform, including crop,stack,norm
        :param scale_size: resize image to shorter size = scale size
        :param input_size: crop image to input size
        :param input_mean: norm param
        :param input_std: norm param
        :return: image_transform
        """
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])

        image_transform = torchvision.transforms.Compose([
            cropping,
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)])

        return image_transform

    def _image_preprocess(self, images):
        """image preprocss
        :param images: 
        :param image_transform: 
        :return: transformed images
        """
        processed_images = self.image_transformer(images)

        return processed_images

    def eval(self, frames):
        """ eval a video
        :return: probs for all classes
        """
        #start_time = time.time()
        transformed_frames = self._image_preprocess(frames)
        transformed_frames.unsqueeze_(0)
        #print("image preprocess time:", time.time() - start_time)

        #start_time = time.time()
        input_var = torch.autograd.Variable(transformed_frames.view(-1, 3, transformed_frames.size(2),
                                            transformed_frames.size(3)), volatile=True)
        rst = self.model(input_var).data.cpu().numpy().copy()
        rst = rst.reshape(-1, self.num_class)
        probs = rst.mean(axis=0)
        #print("eval time:", time.time() - start_time)

        self.confidence_buffer[self.buffer_pos] = probs
        avg_probs = self.confidence_buffer.mean(axis=0)

        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size

        return avg_probs


def do_post(content_id, dianqiu_prob):
    inter = 'http://180.168.69.13:8050/api/penalty/penaltyControl/sendMessage'

    appid = '104020210403'
    key = '72099BD40A287CB19A4D68F0448EE3DD'

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    msg = {'contentId':'','actionType':'','score':'','actionTime':''}
    msg['contentId'] = content_id
    msg['actionType'] = '1'
    msg['score'] = dianqiu_prob
    msg['actionTime'] = time

    message_content = base64.b64encode(json.dumps(msg).encode())

    signdata ='{}{}'.format(time, key)
    hl = hashlib.md5()
    hl.update(signdata.encode(encoding='utf-8'))
    sign = hl.hexdigest()

    param = {'appid':appid, 'messageContent':message_content, 'time':time, 'sign':sign}

    response = requests.get(inter, param)

    print(response.url)

    if response.status_code == requests.codes.ok:
        response.encoding = 'utf-8'
        response_json = json.loads(response.text)
        if response_json['code'] != 0:
            print('post failed:', response_json)
    else:
        print('post failed:', response)


def main(stream_path, content_id, is_stream=True):
    """main entrance for football inference
    :param stream_path: input stream address or video file path
    :param content_id: post id
    :param is_stream: is stream or video file
    :return: 
    """
    football_infer_handler = football_inference(Config.weights, Config.arch, Config.modality,
                                                Config.num_class, Config.gpu, Config.scale_size,
                                                Config.input_size, Config.input_mean, Config.input_std,
                                                Config.buffer_size)
    cap = cv2.VideoCapture(stream_path)

    connect_count = 5
    test_status = 0 # test_status: 0 do inference, others skip
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if test_status == 0:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                probs = football_infer_handler.eval(frame)
                dianqiu_prob = probs[Config.dianqiu_class_index]
                dianqiu_label = dianqiu_prob > Config.dianqiu_class_thresh
                #if dianqiu_label:
                    #do_post(content_id, dianqiu_prob)
            test_status = (test_status + 1) % Config.test_interval
        else:
            stream_handle.destroy()
            '''
        elif is_stream: #reconnect for several times
            print('Stream is wrong, try to connect in', str(connect_count), 'times')
            while connect_count > 0:
                stream_is_init = stream_handle.init(stream_path)
                if stream_is_init:
                    football_infer_handler.reset()
                    test_status = 0
                    connect_count = 5
                    break
                else:
                    sleep(5)
                connect_count -= 1

                if connect_count == 0: # if stream cannot open, destroy the nvvl video reader
                    stream_handle.destroy()
        
            '''


if __name__ == '__main__':
    #stream_path = 'rtmp://101.89.132.156/live/footballgame'
    stream_path = '/home/atlab/Workspace/kaiyu/football-Demo/Alg-VideoAlgorithm/demos/football/test_data/football_PK_test.mp4'
    content_id = '12345'
    main(stream_path, content_id)
