#!/usr/bin/env python
# -*- coding:utf-8 -*-

from qiniu import QiniuMacAuth
import numpy as np

import json
import requests
import struct


class PlayerFaceReco(object):
    def __init__(self):
        self.access_key = "2jtSAmw7Dcbp0zzNH7JCYXkr0iQwMzlSgVSZvU4Y"  # 密钥
        self.secret_key = "Vr5Ys_YRtGjxix4e6LnSOJH5GXDGiKUoCtUPxkqA"
        self.source_path = './feature_lib/face_reco_v2.json'  # 球星特征库的路径

    # def get_pic_path(self):
    #     '''
    #     获取待识别的图片路径
    #     :return:
    #     ：这边就直接返回了
    #     '''
    #     return ['http://p97qu3dad.bkt.clouddn.com/1.jpg']

    def face_reco(self, img_url):
        """
        获取人脸检测的信息
        :param img_url:bucket上面的图片url
        :return:
        """
        req_data = {"data": {"uri": img_url}}
        req_url = 'http://argus.atlab.ai/v1/face/detect'
        token = QiniuMacAuth(self.access_key, self.secret_key).token_of_request(
            method='POST',
            host='argus.atlab.ai',
            url="/v1/face/detect",
            content_type='application/json',
            qheaders='',
            body=json.dumps(req_data)
        )
        token = 'Qiniu ' + token
        headers = {"Content-Type": "application/json", "Authorization": token}
        response = requests.post(req_url, headers=headers, data=json.dumps(req_data))
        return response.text

    def facex_feature(self, img_url, pts):
        """
        获取人脸的特征信息，返回512维度的特征表示
        :param pts:
        :param img_url:
        :return:
        """
        req_data = {'data': {'uri': img_url, 'attribute': {'pts': pts}}}
        req_url = 'http://serve.atlab.ai/v1/eval/facex-feature-v2'
        token = QiniuMacAuth(self.access_key, self.secret_key).token_of_request(
            method='POST',
            host='serve.atlab.ai',
            url="/v1/eval/facex-feature-v2",
            content_type='application/json',
            qheaders='',
            body=json.dumps(req_data)
        )
        token = 'Qiniu ' + token
        headers = {"Content-Type": "application/json", "Authorization": token}

        face_info_feature = []
        try:
            response = requests.post(req_url, headers=headers, data=json.dumps(req_data))
            for val in struct.unpack('>512f', response.content):
                face_info_feature.append(val)
            # print 'face_info', face_info
        except Exception as e:
            print(e)

        return face_info_feature

    def get_labelX_json(self, img_url, pts, en_name):
        """
        转化成labelx的json数据类型,添加训练数据的时候用的
        :return:
        """
        labelX_josn = {'url': img_url,
                       'type': 'image',
                       'label': [
                           {'name': en_name,
                            'type': 'detection',
                            'version': '1',
                            'data': [
                                {'bbox': pts,
                                 "class": "person"}
                            ]}
                       ]}

        return labelX_josn

    def calculate_cosine_distance(self, feature_library_matrix, pic_feature_vector):
        """
        获取最相似的特征索引
        :param feature_library_matrix:
        :param pic_feature_vector:
        :return:最相似特征的索引，最高的相似度
        """

        library_matrix_normal_all = []
        for item in feature_library_matrix:
            library_matrix_normal_all.append(np.linalg.norm(item))  # 生成特征矩阵的向量normal
        feature_vector_normal = np.linalg.norm(pic_feature_vector)  # 生成检测图片的向量normal

        # 进行矩阵点乘预算
        dot_result = np.dot(feature_library_matrix, pic_feature_vector)

        # 计算余弦距离
        cosine_result = []
        for i in range(len(dot_result)):
            cosine_result.append(dot_result[i][0] / (library_matrix_normal_all[i] * feature_vector_normal))

        # #获取最相似的top10下标
        # print(max(cosine_result))
        # result_top_10 = []
        # for j in range(0,10):
        #     result_top_10.append(cosine_result.index(sorted(cosine_result)[(j-10)]))

        return cosine_result.index(max(cosine_result)), max(cosine_result)

    def get_most_similarity_label(self, video_pic_feature, player_feature_library_url):
        """
        获取与图片识别中的球星最相似的球星姓名
        :param video_pic_feature: 视频分解后图片中的球星的特征
        :param player_feature_library_url:球星特征库的地址
        :return:最相似的球星姓名
        """

        feature_library = []
        feature_label = []
        with open(player_feature_library_url, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                player_feature_library = json.loads(line)
                player_feature = player_feature_library['feature']  # 获取特征库中每一个特征
                feature_library.append(player_feature)

                player_feature_label = player_feature_library['label']  # 获取特征库中每一个标签
                feature_label.append(player_feature_label)

        # print feature_library
        feature_library_matrix = np.array(feature_library)  # 生成特征库矩阵
        # print(feature_library_matrix.shape)
        pic_feature_vector = np.array(video_pic_feature)[:, None]  # 一张生成待检测图片的列向量，注意是一张，直接用矩阵相乘即可
        # 获取最相似的特征索引，最高的相似特征
        get_most_similarity_feature, highest_similarity = self.calculate_cosine_distance(feature_library_matrix,
                                                                                         pic_feature_vector)

        predict_player_name = feature_label[get_most_similarity_feature]
        return predict_player_name, highest_similarity

    def main(self, pic_path):
        """
        主函数，
        :param pic_path: 图片路径
        :return: 预测球星姓名，相似度
        """
        # print('------------')
        pic_url = pic_path
        response_text = player_face_reco.face_reco(pic_url)
        response_text_json = json.loads(response_text)
        pts = response_text_json['result']['detections'][0]['boundingBox']['pts']

        # 处理图片的特征信息，标签就是球星的全名，然后加上立标index
        face_info_feature = player_face_reco.facex_feature(pic_url, pts)
        predict_player_name, highest_similarity = player_face_reco.get_most_similarity_label(face_info_feature,
                                                                                             self.source_path)
        # print(pic_url, predict_player_name, highest_similarity)

        return predict_player_name


if __name__ == '__main__':
    pic_path = '/Users/harry/PycharmProjects/toys/Demo/face_reco/test_pictures/1.jpg'

    player_face_reco = PlayerFaceReco()
    result = player_face_reco.main(pic_path)  # 调用球星识别接口,返回球星姓名
    print(result)
