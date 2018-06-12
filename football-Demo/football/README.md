# 足球球场事件识别

## Version1 (20180509) : 支持识别／预测点球

### 1. 处理流程

1. 视频（流）输入
2. nvvl 解码视频（流）
3. 逐帧推理或按照一定间隔推理
4. 大于一定阈值的推理结果post给客户，表示推理时刻的视频是点球

### 2. 环境配置

1. dockerfile: [Dockerfile](Dockerfile)。主要依赖包括：anaconda3， pytorch，ffmpeg， nvvl。
2. 模型文件下载：http://p6lnagdlg.bkt.clouddn.com/football-resnet152-0506.pth

### 3. 使用流程

```
python tsn_inference_football.py
```

1. 输入参数：

   | 参数名称        | 参数类型   | 说明         |
   | ----------- | ------ | ---------- |
   | stream_path | string | 视频流地址      |
   | post_url    | string | post API地址 |

2. 配置参数（参考tsn_config.py）：

   | 参数名称                 | 参数类型   | 说明                                   |
   | -------------------- | ------ | ------------------------------------ |
   | num_class            | int    | default: 2。模型类别数                     |
   | dianqiu_class_index  | int    | default: 1。点球类别index                 |
   | dianqiu_class_thresh | float  | default: 0.8。点球类别置信度大于该值时，认为是点球      |
   | test_segment         | int    | default: 1。表示每次推理1帧                  |
   | test_interval        | int    | default: 2。表示隔帧进行推理                  |
   | buffer_size          | int    | default: 5。表示缓存5帧推理结果做平均             |
   | arch                 | string | default: resnet152。模型结构              |
   | weights              | string | 模型文件位置                               |
   | gpu                  | int    | default: 0。使用哪个gpu                   |
   | modality             | string | default: RGB。模型推理模态                  |
   | input_size           | int    | default: 224。模型支持的输入图片大小             |
   | scale_size           | int    | default: 256。图片scale的大小              |
   | input_mean           | list   | default: [0.485, 0.456, 0.406]。输入的均值 |
   | input_std            | list   | default: [0.229, 0.224, 0.225]。输入的方差 |

3. 返回结果：

   ```
   {
       "type" :  'dianqiu',// 类型点球，任意球等
       "score": float,// 置信度
       "timestamp": float //系统时间戳
   }
   ```
