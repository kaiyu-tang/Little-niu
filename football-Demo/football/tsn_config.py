class Config:
	num_class = 4
	dianqiu_class_index = 1
	dianqiu_class_thresh = 0.9

	test_segments = 1 # get 1 frame per time
	test_interval = 2 # infer every 2 frames
	buffer_size = 5 # use 5 frames avg confidence

	arch = 'resnet50'
	weights = '/home/atlab/Workspace/kaiyu/Demo/toys/football-Demo/football/football-resnet50-0528.pth'
	gpu = 0
	modality = 'RGB'
	input_size = 224
	scale_size = 256
	input_mean = [0.485, 0.456, 0.406]
	input_std = [0.229, 0.224, 0.225]
