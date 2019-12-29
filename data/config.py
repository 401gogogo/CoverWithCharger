# config.py
import os
import os.path as osp

# gets home dir cross platform
# window系统运行
# HOME = 'C://Users//oyjp9'
# linux系统运行
HOME = os.path.expanduser("~")

#本机
# SIXray_ROOT = osp.join(HOME,'PycharmProjects','CoverWithCharger','data','Xray20190723')

# .138
SIXray_ROOT = osp.join(HOME, "otherProject/CoverWithCharger/data/Xray20190723/")

# .85
# Trained state_dict file path to open
# SIXray_ROOT = osp.join(HOME, "oyjp/CoverWithCharger/data/Xray20190723/")


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
sixray = {
    'num_classes': 3,
    'lr_steps': (20, 50, 150),
    'max_iter': 12000,
    'max_epoch':100,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SIXRAY',
}



# todo common configFile
# Use CUDA to train model
is_cuda = True

trained_model = 'weights/ssd300_SIXRAY_method_1_2_best(8619).pth'

train_imageSet = osp.join(SIXray_ROOT,'train_test_txt','battery_sub','sub_train_core_coreless.txt')
test_imageSet = osp.join(SIXray_ROOT,'train_test_txt','battery_sub','sub_test_core_coreless.txt')

# todo train.py configFile
dataset = 'SIXRAY'

# Pretrained base model
basenet = 'vgg16_reducedfc.pth'

# Batch size for training
batch_size = 128

# Checkpoint state_dict file to resume training from
resume = None

# Resume training at this iter
start_iter = 0

# Number of workers used in dataloading
num_workers = 4

# initial learning rate
lr = 1e-4

# Momentum value for optim
momentum = 0.9

# Weight decay for SGD
weight_decay = 5e-4

# Gamma update for SGD
gamma = 0.1

# Use visdom for loss visualization
visdom = True

# Directory for saving checkpoint models
train_save_folder = 'weights/'

# todo test.py configFile
# Dir to save test_results
test_save_folder = 'test/'

# Final confidence threshold
visual_threshold = 0.6

# Dummy arg so we can load in Jupyter Notebooks
f = None

# todo eval.py configFile
# File path to save eval_results
val_save_folder = 'eval/'

# Detection confidence threshold
confidence_threshold = 0.01

# Further restrict the number of predictions to parse
top_k = 5

# Cleanup and remove results files following eval
cleanup = False
