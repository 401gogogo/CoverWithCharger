from __future__ import print_function

import warnings

from torch.autograd import Variable

from data import *
from ssd import build_ssd

warnings.filterwarnings('ignore')
import cv2 as cv
from data import SIXray_CLASSES as labels
import numpy as np

if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def test(img_path,anno_path):
    result_root = '../predicted_file'
    result_core = result_root+'/det_test_带电芯充电宝.txt'
    result_coreless = result_root+'/det_test_不带电芯充电宝.txt'

    net = build_ssd('test', 300, 3)  # initialize SSD
    if torch.cuda.is_available():
        print('模型启用GPU')
        net = net.cuda()
    model_path = 'weights/ssd300_SIXRAY_method_1_3_best(9616).pth'
    net.load_weights(model_path)
    print('加载模型',model_path)

    with open(result_core,'w') as f1,open(result_coreless,'w') as f2:
        all_num = len(os.listdir(img_path))
        for n,img in enumerate(os.listdir(img_path)):
            #读取图片
            imgName = img.split('.')[0]
            ori_img = np.array(cv.imread(os.path.join(img_path,img)))
            rgb_image = cv.cvtColor(ori_img, cv.COLOR_BGR2RGB)
            x = cv.resize(ori_img, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
            if torch.cuda.is_available():
                # print('变量启用GPU')
                xx = xx.cuda()
            y = net(xx)
            # print('模型输出成功')

            top_k = 10
            detections = y.data

            # scale each detection back up to the image
            scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.01:
                    score = detections[0, i, j, 0]
                    # print(score.data.cpu().numpy())
                    label_name = labels[i - 1]
                    # print(label_name, score)

                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    # print(pt[0],pt[1],pt[2],pt[3])
                    if label_name == '带电芯充电宝':
                       f1.write('%s %s %s %s %s %s\n'%(str(imgName),str(score.data.cpu().numpy()),str(pt[0]),str(pt[1]),str(pt[2]),str(pt[3])))
                    else:
                       f2.write('%s %s %s %s %s %s\n' % (str(imgName), str(score.data.cpu().numpy()), str(pt[0]), str(pt[1]), str(pt[2]), str(pt[3])))
                    j += 1
            print('当前进度：%s- [%d/%d]' % (str(imgName), int(n + 1), int(all_num)))


if __name__ == '__main__':
    img_path = './data/Xray20190723/cut_Image_core_coreless_battery_sub_2000_500'
    anno_path = './data/Xray20190723/Anno_core_coreless_battery_sub_2000_500'
    test(img_path,anno_path)
