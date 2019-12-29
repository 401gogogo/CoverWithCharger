import os
import sys

import cv2 as cv
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from demoMainUI import Ui_MainWindow

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.autograd import Variable
from data import *
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from data import SIXray_CLASSES as labels
import copy

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)


        self.net = build_ssd('test', 300, 3)  # initialize SSD
        if torch.cuda.is_available():
            print('模型启用GPU')
            self.net = self.net.cuda()
        self.net.load_weights('../weights/ssd300_SIXRAY_method_1_2_best(8619).pth')


        self.readDir = '../data/Xray20190723/cut_Image_core_coreless_battery_sub_2000_500'
        self.annoDir = '../data/Xray20190723/Anno_core_coreless_battery_sub_2000_500'
        self.indicateDir = '../data/Xray20190723/indicateImag'
        self.imageName = None
        self.oriImag = None

    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 0]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 2]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def show_pic(self,object,img_src):
        # 适应显示框
        img_src = cv.resize(img_src, (311, 351))
        object.setPixmap(QPixmap(self.img2pixmap(img_src)))

    def selectPic(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, 'open file', self.readDir, "Image files (*.jpg *.png)")
        # print(self.fname)
        self.lineEdit_picPath.setText(self.fname)
        # if self.fname is None or self.fname == '':
        #     pass
        # else:
        #获取图片名称
        self.imageName = self.fname.split('/')[-1]
        # 展示图片
        self.oriImag = np.array(cv.imread(self.fname))
        self.show_pic(self.label_oriImg,self.oriImag)

        # 显示正确标注图像
        indecateImag = np.array(cv.imread(self.indicateDir + '/' + self.imageName))
        self.show_pic(self.label_incateImg, indecateImag)

    def prePic(self):
        #判断是否选择图片
        if self.oriImag is None or self.fname == '':
            QMessageBox.warning(self, "警告", "请选择图片")
        else:
            #清空列表框
            self.listWidget_confidence.clear()

            pre_img = copy.deepcopy(self.oriImag)
            rgb_image = cv2.cvtColor(self.oriImag, cv2.COLOR_BGR2RGB)
            x = cv2.resize(self.oriImag, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
            if torch.cuda.is_available():
                print('变量启用GPU')
                xx = xx.cuda()
            y = self.net(xx)
            # print('模型输出成功')

            top_k = 10
            detections = y.data

            # scale each detection back up to the image
            scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)


            #获得阈值
            bias = float(self.lineEdit__bias.text())
            # print('获得阈值：',bias)

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= bias:
                    score = detections[0, i, j, 0]
                    label_name = labels[i - 1]
                    # print(label_name, score)
                    self.listWidget_confidence.addItem(label_name+' : '+str(score))
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    # print(pt[0],pt[1],pt[2],pt[3])
                    if label_name == '带电芯充电宝':
                        cv.putText(pre_img, 'Core', (pt[0], pt[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                        cv.rectangle(pre_img, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 255), 4)
                    else:
                        cv.putText(pre_img, 'Coreless', (pt[0], pt[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        cv.rectangle(pre_img, (pt[0], pt[1]), (pt[2], pt[3]), (0, 255, 255), 4)
                    #画预测图
                    self.show_pic(self.label_preImg,pre_img)
                    # 带芯 (255, 0, 255)  不带芯 (0, 255, 255)
                    j += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()

    # 按钮绑定
    myWin.pushButton_selectPic.clicked.connect(myWin.selectPic)
    myWin.pushButton_gerResult.clicked.connect(myWin.prePic)

    myWin.show()
    sys.exit(app.exec_())