import numpy as np
import os.path as osp
import os
from config import HOME, SIXray_ROOT
import random as rd
from shutil import copyfile
import sys
import cv2 as cv


class DataUtils:
    def __init__(self, data_set=['core_3000', 'coreless_3000'], split_rate=0.9, parse='train'):
        assert parse in ('train', 'test', 'val'), 'PARSE ERROR'
        self.data_set = data_set
        self.root = HOME
        self.ori_data_root = osp.join(SIXray_ROOT, '%s')
        self.new_data_root = SIXray_ROOT

        self.ori_annotation_root = osp.join(self.ori_data_root, 'Annotation')
        self.new_annotation_root = osp.join(self.new_data_root, 'Anno_core_coreless_battery_sub_2000_500')

        self.ori_imagesets_root_classes = osp.join(self.ori_data_root, 'ImageSets', 'Main')
        self.new_imagesets_root_all = osp.join(self.new_data_root, 'train_test_txt', 'battery_sub')

        self.ori_image_root = osp.join(self.ori_data_root, 'Image')
        self.new_image_root = osp.join(self.new_data_root, 'cut_Image_core_coreless_battery_sub_2000_500')

        self.indicate_root = osp.join(self.new_data_root, "indicateImag")

        self.split_rate = split_rate

        self._dir_dict = {}
        self._load_data()
        self._load_imagset()

    def _load_data(self):
        '''
        加载数据
        :return:
        '''
        for name in self.data_set:
            assert os.path.isdir(
                osp.join(self.new_data_root, self.ori_annotation_root % name)), 'ERROR,NOT EXIST DATA DIR %s' % (
                osp.join(self.new_data_root, self.ori_annotation_root % name))
            file_list = os.listdir(self.ori_annotation_root % name)
            rd.shuffle(file_list)
            self._dir_dict[name] = file_list

    def _load_imagset(self):
        '''
        加载训练集和验证集
        '''
        self.train_set = []
        self.eval_set = []
        with open(osp.join(self.new_imagesets_root_all, 'sub_train_core_coreless.txt'), 'r') as f1, open(
                osp.join(self.new_imagesets_root_all, 'sub_test_core_coreless.txt'), 'r') as f2:
            for line1 in f1:
                self.train_set.append(line1[:-1])

            for line2 in f2:
                self.eval_set.append(line2[:-1])

        # print(len(self.train_set),self.train_set)
        # print(len(self.eval_set), self.eval_set)

    def ori_split_classes_txt(self):
        '''
        按照类别划分不同的数据集txt文件
        :return:
        '''
        print('开始划分数据')
        for key, value in self._dir_dict.items():
            train_num = int(self.split_rate * len(value))
            train_data = value[:train_num]
            val_data = value[train_num:]
            # print(len(train_data), len(val_data))
            # 存储文件
            with open(osp.join(self.ori_imagesets_root_classes % key, 'train_' + key + '.txt'), 'w') as f1:
                i = 0
                for data in train_data:
                    f1.write(data.split('.')[0])
                    f1.write('\n')
                    i += 1
                    print('当前处理 %s文件,生成train文件[%d/%d]' % (str(key), int(i), int(len(train_data))))

            with open(osp.join(self.ori_imagesets_root_classes % key, 'val_' + key + '.txt'), 'w') as f2:
                i = 0
                for data in val_data:
                    f2.write(data.split('.')[0])
                    f2.write('\n')
                    i += 1
                    print('当前处理 %s文件,生成val文件[%d/%d]' % (str(key), int(i), int(len(val_data))))
        print('处理完成。。。。。')

    def ori_split_all_txt(self):
        '''
        将所有文件整合后划分训练和验证数据集文件txt
        :return:
        '''
        print('开始划分数据')
        train_list = []
        val_list = []
        for key, value in self._dir_dict.items():
            train_num = int(self.split_rate * len(value))
            train_data = value[:train_num]
            val_data = value[train_num:]
            train_list += train_data
            val_list += val_data
        print('len(train_list):%d,len(val_list):%d' % (int(len(train_list)), int(len(val_list))))

        train_list = [x.split('.')[0] for x in train_list]
        val_list = [x.split('.')[0] for x in val_list]
        rd.shuffle(train_list)
        rd.shuffle(val_list)

        with open(osp.join(self.new_imagesets_root_all, 'sub_train_core_coreless.txt'), 'w') as f1, open(
                osp.join(self.new_imagesets_root_all, 'sub_test_core_coreless.txt'), 'w') as f2:
            for file in train_list:
                f1.write(file)
                f1.write('\n')
            for file in val_list:
                f2.write(file)
                f2.write('\n')
        print('build all txt over')

    def merge_image_dir(self):
        '''
        将不同类别的图片合并到一起
        :return:
        '''
        classes_num = len(self.data_set)
        classes_i = 1
        for name in self.data_set:
            curren_classes_total = len(os.listdir(self.ori_image_root % name))
            curren_classes = 0
            for file in os.listdir(self.ori_image_root % name):
                source = osp.join(self.ori_image_root % name, file)
                target = osp.join(self.new_image_root, file)
                try:
                    copyfile(source, target)
                    curren_classes += 1
                    print('当前进度： %s类[%d/%d]，[%d/%d]' % (
                        str(name), int(curren_classes), int(curren_classes_total), int(classes_i), int(classes_num)))
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                    exit(1)
                except:
                    print("Unexpected error:", sys.exc_info())
                    exit(1)
            classes_i += 1
            print('%s 类图片拷贝完毕' % (str(name)))
        print('合并图片完毕')

    def merge_anno_dir(self):
        '''
        将不同类别的文件标注文件整合到一个目录中
        :return:
        '''
        classes_num = len(self.data_set)
        classes_i = 1
        for name in self.data_set:
            curren_classes_total = len(os.listdir(self.ori_annotation_root % name))
            curren_classes = 0
            for file in os.listdir(self.ori_annotation_root % name):
                source = osp.join(self.ori_annotation_root % name, file)
                target = osp.join(self.new_annotation_root, file)
                try:
                    copyfile(source, target)
                    curren_classes += 1
                    print('当前进度： %s类[%d/%d]，[%d/%d]' % (
                        str(name), int(curren_classes), int(curren_classes_total), int(classes_i), int(classes_num)))
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                    exit(1)
                except:
                    print("Unexpected error:", sys.exc_info())
                    exit(1)
            classes_i += 1
            print('%s 类注释文件拷贝完毕' % (str(name)))
        print('合并注释文件完毕')

    def is_train_or_eval(self, image):
        '''
        判断需要标注的图片在测试集中还是训练集中
        '''
        if image in self.train_set:
            return 'train'
        else:
            return 'eval'

    def batach_indicate(self):
        '''
        批量标注图片
        :return:
        '''
        img_total = len(os.listdir(self.new_image_root))
        cur_img = 0
        for img in os.listdir(self.new_image_root):
            np_img = np.array(cv.imread(os.path.join(self.new_image_root, img)))
            with open(os.path.join(self.new_annotation_root, img.replace('.jpg', '.txt')), encoding='utf-8') as f:
                for line in f:
                    classesName = line.strip().split(' ')[1]
                    left_x = int(line.strip().split(' ')[2])
                    left_y = int(line.strip().split(' ')[3])
                    right_x = int(line.strip().split(' ')[4])
                    right_y = int(line.strip().split(' ')[5])

                    # w = right_x - left_x
                    # h = right_y - left_y

                    # 画框 显示图片
                    if classesName == '不带电芯充电宝':
                        text = 'coreless'
                        cv.putText(np_img, str(text), (left_x, left_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        cv.rectangle(np_img, (left_x, left_y), (right_x, right_y), (0, 255, 255), 4)
                    elif classesName == '带电芯充电宝':
                        text = 'core'
                        cv.putText(np_img, str(text), (left_x, left_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                        cv.rectangle(np_img, (left_x, left_y), (right_x, right_y), (255, 0, 255), 4)
                    else:
                        text = 'other'
                        cv.putText(np_img, str(text), (left_x, left_y), cv.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 1)
                        cv.rectangle(np_img, (left_x, left_y), (right_x, right_y), (30, 144, 255), 4)
            # 保存图片
            if self.is_train_or_eval(img.split('.')[0]) == 'train':
                assert os.path.isdir(os.path.join(self.indicate_root, 'train')), 'No Find,' + os.path.isdir(
                    os.path.join(self.indicate_root, 'train'))
                cv.imwrite(os.path.join(self.indicate_root, 'train', img.replace('.txt', '.jpg')), np_img)
            else:
                assert os.path.isdir(os.path.join(self.indicate_root, 'test')), 'No Find,' + os.path.isdir(
                    os.path.join(self.indicate_root, 'test'))
                cv.imwrite(os.path.join(self.indicate_root, 'test', img.replace('.txt', '.jpg')), np_img)

            cur_img += 1
            print('当前进度：[%d/%d]' % (int(cur_img), int(img_total)))


if __name__ == '__main__':
    util = DataUtils()
    # util.split_classes()
    # util.ori_split_all_txt()
    # util.merge_image_dir()
    # util.merge_anno_dir()
    util.batach_indicate()
