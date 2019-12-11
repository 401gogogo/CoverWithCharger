"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os
from .config import HOME, SIXray_ROOT
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import logging
from data.config import train_imageSet

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("logger.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


SIXray_CLASSES = (
    '带电芯充电宝', '不带电芯充电宝'
)



class SIXrayAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            # zip(SIXray_CLASSES, range(len(SIXray_CLASSES))))
            zip(SIXray_CLASSES, range(len(SIXray_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            it has been changed to the path of annotation-2019-07-10
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # print (idx)

        # 遍历Annotation
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2]) / width
            # 只读取V视角的
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:  # 是这么个意思吧？
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            logger.info(str(idx))
            logger.info(str(bndbox))
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        logger.info('len(res):'+str(len(res)))
        logger.info('res:' + str(res))

        if len(res) == 0:
            return [[0, 0, 0, 0, 3]]
        return res


class SIXrayDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 # image_sets=[('core_3000', 'train'), ('coreless_3000', 'train')],
                 image_sets=train_imageSet,
                 transform=None, target_transform=SIXrayAnnotationTransform(),
                 # dataset_name='SIXray'):
                 dataset_name='SIXray'):
        self.root = root
        # self.root = SIXray_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s' % self.root, 'Anno_core_coreless_battery_sub_2000_500', '%s.txt')
        self._imgpath = osp.join('%s' % self.root, 'cut_Image_core_coreless_battery_sub_2000_500', '%s.jpg')

        self.ids = list()

        with open(self.image_set, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ids.append(line.strip('\n'))

    def __getitem__(self, index):
        # im, gt, h, w, og_im = self.pull_item(index)
        im, gt, h, w, og_img = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]


        target = self._annopath % img_id  # 注释目录

        img = cv2.imread(self._imgpath % img_id)

        height, width, channels = img.shape
        og_img = img

        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(a2, 0, a1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img
        # return torch.from_numpy(img), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        anno = self._annopath % img_id
        gt = self.target_transform(anno, width, height, img_id)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
