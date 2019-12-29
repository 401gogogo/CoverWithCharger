# 北航2019秋机器学习大作业 #
+ 此项目是北航2019年机器学习大作业的项目地址，小组成员为**欧阳金鹏ZF1921536**&**叶慕聪ZF1921355**.
## Introduce ##
+ 针对此次大作业，我们充分了解了目标检测的相关算法，对主流算法进行了对比，最终我们选取**SSD**作为模型。我们先对数据集进行简单交叉验证(**90%为训练集，10%为测试集**)，并在GPU上进行训练，多次调整参数使其得到更好的效果。经过**8.4K**次迭代，最终得到的mAP为**86.19**
## Detial ##
#### Environment  ####
+ **python3.7**: python使用的版本为3.7
+ **PyTorch 1.1**: pytorch版本为1.1
+ **Work on GPU or CPU**: 使用DistributedDataParallel进行多GPU并行计算。我们没有使用CPU进行测试，但理论上支持CPU.
+ **Hardware**: 2张NVIDIA Tesla V100 32GB
### Usage ###
+ 我们提供的模型在CoverWithCharger/weigths
###### 1. Install pytorch
+ 我们使用的python版本为3.7
+ 你可以点击此处来安装pytorch [this](https://github.com/pytorch/pytorch)
###### 2. Clone the repository:
git https://github.com/401gogogo/CoverWithCharger
###### 3. Dataset
+ Our dataset is private
###### 4. Evaluation
```
cd CoverWithCharger
python test.py  #生成预测结果txt
python calculate_map_test.py --predicted_file '预测结果文件路径'
```
###### 5. Training:
```
cd CoverWithCharger
python train.py 
```
### Result ###
我们提供我们的训练数据和结果，如下图所示：
![mAP](/result1_3.jpg)
![Loss](/1_2_iter.png)
