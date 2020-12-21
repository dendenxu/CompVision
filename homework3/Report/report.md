# EigenFace Implementation

大家好，以下为第三次作业内容及注意事项：

﻿

#### **内容：**

自己写代码实现 Eigenface 人脸识别的训练与识别过程：

1. 假设每张人脸图像只有一张人脸，且两只眼睛位置已知（即可人工标注给出）。每张图像的眼睛位置存在相应目录下的一个与图像文件名相同但后缀名为 txt 的文本文件里，文本文件中用一行、以空格分隔的4个数字表示，分别对应于两只眼睛中心在图像中的位置；

1. 实现两个程序过程（两个执行文件），分别对应训练与识别；

1. 自己构建一个人脸库（至少 40 人，包括自己），课程主页提供一个人脸库可选用；

1. 不能直接调用 OpenCV 里面与 Eigenface 相关的一些函数，特征值与特征向量求解函数可以调用；只能用 C/C++/Python，不能用其他编程语言；GUI只能用 OpenCV 自带的 HighGUI，不能用QT或其他的；平台可以用 Win/Linux/MacOS，建议 Win 优先；

1. 训练程序格式大致为：“mytrain.exe <*能量百分比>* <*model文件名> <其他参数>*…”，用能量百分比决定取多少个特征脸，将训练结果输出保存到 model 文件中。同时将前 10 个特征脸拼成一张图像，然后显示出来；

1. 识别程序格式大致为：“mytest.exe <*人脸图像文件名> <model文件名> <其他参数>*…”，将 model 文件装载进来后，对输入的人脸图像进行识别，并将识别结果叠加在输入的人脸图像上显示出来，同时显示人脸库中跟该人脸图像最相似的图像。

**注意：**

1. 可任意使用钉钉群内给出的三个数据集，其中BioFaceDatabase内.eye文件包含对应图片眼睛位置；Caltec Database内ImageData.mat包含人脸bounding box位置；JAFFE无额外标注；
1. 在实验报告或readme中注明运行环境和运行方法；
1. 截止时间为2021/01/01 23:59。



## Basic Info

- Student ID: 3180105504

- Student Name: Xu Zhen

- Instructor Name: Song Mingli

- Course Name: Computer Vision

- Homework Name: ==EigenFace Implementation==

- Basic Requirements:
    - Implement an EigenFace face recognition trainer/tester pair
    - Construct a face database of at least 40 persons (including the author)
    - ==Don't== use OpenCV's API for eigenfaces
    - APIs relating to eigenvalues/eigenvectors ==can== be used
    - No Qt for GUI, only HighGUI that comes with OpenCV can be used

## Experiment Principles



