# EigenFace Implementation

大家好，以下为第三次作业内容及注意事项：

 **内容：**

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
    
        I really don't know why this requirements should exist
    
        OpenCV uses Qt as a backend to display stuff, at least it does so with modern Python interfaces
    
        This is a screenshot when I accidentally discovered this trying to open a window with an SSH connection to my server
    
        ![image-20201222210811073](report.assets/image-20201222210811073.png)

## Experiment Principles

### Matrix & Eigenvalues & Eigenvectors

==Matrix is a form of transformation (square matrices)==

This is especially useful when we're trying to unveal the true nature of a lot of linear algebra concepts with the help of geometry (visualization).

Of course, one can always write abstract symbols and prove one's way throughout the linear algebra sea without having to draw even a single checkerboard, it's still gonna be extremely helpful, especially for us beginners, to understand some of the most basic concepts so deeply that, they're like imprinted into our heads.

Hereby I highly recommend the series brought to you by *3Blue1Brown*, a famous YouTuber, and a mathematician. Link below

[Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

I found these masterpiece series when trying to understand eigen stuff better (eigenvalues, eigenvectors, eigenbasis and of course, eigenfaces)



The most important takeways are:

- Matrices (square) can and should be viewed as a transform where each columns represents the new basis's location

    We can visualize this process by **dragging** [1, 0] to the first column of the transformation matrix, and [0, 1] to the second, assuming a 2 by 2 matrix

- Eigenvectors are vectors that doesn't get knocked off their span when performing the transfromation decribed in the previous term

    Typically, for a 2 by 2 matrix, there's two directions, which, during the **dragging** in the previous item, don't rotate around, but just scales up to a certain degree

    And that **certain degree**, are the eigenvalues



OpenCV gives us some API for computing eigenvectors and eigenvalues. Unfortunately, those direct methods are a bit outdated and don't exist anymore in modern interfaces (especially so if we're using Python as the programming language for performing these CV tasks).

But as we can probably guess from the importance of eigen stuff, there's tons of other resources for us to call. And they all got optimized the heck out of them because people use them sooooo often.



### PCA: Principal Component Analysis

==Intuitively, principal components are the axes, onto which if data points are projected, the projections' distances to the origin get the largest variance==

Similar to eigen stuff, we find it much easier to understand PCA with a few visualization and a detailed illustration with a lot of examples.

This is the script that inspired me the most. Masterpiece indeed. As a techincal report, it got over 2500 citations on Google Scholar

[A tutorial on Principal Components Analysis](https://ourarchive.otago.ac.nz/bitstream/handle/10523/7534/OUCS-2002-12.pdf)



The main takeways are:

- Principal Components are those axes that preserve information about the original data the most

    Take a 2-d checkerboard for example, there's a bunch of points on this plane and they roughly form a straight line

    For simplicity, we assume this rough line to be `y=x`

    Then intuitively, if we memorize the points as the distance to the origin, we'll lose one degree of freedom

    But we'll be able to roughly reconstruct the points by drawing them on the line `y=x` using there distance to the origin

    And this `y=x` would be the **Principal Components** for those bunch of points

- Mathematically, one can compute the principal components by getting the eigenvectors sorted by their corresponding eigenvalues reversely of the covariance matrix of those data points

    A long sentence, right?

    A covariance matrix is a convenient way to write down all the covariance of the data points in question, where its index indicates the orginal data. For example, the value at `[i, j]` is the covariance of `data[i]` and `data[j]`

- Mathematically, if

    