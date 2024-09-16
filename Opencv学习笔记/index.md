# 图像处理与Opencv

# Introduction

​	OpenCv是应用广泛的开源图像处理库，我们以其为基础，介绍相关的图像处理方法:包括基本的图像处理方法:几何变换，形态学变换，图像平滑，直方图操作，模板匹配，霍夫变换等;特征提取和描述方法:理解角点特征，Harris和Shi-Tomas算法，SIFT/SURF算法，Fast算法，ORB算法等;还有OpenCV在视频操作中的应用，最后的案例是使用OpenCV进行人脸检测。

# 1. OpenCV简介

- 图像的起源和数字图像
- OpenCV的简介及其部署方法
- OpenCV中包含的主要模块。

## 1.1 图像处理简介



1 图像的起源

​	1.1 <u>图象是什么</u>

​	图像是人类视觉的基础，是自然景物的客观反映，是人类认识世界和人类本身的重要源泉。“图”是物体反射或透射光的分布，“像“是人的视觉系统所接受的图在人脑中所形版的印象或认识，照片、绘画、剪贴画、地图、书法作品、手写汉学、传真、卫星云图、影视画面、X光片、脑电图、心电图等都是图像。--姚敏.数字图像处理:机械工业出版社，2014年。

![image-20240916080212034](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916080212034.png)

​	1.2 <u>模拟图像和数字图像</u>

​	图像起源于1826年前后法国科学家Joseph Nicéphore Niépce发明的第一张可永久保存的照片，属于模拟图像。模拟图像又称连续图像，它通过某种物理量(如光、电等)的强弱变化来纪录图像亮度信息，所以是连续变换的。模拟信号的特点是**容易受干扰**，如今已经基本全面被数字图像替代。

​	在第一次世界大战后，1921年美国科学家发明了Bartlane System，并从伦敦传到纽约传输了第一幅数字图像，其亮度用离散数值表示，将图片编码成5个灰度级，如下图所示，通过海底电缆进行传输。在发送端图片被编码并使用打孔带记录，通过系统传输后在接收方使用特殊的打印机恢复成图像。



​	1950年左右，计算机被发明，数字图像处理学科正式诞生，
​	

模拟图像和数字图像的对比，我们可以看一下:

![image-20240916080836751](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916080836751.png)



2 数字图像的的表示

​	2.1 <u>位数</u>

​	计算机采用0/1编码的系统，数字图像也是利用0/1来记录信息，我们平常接触的图像都是8位数图像，包含0~255灰度，其中0，代表最黑，1，表示最白。

![image-20240916081220501](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916081220501.png)

人眼对灰度更敏感一些，在16位到32位之间。

![image-20240916081334366](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916081334366.png)

​	2.2 <u>图像的分类</u>

二值图像:
	一幅二值图像的二维矩阵仅由0、1两个值构成,“0”代表黑色,“1”代白色。由于每一像素(矩阵中每一元素)取值价有0、1两种可能，所以计算机中二值图像的数据类型通常为1个二进制位。二值图像通常用于**文字**、**线条图**的扫描识别(OCR)和**掩膜图像**的存储。

灰度图:
	每个像素只有一个采样颜色的图像，这类图像通常显示为从最暗黑色到最亮的白色的灰度，尽管理论上这个采样可以任何颜色的不同深浅，甚至可以是不同亮度上的不同颜色。灰度图像与黑白图像不同，在计算机图像领域中黑白图像只有黑色与白色两种颜色;但是，灰度图像在黑色与白色之间还有许多级的颜色深度。灰度图像经常是在单个电磁波频谱如可见光内测量每个像素的亮度得到的，用于显示的灰度图像通常用每个采样像素8位的非线性尺度来保存，这样可以有256级灰度(如果用16位，则有65536级)

彩色图:
	每个像素通常是由红(R)、绿(G)、蓝(B)三个分量来表示的，分量介于(0，255)。RGB图像与索引图像一样都可以用来表示彩色图像。与索引图像一样，它分别用红(R)、绿(G)、蓝(B)三原色的组合来表示每个像素的颜色。但与索引图像不同的是，RGB图像每一个像素的颜色值(由RGB三原色表示)直接存放在图像矩阵中，由于每一像素的颜色需由R、G、B三个分量来表示，M、N分别表示图像的行列数，三个MxN的二维矩阵分别表示各个像素的R、G、B三个颜色分量。RGB图像的数据类型一般为8位无符号整形，通常用于表示和存放真彩色图像。

---



总结

1. 图像是什么
   图:物体反射或透射光的分布
   像:人的视觉系统所接受的图在人脑中所形版的印象或认识

2. 模拟图像和数字图像

   模拟图像:连续存储的数据

   数字图像:分级存储的数据

3. 数字图像

   位数:图像的表示，常见的就是8位

   分类:二值图像，灰度图像和彩色图像



## 1.2 OpenCV简介及安装方法

1 什么是OpenCV

​	1.1 <u>OpenCV简介</u>

![image-20240916082139273](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916082139273.png)

​	OpenCV是一款由Intel公司俄罗斯团队发起并参与和维护的一个计算机视觉处理开源软件库，支持与计算机视觉和机器学习相关的众多算法，并且正在日益扩展。



OpenCV的优势:

- 编程语言

  OpenCV基于C++实现，同时提供`python,Ruby,Matlab`等语言的接口。OpenCV-Python是OpenCV的Python API，结合了OpenCV C++ API和Python语言的最佳特性。

- 跨平台

  可以在`不同的系统平台`上使用，包括Windows，Linux，OSX，Android和i0S。基于CUDA和OpenCL的高速GPU操作接口也在积极开发中

- 活跃的开发团队

- 丰富的API

  完善的传统计算机视觉算法，涵盖主流的机器学习算法，同时添加了对深度学习的支持

1.2 OpenCV-Python

​	OpenCV-Python是一个Python绑定库，旨在解决计算机视觉问题。

​	Python是一种由Guido van Rossum开发的通用编程语言，它很快就变得非常流行，主要是因为它的简单性和代码可读性。它使程序员能够用更少的代码行表达思想，而不会降低可读性。

与C/C++等语言相比，Python速度较慢。也就是说，Python可以使用C/C++轻松扩展，这使我们可以在C/C++中编写计算密集型代码，并创建可用作Python模块的Python包装器。这给我们带来了两个好处:首先，代码与原始C/C++代码一样快(因为它是在后台工作的实际C++代码)，其次，在Python中编写代码比使用C/C++更容易。OpenCV-Python是原始OpenCVC++实现的Python包装器。

​	OpenCV-Python使用Numpy，这是一个高度优化的数据库操作库，具有MATLAB风格的语法。所有OpenCv数组结构都转换为Numpy数组。这也使得与使用Numpy的其他库(如SciPy和Matplotlib)集成更容易。



2 <u>OpenCV部署方法</u> 

​	安装OpenCV之前需要先安装numpy,matplotlib

​	创建Python虚拟环境cv,在cv中安装即可

​	先安装OpenCV-Pytion,由于一些经典的算法被申请了版权，新版本有很大的限制，所以选用3.4.3以下的版本

```shell
pip install opencv-python==3.4.2.17(python版本过高安装不了)
pip install opencv-python==3.4.1.15(python版本过高安装不了)
```

[Links for opencv-contrib-python (aliyun.com)](https://mirrors.aliyun.com/pypi/simple/opencv-contrib-python/)

[Links for opencv-python (tsinghua.edu.cn)](https://pypi.tuna.tsinghua.edu.cn/simple/opencv-python/)



现在可以测试下是否安装成功，运行以下代码无报错则说明安装成功。

```python
import cv2

l = cv2.imread("D:\\desktop\\1.png")
cv2.imshow("image",l)
cv2.waitKey(0)
```

如果我们要利用SIFT和SURF等进行特征提取时，还需要安装:

```
pip install opencv-contrib-python==3.4.1.15
```

---



总结

1. OpenCV是计算机视觉的开源库

   优势:

   - 支持多种编程语言
   -  跨平台
   - 活跃的开发团队
   - 丰富的API

## 1.3 OpenCV的模块

1 <u>0penCV的模块</u>

下图列出了OpenCV中包含的各个模块:

![image-20240916093536858](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20240916093536858.png)

其中core、highgui、imgproc是最基础的模块，该课程主要是围绕这几个模块展开的，分别介绍如下

- **features2d模块**用于提取图像特征以及特征匹配，nonfree模块实现了一些专利算法，如sift特征。
- **objdetect模块**实现了一些目标检测的功能，经典的基于Haar、LBP特征的人脸检测，基于HOG的行人、汽车等目标检测，分类器使用Cascade Classification（级联分类）和Latent SVM等。
- **stitching模块**实现了图像拼接功能。
- **FLANN模块**（Fast Library for Approximate Nearest Neighbors），包含快速近似最近邻搜索FLANN 和聚类Clustering算法。
- **ml模块**机器学习模块（SVM，决策树，Boosting等等）。
- **photo模块**包含图像修复和图像去噪两部分。
- **video模块**针对视频处理，如背景分离，前景检测、对象跟踪等。
- **calib3d模块**即Calibration（校准）3D，这个模块主要是相机校准和三维重建相关的内容。包含了基本的多视角几何算法，单个立体摄像头标定，物体姿态估计，立体相似性算法，3D信息的重建等等。
- **G-API模块**包含超高效的图像处理pipeline引擎

---



**总结**

1. OpenCV 的模块

   core：最核心的数据结构

   highgui：视频与图像的读取、显示、存储

   imgproc：图像处理的基础方法

   features2d：图像特征以及特征匹配



# 2. OpenCV基本操作

## 2.1 OpenCV基本操作

### 2.1.1 图像的IO操作

<u>读取图像</u>

API

```python
cv.imread()
```

参数：

- 要读取的图像

- 读取方式的标志

  - cv.IMREAD*COLOR：以彩色模式加载图像，任何图像的透明度都将被忽略。这是默认参数。

  - cv.IMREAD*GRAYSCALE：以灰度模式加载图像

  - cv.IMREAD_UNCHANGED：包括alpha通道的加载图像模式。

    **可以使用1、0或者-1来替代上面三个标志**

- 参考代码

  ```python
  import numpy as np
  import cv2 as cv
  # 以灰度图的形式读取图像
  img = cv.imread('1.jpg',0)
  ```

**注意：如果加载的路径有错误，不会报错，会返回一个None值**

<u>显示图像</u>

 API

```python
cv.imshow()
```

参数：

- 显示图像的窗口名称，以字符串类型表示
- 要加载的图像

**注意：在调用显示图像的API后，要调用cv.waitKey()给图像绘制留下时间，否则窗口会出现无响应情况，并且图像无法显示出来**。

另外我们也可使用matplotlib对图像进行展示。

参考代码

```python
# opencv中显示
cv.imshow('image',img)
cv.waitKey(0)//永远
# matplotlib中展示
plt.imshow(img[:,:,::-1])
```

<u>保存图像</u>

API

```python
cv.imwrite()
```

参数：

- 文件名，要保存在哪里
- 要保存的图像

参考代码

```python
cv.imwrite('messigray.png',img)
```

总结

我们通过加载灰度图像，显示图像，如果按's'并退出则保存图像，或者按ESC键直接退出而不保存。

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1 读取图像
img = cv.imread('messi5.jpg',0)
# 2 显示图像
# 2.1 利用opencv展示图像
cv.imshow('image',img)
# 2.2 在matplotplotlib中展示图像
plt.imshow(img[:,:,::-1])
plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
plt.show()
k = cv.waitKey(0)
# 3 保存图像
cv.imwrite('messigray.png',img)
```

### 2.1.2 绘制几何图形

<u>绘制直线</u>

```
cv.line(img,start,end,color,thickness)
```

参数：

- img:要绘制直线的图像
- Start,end: 直线的起点和终点
- color: 线条的颜色
- Thickness: 线条宽度

<u>绘制圆形</u>

```python
cv.circle(img,centerpoint, r, color, thickness)
```

参数：

- img:要绘制圆形的图像
- Centerpoint, r: 圆心和半径
- color: 线条的颜色
- Thickness: 线条宽度，为-1时生成闭合图案并填充颜色

<u>绘制矩形</u>

```python
cv.rectangle(img,leftupper,rightdown,color,thickness)
```

参数：

- img:要绘制矩形的图像
- Leftupper, rightdown: 矩形的左上角和右下角坐标
- color: 线条的颜色
- Thickness: 线条宽度

<u>向图像中添加文字</u>

```python
cv.putText(img,text,station, font, fontsize,color,thickness,cv.LINE_AA)
```

参数：

- img: 图像
- text：要写入的文本数据
- station：文本的放置位置
- font：字体
- Fontsize :字体大小

<u>效果展示</u>

我们生成一个全黑的图像，然后在里面绘制图像并添加文字

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1 创建一个空白的图像
img = np.zeros((512,512,3), np.uint8)
# 2 绘制图形
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(447,63), 63, (0,0,255), -1)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
# 3 图像展示
plt.imshow(img[:,:,::-1])
plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190925154009533](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20190925154009533.png)

### 2.1.3 获取并修改图像中的像素点

我们可以通过行和列的坐标值获取该像素点的像素值。对于BGR图像，它返回一个蓝，绿，红值的数组。对于灰度图像，仅返回相应的强度值。使用相同的方法对像素值进行修改。

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg')
# 获取某个像素点的值
px = img[100,100]
# 仅获取蓝色通道的强度值
blue = img[100,100,0]
# 修改某个位置的像素值
img[100,100] = [255,255,255]
```

### 2.1.4 获取图像的属性

图像属性包括行数，列数和通道数，图像数据类型，像素数等。

![image-20191016151042764](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20191016151042764.png)

### 2.1.5 图像通道的拆分与合并

有时需要在B，G，R通道图像上单独工作。在这种情况下，需要将BGR图像分割为单个通道。或者在其他情况下，可能需要将这些单独的通道合并到BGR图像。你可以通过以下方式完成。

```python
# 通道拆分
b,g,r = cv.split(img)
# 通道合并
img = cv.merge((b,g,r))
```

### 2.1.6 色彩空间的改变

OpenCV中有150多种颜色空间转换方法。最广泛使用的转换方法有两种，BGR↔Gray和BGR↔HSV。

API：

```python
cv.cvtColor(input_image，flag)
```

参数：

- input_image: 进行颜色空间转换的图像
- flag: 转换类型
  - cv.COLOR_BGR2GRAY : BGR↔Gray
  - cv.COLOR_BGR2HSV: BGR→HSV

------

**总结：**

1. 图像IO操作的API：

   cv.imread(): 读取图像

   cv.imshow()：显示图像

   cv.imwrite(): 保存图像

2. 在图像上绘制几何图像

   cv.line(): 绘制直线

   cv.circle(): 绘制圆形

   cv.rectangle(): 绘制矩形

   cv.putText(): 在图像上添加文字

3. 直接使用行列索引获取图像中的像素并进行修改

4. 图像的属性

   ![image-20191016151119554](F:\Documents\GitHub\hwj-s-study-notes\Opencv学习笔记\assets\image-20191016151119554.png)

5. 拆分通道：cv.split()

   通道合并：cv.merge()

6. 色彩空间的改变

   cv.cvtColor(input_image，flag)