# 图像处理与Opencv

# Introduction

​	OpenCv是应用广泛的开源图像处理库，我们以其为基础，介绍相关的图像处理方法:包括基本的图像处理方法:几何变换，形态学变换，图像平滑，直方图操作，模板匹配，霍夫变换等;特征提取和描述方法:理解角点特征，Harris和Shi-Tomas算法，SIFT/SURF算法，Fast算法，ORB算法等;还有OpenCV在视频操作中的应用，最后的案例是使用OpenCV进行人脸检测。

# 1. OpenCV简介

- 图像的起源和数字图像
- OpenCV的简介及其部署方法
- OpenCV中包含的主要模块

## 1.1 图像处理简介



1 图像的起源

​	1.1 <u>图象是什么</u>

​	图像是人类视觉的基础，是自然景物的客观反映，是人类认识世界和人类本身的重要源泉。“图”是物体反射或透射光的分布，“像“是人的视觉系统所接受的图在人脑中所形版的印象或认识，照片、绘画、剪贴画、地图、书法作品、手写汉学、传真、卫星云图、影视画面、X光片、脑电图、心电图等都是图像。--姚敏.数字图像处理:机械工业出版社，2014年。

![image-20240916080212034](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511053.png)

​	1.2 <u>模拟图像和数字图像</u>

​	图像起源于1826年前后法国科学家Joseph Nicéphore Niépce发明的第一张可永久保存的照片，属于模拟图像。模拟图像又称连续图像，它通过某种物理量(如光、电等)的强弱变化来纪录图像亮度信息，所以是连续变换的。模拟信号的特点是**容易受干扰**，如今已经基本全面被数字图像替代。

​	在第一次世界大战后，1921年美国科学家发明了Bartlane System，并从伦敦传到纽约传输了第一幅数字图像，其亮度用离散数值表示，将图片编码成5个灰度级，如下图所示，通过海底电缆进行传输。在发送端图片被编码并使用打孔带记录，通过系统传输后在接收方使用特殊的打印机恢复成图像。



​	1950年左右，计算机被发明，数字图像处理学科正式诞生，
​	

模拟图像和数字图像的对比，我们可以看一下:

![image-20240916080836751](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511054.png)



2 数字图像的的表示

​	2.1 <u>位数</u>

​	计算机采用0/1编码的系统，数字图像也是利用0/1来记录信息，我们平常接触的图像都是8位数图像，包含0~255灰度，其中0，代表最黑，1，表示最白。

![image-20240916081220501](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511055.png)

人眼对灰度更敏感一些，在16位到32位之间。

![image-20240916081334366](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511056.png)

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

![image-20240916082139273](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511057.png)

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
pip install opencv-python==3.4.2.17
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
pip install opencv-contrib-python==3.4.2.17
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

![image-20240916093536858](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511058.png)

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

![image-20190925154009533](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511059.png)

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

![image-20191016151042764](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511060.png)

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

   ![image-20191016151119554](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511061.png)

5. 拆分通道：cv.split()

   通道合并：cv.merge()

6. 色彩空间的改变

   cv.cvtColor(input_image，flag)

## 2.2 算数操作

### 2.2.1 图像的加法

​	可以使用OpenCV的cv.add()函数把两幅图像相加，或者可以简单地通过numpy操作添加两个图像，如res = img1 + img2。两个图像应该具有相同的大小和类型，或者第二个图像可以是标量值。

**注意：OpenCV加法和Numpy加法之间存在差异。OpenCV的加法是饱和操作，而Numpy添加是模运算。**

参考以下代码：

```python
>>> x = np.uint8([250])
>>> y = np.uint8([10])
>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]
>>> print( x+y )          # 250+10 = 260 % 256 = 4
[4]
```

​	这种差别在你对两幅图像进行加法时会更加明显。OpenCV 的结果会更好一点。所以我们尽量使用 OpenCV 中的函数。

我们将下面两幅图像：

![image-20191016154526370](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511062.png)

代码：

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img1 = cv.imread("view.jpg")
img2 = cv.imread("rain.jpg")

# 2 加法操作
img3 = cv.add(img1,img2) # cv中的加法
img4 = img1+img2 # 直接相加

# 3 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img3[:,:,::-1])
axes[0].set_title("cv中的加法")
axes[1].imshow(img4[:,:,::-1])
axes[1].set_title("直接相加")
plt.show()
```

结果如下所示：

![image-20191016154714377](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511063.png)

### 2.2.2 图像的混合

​	这其实也是加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉。图像混合的计算公式如下：

> g(x) = (1−α)f0(x) + αf1(x)

通过修改 α 的值（0 → 1），可以实现非常炫酷的混合。

现在我们把两幅图混合在一起。第一幅图的权重是0.7，第二幅图的权重是0.3。函数cv2.addWeighted()可以按下面的公式对图片进行混合操作。

> dst = α⋅img1 + β⋅img2 + γ

这里γ取为零。

参考以下代码：

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img1 = cv.imread("view.jpg")
img2 = cv.imread("rain.jpg")

# 2 图像混合
img3 = cv.addWeighted(img1,0.7,img2,0.3,0)

# 3 图像显示
plt.figure(figsize=(8,8))
plt.imshow(img3[:,:,::-1])
plt.show()
```

窗口将如下图显示：

![image-20191016161128720](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181511064.png)

------

**总结**

1. 图像加法：将两幅图像加载一起

   cv.add()

2. 图像的混合：将两幅图像按照不同的比例进行混合

   cv.addweight()

注意：这里都要求两幅图像是相同大小的。



# 3. OpenCV图像处理

**主要内容**

- 图像的几何变换
- 图像的形态学转换
- 图像的平滑方法
- 直方图的方法
- 边缘检测的方法
- 模板匹配和霍夫变换的应用

## 3.1 几何变换

**学习目标**

- 掌握图像的缩放，平移，旋转等
- 了解数字图像的仿射变换和透射变换

------

### 3.1.1 图像缩放

缩放是对图像的大小进行调整，即使图像放大或缩小。

1. API

   ```python
   cv2.resize(src,dsize,fx=0,fy=0,interpolation=cv2.INTER_LINEAR)
   ```

   参数：

   - src : 输入图像

   - dsize: 绝对尺寸，直接指定调整后图像的大小

   - fx,fy: 相对尺寸，将dsize设置为None，然后将fx和fy设置为比例因子即可

   - interpolation：插值方法，

     ![image-20191016161502727](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730024.png)

2. 示例

3. ```python
   import numpy as np
   import cv2 as cv
   import matplotlib.pyplot as plt
   from matplotlib import rcParams
   
   # 设置字体以避免中文乱码
   rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
   rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
   
   # 1 读取图像
   img1 = cv.imread("view.jpg")
   img2 = cv.imread("rain.jpg")
   
   # 2.1 绝对尺寸
   res = cv.resize(img1, (800, 600))
   
   # 2.2 相对尺寸
   res1 = cv.resize(img1, None, fx=0.5, fy=0.5)
   
   # 3 图像显示
   # 3.1 使用opencv显示图像(不推荐)
   cv.imshow("original", img1)
   cv.imshow("enlarge", res)
   cv.imshow("shrink", res1)
   cv.waitKey(0)
   cv.destroyAllWindows()
   
   # 3.2 使用matplotlib显示图像
   fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
   axes[0].imshow(res[:, :, ::-1])
   axes[0].set_title("绝对尺度（放大）")
   axes[1].imshow(img1[:, :, ::-1])
   axes[1].set_title("原图")
   axes[2].imshow(res1[:, :, ::-1])
   axes[2].set_title("相对尺度（缩小）")
   plt.show()
   ```

   ![image-20190926143500645](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730025.png)

### 3.1.2 图像平移

图像平移将图像按照指定方向和距离，移动到相应的位置。

1. API

```python
cv.warpAffine(img,M,dsize)
```

参数：

- img: 输入图像

- M： 2∗3移动矩阵

  对于(x,y)处的像素点，要把它移动到(*𝑥+𝑡~𝑥~,𝑦+𝑡~𝑦~*)处时，M矩阵应如下设置：
  $$
  M = \left[\begin{matrix}1&0&t_{x}\\0&1&t_{y}\\&\end{matrix}\right]
  $$
  
  注意：将𝑀*M*设置为np.float32类型的Numpy数组。
  
- dsize: 输出图像的大小

  **注意：输出图像的大小，它应该是(宽度，高度)的形式。请记住,width=列数，height=行数。**

- 示例

需求是将图像的像素点移动(50,100)的距离：

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1. 读取图像
img1 = cv.imread("./image/image2.jpg")

# 2. 图像平移
rows,cols = img1.shape[:2]
M = M = np.float32([[1,0,100],[0,1,50]])# 平移矩阵
dst = cv.warpAffine(img1,M,(cols,rows))

# 3. 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img1[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("平移后结果")
plt.show()
```

![image-20190926151127550](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730026.png)

### 3.1.3 图像旋转

​		图像旋转是指图像按照某个位置转动一定角度的过程，旋转中图像仍保持这原始尺寸。图像旋转后图像的水平对称轴、垂直对称轴及中心坐标原点都可能会发生变换，因此需要对图像旋转中的坐标进行相应转换。

那图像是怎么进行旋转的呢？如下图所示：

![image-20191023102648731](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730027.png)

假设图像逆时针旋转𝜃，则根据坐标转换可得旋转转换为:

![image-20191023102927643](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730028.png)

其中：

![image-20191023103038145](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730029.png)

带入上面的公式中，有：

![image-20191023103106568](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730030.png)

也可以写成：
$$
\begin{bmatrix}
x'&y'&1\end{bmatrix}=\begin{bmatrix}x&y&1\end{bmatrix}
=
\begin{bmatrix}
cos\theta&-sin\theta&0\\
sin\theta&cos\theta&0\\
0&0&1
\end{bmatrix}
$$
​	同时我们要修正原点的位置，因为原图像中的坐标原点在图像的左上角，经过旋转后图像的大小会有所变化，原点也需要修正。

​	假设在旋转的时候是以旋转中心为坐标原点的，旋转结束后还需要将坐标原点移到图像左上角，也就是还要进行一次变换。

![image-20191023105453003](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730031.png)
$$
\begin{bmatrix}
𝑥''&𝑦''&1
\end{bmatrix}
=
\begin{bmatrix}
𝑥'&𝑦'&1
\end{bmatrix}
\begin{bmatrix}
1&0&1\\
0&-1&0\\
left&top&1
\end{bmatrix}
$$

$$
\begin{bmatrix}
𝑥&𝑦&1
\end{bmatrix}
=
\begin{bmatrix}
cos\theta&-sin\theta&0\\
sin\theta&cos\theta&0\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
1&0&1\\
0&-1&0\\
left&top&1
\end{bmatrix}
$$



​	**在OpenCV中图像旋转首先根据旋转角度和旋转中心获取旋转矩阵，然后根据旋转矩阵进行变换，即可实现任意角度和任意中心的旋转效果。**

1. API

   ```
   cv2.getRotationMatrix2D(center, angle, scale)
   ```

   参数：

   - center：旋转中心
   - angle：旋转角度
   - scale：缩放比例

   返回：

   - M：旋转矩阵

     调用cv.warpAffine完成图像的旋转

2. 示例

   ```python
   import numpy as np
   import cv2 as cv
   import matplotlib.pyplot as plt
   # 1 读取图像
   img = cv.imread("./image/image2.jpg")
   
   # 2 图像旋转
   rows,cols = img.shape[:2]
   # 2.1 生成旋转矩阵
   M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
   # 2.2 进行旋转变换
   dst = cv.warpAffine(img,M,(cols,rows))
   
   # 3 图像展示
   fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
   axes[0].imshow(img1[:,:,::-1])
   axes[0].set_title("原图")
   axes[1].imshow(dst[:,:,::-1])
   axes[1].set_title("旋转后结果")
   plt.show()
   ```

   ![image-20190926152854704](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730032.png)

### 3.1.4 仿射变换

​	图像的仿射变换涉及到图像的形状位置角度的变化，是深度学习预处理中常到的功能,仿射变换主要是对图像的缩放，旋转，翻转和平移等操作的组合。

​	那什么是图像的仿射变换，如下图所示，图1中的点1, 2 和 3 与图二中三个点一一映射, 仍然形成三角形, 但形状已经大大改变，通过这样两组三点（感兴趣点）求出仿射变换， 接下来我们就能把仿射变换应用到图像中所有的点中，就完成了图像的仿射变换。

![image-20191023115222617](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730033.png)

在OpenCV中，仿射变换的矩阵是一个2×3的矩阵，
$$
M=
\begin{bmatrix}
A&B
\end{bmatrix}
=
\begin{bmatrix}
a_{00}&a_{01}&b_{0}\\
a_{10}&a_{11}&b_{1}
\end{bmatrix}
$$
其中左边的2×2子矩阵$A$是线性变换矩阵，右边的2×1子矩阵$B$是平移项：
$$
A=\begin{bmatrix}
a_{00}&
a_{01}\\
a_{10}&
a_{11}\end{bmatrix}
,
B=\begin{bmatrix}
b_{0}\\
b_{1}
\end{bmatrix}
$$
对于图像上的任一位置$(x,y)$，仿射变换执行的是如下的操作：
$$
𝑇𝑎𝑓𝑓𝑖𝑛𝑒=𝐴\begin{bmatrix}x\\y\end{bmatrix}+𝐵=𝑀\begin{bmatrix}x\\y\\1\end{bmatrix}
$$
​	需要注意的是，对于图像而言，宽度方向是x，高度方向是y，坐标的顺序和图像像素对应下标一致。所以原点的位置不是左下角而是右上角，y的方向也不是向上，而是向下。

​	在仿射变换中，原图中所有的平行线在结果图像中同样平行。为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。然后cv2.getAﬃneTransform 会创建一个 2x3 的矩阵，最后这个矩阵会被传给函数 cv2.warpAﬃne。

示例

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1 图像读取
img = cv.imread("./image/image2.jpg")

# 2 仿射变换
rows,cols = img.shape[:2]
# 2.1 创建变换矩阵
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
# 2.2 完成仿射变换
dst = cv.warpAffine(img,M,(cols,rows))

# 3 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("仿射后结果")
plt.show()
```

![image-20190926161027173](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730034.png)

### 3.1.5 透射变换

透射变换是视角变化的结果，是指利用透视中心、像点、目标点三点共线的条件，按透视旋转定律使承影面（透视面）绕迹线（透视轴）旋转某一角度，破坏原有的投影光线束，仍能保持承影面上投影几何图形不变的变换。

![image-20191023130051717](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730035.png)

它的本质将图像投影到一个新的视平面，其通用变换公式为：
$$
\begin{bmatrix}𝑥′&𝑦′&𝑧′\end{bmatrix}=
\begin{bmatrix}
𝑢&𝑣&𝑤
\end{bmatrix}
\begin{bmatrix}
𝑎_{00}&𝑎_{01}&𝑎_{02}\\
𝑎_{10}&𝑎_{11}&𝑎_{12}\\
𝑎_{20}&𝑎_{21}&𝑎_{22}
\end{bmatrix}
$$
其中，$(u,v)$是原始的图像像素坐标，w取值为1，$(x=x'/z',y=y'/z')$是透射变换后的结果。后面的矩阵称为透视变换矩阵，一般情况下，我们将其分为三部分：


$$
T=
\begin{bmatrix}
a_{00}&a_{01}&a_{02}\\
a_{10}&a_{11}&a_{12}\\
a_{20}&a_{21}&a_{22}
\end{bmatrix}
=
\begin{bmatrix}
T1&T2\\
T3&a_{22}
\end{bmatrix}
$$
其中：T1表示对图像进行线性变换，T2对图像进行平移，T3表示对图像进行投射变换，𝑎22*a*22一般设为1.

在opencv中，我们要找到四个点，其中任意三个不共线，然后获取变换矩阵T，再进行透射变换。通过函数cv.getPerspectiveTransform找到变换矩阵，将cv.warpPerspective应用于此3x3变换矩阵。

1. 示例

   ```python
   import numpy as np
   import cv2 as cv
   import matplotlib.pyplot as plt
   # 1 读取图像
   img = cv.imread("./image/image2.jpg")
   # 2 透射变换
   rows,cols = img.shape[:2]
   # 2.1 创建变换矩阵
   pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
   pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
   
   T = cv.getPerspectiveTransform(pts1,pts2)
   # 2.2 进行变换
   dst = cv.warpPerspective(img,T,(cols,rows))
   
   # 3 图像显示
   fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
   axes[0].imshow(img[:,:,::-1])
   axes[0].set_title("原图")
   axes[1].imshow(dst[:,:,::-1])
   axes[1].set_title("透射后结果")
   plt.show()
   ```

   ![image-20190926162913916](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730036.png)

### 3.1.6 图像金字塔

图像金字塔是图像多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构。

图像金字塔用于机器视觉和图像压缩，一幅图像的金字塔是一系列以金字塔形状排列的分辨率逐步降低，且来源于同一张原始图的图像集合。其通过梯次向下采样获得，直到达到某个终止条件才停止采样。

金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似，层级越高，图像越小，分辨率越低。

![timg](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730037.jpeg)

1. API

   ```python
   cv.pyrUp(img)       #对图像进行上采样
   cv.pyrDown(img)        #对图像进行下采样
   ```

2. 示例

   ```python
   import numpy as np
   import cv2 as cv
   import matplotlib.pyplot as plt
   # 1 图像读取
   img = cv.imread("./image/image2.jpg")
   # 2 进行图像采样
   up_img = cv.pyrUp(img)  # 上采样操作
   img_1 = cv.pyrDown(img)  # 下采样操作
   # 3 图像显示
   cv.imshow('enlarge', up_img)
   cv.imshow('original', img)
   cv.imshow('shrink', img_1)
   cv.waitKey(0)
   cv.destroyAllWindows()
   ```

   ![image-20190926114816933](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730038.png)

------

**总结**

1. 图像缩放：对图像进行放大或缩小

   cv.resize()

2. 图像平移：

   指定平移矩阵后，调用cv.warpAffine()平移图像

3. 图像旋转：

   调用cv.getRotationMatrix2D获取旋转矩阵，然后调用cv.warpAffine()进行旋转

4. 仿射变换：

   调用cv.getAffineTransform将创建变换矩阵，最后该矩阵将传递给cv.warpAffine()进行变换

5. 透射变换：

   通过函数cv.getPerspectiveTransform()找到变换矩阵，将cv.warpPerspective()进行投射变换

6. 金字塔

   图像金字塔是图像多尺度表达的一种，使用的API：

   cv.pyrUp(): 向上采样

   cv.pyrDown(): 向下采样



## 3.2 形态学操作

**学习目标**

- 理解图像的邻域，连通性
- 了解不同的形态学操作：腐蚀，膨胀，开闭运算，礼帽和黑帽等，及其不同操作之间的关系

------

### 3.2.1 连通性

在图像中，最小的单位是像素，每个像素周围有8个邻接像素，常见的邻接关系有3种：4邻接、8邻接和D邻接。分别如下图所示：

![image-20190926185646667](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730039.png)

- 4邻接：像素p(x,y)的4邻域是：(x+1,y)；(x-1,y)；(x,y+1)；(x,y-1)，用𝑁~4~(𝑝)*N*~4~(*p*)表示像素p的4邻接
- D邻接：像素p(x,y)的D邻域是：对角上的点 (x+1,y+1)；(x+1,y-1)；(x-1,y+1)；(x-1,y-1)，用𝑁𝐷(𝑝)*N**D*(*p*)表示像素p的D邻域
- 8邻接：像素p(x,y)的8邻域是： 4邻域的点 ＋ D邻域的点，用𝑁8(𝑝)*N*8(*p*)表示像素p的8邻域

**连通性**是描述区域和边界的重要概念，两个像素连通的两个必要条件是：

1. 两个像素的位置是否相邻
2. 两个像素的灰度值是否满足特定的相 似性准则（或者是否相等

根据连通性的定义，有4联通、8联通和m联通三种。

- 4联通：对于具有值𝑉*V*的像素𝑝和𝑞，如果𝑞在集合𝑁~4~(𝑝)中，则称这两个像素是4连通。

- 8联通：对于具有值𝑉*V*的像素𝑝和𝑞，如果𝑞在集 合𝑁8(𝑝)中，则称这两个像素是8连通。

  ![image-20190926185504256](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730041.png)

- 对于具有值𝑉的像素𝑝和𝑞，如果:

  1. 𝑞在集合𝑁4(𝑝)*N*4(*p*)中，或
  2. 𝑞在集合𝑁~𝐷~(𝑝)中，并且𝑁4(𝑝)*N*4(*p*)与𝑁4(𝑞)*N*4(*q*)的交集为空（没有值𝑉*V*的像素）

  则称这两个像素是𝑚连通的，即4连通和D连通的混合连通。

  ![image-20190927101630929](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730042.png)

形态学操作

​	形态学转换是基于图像形状的一些简单操作。它通常在二进制图像上执行。腐蚀和膨胀是两个基本的形态学运算符。然后它的变体形式如开运算，闭运算，礼帽黑帽等。

### 3.2.2 腐蚀和膨胀

​	腐蚀和膨胀是最基本的形态学操作，腐蚀和膨胀都是针对高亮部分而言的。

​	膨胀就是使图像中高亮部分扩张，效果图拥有比原图更大的高亮区域；腐蚀是原图中的高亮区域被蚕食，效果图拥有比原图更小的高亮区域。膨胀是求局部最大值的操作，腐蚀是求局部最小值的操作。

1. **腐蚀**

   具体操作是：用一个结构元素扫描图像中的每一个像素，用结构元素中的每一个像素与其覆盖的像素做“与”操作，如果都为1，则该像素为1，否则为0。如下图所示，结构A被结构B腐蚀后：
   
   ![image-20190927105316401](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730043.png)

​	腐蚀的**作用**是消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点。

**API**：

```python
   cv.erode(img,kernel,iterations)
```

参数：

- img: 要处理的图像
- kernel: 核结构
- iterations: 腐蚀的次数，默认是1

1. **膨胀**

​	具体操作是：用一个结构元素扫描图像中的每一个像素，用结构元素中的每一个像素与其覆盖的像素做“或”操作，如果都为0，则该像素为0，否则为1。如下图所示，结构A被结构B膨胀后：

![image-20190927110711458](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730044.png)	膨胀的作用是将与物体接触的所有背景点合并到物体中，使目标增大，可添补目标中的孔洞。

**API**：

```python
   cv.dilate(img,kernel,iterations)
```

参数：

- img: 要处理的图像
- kernel: 核结构
- iterations: 腐蚀的次数，默认是1

1. **示例**

我们使用一个5*5的卷积核实现腐蚀和膨胀的运算：

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)  # 使用黑体字体

# 1 读取图像
img = cv.imread("./image/letter.png")
if img is None:
    raise FileNotFoundError("图像文件未找到，请检查路径。")

# 2 创建核结构
kernel = np.ones((5, 5), np.uint8)

# 3 图像腐蚀和膨胀
erosion = cv.erode(img, kernel)  # 腐蚀
dilate = cv.dilate(img, kernel)  # 膨胀

# 4 图像展示
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0].set_title("原图", fontproperties=font)
axes[1].imshow(cv.cvtColor(erosion, cv.COLOR_BGR2RGB))
axes[1].set_title("腐蚀后结果", fontproperties=font)
axes[2].imshow(cv.cvtColor(dilate, cv.COLOR_BGR2RGB))
axes[2].set_title("膨胀后结果", fontproperties=font)
plt.show()
```

![image-20190927151844574](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730045.png)

### 3.2.3 开闭运算

​	开运算和闭运算是将腐蚀和膨胀按照一定的次序进行处理。 但这两者并不是可逆的，即先开后闭并不能得到原来的图像。

1. **开运算**

   ​	开运算是**先腐蚀后膨胀**，其**作用**是：分离物体，消除小区域。**特点**：消除噪点，去除小的干扰块，而不影响原来的图像。

   ![image-20190927142206425](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730046.png)

2. **闭运算**

   闭运算与开运算相反，是先膨胀后腐蚀，**作用**是消除/“闭合”物体里面的孔洞，**特点**：可以填充闭合区域。

   ![image-20190927142923777](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730047.png)

3. **API**

   ```
   cv.morphologyEx(img, op, kernel)
   ```

   参数：

   - img: 要处理的图像
   - op: 处理方式：若进行开运算，则设为cv.MORPH_OPEN，若进行闭运算，则设为cv.MORPH_CLOSE
   - Kernel： 核结构

4. **示例**

   使用10*10的核结构对卷积进行开闭运算的实现。

   ```python
   import numpy as np
   import cv2 as cv
   import matplotlib.pyplot as plt
   # 1 读取图像
   img1 = cv.imread("./image/image5.png")
   img2 = cv.imread("./image/image6.png")
   # 2 创建核结构
   kernel = np.ones((10, 10), np.uint8)
   # 3 图像的开闭运算
   cvOpen = cv.morphologyEx(img1,cv.MORPH_OPEN,kernel) # 开运算
   cvClose = cv.morphologyEx(img2,cv.MORPH_CLOSE,kernel)# 闭运算
   # 4 图像展示
   fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
   axes[0,0].imshow(img1)
   axes[0,0].set_title("原图")
   axes[0,1].imshow(cvOpen)
   axes[0,1].set_title("开运算结果")
   axes[1,0].imshow(img2)
   axes[1,0].set_title("原图")
   axes[1,1].imshow(cvClose)
   axes[1,1].set_title("闭运算结果")
   plt.show()
   ```

   ![image-20190927153400823](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730048.png)

### 3.2.4 礼帽和黑帽

1. **礼帽运算**

   原图像与“开运算“的结果图之差，如下式计算：

   ![image-20190927144145071](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730049.png)

   　　因为开运算带来的结果是放大了裂缝或者局部低亮度的区域，因此，从原图中减去开运算后的图，得到的效果图突出了比原图轮廓周围的区域更明亮的区域，且这一操作和选择的核的大小相关。

   　　礼帽运算用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。

2. **黑帽运算**

   为”闭运算“的结果图与原图像之差。数学表达式为：

   　　![image-20190927144356013](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730050.png)

   黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。

   黑帽运算用来分离比邻近点暗一些的斑块。

3. **API**

   ```
   cv.morphologyEx(img, op, kernel)
   ```

   参数：

   - img: 要处理的图像

   - op: 处理方式：

     ![image-20191016162318033](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730051.png)

   - Kernel： 核结构

4. **示例**

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1 读取图像
img1 = cv.imread("./image/image5.png")
img2 = cv.imread("./image/image6.png")
# 2 创建核结构
kernel = np.ones((10, 10), np.uint8)
# 3 图像的礼帽和黑帽运算
cvOpen = cv.morphologyEx(img1,cv.MORPH_TOPHAT,kernel) # 礼帽运算
cvClose = cv.morphologyEx(img2,cv.MORPH_BLACKHAT,kernel)# 黑帽运算
# 4 图像显示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img1)
axes[0,0].set_title("原图")
axes[0,1].imshow(cvOpen)
axes[0,1].set_title("礼帽运算结果")
axes[1,0].imshow(img2)
axes[1,0].set_title("原图")
axes[1,1].imshow(cvClose)
axes[1,1].set_title("黑帽运算结果")
plt.show()
```

![image-20190927154018177](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730052.png)

------

**总结**

1. 连通性 邻接关系：4邻接，8邻接和D邻接

   连通性：4连通，8连通和m连通

2. 形态学操作

   - 腐蚀和膨胀：

     腐蚀：求局部最大值

     膨胀：求局部最小值

   - 开闭运算：

     开：先腐蚀后膨胀

     闭：先膨胀后腐蚀

   - 礼帽和黑帽：

     礼帽：原图像与开运算之差

     黑帽：闭运算与原图像之差



## 3.3图像平滑

**学习目标**

- 了解图像中的噪声类型
- 了解平均滤波，高斯滤波，中值滤波等的内容
- 能够使用滤波器对图像进行处理

------

### 3.3.1 图像噪声

​	由于图像采集、处理、传输等过程不可避免的会受到噪声的污染，妨碍人们对图像理解及分析处理。常见的图像噪声有高斯噪声、椒盐噪声等。

<u>**椒盐噪声**</u>

​	椒盐噪声也称为脉冲噪声，是图像中经常见到的一种噪声，它是一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）。椒盐噪声的成因可能是影像讯号受到突如其来的强烈干扰而产生、类比数位转换器或位元传输错误等。例如失效的感应器导致像素值为最小值，饱和的感应器导致像素值为最大值。

![image-20190927163654718](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730053.png)

<u>**高斯噪声**</u>

​	高斯噪声是指噪声密度函数服从高斯分布的一类噪声。由于高斯噪声在空间和频域中数学上的易处理性，这种噪声(也称为正态噪声)模型经常被用于实践中。高斯随机变量z的概率密度函数由下式给出：
$$
p(z)=\frac{1}{\sqrt{2\pi\sigma}}e^{\frac{-(z-u)^2}{2\sigma^2}}
$$


​	其中z表示灰度值，μ表示z的平均值或期望值，σ表示z的标准差。标准差的平方$\sigma^2$称为z的方差。高斯函数的曲线如图所示。

![image-20190927164749878](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730054.png)

![image-20190927164045611](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730055.png)

**<u>图像平滑简介</u>**

​	图像平滑从信号处理的角度看就是**去除**其中的**高频信息**，**保留低频信息**。因此我们可以对图像实施低通滤波。低通滤波可以去除图像中的噪声，对图像进行平滑。

根据滤波器的不同可分为均值滤波，高斯滤波，中值滤波， 双边滤波。

### 3.3.2 均值滤波

​	采用均值滤波模板对图像噪声进行滤除。令$S_{xy}$表示中心在(x, y)点，尺寸为$m×n$ 的矩形子图像窗口的坐标组。 均值滤波器可表示为：
$$
\hat{f}(x, y) = \frac{1}{mn} \sum_{(s,t) \in S_{xy}} g(s, t)
$$
由一个归一化卷积框完成的。它只是用卷积框覆盖区域所有像素的平均值来代替中心元素。

例如，3x3标准化的平均过滤器如下所示：
$$
K=\frac{1}{9}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}
$$
均值滤波的优点是算法简单，计算速度较快，缺点是在去噪的同时去除了很多细节部分，将图像变得模糊。

API:

```
cv.blur(src, ksize, anchor, borderType)
```

参数:

- src：输入图像
- ksize：卷积核的大小
- anchor：默认值 (-1,-1) ，表示核中心
- borderType：边界类型

示例：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1 图像读取
img = cv.imread('./image/dogsp.jpeg')
# 2 均值滤波
blur = cv.blur(img,(5,5))
# 3 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('均值滤波后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190928102258185](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730056.png)

### 3.3.3 高斯滤波

二维高斯是构建高斯滤波器的基础，其概率分布函数如下所示：

$$
G(x,y)=\frac{1}{2\pi\sigma^2}\exp\left\{-\frac{x^2+y^2}{2\sigma^2}\right\}
$$
G(x,y)的分布是一个突起的帽子的形状。这里的σ可以看作两个值，一个是x方向的标准差$\sigma_{x}$，另一个是y方向的标准差$\sigma_{y}$

![image-20190928104118332](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730058.png)	当$\sigma_{x}$和$\sigma_{y}$取值越大，整个形状趋近于扁平；当，$\sigma_{x}$和$\sigma_{y}$越小整个形状越突起。

​	正态分布是一种钟形曲线，越接近中心，取值越大，越远离中心，取值越小。计算平滑结果时，只需要将"中心点"作为原点，其他点按照其在正态曲线上的位置，分配权重，就可以得到一个加权平均值。

​	高斯平滑在从图像中去除高斯噪声方面非常有效。

**高斯平滑的流程：**

- 首先确定权重矩阵

假定中心点的坐标是（0,0），那么距离它最近的8个点的坐标如下：

![image-20190928110341406](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730059.png)

更远的点以此类推。

为了计算权重矩阵，需要设定σ的值。假定σ=1.5，则模糊半径为1的权重矩阵如下：

![image-20190928110425845](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730060.png)

这9个点的权重总和等于0.4787147，如果只计算这9个点的加权平均，还必须让它们的权重之和等于1，因此上面9个值还要分别除以0.4787147，得到最终的权重矩阵。

**![image-20190928110455467](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730061.png)**

- 计算高斯模糊

有了权重矩阵，就可以计算高斯模糊的值了。

假设现有9个像素点，灰度值（0-255）如下：

![image-20190928110522272](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730062.png)

每个点乘以对应的权重值：

![image-20190928110551897](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730063.png)

得到

![image-20190928110613880](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730064.png)

将这9个值加起来，就是中心点的高斯模糊的值。

对所有点重复这个过程，就得到了高斯模糊后的图像。如果原图是彩色图片，可以对RGB三个通道分别做高斯平滑。

API：

```python
cv2.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
```

参数：

- src: 输入图像
- ksize:高斯卷积核的大小，**注意** ： 卷积核的宽度和高度都应为奇数，且可以不同
- sigmaX: 水平方向的标准差
- sigmaY: 垂直方向的标准差，默认值为0，表示与sigmaX相同
- borderType:填充边界类型

**示例**：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt, rcParams
#设置中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False


# 1 图像读取
img = cv.imread('./image/dogGauss.jpeg')
# 2 高斯滤波
blur = cv.GaussianBlur(img,(3,3),1)
# 3 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('高斯滤波后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190928111903926](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730065.png)

### 3.3.4 中值滤波

​	中值滤波是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值。

​	中值滤波对椒盐噪声（salt-and-pepper noise）来说尤其有用，因为它不依赖于邻域内那些与典型值差别很大的值。

API：

```python
cv.medianBlur(src, ksize )
```

参数：

- src：输入图像
- ksize：卷积核的大小

示例：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1 图像读取
img = cv.imread('./image/dogsp.jpeg')
# 2 中值滤波
blur = cv.medianBlur(img,5)
# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('中值滤波后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190928102319410](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730066.png)

------

**总结**

1. 图像噪声

   - 椒盐噪声：图像中随机出现的白点或者黑点
   - 高斯噪声：噪声的概率密度分布是正态分布

2. 图像平滑

   - 均值滤波：算法简单，计算速度快，在去噪的同时去除了很多细节部分，将图像变得模糊

     cv.blur()

   - 高斯滤波: 去除高斯噪声

     cv.GaussianBlur()

   - 中值滤波: 去除椒盐噪声

     cv.medianBlur()



## 3.4 直方图

**学习目标**

- 掌握图像的直方图计算和显示
- 了解掩膜的应用
- 熟悉直方图均衡化，了解自适应均衡化

------

### 3.4.1 灰度直方图

**<u>原理</u>**

​	直方图是对数据进行统计的一种方法，并且将统计值组织到一系列实现定义好的 bin 当中。其中， bin 为直方图中经常用到的一个概念，可以译为 “直条” 或 “组距”，其数值是从数据中计算出的特征统计量，这些数据可以是诸如梯度、方向、色彩或任何其他特征。

  图像直方图（Image Histogram）是用以表示数字图像中**亮度分布**的直方图，标绘了图像中每个亮度值的像素个数。这种直方图中，横坐标的左侧为较暗的区域，而右侧为较亮的区域。因此一张较暗图片的直方图中的数据多集中于左侧和中间部分，而整体明亮、只有少量阴影的图像则相反。

![image-20190928144352467](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730067.png)

注意：直方图是根据灰度图进行绘制的，而不是彩色图像。   假设有一张图像的信息（灰度值 0 - 255，已知数字的范围包含 256 个值，于是可以按一定规律将这个范围分割成子区域（也就是 bins）。如：
$$
[0,255]=[0,15]\bigcup[16,30]\cdots\bigcup[240,255]
$$
然后再统计每一个 bin(i) 的像素数目。可以得到下图（其中 x 轴表示 bin，y 轴表示各个 bin 中的像素个数）：

     ![image-20190928145730979](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730068.png)

直方图的一些**术语和细节**：

- dims：需要统计的特征数目。在上例中，dims = 1 ，因为仅仅统计了灰度值。
- bins：每个特征空间子区段的数目，可译为 “直条” 或 “组距”，在上例中， bins = 16。
- range：要统计特征的取值范围。在上例中，range = [0, 255]。

直方图的**意义**：

- 直方图是图像中像素强度分布的图形表达方式。   
- 它统计了每一个强度值所具有的像素个数。
- 不同的图像的直方图可能是相同的

### 3.4.2 直方图的计算和绘制

我们使用OpenCV中的方法统计直方图，并使用matplotlib将其绘制出来。

API：

```python
cv2.calcHist(images,channels,mask,histSize,ranges[,hist[,accumulate]])
```

参数：

- images: 原图像。当传入函数时应该用中括号 [] 括起来，例如：[img]。
- channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。 　　
- mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并使用它。（后边有例子） 　　
- histSize:BIN 的数目。也应该用中括号括起来，例如：[256]。 　　
- ranges: 像素值范围，通常为 [0，256]

示例：

如下图，绘制相应的直方图

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 2 统计灰度图
histr = cv.calcHist([img],[0],None,[256],[0,256])
# 3 绘制灰度图
plt.figure(figsize=(10,6),dpi=100)
plt.plot(histr)
plt.grid()
plt.show()
```

![image-20190928155000064](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730069.png)

### 3.4.3 掩膜的应用

​	掩膜是用选定的图像、图形或物体，对要处理的图像进行遮挡，来控制图像 处理的区域。

​	在数字图像处理中，我们通常使用二维矩阵数组进行掩膜。掩膜是由0和1组成一个二进制图像，利用该掩膜图像要处理的图像进行掩膜，其中1值的区域被处理，0 值区域被屏蔽，不会处理。

掩膜的主要用途是：

- 提取感兴趣区域：用预先制作的感兴趣区掩模与待处理图像进行”与“操作，得到感兴趣区图像，感兴趣区内图像值保持不变，而区外图像值都为0。
- 屏蔽作用：用掩模对图像上某些区域作屏蔽，使其不参加处理或不参加处理参数的计算，或仅对屏蔽区作处理或统计。
- 结构特征提取：用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。
- 特殊形状图像制作

​	掩膜在遥感影像处理中使用较多，当提取道路或者河流，或者房屋时，通过一个掩膜矩阵来对图像进行像素过滤，然后将我们需要的地物或者标志突出显示出来。

​	我们使用cv.calcHist（）来查找完整图像的直方图。 如果要查找图像某些区域的直方图，该怎么办？ 只需在要查找直方图的区域上创建一个白色的掩膜图像，否则创建黑色， 然后将其作为掩码mask传递即可。

示例：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1. 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 2. 创建蒙版
mask = np.zeros(img.shape[:2], np.uint8)
mask[400:650, 200:500] = 255
# 3.掩模
masked_img = cv.bitwise_and(img,img,mask = mask)
# 4. 统计掩膜后图像的灰度图
mask_histr = cv.calcHist([img],[0],mask,[256],[1,256])
# 5. 图像展示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img,cmap=plt.cm.gray)
axes[0,0].set_title("原图")
axes[0,1].imshow(mask,cmap=plt.cm.gray)
axes[0,1].set_title("蒙版数据")
axes[1,0].imshow(masked_img,cmap=plt.cm.gray)
axes[1,0].set_title("掩膜后数据")
axes[1,1].plot(mask_histr)
axes[1,1].grid()
axes[1,1].set_title("灰度直方图")
plt.show()
```

![image-20190928160241831](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730070.png)

### 3.4.5  直方图均衡化

**<u>原理与应用</u>**

想象一下，如果一副图像中的大多数像素点的像素值都集中在某一个小的灰度值值范围之内会怎样呢？如果一幅图像整体很亮，那所有的像素值的取值个数应该都会很高。所以应该把它的直方图做一个横向拉伸（如下图），就可以扩大图像像素值的分布范围，提高图像的对比度，这就是直方图均衡化要做的事情。

![image-20190928162111755](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730071.png)

“直方图均衡化”是把原始图像的灰度直方图从比较集中的某个灰度区间变成在更广泛灰度范围内的分布。直方图均衡化就是对图像进行非线性拉伸，重新分配图像像素值，使一定灰度范围内的像素数量大致相同。

这种方法提高图像整体的对比度，特别是有用数据的像素值分布比较接近时，在X光图像中使用广泛，可以提高骨架结构的显示，另外在曝光过度或不足的图像中可以更好的突出细节。

使用opencv进行直方图统计时，使用的是：

API：

```python
dst = cv.equalizeHist(img)
```

参数：

- img: 灰度图像

返回：

- dst : 均衡化后的结果

示例：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1. 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 2. 均衡化处理
dst = cv.equalizeHist(img)
# 3. 结果展示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(dst,cmap=plt.cm.gray)
axes[1].set_title("均衡化后结果")
plt.show()
```

![image-20190928163431354](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730072.png)

### 3.4.5 自适应的直方图均衡化

上述的直方图均衡，我们考虑的是图像的全局对比度。 的确在进行完直方图均衡化之后，图片背景的对比度被改变了，在猫腿这里太暗，我们丢失了很多信息，所以在许多情况下，这样做的效果并不好。如下图所示，对比下两幅图像中雕像的画面，由于太亮我们丢失了很多信息。

![image-20191024105039014](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730073.png)

为了解决这个问题， 需要使用自适应的直方图均衡化。 此时， 整幅图像会被分成很多小块，这些小块被称为“tiles”（在 OpenCV 中 tiles 的 大小默认是 8x8），然后再对每一个小块分别进行直方图均衡化。 所以在每一个的区域中， 直方图会集中在某一个小的区域中）。如果有噪声的话，噪声会被放大。为了避免这种情况的出现要使用对比度限制。对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话，就把 其中的像素点均匀分散到其他 bins 中，然后在进行直方图均衡化。

![image-20191024105353109](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730074.png)

最后，为了 去除每一个小块之间的边界，再使用双线性差值，对每一小块进行拼接。

API：

```python
cv.createCLAHE(clipLimit, tileGridSize)
```

参数：

- clipLimit: 对比度限制，默认是40
- tileGridSize: 分块的大小，默认为8∗88∗8

示例：

```python
import numpy as np
import cv2 as cv
# 1. 以灰度图形式读取图像
img = cv.imread('./image/cat.jpeg',0)
# 2. 创建一个自适应均衡化的对象，并应用于图像
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
# 3. 图像展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(cl1,cmap=plt.cm.gray)
axes[1].set_title("自适应均衡化后的结果")
plt.show()
```

![image-20190928165605432](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730075.png)

------

**总结**

1. 灰度直方图：

   - 直方图是图像中像素强度分布的图形表达方式。
   - 它统计了每一个强度值所具有的像素个数。
   - 不同的图像的直方图可能是相同的

   cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）

2. 掩膜

   创建蒙版，透过mask进行传递，可获取感兴趣区域的直方图

3. 直方图均衡化：增强图像对比度的一种方法

   cv.equalizeHist(): 输入是灰度图像，输出是直方图均衡图像

4. 自适应的直方图均衡

   将整幅图像分成很多小块，然后再对每一个小块分别进行直方图均衡化，最后进行拼接

   clahe = cv.createCLAHE(clipLimit, tileGridSize)



## 3.5 边缘检测

**学习目标**

- 了解Sobel算子，Scharr算子和拉普拉斯算子
- 掌握canny边缘检测的原理及应用

------

### 3.5.1原理

边缘检测是图像处理和计算机视觉中的基本问题，边缘检测的目的是标识数字图像中亮度变化明显的点。图像属性中的显著变化通常反映了属性的重要事件和变化。边缘的表现形式如下图所示：

![image-20191024160953045](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730076.png)

图像边缘检测大幅度地减少了数据量，并且剔除了可以认为不相关的信息，保留了图像重要的结构属性。有许多方法用于边缘检测，它们的绝大部分可以划分为两类：**基于搜索**和**基于零穿越**。

- 基于搜索：通过寻找图像一阶导数中的最大值来检测边界，然后利用计算结果估计边缘的局部方向，通常采用梯度的方向，并利用此方向找到局部梯度模的最大值，代表算法是Sobel算子和Scharr算子。

  ![image-20190929104240226](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730077.png)

- 基于零穿越：通过寻找图像二阶导数零穿越来寻找边界，代表算法是Laplacian算子。

  ![image-20190929104430480](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730078.png)

### 3.5.2 Sobel检测算子

Sobel边缘检测算法比较简单，实际应用中效率比canny边缘检测效率要高，但是边缘不如Canny检测的准确，但是很多实际应用的场合，sobel边缘却是首选，Sobel算子是高斯平滑与微分操作的结合体，所以其抗噪声能力很强，用途较多。尤其是效率要求较高，而对细纹理不太关心的时候。

**<u>方法</u>**

对于不连续的函数，一阶导数可以写作：
$$
f'(x)=f(x)-f(x-1)
$$
或
$$
f'(x)=f(x+1)-f(x)
$$
所以有：
$$
f^{\prime}(x)=\frac{f(x+1)-f(x-1)}2
$$
假设要处理的图像为𝐼*I*，在两个方向求导:

-  **水平变化**: 将图像𝐼*I* 与奇数大小的模版进行卷积，结果为𝐺~x~。比如，当模板大小为3时, 𝐺~𝑥~为:

$$
G_x=\begin{bmatrix}-1&0&+1\\-2&0&+2\\-1&0&+1\end{bmatrix}*I
$$



-  **垂直变化**: 将图像𝐼*I*与奇数大小的模板进行卷积，结果为𝐺~𝑦~。比如，当模板大小为3时, 𝐺~𝑦~为:

$$
G_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\+1&+2&+1\end{bmatrix}*I
$$

在图像的每一点，结合以上两个结果求出：
$$
G=\sqrt{G_x^2+G_y^2}
$$
统计极大值所在的位置，就是图像的边缘。

**注意**：当内核大小为3时, 以上Sobel内核可能产生比较明显的误差， 为解决这一问题，我们使用Scharr函数，但该函数仅作用于大小为3的内核。该函数的运算与Sobel函数一样快，但结果却更加精确，其计算方法为:
$$
\begin{gathered}G_x=\begin{bmatrix}-3&0&+3\\-10&0&+10\\-3&0&+3\end{bmatrix}*I\\G_y=\begin{bmatrix}-3&-10&-3\\0&0&0\\+3&+10&+3\end{bmatrix}*I\end{gathered}
$$
**<u>应用</u>**

利用OpenCV进行sobel边缘检测的API是：

```python
Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
```

参数：

- src：传入的图像

- ddepth: 图像的深度

- dx和dy: 指求导的阶数，0表示这个方向上没有求导，取值为0、1。

- ksize: 是Sobel算子的大小，即卷积核的大小，必须为奇数1、3、5、7，默认为3。

  注意：如果ksize=-1，就演变成为3x3的Scharr算子。

- scale：缩放导数的比例常数，默认情况为没有伸缩系数。

- borderType：图像边界的模式，默认值为cv2.BORDER_DEFAULT。

Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。处理完图像后，再使用cv2.convertScaleAbs()函数将其转回原来的uint8格式，否则图像无法显示。

Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted( )函数将其组合起来

```pyh
Scale_abs = cv2.convertScaleAbs(x)  # 格式转换函数
result = cv2.addWeighted(src1, alpha, src2, beta) # 图像混合
```

示例：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 读取图像
img = cv.imread('./image/horse.jpg',0)
# 2 计算Sobel卷积结果
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)
# 3 将数据进行转换
Scale_absX = cv.convertScaleAbs(x)  # convert 转换  scale 缩放
Scale_absY = cv.convertScaleAbs(y)
# 4 结果合成
result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
# 5 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(result,cmap = plt.cm.gray),plt.title('Sobel滤波后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190929141752847](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730079.png)

将上述代码中计算sobel算子的部分中将ksize设为-1，就是利用Scharr进行边缘检测。

```Python
x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize = -1)
y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize = -1)
```

![image-20190929141636521](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730080.png)

### 3.5.3 Laplacian算子

Laplacian是利用二阶导数来检测边缘 。 因为图像是 “*2维*”, 我们需要在两个方向求导，如下式所示：
$$
\Delta src=\frac{\partial^2src}{\partial x^2}+\frac{\partial^2src}{\partial y^2}
$$
*

那不连续函数的二阶导数是：
$$
f''(x)=f'(x+1)-f'(x)=f(x+1)+f(x-1)-2f(x)
$$
那使用的卷积核是：
$$
kernel=\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}
$$


```python
laplacian = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
```

参数：

- Src: 需要处理的图像，
- Ddepth: 图像的深度，-1表示采用的是原图像相同的深度，目标图像的深度必须大于等于原图像的深度；
- ksize：算子的大小，即卷积核的大小，必须为1,3,5,7。

示例：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 读取图像
img = cv.imread('./image/horse.jpg',0)
# 2 laplacian转换
result = cv.Laplacian(img,cv.CV_16S)
Scale_abs = cv.convertScaleAbs(result)
# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(Scale_abs,cmap = plt.cm.gray),plt.title('Laplacian检测后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190929145507862](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730081.png)

### 3.5.4 Canny边缘检测

Canny 边缘检测算法是一种非常流行的边缘检测算法，是 John F. Canny 于 1986年提出的，被认为是最优的边缘检测算法。

**<u>原理</u>**

Canny边缘检测算法是由4步构成，分别介绍如下：

- 第一步：噪声去除

  ###### 由于边缘检测很容易受到噪声的影响，所以首先使用$5*5$高斯滤波器去除噪声，在图像平滑那一章节中已经介绍过。

- 第二步：计算图像梯度

对平滑后的图像使用 Sobel 算子计算水平方向和竖直方向的一阶导数（Gx 和 Gy）。根据得到的这两幅梯度图（Gx 和 Gy）找到边界的梯度和方向，公式如下:
$$
Edge_Gradient\left(G\right)=\sqrt{G_x^2+G_y^2}
$$

$$
Angle\left(\theta\right)=tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

如果某个像素点是边缘，则其梯度方向总是垂直与边缘垂直。梯度方向被归为四类：垂直，水平，和两个对角线方向。

- 第三步：非极大值抑制

在获得梯度的方向和大小之后，对整幅图像进行扫描，去除那些非边界上的点。对每一个像素进行检查，看这个点的梯度是不是周围具有相同梯度方向的点中最大的。如下图所示：

![image-20190929153926063](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730083.png)

A点位于图像的边缘，在其梯度变化方向，选择像素点B和C，用来检验A点的梯度是否为极大值，若为极大值，则进行保留，否则A点被抑制，最终的结果是具有“细边”的二进制图像。

- 第四步：滞后阈值

现在要确定真正的边界。 我们设置两个阈值： minVal 和 maxVal。 当图像的灰度梯度高于 maxVal 时被认为是真的边界， 低于 minVal 的边界会被抛弃。如果介于两者之间的话，就要看这个点是否与某个被确定为真正的边界点相连，如果是就认为它也是边界点，如果不是就抛弃。如下图：

![image-20190929155208751](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730084.png)

如上图所示，A 高于阈值 maxVal 所以是真正的边界点，C 虽然低于 maxVal 但高于 minVal 并且与 A 相连，所以也被认为是真正的边界点。而 B 就会被抛弃，因为低于 maxVal 而且不与真正的边界点相连。所以选择合适的 maxVal 和 minVal 对于能否得到好的结果非常重要。

**<u>应用</u>**

在OpenCV中要实现Canny检测使用的API:

```python
canny = cv2.Canny(image, threshold1, threshold2)
```

参数：

- image:灰度图，
- threshold1: minval，较小的阈值将间断的边缘连接起来
- threshold2: maxval，较大的阈值检测图像中明显的边缘

示例：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 图像读取
img = cv.imread('./image/horse.jpg',0)
# 2 Canny边缘检测
lowThreshold = 0
max_lowThreshold = 100
canny = cv.Canny(img, lowThreshold, max_lowThreshold) 
# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = plt.cm.gray),plt.title('Canny检测后结果')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20190929160959357](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730085.png)

------

**总结**

1. 边缘检测的原理

   - 基于搜索：利用一阶导数的最大值获取边界
   - 基于零穿越：利用二阶导数为0获取边界

2. Sobel算子

   基于搜索的方法获取边界

   cv.sobel()

   cv.convertScaleAbs()

   cv.addweights()

3. Laplacian算子

   基于零穿越获取边界

   cv.Laplacian()

4. Canny算法

   流程：

   - 噪声去除：高斯滤波
   - 计算图像梯度：sobel算子，计算梯度大小和方向
   - 非极大值抑制：利用梯度方向像素来判断当前像素是否为边界点
   - 滞后阈值：设置两个阈值，确定最终的边界

5 算子比较

![image-20191024180930270](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730086.png)

[
  ](section5.html)





## 3.6 模版匹配和霍夫变换

**学习目标**

- 掌握模板匹配的原理，能完成模板匹配的应用
- 理解霍夫线变换的原理，了解霍夫圆检测
- 知道使用OpenCV如何进行线和圆的检测

------

### 3.6.1 模板匹配

**<u>原理</u>**

所谓的模板匹配，就是在给定的图片中查找和模板最相似的区域，该算法的输入包括模板和图片，整个任务的思路就是按照滑窗的思路不断的移动模板图片，计算其与图像中对应区域的匹配度，最终将匹配度最高的区域选择为最终的结果。

**实现流程：**

- 准备两幅图像：

  1.原图像(I)：在这幅图中，找到与模板相匹配的区域

  2.模板(T)：与原图像进行比对的图像块

  ![Template_Matching_Template_Theory_Summary](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730087.jpg)

- 滑动模板图像和原图像进行比对：

![Template_Matching_Template_Theory_Sliding](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730088.jpg)

将模板块每次移动一个像素 (从左往右，从上往下)，在每一个位置，都计算与模板图像的相似程度。

- 对于每一个位置将计算的相似结果保存在结果矩阵（R）中。如果输入图像的大小（WxH）且模板图像的大小(wxh)，则输出矩阵R的大小为（W-w + 1,H-h + 1）将R显示为图像，如下图所示：

![Template_Matching_Template_Theory_Result](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730089.jpg)

- 获得上述图像后，查找最大值所在的位置，那么该位置对应的区域就被认为是最匹配的。对应的区域就是以该点为顶点，长宽和模板图像一样大小的矩阵。

**<u>实现</u>**

我们使用OpenCV中的方法实现模板匹配。

API：

```
res = cv.matchTemplate(img,template,method)
```

参数：

- img: 要进行模板匹配的图像
- Template ：模板
- method：实现模板匹配的算法，主要有：
  1. 平方差匹配(CV_TM_SQDIFF)：利用模板与图像之间的平方差进行匹配，最好的匹配是0，匹配越差，匹配的值越大。
  2. 相关匹配(CV_TM_CCORR)：利用模板与图像间的乘法进行匹配，数值越大表示匹配程度较高，越小表示匹配效果差。
  3. 利用相关系数匹配(CV_TM_CCOEFF)：利用模板与图像间的相关系数匹配，1表示完美的匹配，-1表示最差的匹配。

完成匹配后，使用cv.minMaxLoc()方法查找最大值所在的位置即可。如果使用平方差作为比较方法，则最小值位置是最佳匹配位置。

**示例：**

在该案例中，载入要搜索的图像和模板，图像如下所示：

![wulin2](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730090.jpeg)

模板如下所示：

![wulin-0430810](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730091.jpeg)

通过matchTemplate实现模板匹配，使用minMaxLoc定位最匹配的区域，并用矩形标注最匹配的区域。

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 图像和模板读取
img = cv.imread('./image/wulin2.jpeg')
template = cv.imread('./image/wulin.jpeg')
h,w,l = template.shape
# 2 模板匹配
# 2.1 模板匹配
res = cv.matchTemplate(img, template, cv.TM_CCORR)
# 2.2 返回图像中最匹配的位置，确定左上角的坐标，并将匹配位置绘制在图像上
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# 使用平方差时最小值为最佳匹配位置
# top_left = min_loc
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img, top_left, bottom_right, (0,255,0), 2)
# 3 图像显示
plt.imshow(img[:,:,::-1])
plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20191007144614688](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730092.png)

拓展：模板匹配不适用于尺度变换，视角变换后的图像，这时我们就要使用关键点匹配算法，比较经典的关键点检测算法包括SIFT和SURF等，主要的思路是首先通过关键点检测算法获取模板和测试图片中的关键点；然后使用关键点匹配算法处理即可，这些关键点可以很好的处理尺度变化、视角变换、旋转变化、光照变化等，具有很好的不变性。

### 3.6.2 霍夫变换

霍夫变换常用来提取图像中的直线和圆等几何形状，如下图所示：

![image-20191007151122386](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730093.png)

**<u>2.1 原理</u>**

1. **原理**

在笛卡尔坐标系中，一条直线由两个点$A=(x_1,y_1)$和$B=(x_2,y_2)$确定，如下图所示:

![image-20191007153126537](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730094.png)

将直线y=kx+q可写成关于(𝑘,𝑞)的函数表达式：
$$
\begin{cases}q=-kx_1+y_1\\q=-kx_2+y_2\end{cases}
$$
对应的变换通过图形直观的表示下：

![image-20191007154123721](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730095.png)

变换后的空间我们叫做霍夫空间。即：**笛卡尔坐标系中的一条直线，对应于霍夫空间中的一个点**。反过来，同样成立，霍夫空间中的一条线，对应于笛卡尔坐标系中一个点，如下所示：

 ![image-20191007154350195](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730096.png)

我们再来看下A、B两个点，对应于霍夫空间的情形：

![image-20191007154546905](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730097.png)

在看下三点共线的情况：

![image-20191007160434136](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730098.png)

可以看出如果**在笛卡尔坐标系的点共线，那么这些点在霍夫空间中对应的直线交于一点**。

如果不止存在一条直线时，如下所示：

![image-20191007160932077](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730099.png)

我们选择尽可能多的直线汇成的点，上图中三条直线汇成的A、B两点，将其对应回笛卡尔坐标系中的直线：

![image-20191007161219204](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730100.png)

到这里我们似乎已经完成了霍夫变换的求解。但如果像下图这种情况时：

![image-20191007161417485](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730101.png)

上图中的直线是𝑥=2，那(𝑘,𝑞)怎么确定呢？

为了解决这个问题，我们考虑将笛卡尔坐标系转换为极坐标。

![image-20191007165431682](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730102.png)

在极坐标下是一样的，极坐标中的点对应于霍夫空间的线，这时的霍夫空间是不在是参数(𝑘,𝑞)的空间，而是(𝜌,𝜃)的空间，𝜌是原点到直线的垂直距离，𝜃表示直线的垂线与横轴顺时针方向的夹角，垂直线的角度为0度，水平线的角度是180度。

![image-20191007163203594](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730103.png)

我们只要求得霍夫空间中的交点的位置，即可得到原坐标系下的直线。

**实现流程**

假设有一个大小为100∗100的图片，使用霍夫变换检测图片中的直线，则步骤如下所示：

- 直线都可以使用(𝜌,𝜃)表示，首先创建一个2D数组，我们叫做**累加器**，初始化所有值为0，行表示𝜌 ，列表示𝜃 。

  ![image-20191007170330026](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730104.png)

  该数组的大小决定了结果的准确性，若希望角度的精度为1度，那就需要180列。对于𝜌，最大值为图片对角线的距离，如果希望精度达到像素级别，行数应该与图像的对角线的距离相等。

- 取直线上的第一个点(𝑥,𝑦)，将其带入直线在极坐标中的公式中，然后遍历𝜃的取值：0，1，2，...，180，分别求出对应的𝜌值，如果这个数值在上述累加器中存在相应的位置，则在该位置上加1.

- 取直线上的第二个点，重复上述步骤，更新累加器中的值。对图像中的直线上的每个点都直线以上步骤，每次更新累加器中的值。

- 搜索累加器中的最大值，并找到其对应的(𝜌,𝜃)，就可将图像中的直线表示出来。

  ![image70-0440438](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730105.gif)

2.2 **<u>霍夫线检测</u>**

在OpenCV中做霍夫线检测是使用的API是：

```
cv.HoughLines(img, rho, theta, threshold)
```

参数：

- img: 检测的图像，要求是二值化的图像，所以在调用霍夫变换之前首先要进行二值化，或者进行Canny边缘检测

- rho、theta: 𝜌和𝜃的精确度

- threshold: 阈值，只有累加器中的值高于该阈值时才被认为是直线。

  霍夫线检测的整个流程如下图所示，这是在stackflow上一个关于霍夫线变换的解释：

  ![image-20191007184838549](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730106.png)

  **示例：**

  检测下述图像中的直线：

  ![rili](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730107.jpg)

```python
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
# 1.加载图片，转为二值图
img = cv.imread('./image/rili.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)

# 2.霍夫直线变换
lines = cv.HoughLines(edges, 0.8, np.pi / 180, 150)
# 3.将检测的线绘制在图像上（注意是极坐标噢）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))
# 4. 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(img[:,:,::-1]),plt.title('霍夫变换线检测')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20191007184301611](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730108.png)

**<u>霍夫圆检测[了解]</u>**

1. 原理

   圆的表示式是：
   $$
   (x-a)^2+(y-b)^2=r
   $$
   其中*a*和*b*表示圆心坐标，*r*表示圆半径，因此标准的霍夫圆检测就是在这三个参数组成的三维空间累加器上进行圆形检测，此时效率就会很低，所以OpenCV中使用**霍夫梯度法**进行圆形的检测。

   霍夫梯度法将霍夫圆检测范围两个阶段，第一阶段检测圆心，第二阶段利用圆心推导出圆半径。

   - 圆心检测的原理：圆心是圆周法线的交汇处，设置一个阈值，在某点的相交的直线的条数大于这个阈值就认为该交汇点为圆心。
   - 圆半径确定原理：圆心到圆周上的距离（半径）是相同的，确定一个阈值，只要相同距离的数量大于该阈值，就认为该距离是该圆心的半径。

   原则上霍夫变换可以检测任何形状，但复杂的形状需要的参数就多，霍夫空间的维数就多，因此在程序实现上所需的内存空间以及运行效率上都不利于把标准霍夫变换应用于实际复杂图形的检测中。霍夫梯度法是霍夫变换的改进，它的目的是减小霍夫空间的维度，提高效率。

2. API

   在OpenCV中检测图像中的圆环使用的是API是：

   ```python
   circles = cv.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0,maxRadius=0 )
   ```

   参数：

   - image：输入图像，应输入灰度图像
   - method：使用霍夫变换圆检测的算法，它的参数是CV_HOUGH_GRADIENT
   - dp：霍夫空间的分辨率，dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推
   - minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
   - param1：边缘检测时使用Canny算子的高阈值，低阈值是高阈值的一半。
   - param2：检测圆心和确定半径时所共有的阈值
   - minRadius和maxRadius为所检测到的圆半径的最小值和最大值

   返回：

   - circles：输出圆向量，包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径

3. 实现

   由于霍夫圆检测对噪声比较敏感，所以首先对图像进行中值滤波。

   ```python
   import cv2 as cv
   import numpy as np
   import matplotlib.pyplot as plt
   # 1 读取图像，并转换为灰度图
   planets = cv.imread("./image/star.jpeg")
   gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
   # 2 进行中值模糊，去噪点
   img = cv.medianBlur(gay_img, 7)  
   # 3 霍夫圆检测
   circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=0, maxRadius=100)
   # 4 将检测结果绘制在图像上
   for i in circles[0, :]:  # 遍历矩阵每一行的数据
       # 绘制圆形
       cv.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
       # 绘制圆心
       cv.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
   # 5 图像显示
   plt.figure(figsize=(10,8),dpi=100)
   plt.imshow(planets[:,:,::-1]),plt.title('霍夫变换圆检测')
   plt.xticks([]), plt.yticks([])
   plt.show()
   ```

   ![image-20191008105125382](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409181730109.png)

------

**总结：**

1. 模板匹配

   原理：在给定的图片中查找和模板最相似的区域

   API：利用cv.matchTemplate()进行模板匹配，然后

   使用cv.minMaxLoc()搜索最匹配的位置。

2. 霍夫线检测

   原理：将要检测的内容转换到霍夫空间中，利用累加器统计最优解，将检测结果表示处理

   API：cv2.HoughLines()

   注意：该方法输入是的二值化图像，在进行检测前要将图像进行二值化处理

3. 霍夫圆检测

   方法：霍夫梯度法

   API：cv.HoughCircles()