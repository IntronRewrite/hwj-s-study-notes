# hwj-s-study-notes
# 个人笔记

# 2024

## 9月

#### 2024/9/14

学习Linux     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Linux学习笔记/index.html)

#### 2024/9/15

学习Linux     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Linux学习笔记/index.html)

#### 2024/9/16

学习图像处理与OpenCV     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/OpenCV学习笔记/index.html)

#### 2024/9/17

中秋，不上班

#### 2024/9/18

学习图像处理与OpenCV    

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/OpenCV学习笔记/index.html)

#### 2024/9/19

学习图像处理与OpenCV    

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/OpenCV学习笔记/index.html)

#### 2024/9/19

OpenCV完结，图像处理还未学透彻。   

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/OpenCV学习笔记/index.html)

#### 2024/9/20-2024/9/27

学习YOLO，看代码

#### 2024/9/28

看R-CNN,Fast-RCNN,Faster-RCNN

## 10月

#### 2024/9/29-2024/10/12

看R-CNN,Fast-RCNN,Faster-RCNN,ResNet

#### 2024/10/13-2024/10/15

学习python算法

学习python教程二，三，四章

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Python学习笔记/index.html)

[复习地址 - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

#### 2024/10/16

[2.1. 数据操作 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/ndarray.html)

![image-20241016215412346](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013878.png)

复习python所学

[Python入门基础教程(附Python题库) - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

![image-20241016222602935](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013880.png)

### 2024/10/17

[2.3. 线性代数 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html)

![image-20241018081419016](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013881.png)

### 2024/10/18

python学习（1）

![image-20241020094326982](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013882.png)

[2.3. 线性代数 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html)

### 2024/10/19

pythoch学习(2)

![image-20241020095335082](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013883.png)

[深入浅出PyTorch — 深入浅出PyTorch](https://datawhalechina.github.io/thorough-pytorch/index.html#)

安装win11，修复网络连接

### 2024/10/20

pythoch学习(3)

[深入浅出PyTorch — 深入浅出PyTorch](https://datawhalechina.github.io/thorough-pytorch/index.html#)

数据读入
模型构建
模型初始化
损失函数
训练和评估

使用模块搭建复杂网络

tensorboard可视化，可视化网络参数，可视化训练过程

动态调整学习率

模型微调，使用预训练模型

半精度训练

argparse进行调参

#### 实现FashionMNIST分类



![image-20241022032113830](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013885.png)

![image-20241022032440997](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013886.png)

#### FashionMNIST数据集

|              lr              | epochs | Training Loss | Validation Loss | Accuracy  |
| :--------------------------: | :----: | :-----------: | :-------------: | :-------: |
|             0.1              |   20   |   2.302978    |    2.303285     | 0.100160  |
|             0.05             |   20   |   0.817705    |    0.816947     | 0.672877  |
|             0.01             |   20   |   0.349256    |    0.339863     | 0.870192  |
|             0.01             |  401   |   0.312429    |    0.316438     | 0.884415  |
|    0.01(normal_,bia=0.3)     |   20   |   2.361437    |    2.302825     | 0.100060  |
| 0.01(xavier_uniform_,bia=0)  |   20   |   0.318755    |    0.320410     | 0.881811  |
|      0.1（lr*0.1/10轮）      |   40   |   2.298950    |    2.299133     | 0.099860  |
|      0.01(lr*0.1/10轮)       |   40   |   0.258692    |    0.264882     | 0.900841  |
|       0.01(lr*0.8/轮)        |   40   |   0.218625    |    0.246748     | 0.910156  |
|       0.01(lr*0.85/轮)       |   80   |   0.206507    |    0.238813     | 0.913762  |
| 0.01(kaiming_uniform_,bia=0) |   20   |   0.372171    |    0.336938     | 0.871494  |
| 0.01(kaiming_uniform_,bia=0) |   40   |   0.351744    |    0.324420     | 0.877704  |
| 0.01(kaiming_normal_,bia=0)  |   40   |   0.323300    |    0.323402     | 0.885517  |
|   0.01(10轮*0.1，耐心*0.1)   |  100   |   0.278295    |    0.271341     | 0.899439  |
|   0.01(10轮*0.5，耐心*0.5)   |  100   |   0.244082    |    0.253909     | 0.905749  |
|  0.01(0.95每轮，耐心*0.5))   |  100   |   0.171156    |    0.248037     | :0.913161 |
|  0.01(0.95每轮，耐心*0.5))   |  100   |   0.194997    |    0.250352     | 0.911659  |
|   0.01(0.95每轮，耐心*0.5)   |  100   |   0.170374    |    0.255083     | 0.914163  |
|            0.005             |   20   |   0.252192    |    0.276143     | 0.902344  |
|            0.001             |   20   |   0.175485    |    0.230183     | 0.919271  |
|            0.0001            |   20   |   0.334874    |    0.322388     | 0.884115  |
|           0.00001            |   20   |   0.584782    |    0.584782     | 0.777344  |

### 2024/10/21

整理所学，做markdown文档准备开组会

[组会 markdown](https://intronrewrite.github.io/hwj-s-study-notes/组会/2024-10-22组会.html)

### 2024/10/22

学习python（2）

![image-20241028094612375](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013887.png)

[Python入门基础教程(附Python题库) - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

### 2024/10/23

学习python（3）

![image-20241028094559698](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013888.png)

[Python入门基础教程(附Python题库) - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

### 2024/10/24

学习python（4）

![image-20241028094544818](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013889.png)

### 2024/10/25

SSD论文阅读(1)

### 2024/10/26

SSD论文阅读(2)

### 2024/10/27

SSD论文阅读（3）

### 2024/10/28

[41 物体检测和数据集【动手学深度学习v2】](https://www.bilibili.com/video/BV1Lh411Y7LX/)

[42 锚框【动手学深度学习v2】](https://www.bilibili.com/video/BV1aB4y1K7za/)

[44 物体检测算法：R-CNN，SSD，YOLO【动手学深度学习v2】](https://www.bilibili.com/video/BV1Db4y1C71g/)

### 2024/10/29

[13.1. 图像增广 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/image-augmentation.html)

[13.2. 微调 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/fine-tuning.html)

[13.3. 目标检测和边界框 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/bounding-box.html)

[13.5. 多尺度目标检测 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/multiscale-object-detection.html)

SSD算法实现(1)

[13.7. 单发多框检测（SSD） — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/ssd.html)

### 2024/10/30

SSD算法实现(2)

[13.7. 单发多框检测（SSD） — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/ssd.html)

### 2024/10/31

敲遗传算法解决仓库分配问题

## 11月

### 2024/11/01

敲遗传算法解决仓库分配问题

![best_fitness](F:\Documents\GitHub\hwj-s-study-notes\assets\202411011951779.svg+xml)

![c4db41e9e5ee61bc1b7b85e5043e84e](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013890.png)

![b69f1235feb9389338af550c8e8f133](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013891.png)

SSD算法实现(3)

[13.7. 单发多框检测（SSD） — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_computer-vision/ssd.html)

### 2024/11/02

![image-20241103110939523](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411092013892.png)

### 2024/11/03

注意力机制

### 2024/11/04-2024/11/05

组会

### 2024/11/06

看点云匹配综述

### 2021/11/07

- [x] 整理2024年CVPR会议论文
- [x] 整理2023年CVPR会议论文
- [x] 整理2022年CVPR会议论文
- [x] 整理2021年CVPR会议论文

### 2024/11/08

**整理2023年CVPR会议论文，提取重点**

[论文材料整理](https://intronrewrite.github.io/hwj-s-study-notes/Point-cloud-registration/点云配准相关论文(cpvr2023)/cpvr2023.html)

### 2024/11/09

学习CMake

[参考资料](https://intronrewrite.github.io/hwj-s-study-notes/slam/CMake/CMake.html)

[视频](https://www.bilibili.com/video/BV1fy4y1b7TC)

五星好评！！！

2024/11/10

2024/11/11

看PointNet++

[论文阅读笔记 PointNet++ : Deep Hierarchical Feature Learning on Point Sets in a Metric Space - 知乎](https://zhuanlan.zhihu.com/p/399432095)

### 2024/11/12

看PointNet++

### 2024/11/13

- [x] 看PointNet代码

- [x] 看PointNet++代码

### 2024/11/14

看KPConv

### 2024/11/15

看roreg,看不懂，草

### 2024/11/16

看slam，rerog有刚性变换，李群相关知识

### 2024/11/17

玩去了

### 2024/11/18

- [x] 看slam刚性变换
- [x] 看slam李群与李代数

### 2024/11/19

在看几遍slam李群与李代数

### 2024/11/20

整理4PCS方法

还在看slam李群与李代数

看视觉SLAM十四讲 5相机与图像

.....有必要跟着写一遍
