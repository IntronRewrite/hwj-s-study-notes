# hwj-s-study-notes
# 个人笔记

#### 2024/9/14

学习Linux     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Linux学习笔记/index.html)

#### 2024/9/15

学习Linux     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Linux学习笔记/index.html)

#### 2024/9/16

学习图像处理与OpenCV     
[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Opencv学习笔记/index.html)

#### 2024/9/17

中秋，不上班

#### 2024/9/18

学习图像处理与OpenCV    

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Opencv学习笔记/index.html)

#### 2024/9/19

学习图像处理与OpenCV    

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Opencv学习笔记/index.html)

#### 2024/9/19

OpenCV完结，图像处理还未学透彻。   

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Opencv学习笔记/index.html)

#### 2024/9/20-2024/9/27

学习YOLO，看代码

#### 2024/9/28

看R-CNN,Fast-RCNN,Faster-RCNN

#### 2024/9/29-2024/10/12

看R-CNN,Fast-RCNN,Faster-RCNN,ResNet

#### 2024/10/13-2024/10/15

学习python算法

学习python教程二，三，四章

[笔记](https://intronrewrite.github.io/hwj-s-study-notes/Python学习笔记/index.html)

[复习地址 - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

#### 2024/10/16

[2.1. 数据操作 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/ndarray.html)

![image-20241016215412346](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202410162155164.png)

复习python所学

[Python入门基础教程(附Python题库) - C语言网 (dotcpp.com)](https://www.dotcpp.com/course/python/)

![image-20241016222602935](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202410162226980.png)

### 2024/10/17

[2.3. 线性代数 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html)

![image-20241018081419016](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202410201842852.png)

### 2024/10/18

python学习（1）

![image-20241020094326982](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202410201842854.png)

[2.3. 线性代数 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html)

### 2024/10/19

pythoch学习(2)

![image-20241020095335082](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202410201842855.png)

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

### 实现FashionMNIST分类



![image-20241022032113830](F:\Documents\GitHub\hwj-s-study-notes\assets\202410220332744.png)

![image-20241022032440997](F:\Documents\GitHub\hwj-s-study-notes\assets\202410220332745.png)

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

[组会 markdown](https://intronrewrite.github.io/hwj-s-study-notes/tree/main/组会记/2024-10-22组会.html)

