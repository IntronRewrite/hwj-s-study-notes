# R-CNN

**地位**

RCNN 目标检测领域首次使用深度学习和卷积神经网络取得显著进展。

RCNN之后，目标检测由深度学习一统天下。

主要分为两个派系。

两阶段目标检测：先提取候选框，在逐一甄别分类

单阶段目标检测：不需要提取候选框，输入图像，直接输出目标检测结果（代表作YOLO，SSD，RetinaNet）



图像分类——输入图像，识别类别

目标定位（对象定位）

目标检测（对象检测）——识别类别，找到框

语义分割——找到每个像素的类别

实例分割——找到每个像素的类别的基础上，区分每一个物体

关键点检测——识别多个关键点的坐标



![image-20240929155617597](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929155617597.png)

不管是单阶段还是两阶段，都是基于深度学习。都随着深度学习图像分类主干网络的性能提升一起进步

![image-20240929160148755](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929160148755.png)

### 产生候选框-Selective Search

![image-20240929162223926](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929162223926.png)

![image-20240929162248870](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929162248870.png)

### 将侯选框缩放至227x227固定大小

作者选择连带邻近像素的非等比例缩放（王周围扩展16px,多余部分填充为整张图像平均值，归一化后减为0）

![image-20240929162435126](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929162435126.png)

为什么不直接用softmax分类而要用线性SVM分类 ?

![image-20240929163635410](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20240929163635410.png)