# ORB-SLAM2源码解析

## ORB-SLAM2简介

### ORB-SLAM2特点

2015年，西班牙萨拉戈萨大学机器人感知与实时研究组开源了ORB-SLAM第一个版本，由于其出色的效

果受到广泛关注。该团队分别在2016年和2020年开源了第二个版本ORB-SLAM2和第三个版本ORB

SLAM3。

ORB-SLAM2有如下特点：

1. 首个（发布时）支持单目，双目和RGB-D相机的完整的开源SLAM方案，具有回环检测和重新定位的功能。

2. 能够在CPU上进行实时工作，可以用于移动终端如 移动机器人、手机、无人机、汽车。

   特征点法的巅峰之作，定位精度极高，可达厘米级。

3. 能够实时计算出相机的位姿，并生成场景的稀疏三维重建地图。

4. 代码非常整洁，包含很多实际应用中的技巧，非常实用。

5. 支持仅定位模式，该模式适用于轻量级以及在地图已知情况下长期运行，此时不使用局部建图和回环检测的线程

6. 双目和RGB-D相对单目相机的主要优势在于，可以直接获得深度信息，不需要像单目情况中那样做一个特定的SFM初始化。

### 算法流程框架

**主体框架**

- 输入。有三种模式可以选择：单目模式、双目模式和RGB-D模式。
- 跟踪。初始化成功后首先会选择参考关键帧跟踪，然后大部分时间都是恒速模型跟踪，当跟踪丢失

的时候启动重定位跟踪，在经过以上跟踪后可以估计初步的位姿，然后经过局部地图跟踪对位姿进

行进一步优化。同时会根据条件判断是否需要将当前帧新建为关键帧。

- 局部建图。输入的关键帧来自跟踪里新建的关键帧。为了增加局部地图点数目，局部地图里关键帧

之间会重新进行特征匹配，生成新的地图点，局部BA会同时优化共视图里的关键帧位姿和地图点，

优化后也会删除不准确的地图点和冗余的关键帧。

- 闭环。通过词袋来查询数据集检测是否闭环，计算当前关键帧和闭环候选关键帧之间的Sim3位姿，

仅在单目时考虑尺度，双目或RGB-D模式下尺度固定为1。然后执行闭环融合和本质图优化，使得

所有关键帧位姿更准确。

- 全局BA。优化所有的关键帧及其地图点。

- 位置识别。需要导入离线训练好的字典，这个字典是由视觉词袋模型构建的。新输入的图像帧需要

先在线转化为词袋向量，主要应用于特征匹配、重定位、闭环。

- 地图。地图主要由地图点和关键帧组成。关键帧之间根据共视地图点数目组成了共视图，根据父子

关系组成了生成树。

![image-20241127162621420](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815727.png)



**数据输入的预处理**

为了兼容不同相机（双目相机与RGBD相机）,需要对输入数据进行预处理，使得交给后期处理的数

据格式一致，具体流程如下：

![image-20241127162658916](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815728.png)

### 详细安装教程

**安装第三方库**

在刚装好的 `Ubuntu 16.04` 系统上安装 `ORB-SLAM2`

`Pangolin` 、 `OpenCV` 、 `Eigen` 、 `g2o 与 DBoW2` （ `ORB-SLAM2` 自带）

**安装前的准备：** 安装 `vim` 、 `cmake` 、 `git` 、 `gcc` 、 `g++`

```shell
sudo apt-get install vim cmake
sudo apt-get install git
sudo apt-get install gcc g++
```



**安装Pangolin(建议源码安装)**

1) 安装依赖项

```shell
sudo apt-get install libglew-dev
sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
sudo apt-get install libpython2.7-dev
```

2) 安装 Pangolin

```shell
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake -DCPP11_NO_BOOSR=1 ..
make -j
```



**安装OpenCV3.4** (建议源码安装 安装时间较长 耐心等待 ) 

1) 安装依赖项

```shell
sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev
libjpeg.dev
sudo apt-get install libtiff4.dev libswscale-dev libjasper-dev
```



2) 安装 OpenCV3.4

进入下载的安装压缩包，解压到某文件夹，然后进去该文件夹建立build文件夹 编译文件夹

```shell
cd opencv-3.4.5
mkdir build
cd build
cmake ..
make
sudo make install 
```



3) 配置环境变量

```shell
sudo vim /etc/ld.so.conf.d/opencv.conf
```

在打开的空白文件中添加 `/usr/local/lib`

执行 `sudo ldconfig` ,使配置的环境变量生效

4) 配置` .bashrc `，末尾添加下面两行

```shell
//打开.bashrc
sudo vim /etc/bash.bashrc
//添加以下两行内容到.bashrc
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```



5) `source` 与 `update`

```shell
source /etc/bash.bashrc
sudo updatedb
```



6) 测试是否正常安装 (成功会出现带 `“hello opcv”` 字样的窗口)

```shell
cd opencv-3.4.5/samples/cpp/example_cmake
cmake .
make
./opencv_example
```



安装Eigen3.3.7 (建议源码安装) 

解压缩安装包

安装

```shell
cd eigen-git-mirror
mkdir build
cd build
cmake ..
sudo make install
\#安装后 头文件安装在/usr/local/include/eigen3/
\#移动头文件
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include
```



备注：在很多程序中 `include` 时经常使用 `#include <Eigen/Dense>` 而不是使用 `#include`

`<eigen3/Eigen/Dense>` 所以要做下处理

**安装运行ORB-SLAM2及常见问题解决办法**

安装 运行ORB_SLAM2(如果在ROS下 推荐工程目录: `orbslam_ws/src` )

```shell
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```





编译时如果有如下错误：**在对应的头文件中加上** `#include <unistd.h>`

![image-20241127163638143](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815729.png)





如果需要在 `ROS` 环境下运行 `ORB_SLAM2` :

```shell
chmod +x build_ros.sh
export
ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/orbslam_ws/src/ORB_SLAM2/Examples/ROS
./build_ros.sh
```



编译时如果提示 `boost`库 相关错误： 修改`Examples/ROS/ORB_SLAM2/`文件夹下的`CMakeLists.txt`文件

![image-20241127163802244](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815730.png)

单目运行方法

```shell
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml
Data/rgbd_dataset_freburg1_desk
```



### TUM 数据集介绍及使用

**RGBD模式下在TUM数据集上的表现**

![image-20241127163915783](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815731.png)





**TUM RGB-D 数据集 简介**

​	注意两点：深度的表达方式 以及 数据的存储格式

![image-20241127164032596](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815732.png)

**RGBD模式下运行时数据预处理**

关于associate.py

注意：只能在Python2 环境下运行

**Python下运行**

```shell
python associate.py rgb.txt depth.txt > associate.txt
python associate.py associate.txt groundtruth.txt > associate_with_groundtruth.txt
```

注意：

直接association后出问题，生成的结果

associate.txt 1641行

associate_with_groundtruth.txt 1637行

也就是说，associate的不一定有groundtruth，所以要以associate_with_groundtruth.txt的关联结果为

准

### 可视化运行效果解析

运行的中间结果截图如下所示，对重点信息进行了标注

![image-20241127164220361](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815733.png)

不同颜色地图点的含义解析

- **红色**点表示参考地图点，其实就是tracking里的local mappoints

```C++
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}
```



- **黑色**点表示所有地图点，红色点属于黑色点的一部分

```C++
void Tracking::UpdateLocalMap()

{

    // This is for visualization

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update

    UpdateLocalKeyFrames();

    UpdateLocalPoints();

}

void MapDrawer::DrawMapPoints()

{

    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();

    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())

    return;

    glPointSize(mPointSize);

    glBegin(GL_POINTS);

    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)

    {

        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))

        continue;

        cv::Mat pos = vpMPs[i]->GetWorldPos();

        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));**1.6** **变量命名规范**

    }

    glEnd();

    glPointSize(mPointSize);

    glBegin(GL_POINTS);

    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end();

    sit!=send; sit++)

    {

        if((*sit)->isBad())

        continue;

        cv::Mat pos = (*sit)->GetWorldPos();

        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();

}
```



![image-20241127164504327](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815734.png)







后面会有大量的源码详解，在介绍之前，我们有必要先了解一下在ORB-SLAM2中变量的命名规则，这对

我们学习代码非常有用。

以小写字母m（member的首字母）开头的变量表示类的成员变量。比如：

```C++
int mSensor;
int mTrackingState;
std::mutex mMutexMode;
```

对于某些复杂的数据类型，第2个甚至第3个字母也有一定的意义，比如：

mp开头的变量表示指针（pointer）型类成员变量；

```C++
Tracking* mpTracker;
LocalMapping* mpLocalMapper;
LoopClosing* mpLoopCloser;
Viewer* mpViewer;
```

mb开头的变量表示布尔（bool）型类成员变量；

```C++
bool mbOnlyTracking;
```

mv开头的变量表示向量（vector）型类成员变量；

```C++
std::vector<int> mvIniLastMatches;
std::vector<cv::Point3f> mvIniP3D;
```

mpt开头的变量表示指针（pointer）型类成员变量，并且它是一个线程（thread）；

```C++
std::thread* mptLocalMapping;
std::thread* mptLoopClosing;
std::thread* mptViewer;
```

ml开头的变量表示列表（list）型类成员变量；

mlp开头的变量表示列表（list）型类成员变量，并且它的元素类型是指针（pointer）；

mlb开头的变量表示列表（list）型类成员变量，并且它的元素类型是布尔型（bool）；

```C++
list<double> mlFrameTimes;
list<bool> mlbLost;
list<cv::Mat> mlRelativeFramePoses;
list<KeyFrame*> mlpReferences;
```



## ORB特征点提取

###  关键点和描述子

FAST关键点

![image-20241127165351080](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815735.png)

- 选取像素p，假设它的亮度为Ip；
- 设置一个阈值T（比如Ip的20%）；
- 以像素p为中心，选取半径为3的圆上的16个像素点；
- 假如选取的圆上，有连续的N个点的亮度大于Ip+T或小于Ip-T，那么像素p可以被认为是特征点；
- 循环以上4步，对每一个像素执行相同操作。

**FAST 描述子**

论文：BRIEF: Binary Robust Independent Elementary Features

BRIEF算法的核心思想是在关键点P的周围以一定模式选取N个点对，把这N个点对的比较结果组合起来

作为描述子。为了保持踩点固定，工程上采用特殊设计的固定的pattern来做

![image-20241127165420012](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815736.png)

###  灰度质心法

原始的FAST关键点没有方向信息，这样当图像发生旋转后，brief描述子也会发生变化，使得特征点对旋转不鲁棒

**解决方法：orientated FAST**

使用灰度质心法计算特征点的方向，

**什么是灰度质心法？**

下面重点说一下如何计算灰度质心。

- 第1步：我们定义该区域图像的矩为：

$$
m_{pq}=\sum_{x,y}x^py^qI(x,y),\quad p,q=\{0,1\}
$$

​	式中，$p,q$取0或者1；$I(x$,$y)$表示在像素坐标$(x,y)$处图像的灰度值；$m_pq$表示图像的矩。
​	在半径为$R$的圆形图像区域，沿两个坐标轴$x,y$方向的图像矩分别为：
$$
\begin{aligned}m_{10}&=\sum_{x=-R}^R\sum_{y=-R}^RxI(x,y)\\m_{01}&=\sum_{x=-R}^R\sum_{y=-R}^RyI(x,y)\end{aligned}
$$
​	圆形区域内所有像素的灰度值总和为：
$$
m_{00}=\sum_{x=-R}^R\sum_{y=-R}^RI(x,y)
$$

- 第2步：图像的质心为：

$$
C=(c_x,c_y)=\left(\frac{m_{10}}{m_{00}},\frac{m_{01}}{m_{00}}\right)
$$



- 第3步：然后关键点的“主方向”就可以表示为从圆形图像形心 指向质心 的方向向量 ,于是关键点的旋转角度记为

$$
\theta=\arctan2\left(c_y,c_x\right)=\arctan2\left(m_{01},m_{10}\right)
$$

以上是灰度质心法求关键点旋转角度的原理。





**在一个圆内计算灰度质心**

下图P为几何中心，Q为灰度质心

![image-20241127170704808](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815738.png)

**思考：为什么是圆？不是正方形？**

ORBSLAM里面是先旋转坐标再从图像中采点提取，并不是先取那块图像再旋转，见

computeOrbDescriptor函数里的这个表达式

`\#define GET_VALUE(idx) \ center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \ // y'* step` 

`cvRound(pattern[idx].x*a - pattern[idx].y*b)]`

会导致下方采集点的时候绿色和黄色部分就是不同的像素

![image-20241127170744665](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815739.png)

下面求圆内的坐标范围

```C++
vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
// 对应从D到B的红色弧线，umax坐标从D到C
for (v = 0; v <= vmax; ++v)
	umax[v] = cvRound(sqrt(hp2 - v * v));

while (umax[v0] == umax[v0 + 1]
// 对应从B到E的蓝色弧线，umax坐标从C到A
for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
{
    while (umax[v0] == umax[v0 + 1])
        ++v0;
    umax[v] = v0;
    	++v0;
}
```





![image-20241127171049673](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815740.png)

参考：

https://blog.csdn.net/liu502617169/article/details/89423494

https://www.cnblogs.com/wall-e2/p/8057448.html

###  特征点角度计算

![image-20241127171111286](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815741.png)

**steer brief**

点v 绕 原点旋转θ 角，得到点v’，假设 v点的坐标是(x, y) ，那么可以推导得到 v’点的坐标（x’, y’)

![image-20241127171129314](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815742.png)

那么
$$
x=rcos(\varphi)\\y=rsin(\varphi)
$$

$$
x'=rcos(\theta+\varphi)=rcos(\theta)cos(\varphi)-rsin(\theta)sin(\varphi)=xcos(\theta)-ysin(\theta)\\y'=rsin(\theta+\varphi)=rsin(\theta)cos(\varphi)+rcos(\theta)sin(\varphi)=xsin(\theta)+ycos(\theta)
$$

得到
$$
\begin{bmatrix}x^{\prime}\\y^{\prime}\end{bmatrix}=\begin{bmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{bmatrix}*\begin{bmatrix}x\\y\end{bmatrix}
$$
参考：

https://www.cnblogs.com/zhoug2020/p/7842808.html



**IC_Angle** **计算技巧**

在一个圆域中算出m10 (x坐标)和m01 (y坐标),计算步骤是先算出中间红线的m10，然后在平行于
x轴算出m10和m01，一次计算相当于图像中的同个颜色的两个line。

![image-20241127171523837](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815743.png)

![image-20241127171529714](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815744.png)



为什么要重载小括号运算符 operator() ？

可以用于仿函数（一个可以实现函数功能的对象）

仿函数（functor）又称为函数对象（function object）是一个能行使函数功能的类。仿函数的语法几乎

和我们普通的函数调用一样，不过作为仿函数的类，都必须重载operator()运算符

1. 仿函数可有拥有自己的数据成员和成员变量，这意味着这意味着仿函数拥有状态。这在一般函数中是不可能的。

2. 仿函数通常比一般函数有更好的速度。

扩展阅读

https://blog.csdn.net/jinzhu1911/article/details/101317367

###  金字塔的计算

图像金字塔对应函数为：ORBextractor::ComputePyramid

![image-20241127171616969](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815745.png)

![image-20241127171640203](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815746.png)

![image-20241127171654472](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815747.png)

![image-20241127171707428](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815748.png)



### 特征点数量的分配计算

图像金字塔层数越高，对应层数的图像分辨率越低，面积(高×宽)越小，所能提取到的特征点数量就越少。所以分配策略就是根据图像的面积来定，将总特征点数目根据面积比例均摊到每层图像上。我们假设需要提取的特征点数目为$N$,金字塔总共有$m$层，第 0 层图像的宽为$W$,高为$H$,对应的面
积$H\cdot W=C$,图像金字塔缩放因子为$s$, $0<s<1$,在 ORB-SLAM2 中，$m=8,s=\frac{1}{1.2}$ 。
那么整个金字塔总的图像面积是：
$$
\begin{aligned}\text{S}&=H\cdot W\cdot(s^2)^0+H\cdot W\cdot(s^2)^1+\cdots+H\cdot W\cdot(s^2)^{(m-1)}\\&=HW\frac{1-(s^2)^m}{1-(s^2)}=C\frac{1-(s^2)^m}{1-(s^2)}\end{aligned}
$$


单位面积应该分配的特征点数量为：
$$
N_{avg}=\frac{N}{S}=\frac{N}{C\frac{1-(s^2)^m}{1-s^2}}=\frac{N(1-s^2)}{C(1-(s^2)^m)}
$$
第 0 层应该分配的特征点数量为：
$$
N_0=\frac{N(1-s^2)}{1-(s^2)^m}
$$
第 i 层应该分配的特征点数量为：
$$
N_i=\frac{N(1-s^2)}{C(1-(s^2)^m)}C(s^2)^i=\frac{N(1-s^2)}{1-(s^2)^m}(s^2)^i
$$
$\begin{aligned}&\text{在ORB-SLAM2 的代码里,不是按照面积均摊的,而是按照面积的开方来均摊特征点的,也就是将上述}\text{公式中的 }s^{2}\text{ 换成 }s\text{ 即可。}\end{aligned}$
$$
\frac{N(1-s)}{1-(s)^m}(s)^i
$$
参考：https://zhuanlan.zhihu.com/p/61738607

### 使用四叉树均匀分布特征点

 ORB特征提取策略对ORB-SLAM2性能的影响：ORB-SLAM2中的ORB特征提取方法相对于OpenCV中的

方法，提高了ORB-SLAM2的轨迹精度和鲁棒性。增加特征提取的均匀性可以提高系统精度，但是似乎会

降低特征提取的重复性。

参见：https://zhuanlan.zhihu.com/p/57235987

对应函数 DistributeOctTree

- 如果图片的宽度比较宽，就先把分成左右w/h份。一般的640×480的图像开始的时候只有一个node。
- 如果node里面的点数>1，把每个node分成四个node，如果node里面的特征点为空，就不要了，删掉。
- 新分的node的点数>1，就再分裂成4个node。如此，一直分裂。
- 终止条件为：node的总数量> [公式] ，或者无法再进行分裂。
- 然后从每个node里面选择一个质量最好的FAST点。

![image-20241127190706655](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815749.png)

参考：https://zhuanlan.zhihu.com/p/61738607

ExtractorNode::DivideNode

![image-20241127190744067](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815750.png)

节点分裂顺序：后加的先分裂

![image-20241127190817197](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815751.png)

###  高斯处理

**高斯模糊**

![image-20241127190829045](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815752.png)

**常用的高斯模板，中间权重最大**

![image-20241127190848943](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815753.png)

![image-20241127190854199](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815754.png)



**高斯公式**
$$
G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
$$


###  特征点去畸变去畸变效果

![image-20241127165304138](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815755.png)

![image-20241127190932150](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815756.png)

去畸变效果

![image-20241127190942769](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815757.png)



##  地图初始化

###  多视图几何基础

**对极约束示意图**

![image-20241127191121538](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815758.png)

**H矩阵求解原理**

特征点对 $p_1,p_2$,用单应矩阵$H_{21}$ 来描述特征点对之间的变换关系

$$
p_2=H_{21}*p_1
$$
我们写成矩阵形式：
$$
\begin{bmatrix}u_2\\v_2\\1\end{bmatrix}=\begin{bmatrix}h_1&h_2&h_3\\h_4&h_5&h_6\\h_7&h_8&h_9\end{bmatrix}\begin{bmatrix}u_1\\v_1\\1\end{bmatrix}
$$
为了化为齐次方程，左右两边同时叉乘$p_2$ ，得到
$$
p_2\times H_{21}*p_1=0
$$
写成矩阵形式：
$$
\begin{bmatrix}0&-1&v_2\\1&0&-u_2\\-v_2&u_2&0\end{bmatrix}\begin{bmatrix}h_1&h_2&h_3\\h_4&h_5&h_6\\h_7&h_8&h_9\end{bmatrix}\begin{bmatrix}u_1\\v_1\\1\end{bmatrix}=0
$$
展开计算得到
$$
v_2=(h_4*u_1+h_5*v_1+h_6)/(h_7*u_1+h_8*v_1+h_9)\\u_2=(h_1*u_1+h_2*v_1+h_3)/(h_7*u_1+h_8*v_1+h_9)
$$
写成齐次方程
$$
-(h_4*u_1+h_5*v_1+h_6)+(h_7*u_1*v_2+h_8*v_1*v_2+h_9*v_2)=0\\h_1*u_1+h_2*v_1+h_3-(h_7*u_1*u_2+h_8*v_1*u_2+h_9*u_2)=0
$$
转化为矩阵形式
$$
\begin{bmatrix}0&0&0&-u_1&-v_1&-1&u_1*v_2&v_1*v_2&v_2\\u_1&v_1&1&0&0&0&-u_1*u_2&-v_1*u_2&-u_2\end{bmatrix}\begin{bmatrix}h_1\\h_2\\h_3\\h_4\\h_5\\h_6\\h_7\\h_8\\h_9\end{bmatrix}=0
$$
等式左边两项分别用A, X表示，则有
$$
AX=0
$$
一对点提供两个约束等式，单应矩阵H总共有9个元素，8个自由度（尺度等价性），所以需要4对点提供8个约束方程就可以求解。

**哪个奇异向量是最优解？**

为什么 $V^T$的第9个奇异向量就是最优解？

$Ah=0$对应的代价函数
$$
f(h)=\frac{1}{2}(Ah)^T(Ah)=\frac{1}{2}h^TA^TAh
$$
最优解是导数为0
$$
\begin{aligned}&\frac{df}{dh}=0\\&A^TAh=0\end{aligned}
$$
问题就转换为求$A^TA$的最小特征值向量
$$
A^TA=((UDV^T)^T(UDV^T))=VD^TU^TUDV^T=VD^TDV^T
$$
可见$A^TA$的特征向量就是$V^T$的特征向量。因此求解得到V 之后取出最后一行奇异值向量作为f的最优值，然后整理成3维矩阵形式。(其实其他行的奇异值向量也是一个解，但是不是最优解)





**求解基础矩阵F**

**推导F矩阵约束方程**

特征点对$p_1,p_2$,用基础矩阵$F_{21}$ 来描述特征点对之间的变换关系

$$
p_2^T*F_{21}*p_1=0
$$
我们写成矩阵形式：
$$
[\begin{array}{ccc}u_2&v_2&1\end{array}]\left[\begin{array}{ccc}f_1&f_2&f_3\\f_4&f_5&f_6\\f_7&f_8&f_9\end{array}\right]\left[\begin{array}{c}u_1\\v_1\\1\end{array}\right]=0
$$
为方便计算先展开前两项得到
$$
a=f_1*u_2+f4*v_2+f_7;\\b=f_2*u_2+f_5*v_2+f_8;\\c=f_3*u_2+f_6*v_2+f_9;
$$
那么，上面的矩阵可以化为
$$
[\begin{matrix}a&b&c\end{matrix}]\left[\begin{matrix}u_1\\v_1\\1\end{matrix}\right]=0
$$
展开后：
$$
a*u_1+b*v_1+c=0
$$


带入前面a，b,c表达式子整理得到
$$
f_1*u_1*u_2+f_2*v_1*u_2+f_3*u_2+f_4*u_1*v_2+f_5*v_1*v_2+f_6*v_2+f_7*u_1+f_8*v_1+f_9=0
$$
转化为矩阵形式
$$
\begin{bmatrix}u_1*u_2&v_1*u_2&u_2&u_1*v_2&v_1*v_2&v_2&u_1&v_1&1\end{bmatrix}\begin{bmatrix}f_1\\f_2\\f_3\\f_4\\f_5\\f_6\\f_7\\f_8\\f_9\end{bmatrix}=0
$$
等式左边两项分别用A，f表示，则有

$$
Af=0
$$
一对点提供一个约束方程，基础矩阵F总共有9个元素，7个自由度(尺度等价性，秩为2),所以8对点提供8个约束方程就可以求解F。



**SVD**

SVD分解结果
$$
A=UDV^T
$$
假设我们使用8对点求解，A 是 8x9 矩阵，分解后

U 是左奇异向量，它是一个8x8的 正交矩阵，

V 是右奇异向量，是一个 9x9 的正交矩阵， $V^T$是V的转置

D是一个8 x 9 对角矩阵，除了对角线其他元素均为0，对角线元素称为奇异值，一般来说奇异值是按照从大到小的顺序降序排列。因为每个奇异值都是一个残差项，因此最后一个奇异值最小，其含义就是最优的残差。因此其对应的奇异值向量就是最优值，即最优解。

$V^T$中的每个列向量对应着D中的每个奇异值，最小二乘最优解就是$V^T$对应的第9个列向量，也就是基础矩阵F的元素。这里我们先记做 Fpre,因为这个还不是最终的F



**F矩阵秩为2**

基础矩阵 F 有个很重要的性质，就是秩为2，可以进一步约束求解准确的F

上面的方法使用$V^T$对应的第9个列向量构造的$F_{pre}$ 秩通常不为2，我们可以继续进行SVD分解。
$$
F_{pre}=UDV^T=U\begin{bmatrix}\sigma_1&0&0\\0&\sigma_2&0\\0&0&\sigma_3\end{bmatrix}V^T
$$
其最小奇异值人为置为0，这样F矩阵秩为2

$$
F=UDV^T=U\begin{bmatrix}\sigma_1&0&0\\0&\sigma_2&0\\0&0&0\end{bmatrix}V^T
$$
此时的E就早是终得到的其础矩阵。



**单目投影恢复3D点**

匹配特征点对$x,x^{\prime}$,$P,P^{\prime}$ 分别是投影矩阵，他们将同一个空间点X投影到图像上的点$x_1,x_2$

描述为

$$
\begin{array}{l}x=\lambda*P*X\\x'=\lambda*P'*X\end{array}
$$
两个表达式类似，我们以一个通用方程来描述

$$
\begin{bmatrix}x\\y\\1\end{bmatrix}=\lambda\begin{bmatrix}p_1&p_2&p_3&p_4\\p_5&p_6&p_7&p_8\\p_9&p_{10}&p_{11}&p_{12}\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
$$



为方便推导，简单记为
$$
\begin{bmatrix}x\\y\\1\end{bmatrix}=\lambda\begin{bmatrix}-&P_0&-\\-&P_1&-\\-&P_2&-\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
$$
为了化为齐次方程，左右两边同时叉乘，得到
$$
\begin{gathered}
\begin{bmatrix}x\\y\\1\end{bmatrix}_\times\begin{bmatrix}x\\y\\1\end{bmatrix}=\begin{bmatrix}x\\y\\1\end{bmatrix}_\times\lambda\begin{bmatrix}-&P_0&-\\-&P_1&-\\-&P_2&-\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix} \\
\begin{bmatrix}x\\y\\1\end{bmatrix}_\times\lambda\begin{bmatrix}-&P_0&-\\-&P_1&-\\-&P_2&-\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix} \\
\begin{bmatrix}0&-1&y\\1&0&-x\\-y&x&0\end{bmatrix}\begin{bmatrix}-&P_0&-\\-&P_1&-\\-&P_2&-\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix} \\
\left.\left[\begin{array}{c}yP_2-P_1\\P_0-xP_2\\xP_1-yP_0\end{array}\right.\right]\left[\begin{array}{c}X\\Y\\Z\end{array}\right]=\left[\begin{array}{c}0\\0\\0\end{array}\right] 
\end{gathered}
$$
两对匹配点
$$
\begin{bmatrix}yP_2-P_1\\P_0-xP_2\\y'P_2'-P_1'\\P_0'-x'P_2'\end{bmatrix}X=\begin{bmatrix}0\\0\\0\\0\end{bmatrix}
$$


等式左边两项分别用A，X表示，则有
$$
AX=0
$$
SVD求解，右奇异矩阵的最后一行就是最终的解。



### 卡方检验

**为什么要引用卡方检验?**

以特定概率分布为某种情况建模时，事物长期结果较为稳定，能够清晰进行把握。比如抛硬币实验。但是期望与事实存在差异怎么办？偏差是正常的小幅度波动？还是建模错误？此时，利用卡方分布**分析结果**，排除**可疑结果**。

简单来说：当事实与期望不符合情况下使用卡方分布进行检验，看是否系统出了问题，还是属于正常波

动

**卡方分布用途？**

检查实际结果与期望结果之间何时存在显著差异。

1. 检验拟合程度：也就是说可以检验一组给定数据与指定分布的吻合程度。如：用它检验抽奖机收益的

观察频数与我们所期望的吻合程度。 

2. 检验两个变量的独立性：通过这个方法检查变量之间是否存在某种关系。

   

**卡方分布假设检验步骤？**

1. 确定要进行检验的假设（H0）及其备择假设H1.

2. 求出期望E. 

3. 确定用于做决策的拒绝域（右尾）.

4. 根据**自由度**和**显著性水平**查询检验统计量临界值. 

5. 查看检验统计量是否在拒绝域内.

6. 做出决策.

**一个例子：抽奖机之谜**

抽奖机，肯定都不陌生，现在一些商场超市门口都有放置。正常情况下出奖概率是一定的，综合来看，

商家收益肯定大于支出。



倘若突然某段时间内总是出奖，甚是反常，那么到底是某阶段是小概率事件还是有人进行操作了？**抽奖机怎么了？**针对这种现象或者类似这种现象问题则可以借助卡方进行检验，暂且不着急如何检验，还是补充一下基础知识，再逐步深入解决问题。【常规事件中出现非常规现象，如何检查问题所在的情况下

使用卡方分布】

下面是某台抽奖机的期望分布，其中X代表每局游戏的净收益（每局独立事件）：

![image-20241127193700179](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815759.png)

实际观察中玩家收益的频数为：

![image-20241127193740630](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815760.png)

目的：在5%的显著性水平下，看看能否有足够证据证明判定抽奖机被人动了手脚。

![image-20241127193751574](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815761.png)

**要检验的原假设是什么？备择假设是什么？**

**什么是显著性水平？**

显著性水平是估计总体参数落在某一区间内，可能犯错误的概率，用α表示。

显著性水平是假设检验中的一个概念，是指当原假设为正确时人们却把它拒绝了的概率或风险。它是公认的小概率事件的概率值，必须在每一次统计检验之前确定，通常取**α=0.05或α=0.01**。这表明，当作出接受原假设的决定时，其**正确的可能性（概率）为95%或99%**。

卡方分布指出观察频数与期望频数之间差异显著性，和其他假设一样，这取决于显著性水平。

1. 显性水平α进行检验，则写作：（常用的显著性水平1%和5%）

2. 检测标准：卡方分布检验是单尾检验且是右尾，右尾被作为**拒绝域**。于是通过查看检验统计量是否位于右尾的拒绝域以内，来判定期望分布得出结果的可能性。

   ![image-20241127193928186](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815762.png)

3. 卡方概率表的使用：卡方临界值表是给定可以查询的

**问题简述**

抽奖机平常收益者总是商家，突然一段时间总是出奖。本来小概率事件频发，我们利用卡方的检验拟合

优度看看能否有足够证据证明判定抽奖机被人动了手脚

**卡方分布是什么？**

通过一个检验统计量来比较**期望结果**和**实际结果**之间的差别，然后得出观察频数极值的发生概率。

**计算统计量步骤：** （期望频数总和与观察频数总和相等）

1. 表里填写相应的观察频数和期望频数

2. 利用卡方公式计算检验统计量：

O 代表观察到的频数，也就是实际发生的频数。E代表期望频数。

**检验统计量$\chi^{2}$意义**：O与E之间差值越小，检验统计量$\chi^{2}$越小。以E为除数，令差值与期望频数成比例。

**卡方检验的标准：**如果统计量值 很小，说明观察频数和期望频数之间的差别不显著，统计量越大，差

别越显著。

**期望频数E的计算**

期望频数=（观察频数之和（1000）） X （每种结果的概率） 如：X=(-2)的期望频数：977=（0.977）

X（1000）

算出每个x值的实际频率与根据概率分布得出的期望频率进行比较

![image-20241127194139045](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815763.png)

**利用抽奖机的观察频率和期望频率表计算检验统计量**
$$
\begin{aligned}
\text{x}&=(965-977)^2/977 + (10-8)^2/8 + (9-8)^2/8 + (9-6)^2/6 + (7-1)^2/1 \\
&=(-12)^2/977 + 2^2/8 + 1^2/8 + 3^2/6 + 6^2 \\
&=144/977+4/8+1/8+9/6+36 \\
&=0.147+0.5+0.125+1.5+36 \\
&=38.272
\end{aligned}
$$
**根据自由度和显著性水平查询检验统计量临界值**

自由度的影响

**自由度**:用于计算检验统计量的独立变量的数目。

<img src="https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815764.png" alt="image-20241127194426600" style="zoom:150%;" />

当自由度等于1或者2时：卡方分布先高后低的平滑曲线，检验统计量等于较小值的概率远远大于较大值

的概率，即观察频数有可能接近期望频数。

当自由度大于2时：卡方分布先低后高再低，其外形沿着正向扭曲，但当自由度很大时，图形接近正态分

布。

自由度的计算，

对于单行或单列：自由度 = 组数-限制数

对于表格类：自由度 = (行数 - 1) * (列数 - 1)

![image-20241127194455382](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815765.png)

对于 抽奖机的例子，自由度为5-1=4





**自由度为4, 5%显著水平的拒绝域是多少？**

查表，自由度F=4，显著性为0.05，对应拒绝域 为
$$
\chi^2>9.49
$$
也就是说检验统计量大于9.49 位于拒绝域内决策原则

![image-20241127194538300](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815766.png)

如果位于拒绝域内我们拒绝原假设H0，**接受**H1。

如果不在拒绝域内我们**接受**原假设H0，拒绝H1

检验统计量38.272 > 9.49 位于拒绝域内

于是拒绝原假设：抽奖机每局收益如何概率分布，也就是说**抽奖机被人动了手脚**

**检验统计量拒绝域内外判定：**

1. 求出检验统计量a

2. 通过自由度和显著性水平查到拒绝域临界值b

3. a>b则位于拒绝域内，反之，位于拒绝域外。

**ORB-SLAM2中的卡方检测剔除外点策略**

误差的定义：

就特征点法的视觉SLAM而言，一般会计算重投影误差。具体而言，记 $\mathbf{u}$为特征点的2D位置，$\mathbf{\bar{u}}$为由地图点投影到图像上的2D位置。重投影误差为

$$
\mathbf{e}=\mathbf{u}-\mathbf{\bar{u}}\quad\mathrm{~(1)}
$$
重投影误差服从高斯分布
$$
\mathbf{e}\sim\mathcal{N}(\mathbf{0},\mathbf{\Sigma})\quad(2)
$$
 其中，协方差**$\Sigma$**一般根据**特征点提取的金字塔层级**确定。具体的，记提取ORB特征时，图像金字塔的每层缩小尺度为**$s$** (ORB-SLAM中为1.2)。在ORB-SLAM中假设第0层的标准差为**$p$**个pixel (ORB-SLAM 中设为了1个pixel);那么，一个在金字塔第**$n$**层提取的特征的重投影误差的协方差为
$$
\boldsymbol{\Sigma}=\left(s^{n}\times p\right)^{2}\begin{bmatrix}1&0\\0&1\end{bmatrix}\quad(3)
$$
回头再看一下式(1)中的误差是一个2维向量，阈值也不好设置吧。那就把它变成一个标量，计算向量的内积**$\boldsymbol{r}$** (向量元素的平方和)。但是，这又有一个问题，不同金字塔层的特征点都用同一个阈值，是不是不合理呢。于是，在计算内积的时候，利用协方差进行加权(协方差表达了不确定度)。金字塔层级越高，特征点提取精度越低，协方差$\Sigma$越大，那么就有了

$$
r=\mathbf{e}^{T}\boldsymbol{\Sigma}^{-1}\mathbf{e}\quad\mathrm{~(4)}
$$
利用协方差加权，起到了归一化的作用。具体的(4)式，可以变为

$$
r=(\boldsymbol{\Sigma}^{-\frac12}\mathbf{e})^T(\boldsymbol{\Sigma}^{-\frac12}\mathbf{e})\quad(5)
$$
而

$$
(\boldsymbol{\Sigma}^{-\frac12}\mathbf{e})\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$
为多维**标准正态分布**。

也就是说**不同金字塔层提取的特征，计算的重投影误差都被归一化了，或者说去量纲化了**，那么，我们只用一个阈值就可以了。

可见：

金字塔层数越高，图像分辨率越低，特征提取的精度也就越低，因此协方差越大



**单目投影为2自由度，在0.05的显著性水平（也就是95%的准确率）下，卡方统计量阈值为5.99**

**双目投影为3自由度，在0.05的显著性水平（也就是95%的准确率）下，卡方统计量阈值为7.81**

双目匹配到的特征点在右图中的x坐标为 $u_{xr}$，重投影后计算得到特征点左图的x坐标 $u_{xl}$，根据视差
$$
disparity=\frac{baseline*fx}{depth}
$$
从而得到重投影后右图中特征点x坐标
$$
u_{xr}=u_{xl}-disparity
$$
disparity就是另一个自由度。

LocalMapping.cc 里面

~~~C++
const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
const float invz1 = 1.0/z1;
if(!bStereo1)
{
    float u1 = fx1*x1*invz1+cx1;
    float v1 = fy1*y1*invz1+cy1;
    float errX1 = u1 - kp1.pt.x;
    float errY1 = v1 - kp1.pt.y;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）自由度2
    if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
    	continue;
}
else
{
    float u1 = fx1*x1*invz1+cx1;
    float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1; // 根据视差公式
    计算假想的右目坐标
    float v1 = fy1*y1*invz1+cy1;
    float errX1 = u1 - kp1.pt.x;
    float errY1 = v1 - kp1.pt.y;
    float errX1_r = u1_r - kp1_ur;
    // 自由度为3
    if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
    	continue;
}
~~~





### 单目SFM地图初始化

**归一化**

对应函数Initializer::Normalize

**原理参考**

Multiple view geometry in computer vision P109 算法4.4

Data normalization is an essential step in the DLT algorithm. It must not be considered optional.

Data normalization becomes even more important for less well conditioned problems,

such as the DLT computation of the fundamental matrix or the trifocal tensor, which

will be considered in later chapters

![image-20241127201649657](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815768.png)

**为什么要归一化？**
$$
Ah=0
$$
矩阵A是利用8点法求基础矩阵的关键，所以Hartey就认为，利用8点法求基础矩阵不稳定的一个主要原因就是**原始的图像像点坐标组成的系数矩阵**A**不好造成的**，而造成A不好的原因是**像点的齐次坐标各个分量的数量级相差太大**。基于这个原因，Hartey提出一种改进的8点法，**在应用8点法求基础矩阵之前，先对像点坐标进行归一化处理，即对原始的图像坐标做同向性变换**，这样就可以减少噪声的干扰，大大的提高8点法的精度。

预先对图像坐标进行归一化有以下好处：

- 能够提高运算结果的精度
- 利用归一化处理后的图像坐标，对任何尺度缩放和原点的选择是不变的。归一化步骤预先为图像坐标选择了一个标准的坐标系中，消除了坐标变换对结果的影响。

归一化操作分两步进行，首先对每幅图像中的坐标进行平移（每幅图像的平移不同）使图像中匹配的点组成的点集的形心（Centroid）移动到原点；接着对坐标系进行缩放使得各个分量总体上有一样的平均值，各个坐标轴的缩放相同的

![image-20241127201829293](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815769.png)

使用归一化的坐标虽然能够在一定程度上消除噪声、错误匹配带来的影响，但还是不够的。

参考：

https://www.cnblogs.com/wangguchangqing/p/8214032.html

**具体归一化操作**

一阶矩就是随机变量的期望，二阶矩就是随机变量平方的期望；一阶绝对矩定义为变量与均值绝对值的

平均。

向量
$$
\boldsymbol{u_1},\boldsymbol{u_2},\ldots\boldsymbol{u_N}
$$
期望
$$
\bar{\boldsymbol{u}}=E(u)
$$
一阶矩绝对矩
$$
|\bar{u}|=\sum_{i=0}^N|x_i-\bar{u}|/N
$$
令新的N维向量维
$$
\boldsymbol{u}^{\prime}=\frac{\boldsymbol{u}-\boldsymbol{\bar{u}}}{|\boldsymbol{\bar{u}}|}
$$
疑问：变换矩阵T为何这样？

答案：就是把上述变换用矩阵表示了而已
$$
\begin{bmatrix}x'\\y'\\1\end{bmatrix}\begin{bmatrix}sX&0&-meanX*sX\\0&sY&-meanY*sY\\0&0&1\end{bmatrix}\begin{bmatrix}x\\y\\1\end{bmatrix}
$$
**SVD分解**

![image-20241127202638763](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815770.png)

**检查位姿的有效性**

对应函数 Initializer::CheckRT
$$
\begin{gathered}
\left.T_{21}=\left[\begin{array}{cc}{R_{21}}&{t_{21}}\\{0^{T}}&{1}\end{array}\right.\right] \\
T_{12}=T_{21}^{-1}=\left[\begin{array}{cc}{{R_{21}^{T}}}&{{-R_{21}^{T}t_{21}}}\\{{0^{T}}}&{1}\end{array}\right] \\
R_{12}=R_{12}^T \\
t_{12}=-R_{21}^Tt_{21} 
\end{gathered}
$$
![image-20241127202741712](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815771.png)

向量余弦夹角：$cos(θ) = a•b / |a||b|$

### 双目地图初始化：稀疏立体匹配

**双目相机**

![image-20241127202819742](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815772.png)

视差公式：z 深度，d （disparity）视差，f 焦距，b （baseline）基线
$$
z=\frac{fb}{d},\quad d=u_{L}-u_{R}.
$$
**稀疏立体匹配原理**

函数ComputeStereoMatches()

两帧图像稀疏立体匹配

*输入：两帧立体矫正后的图像对应的orb特征点集

*过程：

1. 行特征点统计2.粗匹配

3. 精确匹配SAD

4. 亚像素精度优化

5. 最有视差值/深度选择

6. 删除离缺点（outliers）

*输出：稀疏特征点视差图/深度图和匹配结果

![image-20241127202908300](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815773.png)

**亚像素插值**

![image-20241127202942313](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815774.png)

![image-20241127202952622](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815775.png)



## 地图点、关键帧、图

### 地图点

**地图点代表性描述子的计算**

找最有代表性的描述子示意图

最有代表的描述子与其他描述子具有最小的距离中值

![image-20241127203109369](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815776.png)

**地图点法线朝向的计算**

![image-20241127203130989](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815777.png)

**地图点和特征点的区别？**

地图点是三维点，来自真实世界的三维物体，有唯一的id。不同帧里的特征点可能对应三维空间中同一

个三维点，特征点是二维点，是特征提取的点，大部分二维点在三维空间中没有对应地图点

**生成地图点**

关于生成地图点，主要有以下几个地方：

1. 初始化时 前两帧匹配生成地图点

2. local mapping里共视关键帧之间用 LocalMapping::CreateNewMapPoints() 生成地图点

3. Tracking::UpdateLastFrame() 和 Tracking::CreateNewKeyFrame() 中为双目和RGB-D生成了新的

**临时地图点**，单目不生成

### 关键帧

**什么是关键帧**

通俗来说，关键帧就是几帧普通帧里面具有代表性的一帧。

![image-20241127203250502](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815778.png)

**为什么需要关键帧**

- 相近帧之间信息冗余度很高，关键帧是取局部相近帧中最有代表性的一帧，可以**降低信息冗余度**。举例来说，摄像头放在原处不动，普通帧还是要记录的，但关键帧不会增加。

- 关键帧选择时还会对图片质量、特征点质量等进行考察，在Bundle Fusion、RKD SLAM等RGB-DSLAM相关方案中常常用普通帧的深度投影到关键帧上进行深度图优化，一定程度上关键帧是**普通帧滤波和优化的结果**，防止无用的或错误的信息进入优化过程而破坏定位建图的准确性。

- 如果所有帧全部参与计算，不仅浪费了算力，对内存也是极大的考验，这一点在前端vo中表现不明显，但在后端优化里是一个大问题，所以关键帧主要作用是**面向后端优化的算力与精度的折中**，**使得有限的计算资源能够用在刀刃上**，保证系统的平稳运行。假如你放松ORB_SLAM2 关键帧选择条件，大量产生的关键帧不仅耗计算资源，还会导致local mapping 计算不过来，出现误差累积

  

**如何选择关键帧**

选择关键帧主要**从关键帧自身和关键帧与其他关键帧的关系**2方面来考虑。

- **关键帧自身质量要好**，例如不能是非常模糊的图像、特征点数量要充足、特征点分布要尽量均匀等等；

- 关键帧与其他关键帧之间的关系，需要和局部地图中的其他关键帧有一定的共视关系但又不能重复度太高，以达到**既存在约束，又尽量少的信息冗余**的效果。

**选取的指标主要有：**

（1）距离上一关键帧的帧数是否足够多**（时间）**。

比如我每隔固定帧数选择一个关键帧，这样编程简单但效果不好。比如运动很慢的时候，就会选择大量相似的关键帧，冗余，运动快的时候又丢失了很多重要的帧。

（2）距离最近关键帧的距离是否足够远**（空间）**/运动

比如相邻帧根据pose计算运动的相对大小，可以是位移也可以是旋转或者两个都考虑，运动足够大（超过一定阈值）就新建一个关键帧，这种方法比第一种好。但问题是如果对着同一个物体来回扫就会出现大量相似关键帧。

（3）跟踪局部地图质量**（共视特征点数目）**

记录当前视角下跟踪的特征点数或者比例，当相机离开当前场景时（双目或比例明显降低）才会新建关键帧，避免了第2种方法的问题。缺点是数据结构和逻辑比较复杂。

![image-20241127203531996](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815779.png)

在关键帧的运用上，我认为orbslam2做的非常好，跟踪线程选择关键帧标准较宽松，局部建图线程再跟据共视冗余度进行剔除，尤其是在回环检测中使用了以关键帧为代表的帧“簇”的概念，回环筛选中有一步将关键帧前后10帧为一组，计算组内总分，以最高分的组的0.75为阈值，滤除一些组，再在剩下

的组内各自找最高分的一帧作为备选帧，这个方法非常好地诠释了“**关键帧代表局部**”的这个理念。

**关键帧的类型及更新连接关系**

**父子关键帧**

~~~C++
//KeyFrame.h 文件中
bool mbFirstConnection; // 是否是第一次生成树
KeyFrame* mpParent; // 当前关键帧的父关键帧 （共视程度最高的）
std::set<KeyFrame*> mspChildrens; // 存储当前关键帧的子关键帧
~~~



**更新连接关系**

~~~C++
//KeyFrame.cc
KeyFrame::UpdateConnections()
{
    //省略...
    // Step 5 更新生成树的连接
    if(mbFirstConnection && mnId!=0)
    {
    // 初始化该关键帧的父关键帧为共视程度最高的那个关键帧
    mpParent = mvpOrderedConnectedKeyFrames.front();
    // 建立双向连接关系，将当前关键帧作为其子关键帧
    mpParent->AddChild(this);
    mbFirstConnection = false;
    }
}
// 添加子关键帧（即和子关键帧具有最大共视关系的关键帧就是当前关键帧）
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}
// 删除某个子关键帧
void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
// 改变当前关键帧的父关键帧
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    // 添加双向连接关系
    mpParent = pKF;
    pKF->AddChild(this);
}
//获取当前关键帧的子关键帧
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
//获取当前关键帧的父关键帧
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}
// 判断某个关键帧是否是当前关键帧的子关键帧
bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}
~~~



**更新局部关键帧**

~~~C++
void Tracking::UpdateLocalKeyFrames()
{
    //省略...
    // 策略2.2:将自己的子关键帧作为局部关键帧（将邻居的子孙们拉拢入伙）
    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for(set<KeyFrame*>::const_iterator sit=spChilds.begin(),send=spChilds.end(); sit!=send; sit++)
    {
        KeyFrame* pChildKF = *sit;
        if(!pChildKF->isBad())
        {
            if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pChildKF);
                pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                //? 找到一个就直接跳出for循环？
                break;
            }
        }
    }
// 策略2.3:自己的父关键帧（将邻居的父母们拉拢入伙）
    KeyFrame* pParent = pKF->GetParent();
    if(pParent)
    {
    // mnTrackReferenceForFrame防止重复添加局部关键帧
        if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
        {
            mvpLocalKeyFrames.push_back(pParent);
            pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            //! 感觉是个bug！如果找到父关键帧会直接跳出整个循环
            break;
        }
    }
// 省略....
}
~~~





### 共视图 本质图 拓展树

![image-20241127204032913](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815780.png)



**共视图 （Covisibility Graph）**

共视图是无向加权图，每个节点是关键帧，如果两个关键帧之间满足一定的共视关系（至少15个共同观测地图点）他们就连成一条边，边的权重就是共视地图点数目

![image-20241127204110835](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815781.png)

**共视图的作用**

1. 跟踪局部地图，扩大搜索范围

`Tracking:UpdateLocalKeyFrames（）`

2. 局部建图里关键帧之间新建地图点

`LocalMapping:：CreateNewMapPoints（）`

`LocalMapping:SearchlnNeighbors（）`

3. 闭环检测、重定位检测

`LoopClosing:：DetectLoop（）、LoopClosing:CorrectLoop（）`

`KeyFrameDatabase:：DetectLoopCandidates`

`KeyFrameDatabase:：DetectRelocalizationCandidates`

4. 优化

`Optimizer:：OptimizeEssentialGraph`



**本质图（Essential Graph）**

共视图比较稠密，本质图比共视图更稀疏，这是因为本质图的作用是用在闭环矫正时，用相似变换来矫正尺度漂移，把闭环误差均摊在本质图中。本质图中节点也是所有关键帧，但是连接边更少，只保留了联系紧密的边来使得结果更精确。本质图中包含：

1. 扩展树连接关系
2. 形成闭环的连接关系，闭环后地图点变动后新增加的连接关系
3. 共视关系非常好（至少100个共视地图点）的连接关系

**本质图优化**

~~~C++
//Optimizer.cc
Optimizer::OptimizeEssentialGraph()
{
    // 省略....
    // Spanning tree edge
    // Step 4.1：添加扩展树的边（有父关键帧）
    // 父关键帧就是和当前帧共视程度最高的关键帧
    if(pParentKF)
    {
        int nIDj = pParentKF->mnId;
        g2o::Sim3 Sjw;
        LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);
    // 尽可能得到未经过Sim3传播调整的位姿
    if(itj!=NonCorrectedSim3.end())
    	Sjw = itj->second;
    else
    	Sjw = vScw[nIDj];
        
    // 计算父子关键帧之间的相对位姿
    g2o::Sim3 Sji = Sjw * Swi;
    g2o::EdgeSim3* e = new g2o::EdgeSim3();
        
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
    // 希望父子关键帧之间的位姿差最小
    e->setMeasurement(Sji);
    // 所有元素的贡献都一样;每个误差边对总误差的贡献也都相同
    e->information() = matLambda;
    optimizer.addEdge(e);
    }
// 省略....
}
~~~





**本质图优化和全局BA结果对比**

从结果来看，

1. 全局BA存在收敛问题。即使迭代100次，相对均方误差RMSE 也比较高

2. essential graph 优化可以快速收敛并且结果更精确。θmin 表示被选为essential graph至少需要的

共视地图点数目，从结果来看，θmin的大小对精度影响不大，但是较大的θmin值可以显著减少运行时间

3. essential graph 优化 后增加全局 full BA 可以提升精度（但比较有限），但是会耗时较多

![image-20241127204523513](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815782.png)

![image-20241127204530699](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815783.png)



**扩展树（spanning tree）**

子关键帧和父关键帧构成

![image-20241127204621434](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815784.png)

## 特征匹配

### 单目初始化中的特征匹配

**初始化搜索匹配**

对应函数 SearchForInitialization

![image-20241127204934226](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815785.png)

**快速搜索候选匹配特征点**

对应函数 GetFeaturesInArea

![image-20241127204946505](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815786.png)

**通过角度一致性过滤外点中的错误分析**

原因：错误把长度为30的直方图（单个bin角度范围12°）变成了长度为12（单个bin角度范围30°），

影响：弱化了对角度不一致性的错误剔除，导致只有角度差超过30°（本来是12°）才会被去除。

rotHist 预分配了30个bin，实际第13-30个都没有用到

修改前打印输出

![image-20241127205000428](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815787.png)

![image-20241127205009821](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815788.png)

总结，修改后代码

能够更有效的滤掉方向不一致的匹配对，更精确了。

举个例子，假如方向差别6°，

原来的代码：factor = 1/30，假设rot =6°,

计算得到 所在 bin =0; 和正常的角度±1°（bin=0）混在一起，不会被滤掉

改进后代码 factor = 1/12，假设rot =6°,

计算得到 所在 bin =1

会被当做异常值滤掉

经过方向一致性检测后删除的匹配对

![image-20241127205033701](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815789.png)

### 通过视觉词袋进行特征匹配

**直观理解词袋**

![image-20241127205121320](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815790.png)

词袋的大概流程

![image-20241127205144806](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815791.png)

为什么叫 bag of words 而不是 list of words 或者 array of words？

因为丢弃了Word出现的排列顺序、位置等因素，只考虑出现的频率，大大简化了表达，节省了存储空

间，在分析对比相似度的时候非常高效

https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/

**为什么要研究BoW？**

闭环检测：核心就是判断两张图片是否是同一个场景，也就是判断图像的相似性。

如何计算两张图的相似性？

**类帧差方法**

不行，这样做有很多问题

**视角变化**

没办法将两张图的像素进行一一匹配，直接帧差时几乎不能保证是同一个像素点的差。

![image-20241127205224686](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815792.png)

**光照变换**

同一视角，不同时间的光照对比

![image-20241127205235648](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815793.png)

**相机曝光不同**

![image-20241127205247111](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815794.png)

**词袋法**

Bag of words 可以解决这个问题。是以图像特征集合作为visual words，只关心图像中有没有这些words，有多少次，更**符合人类认知方式**，对不同光照、视角变换、季节更替等非常鲁棒。

**加速匹配**

ORB-SLAM2代码中使用的 SearchByBoW（用于关键帧跟踪、重定位、闭环检测SIM3计算），以及局部地图里的SearchForTriangulation，内部实现主要是利用了 BoW中的FeatureVector 来加速特征匹配。

使用FeatureVector 避免了所有特征点的两两匹配，只比较同一个节点下的特征点，极大加速了匹配效率，至于匹配精度，论文 《Bags of Binary Words for Fast Place Recognition in Image Sequences 》中提到在26292 张图片里的 false positive 为0，说明精度是有保证的。实际应用中效果非常不错。

**缺点：**

需要提前加载离线训练好的词袋字典，增加了存储空间。但是带来的优势远大于劣势，而且也有不少改进方法比如用二进制存储等来压缩词袋，减少存储空间，提升加载速度。

**如何制作、生成BoW？**

**为什么BOW一般用BRIEF描述子？**

**速度方面**

因为计算和匹配都非常快，论文中说大概一个关键点计算256位的描述子只需要 17.3µs

因为都是二进制描述子，距离描述通过汉明距离，使用异或操作即可，速度非常快。

而SIFT, SURF 描述子都是浮点型，需要计算欧式距离，会慢很多。

在Intel Core i7 ，2.67GHz CPU上，使用FAST+BRIEF 特征，在26300帧图像中 特征提取+词袋位置识别

耗时 22ms 每帧。

**在精度方面**

先上结论：闭环效果并不比SIFT, SURF之类的高精度特征点差。

具体来看一下对比：

以下对比来自论文《2012，Bags of Binary Words for Fast Place Recognition in Image Sequences，

IEEE TRANSACTIONS ON ROBOTICS》

三种描述子 BRIEF,，SURF64 ，U-SURF128 使用同样的参数，在训练数据集NewCollege，Bicocca25b

上的 Precision-recall 曲线

其中SURF64：带旋转不变性的 64 维描述子

U-SURF128：不带旋转不变性的128维描述子

![image-20241127205428635](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815795.png)

在两个数据中，SURF64 都明显比 U-SURF128 表现的好（曲线下面积更大），可以看到在Bicocca25b

数据集上，BRIEF明显比 U-SURF128 好，比SURF64 也稍微好一些，在NewCollege 数据集上 SURF64

比 BRIEF 更好一点，但是BRIEF也仍然不错。



总之，BRIEF 和 SURF64 效果基本相差不大，可以说打个平手。

**可视化效果**

可视化看一下效果

下图左边图片对是BRIEF 在vocabulary 中同样的Word下的回环匹配结果，同样的特征连成了一条线。

下图右边图像对是同样数据集中SURF64 的闭环匹配结果。

第一行 来看，尽管有一定视角变换，BRIEF 和 SURF64 的匹配结果接近

第二行：BRIEF成功进行了闭环，但是SURF64 没有闭环。原因是SURF64 没有得到足够多的匹配关系。

第三行：BRIEF 闭环失败而SURF64闭环成功。

我们分析一下原因：主要和近景远景有关。因为BRIEF相比SURF64没有尺度不变性，所以在尺度变换较大的近景很容易匹配失败，比如第三行。而在中景和远景，由于尺度变化不大，BRIEF 表现接近甚至优于SURF64

![image-20241127205505116](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815796.png)

不过，我们通过图像金字塔可以解决上述BRIEF的尺度问题。论文中作者也提到了ORB + BRIEF的特征点

主要问题是没有旋转不变性和尺度不变性。不过目前都解决了。

总之，BRIEF的闭环效果值得信赖！

**离线训练** **vocabulary tree（也称为字典）**

首先图像提取ORB 特征点，将描述子通过 k-means 进行聚类，根据设定的树的分支数和深度，从叶子

节点开始聚类一直到根节点，最后得到一个非常大的 vocabulary tree，

1. 遍历所有的训练图像，对每幅图像提取ORB特征点。

2. 设定vocabulary tree的分支数K和深度L。将特征点的每个描述子用 K-means聚类，变成 K个集合，作为vocabulary tree 的第1层级，然后对每个集合重复该聚类操作，就得到了vocabulary tree的第2层级，继续迭代最后得到满足条件的vocabulary tree，它的规模通常比较大，比如ORB-SLAM2使用的离线字典就有108万+ 个节点。

3. 离根节点最远的一层节点称为叶子或者单词 Word。根据每个Word 在训练集中的相关程度给定一个权重weight，训练集里出现的次数越多，说明辨别力越差，给与的权重越低。

**在线图像生成BoW向量**

1. 对新来的一帧图像进行ORB特征提取，得到一定数量（一般几百个）的特征点，描述子维度和vocabulary tree中的一致

2. 对于每个特征点的描述子，从离线创建好的vocabulary tree中开始找自己的位置，从根节点开始，用该描述子和每个节点的描述子计算汉明距离，选择汉明距离最小的作为自己所在的节点，一直遍历到叶子节点。

整个过程是这样的，见下图。紫色的线表示 一个特征点从根节点到叶子节点的过程。

![image-20241127205642086](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815797.png)

村民，村长，市长，省长 类比

**源码解析**

一个描述子转化为Word id， Word weight，节点所属的父节点（距离叶子深度为level up深度的节点）

id 对应的实现代码见：

~~~C++
/**
* @brief 将描述子转化为Word id， Word weight，节点所属的父节点id（这里的父节点不是叶子的上
一层，它距离叶子深度为levelsup）
*
* @tparam TDescriptor
* @tparam F
* @param[in] feature 				特征描述子
* @param[in & out] word_id 			Word id
* @param[in & out] weight 			Word 权重
* @param[in & out] nid 				记录当前描述子转化为Word后所属的 node id，它距离
叶子深度为levelsup
* @param[in] levelsup 				距离叶子的深度
*/
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(const TDescriptor &feature,
WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{
    // propagate the feature down the tree
    vector<NodeId> nodes;
    typename vector<NodeId>::const_iterator nit;
    // level at which the node must be stored in nid, if given
    // m_L: depth levels, m_L = 6 in ORB-SLAM2
    // nid_level 当前特征点转化为的Word 所属的 node id，方便索引
    const int nid_level = m_L - levelsup;
    if(nid_level <= 0 && nid != NULL) *nid = 0; // root
    NodeId final_id = 0; // root
    int current_level = 0;
    do
    {
        ++current_level;
        nodes = m_nodes[final_id].children;
        final_id = nodes[0];
        // 取当前节点第一个子节点的描述子距离初始化最佳（小）距离
        double best_d = F::distance(feature, m_nodes[final_id].descriptor);
        // 遍历nodes中所有的描述子，找到最小距离对应的描述子
        for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
        {
            NodeId id = *nit;
            double d = F::distance(feature, m_nodes[id].descriptor);
            if(d < best_d)
            {
                best_d = d;
                final_id = id;
            }
        }
        // 记录当前描述子转化为Word后所属的 node id，它距离叶子深度为levelsup
        if(nid != NULL && current_level == nid_level)
        *nid = final_id;
    } while( !m_nodes[final_id].isLeaf() );
    // turn node id into word id
    // 取出 vocabulary tree中node距离当前feature 描述子距离最小的那个node的 Word id 和weight
    word_id = m_nodes[final_id].word_id;
    weight = m_nodes[final_id].weight;
}
~~~





**一幅图像里所有特征点转化为两个std::map容器 BowVector 和 FeatureVector 的代码见：**

~~~C++
/**
* @brief 将一幅图像所有的特征点转化为BowVector和FeatureVector
*
* @tparam TDescriptor
* @tparam F
* @param[in] features 图像中所有的特征点
* @param[in & out] v BowVector
* @param[in & out] fv FeatureVector
* @param[in] levelsup 距离叶子的深度
*/
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(
const std::vector<TDescriptor>& features,
BowVector &v, FeatureVector &fv, int levelsup) const
{
    v.clear();
    fv.clear();
    if(empty()) // safe for subclasses
    {
    return;
    }
    // normalize
    // 根据选择的评分类型来确定是否需要将BowVector 归一化
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);
    typename vector<TDescriptor>::const_iterator fit;
    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        unsigned int i_feature = 0;
        // 遍历图像中所有的特征点
        for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
        {
            WordId id; // 叶子节点的Word id
            NodeId nid; // FeatureVector 里的NodeId，用于加速搜索
            WordValue w; // 叶子节点Word对应的权重
            // 将当前描述子转化为Word id， Word weight，节点所属的父节点id（这里的父节点不是叶子
            的上一层，它距离叶子深度为levelsup）
            // w is the idf value if TF_IDF, 1 if TF
            transform(*fit, id, w, &nid, levelsup);
            if(w > 0) // not stopped
            {
                // 如果Word 权重大于0，将其添加到BowVector 和 FeatureVector
                v.addWeight(id, w);
                fv.addFeature(nid, i_feature);
            }
        }
        if(!v.empty() && !must)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
            vit->second /= nd;
        }
    }
    // 省略掉了IDF || BINARY情况的代码
    if(must) v.normalize(norm);
}
~~~



相当于将当前图像信息进行了压缩，并且对后面特征点快速匹配、闭环检测、重定位意义重大。

这两个容器非常重要，下面一个个来解释：

**理解词袋向量BowVector**

它内部实际存储的是这个

`std::map<WordId, WordValue>`

其中 WordId 和 WordValue 表示Word在所有叶子中距离最近的叶子的id 和权重（后面解释）。

同一个Word id 的权重是累加更新的，见代码

~~~C++
void BowVector::addWeight(WordId id, WordValue v)
{
    // 返回指向大于等于id的第一个值的位置
    BowVector::iterator vit = this->lower_bound(id);
    // http://www.cplusplus.com/reference/map/map/key_comp/
    if(vit != this->end() && !(this->key_comp()(id, vit->first)))
    {
        // 如果id = vit->first, 说明是同一个Word，权重更新
        vit->second += v;
    }
    else
    {
        // 如果该Word id不在BowVector中，新添加进来
        this->insert(vit, BowVector::value_type(id, v));
    }
}
~~~



**理解特征向量FeatureVector**

内部实际是个

`std::map<NodeId, std::vector<unsigned int> >`

其中NodeId 并不是该叶子节点直接的父节点id，而是距离叶子节点深度为level up对应的node 的id，对应上面 vocabulary tree 图示里的 Word's node id。为什么不直接设置为父节点？因为后面搜索该Word 的匹配点的时候是在和它具有同样node id下面所有子节点中的Word 进行匹配，搜索区域见图示中的 Word's search region。所以搜索范围大小是根据level up来确定的，level up 值越大，搜索范围越广，速度越慢；level up 值越小，搜索范围越小，速度越快，但能够匹配的特征就越少。

另外 std::vector 中实际存的是NodeId 下所有特征点在图像中的索引。见代码

~~~C++
void FeatureVector::addFeature(NodeId id, unsigned int i_feature)
{
    FeatureVector::iterator vit = this->lower_bound(id);
    // 将同样node id下的特征放在一个vector里
    if(vit != this->end() && vit->first == id)
    {
        vit->second.push_back(i_feature);
    }
    else
    {
        vit = this->insert(vit, FeatureVector::value_type(id,
        std::vector<unsigned int>() ));
        vit->second.push_back(i_feature);
    }
}
~~~



FeatureVector主要用于不同图像特征点快速匹配，加速几何关系验证，比如

ORBmatcher::SearchByBoW 中是这样用的

~~~C++
DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
while(f1it != f1end && f2it != f2end)
{
    // Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
    if(f1it->first == f2it->first)
        // Step 2：遍历KF中属于该node的特征点
        for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
        {
            const size_t idx1 = f1it->second[i1];
            MapPoint* pMP1 = vpMapPoints1[idx1];
            // 省略
            // ..........
~~~



**词典树的保存和加载**

我们将 vocabulary tree翻译为词典树

**如何保存训练好的词典树存储为txt文件？**

~~~C++
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::saveToTextFile(const std::string
&filename) const
{
    fstream f;
    f.open(filename.c_str(),ios_base::out);
    // 第一行打印 树的分支数、深度、评分方式、权重计算方式
    f << m_k << " " << m_L << " " << " " << m_scoring << " " << m_weighting <<endl;
    for(size_t i=1; i<m_nodes.size();i++)
    {
        const Node& node = m_nodes[i];
        // 每行第1个数字为父节点id
        f << node.parent << " ";
        // 每行第2个数字标记是（1）否（0）为叶子（Word）
        if(node.isLeaf())
        f << 1 << " ";
        else
        f << 0 << " ";
        // 接下来存储256位描述子，最后存储节点权重
        f << F::toString(node.descriptor) << " " << (double)node.weight << endl;
    }
    f.close();
}
~~~



**如何加载训练好的词典树文件？**

~~~C++
/**
* @brief 加载训练好的 vocabulary tree，txt格式
*
* @tparam TDescriptor
* @tparam F
* @param[in] filename vocabulary tree 文件名称
* @return true 加载成功
* @return false 加载失败
*/
template<class TDescriptor, class F>
bool TemplatedVocabulary<TDescriptor,F>::loadFromTextFile(const std::string
&filename)
{
    ifstream f;
    f.open(filename.c_str());
    if(f.eof())
    return false;
    m_words.clear();
    m_nodes.clear();
    string s;
    getline(f,s);
    stringstream ss;
    ss << s;
    ss >> m_k; // 树的分支数目
    ss >> m_L; // 树的深度
    int n1, n2;
    ss >> n1;
    ss >> n2;
    if(m_k<0 || m_k>20 || m_L<1 || m_L>10 || n1<0 || n1>5 || n2<0 || n2>3)
    {
        std::cerr << "Vocabulary loading failure: This is not a correct text
        file!" << endl;
        return false;
    }
    m_scoring = (ScoringType)n1; // 评分类型
    m_weighting = (WeightingType)n2; // 权重类型
    createScoringObject();
    // 总共节点（nodes）数，是一个等比数列求和
    //! bug 没有包含最后叶子节点数，应该改为 ((pow((double)m_k, (double)m_L + 2) -
    1)/(m_k - 1))
    //! 但是没有影响，因为这里只是reserve，实际存储是一步步resize实现
    int expected_nodes =
    (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
    m_nodes.reserve(expected_nodes);
    // 预分配空间给 单词（叶子）数
    m_words.reserve(pow((double)m_k, (double)m_L + 1));
    // 第一个节点是根节点，id设为0
    m_nodes.resize(1);
    m_nodes[0].id = 0;
    while(!f.eof())
    {
        string snode;
        getline(f,snode);
        stringstream ssnode;
        ssnode << snode;
        // nid 表示当前节点id，实际是读取顺序，从0开始
        int nid = m_nodes.size();
        // 节点size 加1
        m_nodes.resize(m_nodes.size()+1);
        m_nodes[nid].id = nid;
        // 读每行的第1个数字，表示父节点id
        int pid ;
        ssnode >> pid;
        // 记录节点id的相互父子关系
        m_nodes[nid].parent = pid;
        m_nodes[pid].children.push_back(nid);
        // 读取第2个数字，表示是否是叶子（Word）
        int nIsLeaf;
        ssnode >> nIsLeaf;
        // 每个特征点描述子是256 bit，一个字节对应8 bit，所以一个特征点需要32个字节存储。
        // 这里 F::L=32，也就是读取32个字节，最后以字符串形式存储在ssd
        stringstream ssd;
        for(int iD=0;iD<F::L;iD++)
        {
            string sElement;
            ssnode >> sElement;
            ssd << sElement << " ";
        }
        // 将ssd存储在该节点的描述子
        F::fromString(m_nodes[nid].descriptor, ssd.str());
        // 读取最后一个数字：节点的权重（Word才有）
        ssnode >> m_nodes[nid].weight;
        if(nIsLeaf>0)
        {
            // 如果是叶子（Word），存储到m_words
            int wid = m_words.size();
            m_words.resize(wid+1);
            //存储Word的id，具有唯一性
            m_nodes[nid].word_id = wid;
            //构建 vector<Node*> m_words，存储word所在node的指针
            m_words[wid] = &m_nodes[nid];
        }
        else
        {
            //非叶子节点，直接分配 m_k个分支
            m_nodes[nid].children.reserve(m_k);
        }
    }
	return true;
}
~~~



有几点需要说明：

$K$ 表示树的分支数，$L$ 表示树的深度，这里的深度不考虑根节点$K^0$,是从根节点下面开始算总共有$L$
层深度，最后叶子层总共有$K^{L+1}$个叶子 (Word) 。
总共所有的节点数目是一个等比数列求和问题

等比数列前n项和
$$
S_n=\frac{a_1-a_n\times q}{1-q}
$$
最后所有的节点数目应该是
$$
\frac{K^{L+2}-1}{K-1}
$$
关于权重类型 和 评分类型

~~~C++
/// Weighting type
enum WeightingType
{
    TF_IDF, //0
    TF, //1
    IDF, //2
    BINARY //3
};
/// Scoring type
enum ScoringType
{
    L1_NORM, //0
    L2_NORM, //1
    CHI_SQUARE, //2
    KL, //3
    BHATTACHARYYA,//4
    DOT_PRODUCT, //5
};
~~~



有个地方需要注意：

ORB-SLAM2 中词典初始化时：k = 10, int L = 5, weighting = TF_IDF, scoring = L1_NORM

~~~C++
/**
* Initiates an empty vocabulary
* @param k branching factor
* @param L depth levels
* @param weighting weighting type
* @param scoring scoring type
*/
TemplatedVocabulary(int k = 10, int L = 5,
WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);
// levelsup = 4
mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
~~~



但是实际上ORB-SLAM2 使用时是使用的加载的词典ORBvoc.txt

~~~
10 6 0 0
0 0 252 188 188 242 169 109 85 143 187 191 164 25 222 255 72 27 129 215 237 16 58
111 219 51 219 211 85 127 192 112 134 34 0
0 0 93 125 221 103 180 14 111 184 112 234 255 76 215 115 153 115 22 196 124 110
233 240 249 46 237 239 101 20 104 243 66 33 0
....
~~~



所以真正使用的词典参数是：k = 10, int L = 6, weighting = TF_IDF, scoring = L1_NORM, levelsup=4

**倒排索引**

```C++
// mvInvertedFile[i]表示包含了第i个word id的所有关键帧
 std::vector<list<KeyFrame*> > mvInvertedFile;
```





**直接索引表direct index和逆向（倒排）索引表nverse index。**

直接索引表 是以 图像为基础的，存储图像中的特征以及和该图像相关联的节点 node（vocabulary tree的某一层级）。

直接索引的优势：

通过同一个节点下特征点描述子对应关系 加速几何关系验证，比如用BOW快速匹配

逆向（倒排）索引表 是以 Word 为基础，存储Word的权重以及该Word出现在哪个图像里。这对于查询

数据库非常方便，因为它可以很方便对比哪些图像有共同的Word

![image-20241127211425046](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815798.png)



### 局部建图线程中搜索匹配三角化

对应函数 ORBmatcher::SearchForTriangulation

![image-20241127211454617](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815799.png)



### 闭环线程中Sim3搜索匹配

在闭环线程中使用。目的是为了进一步搜索匹配更多地图点，对应函数 SearchBySim3

![image-20241127211556981](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815800.png)

## 跟踪线程

### 跟踪线程整体流程

ORB-SLAM2跟踪部分主要包括两个阶段，第一个阶段包括三种跟踪方法：用参考关键帧来跟踪、恒速模

型跟踪、重定位跟踪，它们的目的是保证能够“跟的上”，但估计出来的位姿可能没那么准确。第二个阶

段是局部地图跟踪，将当前帧的局部关键帧对应的局部地图点投影到该帧，得到更多的特征点匹配关

系，对第一阶段的位姿再次优化得到相对准确的位姿。

![image-20241127212111400](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815801.png)

### 参考关键帧跟踪

对应函数 tracking::TrackReferenceKeyFrame()

![image-20241127212123570](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815802.png)

**应用场景**：没有速度信息的时候、刚完成重定位、或者恒速模型跟踪失败后使用，大部分时间不用。只利用到了参考帧的信息。

1. 匹配方法是 SearchByBoW，匹配当前帧和关键帧在同一节点下的特征点，不需要投影，速度很快
2. BA优化（仅优化位姿），提供比较粗糙的位姿

**思路:**

当使用运动模式匹配到的特征点数较少时，就会选用关键帧模式跟踪。

思路是：尝试和最近一个关键帧去做匹配。为了快速匹配，利用了bag of words（BoW）来加速匹配

**具体流程**

1. 计算当前帧的BoW；

2. 通过特征点的bow加快当前帧和参考帧之间的特征点匹配。使用函数matcher.SearchByBoW()。

- 对属于同一node（同一node才可能是匹配点）的特征点通过描述子距离进行匹配，遍历该node中特征点，特征点最小距离明显小于次小距离才作为成功匹配点，记录特征点对方向差统计到直方图
- 记录特征匹配成功后每个特征点对应的MapPoint（来自参考帧），用于后续3D-2D位姿优化
- 通过角度投票进行剔除误匹配

3. 将上一帧的位姿作为当前帧位姿的初始值（加速收敛），通过优化3D-2D的重投影误差来获得准确位姿。3D-2D来自第2步匹配成功的参考帧和当前帧，重投影误差 e = (u,v) - project(Tcw*Pw)，只优化位姿Tcw，不优化MapPoints的坐标。

顶点 Vertex: g2o::VertexSE3Expmap()，初始值为上一帧的Tcw

边 Edge（单目）: g2o::EdgeSE3ProjectXYZOnlyPose()，一元边 BaseUnaryEdge

\+ 顶点 Vertex：待优化当前帧的Tcw

\+ 测量值 measurement：MapPoint在当前帧中的二维位置(u,v)

\+ 误差信息矩阵 InfoMatrix: Eigen::Matrix2d::Identity()*invSigma2(与特征点所在的尺度有关)

 +附加信息： 相机内参数： e->fx fy cx cy

​		3d点坐标 ： e->Xw[0] Xw[1] Xw[2] 2d点对应的上一帧的3d点

优化多次，根据边误差，更新2d-3d匹配质量内外点标记，当前帧设置优化后的位姿。

4. 剔除优化后的outlier地图点

lower_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于或等于num的数

字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在

数组中的下标。

upper_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于num的数字，找

到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中

的下标。

例子：http://www.cplusplus.com/reference/algorithm/lower_bound/

### 恒速模型跟踪

对应函数Tracking::TrackWithMotionModel()，示意图如下

![image-20241127214300583](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815803.png)

**应用场景**

大部分时间都用这个跟踪，只利用到了上一帧的信息。

1. 用恒速模型先估计一个初始位姿
2. 用该位姿进行投影匹配 SearchByProjection，候选点来自GetFeaturesInArea，未使用BoW
3. BA优化（仅优化位姿），提供比较粗糙的位姿

**思想**

假设短时间内（相邻帧）物体处于匀速运动状态，可以用上一帧的位姿和速度来估计当前帧的位姿。



移动模式跟踪 跟踪前后两帧 得到 变换矩阵。



上一帧的地图3d点反投影到当前帧图像像素坐标上，在不同尺度下不同的搜索半径内，做描述子匹配 搜

索 可以加快匹配。



在投影点附近根据描述子距离进行匹配（需要>20对匹配，否则匀速模型跟踪失败,运动变化太大时会出

现这种情况），然后以运动模型预测的位姿为初值，优化当前位姿，



优化完成后再剔除外点，若剩余的匹配依然>=10对，则跟踪成功，否则跟踪失败，需要Relocalization

**具体流程**

1. 创建 ORB特征点匹配器 最小距离 < 0.9*次小距离 匹配成功

2. 更新上一帧的位姿和地图点

单目：只计算了上一帧世界坐标系位姿就退出了。Tlr*Trw = Tlw

双目或rgbd相机：根据上一帧有效的深度值产生为上一帧生成新的临时地图点，之所以说是“临时”因为

这些新加的地图点不加入到Map中，只是为了当前帧间跟踪更稳定，用完会删除，过河拆桥啊！

3. 使用当前的运动速度(之前前后两帧位姿变换)和上一帧的位姿来初始化 当前帧的位姿R,t

4. 在当前帧和上一帧之间搜索匹配点（matcher.SearchByProjection)

​	通过投影(使用当前帧的位姿R,t)，对上一帧的特征点(地图点)进行跟踪. 

​	上一帧3d点投影到当前坐标系下，在该2d点半径th范围内搜索可以匹配的匹配点

​	遍历可以匹配的点，计算描述子距离，记录最小的匹配距离，小于阈值的，再记录匹配点特征方向

差值

进行方向验证，剔除方向差直方图统计中，方向差值数量少的点对，保留前三个数量多的点对。

5. 如果找到的匹配点对如果少于20，则扩大搜索半径th=2*th,使用SearchByProjection()再次进行搜

索。

6. 使用匹配点对对当前帧的位姿进行优化 G2O图优化

7. 如果2d-3d匹配效果差，被标记为外点，则当前帧2d点对于的3d点设置为空，留着以后再优化

8. 根据内点的匹配数量，判断 跟踪上一帧是否成功。

![image-20241127214702756](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815804.png)

**坐标系变换**

推导 tlc = Rlw*twc+tlw
$$
\begin{aligned}
&\left.T_{cw}=\left[\begin{array}{cc}R_{cw}&t_{cw}\\0^T&1\end{array}\right.\right] \\
&\left.T_{cw}^{-1}=\left[\begin{array}{cc}R_{cw}^{T}&-R_{cw}^{T}t_{cw}\\0^{T}&1\end{array}\right.\right] \\
&\left.T_{lw}=\left[\begin{array}{cc}{R_{lw}}&{t_{lw}}\\{0^{T}}&{1}\end{array}\right.\right] \\
T_{lc}& =T_{lw}*T_{cw}^{-1} \\
&\left.=\left[\begin{array}{cc}R_{lw}&t_{lw}\\0^T&1\end{array}\right.\right]\left[\begin{array}{cc}R_{cw}^T&-R_{cw}^Tt_{cw}\\0^T&1\end{array}\right] \\
&=\begin{bmatrix}R_{lw}R_{cw}^T&-R_{lw}R_{cw}^Tt_{cw}+t_{lw}\\0^T&1\end{bmatrix} \\
&t_{lc}=-R_{lw}R_{cw}^Tt_{cw}+t_{lw}
\end{aligned}
$$
**6.4** **重定位跟踪**

**重定位跟踪原理**

对应函数Tracking::Relocalization()

**应用场景**

跟踪丢失的时候使用，很少使用。利用到了相似候选帧的信息。

1. 用BoW先找到与该帧相似的候选关键帧（函数DetectRelocalizationCandidates）
2. 遍历候选关键帧，用SearchByBoW快速匹配，
3. 匹配点足够的情况下用EPnP 计算位姿并取出其中内点做BA优化（仅优化位姿），
4. 如果优化完内点较少，通过关键帧投影生成新的匹配（函数SearchByProjection），
5. 对匹配结果再做BA优化（仅优化位姿）。

**思想**

当TrackWithMotionModel 和 TrackReferenceKeyFrame 都没有跟踪成功，位置丢失后，需要在之前的

关键帧中匹配最相近的关键帧，进而求出位姿信息。



使用当前帧的BoW特征映射，在关键帧数据库中寻找相似的候选关键帧，因为这里没有好的初始位姿信

息，需要使用传统的3D-2D匹配点的EPnP算法来求解一个初始位姿，之后再使用最小化重投影误差来优

化更新位姿。

**具体流程**：

1. 计算当前帧的BoW向量和Feature向量

2. 在关键帧数据库中找到与当前帧相似的候选关键帧组

3. 创建 ORB特征点匹配器 最小距离 < 0.75*次小距离 匹配成功。

 ORBmatcher matcher(0.75,true);

4. 遍历每一个候选关键帧使用BOW特征向量加速匹配，匹配太少的去掉，选择符合要求的候选关键帧

用其地图点为其创建pnp优化器

5. 使用PnPsolver 位姿变换求解器,更加3d-2d匹配点

 	6点直接线性变换DLT,后使用QR分解得到 R,t, 或者使用(P3P)，3点平面匹配算法求解

​	这里会结合 Ransac 随采样序列一致性算法，来提高求解的鲁棒性。 

6. EPnP算法迭代估计姿态作为当前帧的初始位姿，使用最小化重投影误差BA算法来优化位姿

7. 如果优化时记录的匹配点对内点数量少于50，想办法再增加匹配点数量：通过投影的方式对之前未匹配的点进行3D-2D匹配，又给了一次重新做人的机会

8. 如果新增的数量加上之前的匹配点数量 大于50，再次使用 位姿优化算法进行优化

9. 如果上面优化后的内点数量还比较少，还想挽留一下，就缩小搜索窗口重新投影匹配（比之前使用更多的地图点了），如果最后匹配次数大于50，就认为是可以勉强扶起来的阿斗，再给BA位优化一次。否则，放弃了（真的已经仁至义尽了！）

10. 如果经过上面一系列的挽救操作，内点数量 大于等于50 ，则重定位成功。

**检测重定位候选关键帧**

对应函数 DetectRelocalizationCandidates

![image-20241127215042051](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815805.png)

**EPnP** **算法原理详解**

**背景介绍**

论文：

*Lepetit V , Fua M N . EPnP: An Accurate O(n) Solution to the PnP Problem[J]. International Journal of*Computer Vision, 2009.*

输入：

1. 个世界坐标系下的3D点，论文中称为3D参考点
2. 这 个3D点投影在图像上的2D坐标
3. 相机内参矩阵 ，包括焦距和主点

输出：相机的位姿

应用：特征点的图像跟踪，需要实时处理有噪声的特征点，对计算精度和效率要求比较高，只需4对匹配点即可求解。

算法的优点：

1. 只需要4对非共面点，对于平面只需要3对点
2. 闭式解，不需要迭代，不需要初始估计值。
3. 精度比较高。和迭代法里精度最高的方法LHM方法精度相当。4. 比较鲁棒，可以处理带噪声的数据。迭代法受到初始估计的影响比较大，会不稳定

![image-20241127215146674](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815806.png)

4. 线性计算复杂度为

5. 平面和非平面都适用

![image-20241127215210365](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815807.png)

6. 平面和非平面都适用



原理和步骤：

我们目前知道 个世界坐标系下的3D点及其在图像上的2D投影点，还有相机内参，目的是为了求世界坐

标系到相机坐标系下的位姿变换 R,t.

EPnP的思路就是先把2D图像点通过内参变换到相机坐标系下的3D点，然后用ICP来求解3D-3D的变换就

得到了位姿。那么问题的核心就转化为如何通过2D信息，加上一些约束，来得到相机坐标系下的3D点。

因为我们这里的位姿变换是欧式空间下的刚体变换，所以点之间的相对距离信息在不同坐标系下是不变

的。我们称之为**刚体结构不变性**。后面就是紧紧围绕这个特性来求解的。

1. 首先我们对3D点的表达方式进行了新的定义。之前不管是世界坐标系还是相机坐标系下的3D点，

它们都是相对于自己坐标系下的原点的。那么两个坐标系原点不同，坐标的量级可能差异非常大，

比如相机坐标系下3D点坐标范围可能是10-100之间，世界坐标系下坐标可能是1000-10000之间，

这对求解优化都是不利的。所以我们要统一一下量级。可以理解为归一化吧，这在求基础矩阵、单

应矩阵时都是常规手段。

具体来说，我们对每个坐标系定义**4****个控制点**，其中一个是质心（也就是各个方向均值），其他3个

用PCA从三个主方向选取，这4个控制点可以认为是参考基准，类似于坐标系里的基。所有的3D点

都表达为这4个参考点的线性组合。这些系数我们称之为权重，为了不改变数据的相对距离，权重

和必须为1。这样，我们就可以**用世界坐标系或相机坐标系下的4个控制点表示所有的世界坐标系或**

**相近坐标系下的3D点。**

2. 利用投影方程将图像2D点恢复相机坐标系下3D点（未知量）。经过整理后，一组点对可以得到2个

方程。我们待求的相机坐标系下3D点对应的4个控制点，每个控制点3个分量，总共12个未知数组

成的一个向量。

3. 用SVD分解可以求解上述向量，但是因为恢复的相机坐标系下3D点还有个尺度因子 ， 这里我们根

据结构信息不变性作为约束，求解。

4. 最后用高斯牛顿法优化上述求解的 $\beta$。

注：ORB-SLAM里使用的EPnP是直接拷贝OpenCV里的源码：modules → calib3d → src → epnp.cpp

**统一变量格式：**

首先统一一下变量的定义格式：

用上标 $^{w}$和 $^{c}$分别表示在世界坐标系和相机坐标系中的坐标

$n$个3D参考点在世界坐标系中的坐标是已知的输入，记为

$$
\mathbf{p}_i^w,\quad i=1,\ldots,n
$$
$n$个3D参考点在相机坐标系下的坐标是未知的，记为

$$
\mathbf{p}_i^c,\quad i=1,\ldots,n
$$
$n$个3D参考点在相机坐标系下对应的n个2D投影坐标是已知的，记为

$$
\mathbf{u}_i,\quad i=1,\ldots,n
$$
4个控制点在世界坐标系下的坐标为
$$
\mathbf{c}_j^w,j=1,\cdots,4
$$
4个控制点在相机坐标系下的坐标是我们未知的，记为
$$
\mathbf{c}_j^c,j=1,\cdots,4
$$
注意以上坐标都是非齐次坐标，后面也都是非齐次坐标。

4个控制点系数$\alpha _ij$, $i= 1, \ldots , n$, $j= 1, \ldots , 4$,也就是论文中的homogeneous barycentrid coordinates,我们翻译为齐次重心坐标。**同一3D点在世界坐标系下和相机坐标系下的控制点系数相同**。后面会给出证明。





**控制点如何选取？**



理论上，控制点的坐标可以任意选取。但在实践中，作者发现了一种可以提高结果稳定性的控制点选择

方法。具体如下

1. 将参考点的质心（或者称为重心、均值中心）设置为其中一个控制点，表达式见下。这是有一定物理意义的，因为后续会使用质心对坐标点进行归一化。
   $$
   \mathbf{c}_1^w=\frac1n\sum_{i=1}^n\mathbf{p}_i^w
   $$

2. 剩下的3个控制点从数据的三个主方向上选取。

我们对世界坐标系下3D点集合$\left\{\mathbf{p}_i^w,i=1,\ldots,n\right\}$去质心后得到

$$
\mathbf{A}=\left[\begin{array}{c}(\mathbf{p}_1^w)^T-(\mathbf{c}_1^w)^T\\\vdots\\(\mathbf{n}_-^w)^T-(\mathbf{c}_-^w)^T\end{array}\right]
$$
A是一个$n\times3$的矩阵，那么$A^TA$就是$3\times3$方阵，通过对矩阵$A^TA$进行特征值分解，得到三个特征值$\lambda_1^w,\lambda_2^w,\lambda_3^w$,它们对应的特征向量为 $v_1^w,v_2^w,v_3^w$

将剩余的3个控制点表示为


$$
\begin{gathered}
\mathbf{c}_2^w =\mathbf{c}_1^w+\sqrt{\frac{\lambda_1^w}n}v_1^w \\
\mathbf{c}_3^w =\mathbf{c}_{1}^{w}+\sqrt{\frac{\lambda_{2}^{w}}{n}}v_{2}^{w} \\
\mathbf{c}_4^w =\mathbf{c}_{1}^{w}+\sqrt{\frac{\lambda_{3}^{w}}{n}}v_{3}^{w} 
\end{gathered}
$$
为什么要加$c_1^w$ ？

因为之前去了质心，现在要重新加上。

**计算控制点系数，用控制点重新表达数据**

我们将世界坐标系下3D点的坐标表示为对应控制点坐标的线性组合：
$$
\mathbf{p}_i^w=\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^w, \sum_{j=1}^4\alpha_{ij}=1
$$
在论文中，$\alpha_{ij}$ 称为homogeneous barycentric coordinates,我们翻译为齐次重心坐标，它实际上表达的是世界坐标系下3D点在控制点坐标系下的坐标系数。当控制点$\mathbf{c}_j^w$通过第一步的方法确定后，$\alpha_{ij}$也是唯一确定的。我们来推导一下，上式展开后得到
$$
\mathbf{p}_i^w=\alpha_{i1}\mathbf{c}_1^w+\alpha_{i2}\mathbf{c}_2^w+\alpha_{i3}\mathbf{c}_3^w+\alpha_{i4}\mathbf{c}_4^w
$$
 3D点的重心为$\mathbf{c}_1^w$,也是我们的第一个控制点。上式左右分别减去重心

$$
\begin{aligned}\mathbf{p}_{i}^{w}-\mathbf{c}_{1}^{w}&=\alpha_{i1}\mathbf{c}_{1}^{w}+\alpha_{i2}\mathbf{c}_{2}^{w}+\alpha_{i3}\mathbf{c}_{3}^{w}+\alpha_{i4}\mathbf{c}_{4}^{w}-\mathbf{c}_{1}^{w}\\&=\alpha_{i1}\mathbf{c}_{1}^{w}+\alpha_{i2}\mathbf{c}_{2}^{w}+\alpha_{i3}\mathbf{c}_{3}^{w}+\alpha_{i4}\mathbf{c}_{4}^{w}-(\alpha_{i1}+\alpha_{i2}+\alpha_{i3}+\alpha_{i4})\mathbf{c}_{1}^{w}\\&=\alpha_{i2}(\mathbf{c}_{2}^{w}-\mathbf{c}_{1}^{w})\:+\alpha_{i3}(\mathbf{c}_{3}^{w}-\mathbf{c}_{1}^{w})+\alpha_{i4}(\mathbf{c}_{4}^{w}-\mathbf{c}_{1}^{w})\\&=\left[\begin{array}{ccc}\mathbf{c}_{2}^{w}-\mathbf{c}_{1}^{w}&\mathbf{c}_{3}^{w}-\mathbf{c}_{1}^{w}&\mathbf{c}_{4}^{w}-\mathbf{c}_{1}^{w}\end{array}\right]\left[\begin{array}{c}\alpha_{i2}\\\alpha_{i3}\\\alpha_{i4}\end{array}\right]\end{aligned}
$$


那么，世界坐标系下控制点的系数可以这样计算得到
$$
\begin{bmatrix}\alpha_{i2}\\\alpha_{i3}\\\alpha_{i4}\end{bmatrix}=\begin{bmatrix}\mathbf{c}_2^w-\mathbf{c}_1^w&\mathbf{c}_3^w-\mathbf{c}_1^w&\mathbf{c}_4^w-\mathbf{c}_1^w\end{bmatrix}^{-1}(\mathbf{p}_i^w-\mathbf{c}_1^w)  
$$

$$
\alpha_{i1}=1-\alpha_{i2}-\alpha_{i3}-\alpha_{i4}
$$

以上是世界坐标系下的推导，那么，在相机坐标系下， 满足如下的对应关系吗？
$$
\mathbf{p}_i^c=\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^c, \sum_{j=1}^4\alpha_{ij}=1
$$
这里先给出结论：**同一个3D点在世界坐标系下对应控制点的系数$\alpha_{ij}$ 和其在相机坐标系下对应控制点的系数相同**。也就是说，**我们可以预先在世界坐标系下求取控制点系数$\alpha_{ij}$ ,然后将其作为已知量拿到相机坐标系下使用**。

口说无凭，我们来推导一下，假设待求的相机位姿为T，那么
$$
\begin{aligned}
\begin{bmatrix}\mathbf{p}_i^c\\1\end{bmatrix}& =T\left[\begin{array}{c}\mathbf{p}_i^w\\1\end{array}\right] \\
&=T\begin{bmatrix}\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^w\\\sum_{j=1}^4\alpha_{ij}\end{bmatrix} \\
&=\sum_{j=1}^4\alpha_{ij}T\left[\begin{array}{c}\mathbf{c}_j^w\\1\end{array}\right] \\
&=\sum_{j=1}^4\alpha_{ij}\left[\begin{array}{c}\mathbf{c}_j^c\\1\end{array}\right]
\end{aligned}
$$
所以以下结论成立
$$
\mathbf{p}_i^c=\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^c
$$
以上推导使用了对权重$\alpha_{ij}$的重要约束条件$\sum_{j=1}^4\alpha_{ij}=1$,如果没有该约束，那么上述结论不成立。

到目前为止，我们已经根据世界坐标系下3D点$\mathbf{p}_i^w$ 求出了世界坐标系下的4个控制点$\mathbf{c}_j^w,j=1,\cdots,4$, 以及每个3D点对应的控制点系数$\alpha_{ij}$,前面说过，**同一个3D点在世界坐标系下对应控制点的系数$\alpha_{ij}$ 和其在相机坐标系下对应控制点的系数相同**。所以如果我们能把4个控制点在相机坐标系下的坐标
$\mathbf{c}_j^w,j=1,\cdots,4$求出来，就可以得到世界坐标系下3D点在相机坐标系下的坐标$\mathbf{c}_j^c,j=1,\cdots,4$了。就可以根据ICP求解位姿了。



**透视投影关系构建约束**





记$w_i$ 为投影尺度系数，$K$为相机内参矩阵，$\mathbf{u}_i$ 为相机坐标系下3D参考点 $\mathbf{p}_i^c$对应的2D投影坐标，根据i相机投影原理可得
$$
w_i\left[\begin{array}{c}\mathbf{u}_i\\1\end{array}\right]=\mathbf{K}\mathbf{p}_i^c=\mathbf{K}\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^c
$$

记控制点$\mathbf{c}_j^c$坐标为$[x_j^c,y_j^c,z_j^c]^T$,$f_u,f_v$ 是焦距，$u_c,v_c$ 是主点坐标，上式可以化为

$$
w_i\begin{bmatrix}u_i\\v_i\\1\end{bmatrix}=\begin{bmatrix}f_u&0&u_c\\0&f_v&v_c\\0&0&1\end{bmatrix}\sum_{j=1}^4\alpha_{ij}\begin{bmatrix}x_j^c\\y_j^c\\z_j^c\end{bmatrix}
$$


根据最后一行可以推出
$$
w_i=\sum_{j=1}^4\alpha_{ij}z_j^c,\quad i=1,\ldots,n
$$
消去最后一行，我们把上面矩阵展开写成等式右边为0的表达式，所以实际上每个点对可以得到2个方程
$$
\sum_{j=1}^4\alpha_{ij}f_ux_j^c+\alpha_{ij}\left(u_c-u_i\right)z_j^c=0\\\sum_{j=1}^4\alpha_{ij}f_vy_j^c+\alpha_{ij}\left(v_c-v_i\right)z_j^c=0
$$
这里的待求的未知数是12个相机坐标系下控制点坐标 $\left\{\left(x_j^c,y_j^c,z_j^c\right)\right\},j=1,\ldots,4,$我们把 个匹配点

对全部展开，再写成矩阵的形式
$$
\begin{aligned}\begin{bmatrix}\alpha_{11}f_u&0&\alpha_{11}\left(u_c-u_1\right)&\cdots&\alpha_{14}f_u&0&\alpha_{14}\left(u_c-u_1\right)\\0&\alpha_{11}f_v&\alpha_{11}\left(v_c-v_1\right)&\cdots&0&\alpha_{14}f_v&\alpha_{14}\left(v_c-v_1\right)\\\vdots\\\alpha_{i1}f_u&0&\alpha_{i1}\left(u_c-u_i\right)&\cdots&\alpha_{i4}f_u&0&\alpha_{i4}\left(u_c-u_i\right)\\0&\alpha_{i1}f_v&\alpha_{i1}\left(v_c-v_i\right)&\cdots&0&\alpha_{i4}f_v&\alpha_{i4}\left(v_c-v_i\right)\\\vdots\\\alpha_{n1}f_u&0&\alpha_{n1}\left(u_c-u_n\right)&\cdots&\alpha_{n4}f_u&0&\alpha_{n4}\left(u_c-u_n\right)\\0&\alpha_{n1}f_v&\alpha_{n1}\left(v_c-v_n\right)&\cdots&0&\alpha_{n4}f_v&\alpha_{n4}\left(v_c-v_n\right)\end{bmatrix}\begin{bmatrix}x_1^c\\y_1^c\\z_1^c\\x_2^c\\y_2^c\\z_2^c\\x_3^c\\y_3^c\\z_3^c\\x_4^c\\y_4^c\\z_4^c\end{bmatrix}=0\end{aligned}
$$
其中，$i=1,\ldots,n$表示点对的数目，我们记左边第1个矩阵为$\mathcal{M}$,它的大小为$2n\times12$,第2个矩阵x是待求的未知量组成的矩阵，它的大小为$12\times1$ ,则上式可以写为
$$
\mathbf{M}\mathbf{x}=\mathbf{0}
$$
**求解**

满足$\mathbf{Mx}=\mathbf{0}$的所有的解$\mathbf{x}$的集合就是$\mathbf{M}$的零空间。零空间 (null space),有时也称为核(kernel)。

$$
\mathbf{x}=\sum_{i=1}^N\beta_i\mathbf{v}_i
$$
其中$\mathbf{v}_i$是$\mathbf{M}$ 的零奇异值对应的右奇异向量，它的维度为 $12\times1$ 。
具体求解方法是通过构建$\mathbf{M}^T\mathbf{M}$ 组成方阵，求解其特征值和特征向量，特征值为0的特征向量即为$\mathbf{v}_i$ 。
这里需要说明的是，不论有多少点对，$\mathbf{M}^T\mathbf{M}$ 的大小永远是$12\times12$,因此计算复杂度是$O(n)$ 的。



**如何确定 $N$？**

因为每个点对可以得到2个约束方程，总共有12个未知数，所以如果有6组点对，我们就能直接求解，此时$N=1$。如果相机的焦距逐渐增大，相机模型更趋近于使用正交相机代替透视相机，零空间的自由度就会增加到$N=4$。



### 我们来看论文中这张图：

横坐标表示通过$\mathbf{M}^T\mathbf{M}$ 特征值分解得到的12个特征值的序号，纵坐标表示对应特征值的大小。

当焦距$f=100$时，我们看局部放大的右图，只有1个特征值是0，所以只要用最后一个特征向量就可以
了。
当焦距$f=10000$时，右图中可以看到第9，10,11,12个特征值都是O|,也就是说只用最后一个特征向量是
没有办法表示的，要用到最后4个特征值对应的特征向量加权才行，这就是最大$N=4$的来源。

![image-20241127221335674](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411280815808.png)

这样，实际上$N$的取值范围是$N=1,2,3,4$ ,ORB-SLAM中的方法是：四种情况我们都试一遍，找出其中使得重投影误差最小的那组解作为最佳的解。

接下来就是如何求$\left\{\beta_i\right\},i=1,\ldots,N$ 。

这里就用到我们前面说的刚体结构不变性，就是不同坐标系下两个点的相对距离是恒定的，也就是
$$
\left\|c_i^c-c_j^c\right\|^2=\left\|c_i^w-c_j^w\right\|^2
$$
后面会用到这个约束条件。

**下面分别讨论：**

**N=1的情况**

$\mathbf{x}=\beta\mathbf{v}$ ,未知数只有1个。记$\mathbf{v}^[i]$ 是$3\times1$ 的列向量，表示$\mathbf{v}$ (大小为$12\times1)$ 的第$i$个 控 制 点 $\mathbf{c} _i^c$ 所占据
的3个元素组成的子向量，例如$\mathbf{v}^{[1]}=[x_1^c,y_1^c,z_1^c]^T$ 代表$\mathbf{v}$ 的前3个元素，代入上述约束公式

$$
\begin{Vmatrix}\beta\mathbf{v}^{[i]}-\beta\mathbf{v}^{[j]}\end{Vmatrix}^2=\begin{Vmatrix}\mathbf{c}_i^w-\mathbf{c}_j^w\end{Vmatrix}^2
$$


4个控制点可以得到$\beta$ 的一个闭式解
$$
\beta=\frac{\sum_{\{i,j\}\in[1:4]}\left\|\mathbf{v}^{[i]}-\mathbf{v}^{[j]}\right\|\cdot\left\|\mathbf{c}_i^w-\mathbf{c}_j^w\right\|}{\sum_{\{i,j\}\in[1:4]}\left\|\mathbf{v}^{[i]}-\mathbf{v}^{[j]}\right\|^2}
$$
其中 $[i,j]\in[1:4]$ 表示$i,j$可以从1到4之间任意取值，也就是从4个值中任意取2个，有$C_{4}^{2}=6$种取值。
**$\mathbf{N}=2$的情况**


此时，$\mathbf{x}=\beta_1\mathbf{v}_1+\beta_2\mathbf{v}_2$,带入到 刚体结构不变性的约束方程

$$
\begin{Vmatrix}\left(\beta_1\mathbf{v}_1^{[i]}+\beta_2\mathbf{v}_2^{[i]}\right)-\left(\beta_1\mathbf{v}_1^{[j]}+\beta_2\mathbf{v}_2^{[j]}\right)\end{Vmatrix}^2=\begin{Vmatrix}\mathbf{c}_i^w-\mathbf{c}_j^w\end{Vmatrix}^2
$$

展开

$$
\begin{aligned}&\|\beta_{1}\left(\mathbf{v}_{1}^{[i]}-\mathbf{v}_{1}^{[j]}\right)+\beta_{2}\left(\mathbf{v}_{2}^{[i]}-\mathbf{v}_{2}^{[j]}\right))\|^{2}=\left\|\mathbf{c}_{i}^{w}-\mathbf{c}_{j}^{w}\right\|^{2}\\&\left[\left(\mathbf{v}_{1}^{[i]}-\mathbf{v}_{1}^{[j]}\right)^{2}\quad\left(\mathbf{v}_{1}^{[i]}-\mathbf{v}_{1}^{[j]}\right)\left(\mathbf{v}_{2}^{[i]}-\mathbf{v}_{2}^{[j]}\right)\quad\left(\mathbf{v}_{2}^{[i]}-\mathbf{v}_{2}^{[j]}\right)^{2}\right]\begin{bmatrix}\beta_{11}\\\beta_{12}\\\beta_{22}\end{bmatrix}=\left\|c_{i}^{w}-c_{j}^{w}\right\|^{2}\end{aligned}
$$
其中引入了三个中间变量如下：

$$
\beta_{11}=\beta_1^2,\beta_{22}=\beta_2^2,\beta_{12}=\beta_1\beta_2
$$
上式就变成了线性方程。总共3个未知数。根据前面描述，4个控制点可以组合构造出6个线性方程，组
成

$$
\mathrm{L}\beta=\rho
$$
其中，$\beta = [ \beta _{11}, \beta _{12}, \beta _{22}] ^T$, $\mathcal{L}$ 大小为6$\times3$,$\beta$大小为3$\times1$,$\rho$ 大小为 $6\times1$。解出 $\beta$后，可以获得两组$\beta_1,\beta_2$的解。再加上一个 条件，控制点在摄像机的前端，即$c_j^c$的$z$分量要大于0，从而$\beta_1,\beta_2$唯一确定。

**$N=$3的情况**

与 $N=2$ 的解法相同。此时，$\mathbf{x}=\beta_1\mathbf{v}_1+\beta_2\mathbf{v}_2+\beta_3\mathbf{v}_3$,带入到 刚体结构不变性的约束方程

$$
\left\|\left(\beta_1\mathbf{v}_1^{[i]}+\beta_2\mathbf{v}_2^{[i]}+\beta_3\mathbf{v}_3^{[i]}\right)-\left(\beta_1\mathbf{v}_1^{[j]}+\beta_2\mathbf{v}_2^{[j]}+\beta_3\mathbf{v}_3^{[j]}\right)\right\|^2=\left\|\mathbf{c}_i^w-\mathbf{c}_j^w\right\|^2
$$
这里的 $\beta=[\beta_{11},\beta_{12},\beta_{13},\beta_{22},\beta_{23},\beta_{33}]^T$ ,大小为$6\times1$,表示待求解未知数个数为6个，L的大小为$6\times6$ 。

$**N=4$的情况**

我们着重来推导$N=4$ 的情况，因为在ORB-SLAM里面$N$是不知道的，代码里实际就是直接采用$N=4$的情况进行计算。此时，$\mathbf{x}=\beta_1\mathbf{v}_1+\beta_2\mathbf{v}_2+\beta_3\mathbf{v}_3+\beta_4\mathbf{v}_4$,带入到 刚体结构不变性的约束方程
$$
\left\|\left(\beta_1\mathbf{v}_1^{[i]}+\beta_2\mathbf{v}_2^{[i]}+\beta_3\mathbf{v}_3^{[i]}+\beta_4\mathbf{v}_4^{[i]}\right)-\left(\beta_1\mathbf{v}_1^{[j]}+\beta_2\mathbf{v}_2^{[j]}+\beta_3\mathbf{v}_3^{[j]}+\beta_4\mathbf{v}_4^{[j]}\right)\right\|^2=\left\|\mathbf{c}_i^w-\mathbf{c}_j^w\right\|^2
$$


注意上述$\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3,\mathbf{v}_4$ 均为大小为$12\times1$的特征向量，对应$\mathbf{M}^T\mathbf{M}$最后4个零特征值。以特征向量$\mathbf{v}_1$为例，$\mathbf{v}_1^{[i]},\mathbf{v}_1^{[j]},[i,j]\in[1:4]$是上述特征向量$\mathbf{v}_1$拆成的4个大小为$3\times1$的向量，$\mathbf{v}_1^{[i]},\mathbf{v}_1^{[j]}$一共有6种不同的组合方式$[\mathbf{v}_1^{[1]},\mathbf{v}_1^{[2]}],[\mathbf{v}_1^{[1]},\mathbf{v}_1^{[3]}],[\mathbf{v}_1^{[1]},\mathbf{v}_1^{[4]}],[\mathbf{v}_1^{[2]},\mathbf{v}_1^{[3]}],[\mathbf{v}_1^{[2]},\mathbf{v}_1^{[4]}],[\mathbf{v}_1^{[3]},\mathbf{v}_1^{[4]}].$

等式左右互换进行化简：
$$
\begin{aligned}\left\|\mathbf{c}_{i}^{w}-\mathbf{c}_{j}^{w}\right\|^{2}&=\left\|\beta_{1}(\mathbf{v}_{1}^{[i]}-\mathbf{v}_{1}^{[j]})+\beta_{2}(\mathbf{v}_{2}^{[i]}-\mathbf{v}_{2}^{[j]})+\beta_{3}(\mathbf{v}_{3}^{[i]}-\mathbf{v}_{3}^{[j]})+\beta_{4}(\mathbf{v}_{4}^{[i]}-\mathbf{v}_{4}^{[j]})\right\|^{2}\\&=\left\|\beta_{1}\cdot\mathbf{d}\mathbf{v}_{1,[i,j]}+\beta_{2}\cdot\mathbf{d}\mathbf{v}_{2,[i,j]}+\beta_{3}\cdot\mathbf{d}\mathbf{v}_{3,[i,j]}+\beta_{4}\cdot\mathbf{d}\mathbf{v}_{4,[i,j]}\right\|^{2}\\&=\beta_{1}^{2}\cdot\mathbf{d}\mathbf{v}_{1,[i,j]}^{2}+2\beta_{1}\beta_{2}\cdot\mathbf{d}\mathbf{v}_{1,[i,j]}\cdot\mathbf{d}\mathbf{v}_{2,[i,j]}+\beta_{2}^{2}\cdot\mathbf{d}\mathbf{v}_{2,[i,j]}+2\beta_{1}\beta_{3}\cdot\mathbf{d}\mathbf{v}_{1,[i,j]}\cdot\mathbf{d}\mathbf{v}_{3,[i,j]}\\&+2\beta_{2}\beta_{3}\cdot\mathbf{d}\mathbf{v}_{2,[i,j]}\cdot\mathbf{d}\mathbf{v}_{3,[i,j]}+\beta_{3}^{2}\cdot\mathbf{d}\mathbf{v}_{3,[i,j]}^{2}+2\beta_{1}\beta_{4}\cdot\mathbf{d}\mathbf{v}_{1,[i,j]}\cdot\mathbf{d}\mathbf{v}_{4,[i,j]}\\&+2\beta_{2}\beta_{4}\cdot\mathbf{d}\mathbf{v}_{2,[i,j]}\cdot\mathbf{d}\mathbf{v}_{4,[i,j]}+2\beta_{3}\beta_{4}\cdot\mathbf{d}\mathbf{v}_{3,[i,j]}\cdot\mathbf{d}\mathbf{v}_{4,[i,j]}+\beta_{4}^{2}\cdot\mathbf{d}\mathbf{v}_{4,[i,j]}^{2}\end{aligned}
$$
上式等式左边记为$\rho_6\times1$,右边记为两个矩阵$\mathbf{L}_{6\times10}\cdot\beta_{10\times1}$相乘，下角标表示矩阵的维度，$[i,j]\in[1:4]$,于是我们可以得到如下结论：

$$
\begin{aligned}&\mathbf{L}_{6\times10}=\left[\mathbf{d}\mathbf{v}_{1}^{2},2\mathbf{d}\mathbf{v}_{1}\cdot\mathbf{d}\mathbf{v}_{2},\mathbf{d}\mathbf{v}_{2}^{2},2\mathbf{d}\mathbf{v}_{1}\cdot\mathbf{d}\mathbf{v}_{3},2\mathbf{d}\mathbf{v}_{2}\cdot\mathbf{d}\mathbf{v}_{3},\mathbf{d}\mathbf{v}_{3}^{2},2\mathbf{d}\mathbf{v}_{1}\cdot\mathbf{d}\mathbf{v}_{4},2\mathbf{d}\mathbf{v}_{2}\cdot\mathbf{d}\mathbf{v}_{4},2\mathbf{d}\mathbf{v}_{3}\cdot\mathbf{d}\mathbf{v}_{4},\mathbf{d}\mathbf{v}_{4}^{2}\right]_{[i,j]\in[1:4]}\\&\beta_{10\times1}=\left[\beta_{1}^{2},\beta_{1}\beta_{2},\beta_{2}^{2},\beta_{1}\beta_{3},\beta_{2}\beta_{3},\beta_{3}^{2},\beta_{1}\beta_{4},\beta_{2}\beta_{4},\beta_{3}\beta_{4},\beta_{4}^{2}\right]^{T}\\&\mathbf{L}_{6\times10}\cdot\beta_{10\times1}=\boldsymbol{\rho}_{6\times1}\end{aligned}
$$
以上$\mathbf{L}_{6\times10}$ 和 $\rho_{6\times1}$是已知的，待求解的是$\beta_{10\times1}.$


正常来说上面我们可以用SVD求解，但是这样存在问题：

1. 10个未知数，但只有6个方程，未知数个数超过了方程数目。

2. 同时求解这么多参数都是独立的，比如求解出来的第1，3个参数对应的$\beta_{1}$和$\beta_{2}$不一定和求解出来的第2个参数$\beta_1\beta_2$ 相等。所以即使求出来也很难确定最终的4个$\beta$ 值。



在ORB-SLAM2代码里作者使用了一种方法：先求初始解，然后再优化得到最优解。



**近似求β初始解**

我们来介绍ORB-SLAM2代码里的实现方法。
因为我们刚开始只要求粗糙的初始解即可，所以我们可以暴力的把$\beta_{10\times1}$ 中某些项置为0。
下面分$N$取不同值来讨论：

**$N=4$的情况**

此时待求量为：$\beta_1,\beta_2,\beta_3,\beta_4$。我们取$\beta_{10\times1}$ 中的第1，2,4,7个元素(不是必须这样取，这里只是源码中
使用的一种方法),总共4个得到如下
$$
\boldsymbol{\beta}_{4\times1}=\begin{bmatrix}\beta_1^2,\beta_1\beta_2,\beta_1\beta_3,\beta_1\beta_4\end{bmatrix}^T
$$
当然对应的$\mathbf{L}_{6\times10}$ 矩阵中每行也取第1，2,4,7个对应元素得到$\mathbf{L}_{6\times4}$,而$\rho_{6\times1}$不变。这样我们只要用SVD
求解规模更小的矩阵就行了。
$$
\mathbf{L}_{6\times4}\cdot\beta_{4\times1}=\rho_{6\times1}
$$
最后得到

$$
\beta_1=\sqrt{\beta_1^2},\quad\beta_2=\frac{\beta_1\beta_2}{\beta_1},\quad\beta_3=\frac{\beta_1\beta_3}{\beta_1},\quad\beta_4=\frac{\beta_1\beta_4}{\beta_1}
$$


**$N=3$的情况**

此时待求量为：$\beta_1,\beta_2,\beta_3$。我们取$\beta_{10\times1}$ 中的第1，2,3,4,5个元素，总共5个得到如下

$$
\boldsymbol{\beta}_{5\times1}=\begin{bmatrix}\beta_1^2,\beta_1\beta_2,\beta_2^2,\beta_1\beta_3,\beta_2\beta_3\end{bmatrix}^T
$$
当然对应的$\mathbf{L}_{6\times10}$ 矩阵中每行也取第1，2,3,4,5对应元素得到$\mathbf{L}_{6\times5}$ ,而$\rho_{6\times1}$ 不变。这样我们只要用SVD求解规模更小的矩阵就行了。

**N=2的情况**

此时待求量为：$\beta _1, \beta _2$。我们取$\beta_{10\times1}$ 中的第1，2,3个元素，总共3个得到如下

$$
\boldsymbol{\beta}_{5\times1}=\begin{bmatrix}\beta_1^2,\beta_1\beta_2,\beta_2^2\end{bmatrix}^T
$$
当然对应的$\mathbf{L}_{6\times10}$ 矩阵中每行也取第1，2,3对应元素得到$\mathbf{L}_{6\times3}$ ,而$\rho_{6\times1}$ 不变。这样我们只要用SVD求解规模更小的矩阵就行了。

$$
\mathbf{L}_{6\times3}\cdot\beta_{3\times1}=\rho_{6\times1}
$$


最后得到

$$
\beta_1=\sqrt{\beta_1^2},\quad\beta_2=\sqrt{\beta_2^2}
$$

**高斯牛顿优化**

我们的目标是优化两个坐标系下控制点间距的差，使得其误差最小，如下所示
$$
f(\boldsymbol{\beta})=\sum_{(i,j\:s.t.\:i<j)}\left(||\mathbf{c}_i^c-\mathbf{c}_j^c||^2-||\mathbf{c}_i^w-\mathbf{c}_j^w||^2\right)
$$

因为我们前面已经计算了$N=4$的情况下$\left\|c_i^c-c_j^c\right\|^2=\left\|c_i^w-c_j^w\right\|^2$的表达式为：

$$
\mathbf{L}_{6\times10}\cdot\boldsymbol{\beta}_{10\times1}=\boldsymbol{\rho}_{6\times1}
$$
我们记待优化目标$\beta$ 为：
$$
\begin{aligned}\beta_{10\times1}&=\begin{bmatrix}\beta_1^2&\beta_1\beta_2&\beta_2^2&\beta_1\beta_3&\beta_2\beta_3&\beta_3^2&\beta_1\beta_4&\beta_2\beta_4&\beta_3\beta_4&\beta_4^2\end{bmatrix}^T\\&=\begin{bmatrix}\beta_{11}&\beta_{12}&\beta_{22}&\beta_{13}&\beta_{23}&\beta_{33}&\beta_{14}&\beta_{24}&\beta_{34}&\beta_{44}\end{bmatrix}^T\end{aligned}
$$
所以上面的误差函数可以写为：

$$
f(\beta)=\mathbf{L}\beta-\rho
$$
上式两边对$\beta$求偏导，由于$\rho$ 和$\beta$ 无关，所以一阶雅克比矩阵

$$
\begin{aligned}\mathbf{J}=\frac{\partial f(\boldsymbol{\beta})}{\boldsymbol{\beta}}&=\left[\begin{array}{ccccc}\frac{\partial f(\boldsymbol{\beta})}{\partial\beta_1}&&\frac{\partial f(\boldsymbol{\beta})}{\partial\beta_2}&&\frac{\partial f(\boldsymbol{\beta})}{\partial\beta_3}&&\frac{\partial f(\boldsymbol{\beta})}{\partial\beta_4}\end{array}\right]\\&=\left[\begin{array}{cccc}\frac{\partial(\mathbf{L}\boldsymbol{\beta})}{\partial\beta_1}&&\frac{\partial(\mathbf{L}\boldsymbol{\beta})}{\partial\beta_2}&&\frac{\partial(\mathbf{L}\boldsymbol{\beta})}{\partial\beta_3}&&\frac{\partial(\mathbf{L}\boldsymbol{\beta})}{\partial\beta_4}\end{array}\right]\end{aligned}
$$
前面我们已经知道L 的维度是 $6\times10$,$\beta$ 的维度是 $10\times1$,我们以L第一行$\mathbf{L}^1$为例来推导

$$
\begin{aligned}\mathbf{L}^{1}\beta&=\begin{bmatrix}L_{1}^{1}&L_{2}^{1}&L_{3}^{1}&L_{4}^{1}&L_{5}^{1}&L_{6}^{1}&L_{7}^{1}&L_{8}^{1}&L_{9}^{1}&L_{10}^{1}\end{bmatrix}\begin{bmatrix}\beta_{11}\\\beta_{12}\\\beta_{22}\\\beta_{13}\\\beta_{23}\\\beta_{33}\\\beta_{14}\\\beta_{24}\\\beta_{34}\\\beta_{34}\end{bmatrix}\\&=L_{1}^{1}\beta_{11}+L_{2}^{1}\beta_{12}+L_{3}^{1}\beta_{22}+L_{4}^{1}\beta_{13}+L_{5}^{1}\beta_{23}+L_{6}^{1}\beta_{33}+L_{7}^{1}\beta_{14}+L_{8}^{1}\beta_{24}+L_{9}^{1}\beta_{34}+L_{10}^{1}\beta_{44}\end{aligned}
$$
分别求偏导后得到
$$
\begin{gathered}
\frac{\partial(\mathbf{L}_{1}\beta)}{\partial\beta_{1}} =2L_{1}^{1}\beta_{1}+L_{2}^{1}\beta_{2}+L_{4}^{1}\beta_{3}+L_{7}^{1}\beta_{4} \\
\frac{\partial(\mathbf{L}_{1}\beta)}{\partial\beta_{2}} =L_{2}^{1}\beta_{1}+2L_{3}^{1}\beta_{2}+L_{5}^{1}\beta_{3}+L_{8}^{1}\beta_{4} \\
\frac{\partial(\mathbf{L}_{1}\beta)}{\partial\beta_{3}} =L_{4}^{1}\beta_{1}+L_{5}^{1}\beta_{2}+2L_{6}^{1}\beta_{3}+L_{9}^{1}\beta_{4} \\
\frac{\partial(\mathbf{L}_{1}\beta)}{\partial\beta_{4}} =L_7^1\beta_1+L_8^1\beta_2+L_9^1\beta_3+2L_{10}^1\beta_4 
\end{gathered}
$$
高斯牛顿法的增量方程：
$$
\begin{aligned}&\mathbf{H}\Delta\mathbf{x}=\mathbf{g}\\&\mathbf{J}^{T}\mathbf{J}\Delta\mathbf{x}=-\mathbf{J}^{T}f(x)\\&\mathbf{J}\Delta\mathbf{x}=-f(x)\end{aligned}
$$
对应非齐次项$-f(\boldsymbol{\beta})=\boldsymbol{\rho}-\mathbf{L}\boldsymbol{\beta}$

**ICP** **求解位姿**

1. 记3D点在世界坐标系下的坐标及对应相机坐标系下的坐标分别是$\mathbf{p}_i^w,\mathbf{p}_i^c,i=1,\ldots,n$
2. 首先分别计算它们的质心：

$$\mathbf{p}_{0}^{w}=\frac1n\sum_{i=1}^n\mathbf{p}_i^w\\\mathbf{p}_{0}^{c}=\frac1n\sum_{i=1}^n\mathbf{p}_i^c$$

3. 计算 $\left\{\mathbf{p}_i^w\right\}_{i=1,\cdots,n}$ 去质心$\mathbf{p}_0^w$后的矩阵A

$$A=\begin{bmatrix}\mathbf{p}_1^{w^T}-\mathbf{p}_0^{w^T}\\\cdots\\\mathbf{p}_n^{w^T}-\mathbf{p}_0^{w^T}\end{bmatrix}$$

4. 计算$\left\{\mathbf{p}_i^c\right\}_{i=1,\cdots,n}$去质心$\mathbf{p}_0^c$后的矩阵$B:$

$$
\mathbf{p}_0^c=\dfrac{1}{n}\sum_{i=1}^n\mathbf{p}_i^c
$$



$$
B=\begin{bmatrix}\mathbf{p}_1^{c^T}-\mathbf{p}_0^{c^T}\\\cdots\\\mathbf{p}_n^{c^T}-\mathbf{p}_0^{c^T}\end{bmatrix}
$$

5. 得到矩阵$H:$

$$
H=B^TA
$$

6. 计算 的SVD分解 :

$$
H=U\Sigma V^T
$$

7. 计算位姿中的旋转$R$

$$
R=UV^T
$$

8. 计算位姿中的平移 :

$$
\mathbf{t}=\mathbf{p}_0^c-R\mathbf{p}_0^w
$$

**总结**

1. 根据计算出来的$\beta,\mathbf{v}$ 得到相机坐标系下的4个控制点坐标$\left\{\mathbf{c}_j=\left(x_j^c,y_j^c,z_j^c\right)\right\},j=1,\ldots,4$
$$
  \mathbf{x}=\sum_{i=1}^N\beta_i\mathbf{v}_i
$$


2. 根据相机坐标系下控制点坐标$\mathbf{c}_j$ 和控制点系数 $\alpha_{ij}$ (通过世界坐标系下3D点计算得到),得到相机坐标系下3D点坐标$\mathbf{p}_i^c$

$$
\mathbf{p}_i^c=\sum_{j=1}^4\alpha_{ij}\mathbf{c}_j^c
$$

3.现在已经有3D点在世界坐标系下的坐标$\mathbf{p}_i^w$以及对应相机坐标系下的坐标$\mathbf{p}_i^c$,用$ICP$求解$R,t$ 即可。



参考：

Lepetit V , Fua M N . EPnP: An Accurate O(n) Solution to the PnP Problem[J]. International Journal of Computer Vision, 2009.

小葡萄 https://zhuanlan.zhihu.com/p/59070440

Jessie https://blog.csdn.net/jessecw79/article/details/82945918

代码：https://github.com/cvlab-epfl/EPnP

### 局部地图跟踪

对应函数tracking::TrackLocalMap 

**应用场景：**前面3种跟踪方式得到当前帧地图点后的后处理，每次跟踪都使用。前提是必须知道当前帧的

位姿和地图点（尽管不准确），利用到了当前帧的两级共视关键帧的信息，使得位姿更加准确。

**具体步骤**：

- 首先根据前面得到的当前帧的地图点来找能观测到当前帧的一级共视关键帧，将这些一级共视关键帧的二级关键共视帧、子关键帧、父关键帧一起作为局部关键帧;

- 取出上述局部关键帧中所有的地图点作为局部地图点；

- 将局部地图点投影到当前帧，去掉不在视野内的无效的地图点，剩下的局部地图点投影到当前帧进行匹配（函数SearchByProjection）

- 对匹配结果再做BA优化（仅优化位姿）


**当前帧：**mCurrentFrame（当前帧是普通帧）

**参考关键帧****:** 与当前帧共视程度最高的关键帧作为参考关键帧，mCurrentFrame.mpReferenceKF

在KeyFrame::UpdateConnections() 里确定关键帧的父子关系（当前帧必须是关键帧）

**父关键帧：**和当前关键帧共视程度最高的关键帧

**子关键帧：**是上述父关键帧的子关键帧

**mvpLocalMapPoints** 示意图

就是下图中红色地图点，用来在TrackLocalMap里跟踪当前帧

![image-20241128084046180](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059561.png)

**mvpLocalKeyFrames示意图**

![image-20241128084848991](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059562.png)

## 局部建图线程



### 删除地图点和关键帧

删除地图点，在MapPoint::SetBadFlag() 被标记为需要删除

删除关键帧 KeyFrame::SetBadFlag()，会更新spanning tree



![image-20241128085054287](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059563.png)

### 关键帧之间新生成地图点

对应函数 CreateNewMapPoints()

![image-20241128085111807](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059564.png)

已知位姿计算两帧之间的F矩阵，对应函数ComputeF12
$$
\begin{gathered}
T_{1w}={\left[\begin{array}{cc}{R_{1w}}&{t_{1w}}\\{0^{T}}&{1}\end{array}\right]} \\
T_{2w}={\left[\begin{array}{cc}{R_{2w}}&{t_{2w}}\\{0^{T}}&{1}\end{array}\right]} \\
T_{12}=T_{1w}{T_{2w}}^{-1}=\left[\begin{array}{cc}{R_{1w}}&{t_{1w}}\\{0^{T}}&{1}\end{array}\right]\left[\begin{array}{cc}{R_{2w}^{T}}&{-R_{2w}^{T}t_{2w}}\\{0^{T}}&{1}\end{array}\right] \\
=\begin{bmatrix}R_{1w}R_{2w}^T&-R_{1w}R_{2w}^Tt_{2w}+t_{1w}\\0^T&1\end{bmatrix} 
\end{gathered}
$$


### 邻域搜索

对应函数SearchInNeighbors()

![image-20241128085141295](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059565.png)

## 闭环检测及矫正线程

### 闭环示意图

![image-20241128085208961](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059566.png)

### 检测闭环候选帧

对应函数 DetectLoopCandidates

![image-20241128085240600](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059567.png)

**闭环连续性检测原理示例**

对应函数 DetectLoop()

![image-20241128085248340](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059568.png)

### Sim(3)算法原理详解

**为什么需要计算Sim3？**

均摊误差、自然过渡。

​	当前关键帧、闭环关键帧之间其实是隔了很多帧的，他们的pose都是由邻近信息得到的，经过了很

久后，很可能有累计的位姿误差（单目尺度漂移所以有尺度误差，双目或RGBD认为没有尺度误差），

如果你闭环时直接根据各自的位姿强制把相隔了很久的两个位姿接上，很可能会导致明显的错位。

​	我们用扫描人脸来做个比喻，可以认为开始扫描的脸和最后闭环的脸之间你是直接强制缝合，那很

可能两张脸接不上，五官错位。

![image-20241128085328549](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059569.png)

​	用SIM3就是把相隔很久的两个要缝合的关键帧（及其周围关键帧）重新建立连接，做一个软过渡，

尽可能的将累计误差分摊到要缝合的关键帧（及其周围关键帧），也就是闭环调整。相当于我们把要缝

合的人脸两侧都做了一定的调整使得缝合不那么生硬，起码 看起来更协调。

**什么是SIM3变换？**

Sim(3) 表示三维空间的相似变换(**Similarity Transformation**)。计算Sim3 实际就是计算这三个参数：

旋转$R$ 、平移$t$ 、尺度因子$s$ 。

理论来说计算Sim3需要3对不共线的点对即可求解。

为什么三对不共线点就可以求解？

我们来感性的理解一下，我们有三对匹配的不共线三维点可以构成两个三角形。我们根据三角形各自的法向量可以得到他们之间的旋转，通过相似三角形面积能够得到尺度，用前面得到的旋转和尺度可以把两个三角形平行放置，通过计算距离可以得到平移。



以上是直观感性的理解，实际在计算的时候需要有严格的数学推导。我们这里使用的方法是来自Berthold K. P. Horn在1987年发表的论文 "*Closed-form solution of absolute orientation using unit quaternions"。该文提出了用三维匹配点构建优化方程，不需要迭代，直接用闭式解求出了两个坐标系之间的旋转、平移、尺度。该方法的优点非常明显：

1. 给定两个坐标系下的至少3个匹配三维点，只需一步即可求得变换关系，不需要迭代，速度很快。

2. 因为不是数值解，不需要像迭代方法那样需要找一个好的初始解。闭式解可以直接求得比较精确的结果。

> [!NOTE]
>
> 数值解(numerical solution)是在特定条件下通过近似计算得出来的一个数值，比如数值逼近。
>
> 闭式解也称为解析解，就是给出解的具体函数形式，从解的表达式中就可以算出任何对应值。



实际上，在SLAM问题中，我们通常能够得到大于3个的三维匹配点，该论文推导了该情况下最小二乘得

到最优解的方法。

另外，论文中利用单位四元数表示旋转，简化了求解的推导。

**目的**

已知至少三个匹配的不共线三维点对，求他们之间的相对旋转、平移、尺度因子。

**热身：3对点计算旋转可以吗？**

假设坐标系1下有三个不共线三维点$P_1,P_2,P_3$ ,他们分别和坐标系2下的三个不共线三维点$Q_1,Q_2,Q_3——$匹配。

![image-20241128085705487](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059570.png)



首先，我们根据坐标系1下的三个不共线三维点来构造一个新的坐标系。
沿着$x$轴上的单位向量$\hat{x}$

$$
\begin{aligned}&x=P_2-P_1\\&\hat{x}=\frac x{||x||}\end{aligned}
$$
沿着 $y$ 轴的单位向量$\hat{y}$

$$
\begin{aligned}&y=\overrightarrow{AP_3}\\&=\overrightarrow{P_1P_3}-\overrightarrow{P_1A}\\&=(P_3-P_1)-[(P_3-P_1)\hat{x}]\hat{x}\\&\hat{y}=\frac y{||y||}\end{aligned}
$$
沿着$z$轴的单位向量$\hat{z}$

$$
\hat{z}=\hat{x}\times\hat{y}
$$
同理，我们对于坐标系2下的$Q_1,Q_2,Q_3$也可以得到沿着3个坐标轴的单位向量$\hat{x^{\prime}},\hat{y^{\prime}},\hat{z^{\prime}}$
我们现在要 计算 坐标系 1 到坐标系2的旋转，记坐标系单位向量构成的基底矩阵为
$$
\begin{aligned}&M_{1}=[\hat{x},\hat{y},\hat{z}]\\&M_{2}=[\hat{x'},\hat{y'},\hat{z'}]\end{aligned}
$$
假设坐标系1下有一个向量$v_1$ ,它在坐标系2下记为$v_2$ ,因为向量本身没有变化，根据坐标系定义有

$$
\begin{aligned}M_1v_1&=M_2v_2\\v_2&=M_2^TM_1v_1\end{aligned}
$$

那么从坐标系1到坐标系2的旋转就是

$$
R=M_2^TM_1
$$


看起来好像没什么问题，但是实际上我们不会这样使用，因为存在如下问题：

1. 这个旋转的结果和选择点的顺序关系密切，我们分别让不同的点做坐标系原点，得到的结果不同。

2. 这种情况不适用于匹配点大于3个的情况。因此实际上我们不会使用以上方法。我们通常能够拿到远大于3个的三维匹配点对，我们会使用最小二乘法来得到更稳定、更精确的结果。

下面进入正文。

**计算SIM3平移**

假设我们得到了$n>3$组匹配的三维点，分别记为$\{P_i\},\{Q_i\}$ ,其中$i=1,\ldots,n$我们的目的是对于
每对匹配点，找到如下的变换关系：

$$
Q_i=sRP_i+t
$$
其中$s$是尺度因子，$R$是旋转，$t$是平移。
如果数据是没有任何噪音的理想数据，理论上我们可以找到满足上述关系的尺度因子、旋转和平移。但实际上数据是不可避免会有噪音和误差，所以我们转换思路，定义一个误差$e_i$ ,我们的**目的就是寻找合适的尺度因子、旋转和平移，使得它在所有数据上的误差最小。**



$$
\begin{aligned}e_{i}&=Q_i-sRP_i-t\\\min_{s,R,t}\sum_{i=1}^n||e_i||^2&=\min_{s,R,t}\sum_{i=1}^n||Q_i-sRP_i-t||^2\end{aligned}
$$
在开始求解之前，我们先定义两个三维点集合中所有三维点的均值(或者称为质心、重心)

$$
\bar{P}=\frac1n\sum_{i=1}^nP_i\\\bar{Q}=\frac1n\sum_{i=1}^nQ_i
$$


我们对每个三维点 $P_i,Q_i$分别减去均值，得到去中心化后的坐标 $P_i^{\prime},Q_i^{\prime}$,则有

$$
\begin{aligned}P_{i}^{\prime}&=P_i-\bar{P}\\Q_{i}^{\prime}&=Q_i-\bar{Q}\\\sum_{i=1}^nP_i^{\prime}&=\sum_{i=1}^n\left(P_i-\bar{P}\right)=\sum_{i=1}^nP_i-n\bar{P}=0\\\sum_{i=1}^nQ_i^{\prime}&=\sum_{i=1}^n\left(Q_i-\bar{Q}\right)=\sum_{i=1}^nQ_i-n\bar{Q}=0\end{aligned}
$$


上面的结论很重要，我们在后面推导的时候要使用。

下面开始推导我们的误差方程：
$$
\begin{aligned}\sum_{i=1}^n\left|\left|e_i\right|\right|^2&=\sum_{i=1}^n\left|\left|Q_i-sRP_i-t\right|\right|^2\\&=\sum_{i=1}^n\left|\left|Q_i^{\prime}+\bar{Q}-sRP_i^{\prime}-sR\bar{P}-t\right|\right|^2\\&=\sum_{i=1}^n\left|\left|\left(Q_i^{\prime}-sRP_i^{\prime}\right)+\underbrace{\left(\bar{Q}-sR\bar{P}-t\right)}_{t_0}\right|\right|^2\\&=\sum_{i=1}^n\left|\left|\left(Q_i^{\prime}-sRP_i^{\prime}\right)\right|\right|^2+2t_0\sum_{i=1}^n(Q_i^{\prime}-sRP_i^{\prime})+n||t_0||^2\end{aligned}
$$
为了推导不显得那样臃肿，其中我们简记

$$
t_0=\bar{Q}-sR\bar{P}-t
$$
根据前面的推导可得等式右边中间项

$$
\sum_{i=1}^n(Q_i'-sRP_i')=\sum_{i=1}^nQ_i'-sR\sum_{i=1}^nP_i'=0
$$
这样我们前面的误差方程可以化简为：

$$
\sum_{i=1}^n||e_i||^2=\sum_{i=1}^n||(Q_i'-sRP_i')||^2+n||t_0||^2
$$
等式右边的两项都是大于等于0的平方项，并且只有第二项里的$t_{0}$和我们要求的平移$t$有关，所以当$t_0=0$时，我们可以得到平移的最优解 $t^*$

$$
\begin{aligned}&t_0=\bar{Q}-sR\bar{P}-t=0\\&t^{*}=\bar{Q}-sR\bar{P}\end{aligned}
$$
为了推导不显得那样臃肿，其中我们简记
$$
t_0=\bar{Q}-sR\bar{P}-t
$$
根据前面的推导可得等式右边中间项

$$
\sum_{i=1}^n(Q_i'-sRP_i')=\sum_{i=1}^nQ_i'-sR\sum_{i=1}^nP_i'=0
$$
这样我们前面的误差方程可以化简为：

$$
\sum_{i=1}^n\left|\left|e_i\right|\right|^2=\sum_{i=1}^n\left|\left|\left(Q_i'-sRP_i'\right)\right|\right|^2+n|\left|t_0\right||^2
$$
等式右边的两项都是大于等于0的平方项，并且只有第二项里的$t_{0}$和我们要求的平移$t$有关，所以当
$t_0=0$时，我们可以得到平移的最优解 $t^*$

$$
\begin{array}{l}t_0=\bar Q-sR\bar P-t=0\\t^*=\bar Q-sR\bar P\end{array}
$$
也就是说我们知道了旋转$R$和尺度 $s$ 就能根据三维点均值做差得到平移 $t$ 了。注意这里平移的方向是$\{P_i\}\to\{Q_i\}$



**计算SIM3尺度因子**

我们的误差函数也可以进一步简化为：

$$
\begin{aligned}\sum_{i=1}^{n}||e_{i}||^{2}&=\sum_{i=1}^n||Q_i^{\prime}-sRP_i^{\prime}||^2\\&=\sum_{i=1}^n||Q_i^{\prime}||^2-2s\sum_{i=1}^nQ_i^{\prime}RP_i^{\prime}+s^2\sum_{i=1}^n||RP_i^{\prime}||^2\end{aligned}
$$
由于向量的模长不受旋转的影响，所以$||RP_i^{\prime}||^2=||P_i^{\prime}||^2$

为了后续更加清晰的表示，我们用简单的符号代替上述式子里的部分内容，所以有

$$
\begin{aligned}\sum_{i=1}^{n}||e_{i}||^{2}&=\underbrace{\sum_{i=1}^n||Q_i^{\prime}||^2}_{S_Q}-2s\underbrace{\sum_{i=1}^nQ_i^{\prime}RP_i^{\prime}}_{D}+s^2\underbrace{\sum_{i=1}^n||P_i^{\prime}||^2}_{S_P}\\&=S_Q-2sD+s^2S_P\end{aligned}
$$
由于$R$是已知的，我们很容易看出来上面是一个以$s$为自变量的一元二次方程，要使得该方程误差最
小，我们可以得到此时尺度$s$ 的取值：

$$
s=\frac{D}{S_P}=\frac{\sum_{i=1}^nQ_i'RP_i'}{\sum_{i=1}^n\left|\left|P_i'\right|\right|^2}
$$
ORB--SLAM2和3里都是使用上述公式求尺度。注意这里尺度的方向是$\{P_i\}\to\{Q_i\}$ 。
但是，到这里还存在一个问题，我们对$P,Q$ 做个调换后得到

$$
\frac{\sum_{i=1}^nP_i'R^TQ_i'}{\sum_{i=1}^n\left\|Q_i'\right\|^2}\neq\frac1s
$$

> [!NOTE]
>
> 我们看到尺度并不具备对称性，也就是从$\{P_i\}\to\{Q_i\}$得到的尺度并不等于从$\{Q_i\}\to\{P_i\}$得到的

但是，到这里还存在一个问题，我们对$P,Q$ 做个调换后得到
$$
\frac{\sum_{i=1}^nP_i'R^TQ_i'}{\sum_{i=1}^n\left\|Q_i'\right\|^2}\neq\frac1s
$$

我们看到尺度并不具备对称性，也就是从$\{P_i\}\to\{Q_i\}$得到的尺度并不等于从$\{Q_i\}\to\{P_i\}$得到的尺度的倒数。这也说明我们前面方法得到的尺度并不稳定。所以需要重新构造误差函数，使得我们得到的尺度是对称的、稳定的。



当然我们不用自己绞尽脑汁去构造，直接搬运论文里大佬的构造方法即可:
$$
\begin{aligned}
\sum_{i=1}^{n}||e_{i}||^{2}& =\sum_{i=1}^{n}||\frac{1}{\sqrt{s}}Q_{i}^{\prime}-\sqrt{s}RP_{i}^{\prime}||^{2} \\
&=\frac{1}{s}\underbrace{\sum_{i=1}^{n}\left|\left|Q_{i}^{\prime}\right|\right|^{2}}_{S_{Q}}-\underbrace{2\sum_{i=1}^{n}Q_{i}^{\prime}RP_{i}^{\prime}}_{D}+\underbrace{s\sum_{i=1}^{n}\left|\left|RP_{i}^{\prime}\right|\right|^{2}}_{S_{P}} \\
&=\frac1sS_Q-2D+sS_P \\
&=(\sqrt{sS_{P}}-\sqrt{\frac{S_{Q}}{s}})^{2}+2(S_{P}S_{Q}-D)
\end{aligned}
$$

> [!NOTE]
>
> 论文中此处有误，但最后结果是对的

上面等式右边第一项只和尺度$s$有关的平方项，第二项和$s$无关，但和旋转$R$有关，因此令第一项为0,我们就能得到最佳的尺度$s^*$
$$
s^*=\sqrt{\frac{S_Q}{S_P}}=\sqrt{\frac{\sum_{i=1}^n\left|\left|Q_i'\right|\right|^2}{\sum_{i=1}^n\left|\left|P_i'\right|\right|^2}}
$$


同时，第二项里的$S_P,S_Q$都是平方项，所以令第二项里的$D=\sum_{i=1}^nQ_i^{\prime}RP_i^{\prime}$最大，可以使得剩下的误差函数最小。

**这里我们总结下对称形式的优势：**

1.  使得尺度的解和旋转、平移都无关

2. 反过来，旋转的确定不受数据选择不同的影响

我们直观理解一下，尺度就是三维点到各自均值中心的距离之和。

**计算旋转**

下面我们考虑用四元数来代替矩阵来表达旋转。

> [!NOTE]
>
> 为什么用四元数而不是矩阵表达旋转？
>
> 1. 因为直接使用矩阵必须要保证矩阵的正交性等约束，这个约束太强了，会带来很多困难。
>
> 2. 四元数只需要保证模值为1的约束，简单很多，方便推导。

开始之前，先来看下四元数的性质。大家可以自行证明。

**性质1、用四元数来对三维点进行旋转**
$$
\dot{p'}=\dot{q}\dot{p}\dot{q}^{-1}=\dot{q}\dot{p}\dot{q}^*
$$
假设空间三维点$P=[x,y,z]$,用一个虚四元数来表示为$\dot{p}=[0,x,y,z]^T$ 。旋转用一个单位四元数$\dot{q}$来表示，则$\dot{p}$旋转后的三维点用四元数表示为： 四元数 $\dot{p^{\prime}}$ 的虚部取出即为旋转后的坐标。其中$\dot{q}^*$ 表示取$\dot{q}$ 的共轭
**性质2：**
三个四元数满足如下条件。直接相乘的形式，表示四元数乘法，中间的·表示向量点乘
$$
\dot{p}\cdot(\dot{r}\dot{q}^*)=(\dot{p}\dot{q})\cdot\dot{r}
$$
性质3：

假设四元数$\dot{r}=[r_0,r_x,r_y,r_z]$ ,则有

$$
\begin{gathered}\dot{r}\dot{q}=\begin{bmatrix}r_0&-r_x&-r_y&-r_z\\r_x&r_0&-r_z&r_y\\r_y&r_z&r_0&-r_x\\r_z&-r_y&r_x&r_0\end{bmatrix}\dot{q}=\mathbb{R}\dot{q}\\\dot{q}\dot{r}=\begin{bmatrix}r_0&-r_x&-r_y&-r_z\\r_x&r_0&r_z&-r_y\\r_y&-r_z&r_0&r_x\\r_z&r_y&-r_x&r_0\end{bmatrix}\dot{q}=\overline{\mathbb{R}}\dot{q}\end{gathered}
$$
其中$\mathbb{R},\overline{\mathbb{R}}$都是$4\times4$的对称矩阵。

下面进入正题。

利用前面的性质，我们现在的的代价函数可以做如下变换

$$
\begin{aligned}\sum_{i=1}^nQ_i^{\prime}RP_i^{\prime}&=\sum_{i=1}^n(\dot{Q}_i^{\prime})\cdot(\dot{q}\:\dot{P}_i^{\prime}\dot{q}^{*})\\&=\sum_{i=1}^n(\dot{Q}_i^{\prime}\dot{q})\cdot(\dot{q}\:\dot{P}_i^{\prime})\\&=\sum_{i=1}^n(\mathbb{R}_{\mathbb{Q},\mathrm{i}}\dot{q})\cdot(\overline{\mathbb{R}_{\mathbb{P},\mathrm{i}}}\dot{q})\\&=\sum_{i=1}^n\dot{q}^T\mathbb{R}_{\mathbb{Q},\mathrm{i}}^\mathrm{T}\overline{\mathbb{R}_{\mathbb{P},\mathrm{i}}}\dot{q}\\&=\dot{q}^T(\sum_{i=1}^n\mathbb{R}_{\mathbb{Q},\mathrm{i}}^\mathrm{T}\overline{\mathbb{R}_{\mathbb{P},\mathrm{i}}})\dot{q}\\&=\dot{q}^TN\dot{q}\end{aligned}
$$
其中：

$$
\begin{aligned}&Q_{i}^{\prime}=[Q_{i,x}^{\prime},Q_{i,y}^{\prime},Q_{i,z}^{\prime}]^{T}\\&P_{i}^{\prime}=[P_{i,x}^{\prime},P_{i,y}^{\prime},P_{i,z}^{\prime}]^{T}\\&\dot{Q}_{i}^{\prime}\dot{q}=\begin{bmatrix}0&-Q_{i,x}^{\prime}&-Q_{i,y}^{\prime}&-Q_{i,z}^{\prime}\\Q_{i,x}^{\prime}&0&-Q_{i,z}^{\prime}&Q_{i,y}^{\prime}\\Q_{i,y}^{\prime}&Q_{i,z}^{\prime}&0&-Q_{i,x}^{\prime}\\Q_{i,z}^{\prime}&-Q_{i,y}^{\prime}&Q_{i,x}^{\prime}&0\end{bmatrix}\dot{q}=\mathbb{R}_{\mathbb{Q},\mathbb{i}}\dot{q}\\&\dot{q}\:\dot{P}_{i}^{\prime}=\begin{bmatrix}0&-P_{i,x}^{\prime}&-P_{i,y}^{\prime}&-,P_{i,z}^{\prime}\\P_{i,x}^{\prime}&0&,P_{i,z}^{\prime}&-P_{i,y}^{\prime}\\P_{i,y}^{\prime}&-,P_{i,z}^{\prime}&0&P_{i,x}^{\prime}\\P_{i,z}^{\prime}&D_{i,z}^{\prime}&P_{i,x}^{\prime}&0\end{bmatrix}\dot{q}=\overline{\mathbb{R}_{\mathbb{P},\mathbb{i}}}\dot{q}\end{aligned}
$$


我们定义

$$
\begin{aligned}\text{M}&=\sum_{i=1}^nP_i^{\prime}Q_i^T\\&=\begin{bmatrix}S_{xx}&S_{xy}&S_{xz}\\S_{yx}&S_{yy}&S_{yz}\\S_{zx}&S_{zy}&S_{zz}\end{bmatrix}\end{aligned}
$$
其中

$$
S_{xx}=\sum_{i=1}^nP_{i,x}Q_{i,x}\\S_{xy}=\sum_{i=1}^nP_{i,x}Q_{i,y}
$$
引入$M$ 是为了方便用其元素来表示$N$,我们将上面的结果代入整理，则有：

$$
\begin{aligned}\text{N}&=\sum_{i=1}^n\mathbb{R}_{\mathbb{Q},i}^\mathbb{T}\overline{\mathbb{R}_{\mathbb{P},i}}\\&=\begin{bmatrix}(S_{xx}+S_{yy}+S_{zz})&S_{yz}-S_{zy}&S_{zx}-S_{xz}&S_{xy}-S_{yx}\\S_{yz}-S_{zy}&(S_{xx}-S_{yy}-S_{zz})&S_{xy}+S_{yx}&S_{zx}+S_{zz}\\S_{zx}-S_{xz}&S_{xy}+S_{yx}&(-S_{xx}+S_{yy}-S_{zz})&S_{yz}+S_{zy}\\S_{xy}-S_{yx}&S_{zx}+S_{xz}&S_{yz}+S_{zy}&(-S_{xx}-S_{yy}+S_{zz})\end{bmatrix}\end{aligned}
$$
然后我们对$N$ 进行特征值分解，求得最大特征值对应的特征向量就是待求的用四元数表示的旋转，注意这里旋转的方向是$\{P_i\}\to\{Q_i\}$。

> [!NOTE]
>
> 论文中写反了。

至此，我们就得到Sim3 的三个参数：旋转$R$、平移$t$、尺度因子$s$。
我们总结一下计算 Sim3 的步骤。

1. 先计算旋转$R$。
   具体来说，先构建 M 矩阵

$$
\begin{aligned}\text{M}&=\sum_{i=1}^nP_i^{\prime}Q_i^{\prime T}\\&=\begin{bmatrix}S_{xx}&S_{xy}&S_{xz}\\S_{yx}&S_{yy}&S_{yz}\\S_{zx}&S_{zy}&S_{zz}\end{bmatrix}\end{aligned}
$$

然后得到矩阵 N,对$N$ 进行特征值分解，求得最大特征值对应的特征向量就是待求的用四元数表示的旋

转$R$。注意这里旋转的方向是$\{P_i\}\to\{Q_i\}.$

2. 根据上面计算的旋转$R$来计算尺度$s$。
   具体来说，可以使用以下两种方法来计算，第一种是具有对称性的尺度(推荐)

$$
s=\sqrt{\frac{S_Q}{S_P}}=\sqrt{\frac{\sum_{i=1}^n\left|\left|Q_i'\right|\right|^2}{\sum_{i=1}^n\left|\left|P_i'\right|\right|^2}}
$$

第二种是不具有对称性的尺度 (ORBSLAM使用)

$$
s=\frac{D}{S_P}=\frac{\sum_{i=1}^nQ_i'RP_i'}{\sum_{i=1}^n\left|\left|P_i'\right|\right|^2}
$$

3. 根据旋转$R$和尺度$s$计算平移位移$t$

$$
t=\bar Q-sR\bar P
$$

以上就是Sim3 推导过程。如有不理解强烈建议看原论文。





**迭代次数的估计**

$\epsilon$ 表示在$N$对匹配点中，随便抽取一对点是内点的概率。为了计算Sim3，我们需要从$N$对点里取三对点，假设是有放回的取，在一次采样中，同时取这三对点都为内点的概率是$\epsilon^3$ ,相反的，这三对点中至少存在一对外点的概率是 $1-\epsilon^3$ 。假设$RANSAC$ 连续进行了$K$次采样，每一次采样中三对点中至少存在一次外点的概率 $p_0=(1-\epsilon^3)^K$ ,那么，$K$次采样中，至少有一次采样中三对点都是内点的概率$p=1-p_0$

代入，得到

$$
K=\frac{\log(1-p)}{\log(1-\epsilon^3)}
$$
**四元数到旋转向量的转换**

假设四元数为$q=[q_0,q_1,q_2,q_3]$,$q_0$为实部，$q_1,q_2,q_3$为虚部，旋转向量$\theta n$ 的旋转轴
$n=[n_x,n_y,n_z]^T$,旋转角度$\theta$,旋转向量到四元数的转换公式：

$$
q=[\cos\frac\theta2,n\sin\frac\theta2]
$$
四元数到旋转向量的转换公式：

$$
\begin{aligned}&\theta=2\arccos q_0\\&[n_x,n_y,n_z]^T=[q_1,q_2,q_3]^T/\sin\frac\theta2\end{aligned}
$$
**Sim3的逆变换矩阵**

假设Sim3的变换矩阵为：

$$
Sim3=\left[\begin{array}{cc}sR&t\\0&1\end{array}\right]
$$
那么，它的逆变换矩阵为：
$$
(Sim3)^{-1}=\begin{bmatrix}sR&t\\0&1\end{bmatrix}^{-1}=\begin{bmatrix}\frac{1}{s}R^T&-\frac{1}{s}R^Tt\\0&1\end{bmatrix}
$$


参考论文：*Closed-form solution of absolute orientation using unit quaternions

**ComputeSim3计算im3**

LoopClosing::ComputeSim3() 里查找vpLoopConnectedKFs，mvpLoopMapPoints

![image-20241128094138057](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059571.png)

### 矫正回环

对应函数 CorrectLoop

**调整位姿**

通过mg2oScw（认为是准的）来进行位姿传播，得到当前关键帧的共视关键帧的世界坐标系下Sim3 位

姿（还没有修正）

![image-20241128094213324](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059572.png)

得到矫正的当前关键帧的共视关键帧位姿后，修正这些关键帧的地图点

![image-20241128094234662](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059573.png)

## BA优化

### 图优化和g2o简介

[从零开始一起学习SLAM | 理解图优化，一步步带你看懂g2o代码]([从零开始一起学习SLAM | 理解图优化，一步步带你看懂g2o代码](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247486858&idx=1&sn=ce458d5eb6b1ad11b065d71899e31a04&chksm=97d7e81da0a0610b1e3e12415b6de1501329920c3074ab5b48e759edbb33d264a73f1a9f9faf&scene=21#wechat_redirect))

[从零开始一起学习SLAM | 掌握g2o顶点编程套路]([从零开始一起学习SLAM | 掌握g2o顶点编程套路](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247486992&idx=1&sn=ecb7c3ef9bd968e51914c2f5b767428d&chksm=97d7eb87a0a062912a9db9fb16a08129f373791fd3918952342d5db46c0bc4880326a7933671&scene=21#wechat_redirect))

[从零开始一起学习SLAM | 掌握g2o边的代码套路]([从零开始一起学习SLAM | 掌握g2o边的代码套路](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247487082&idx=1&sn=d4a27e4c9a76760fffb571f57f4f7719&chksm=97d7ebfda0a062eba412877e9ecf5933f2051f0210c0d56f03267985512d97f2db434ab7356c&scene=21#wechat_redirect))

[参考资料]([ORB SLAM2源码解读(七)：Optimizer类 - 知乎](https://zhuanlan.zhihu.com/p/84466670?from_voters_page=true))



**块求解器**

BlockSolver_6_3 ：表示pose 是6维，观测点是3维

![image-20241128094904688](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059574.png)

**雅可比矩阵**

2 × 6 的雅可比矩阵：
$$
\frac{\partial\boldsymbol e}{\partial\delta\boldsymbol\xi}=-\begin{bmatrix}\frac{f_x}{Z'}&0&-\frac{f_xX'}{Z'^2}&-\frac{f_xX'Y'}{Z'^2}&f_x+\frac{f_xX^2}{Z'^2}&-\frac{f_xY'}{Z'}\\0&\frac{f_y}{Z'}&-\frac{f_yY'}{Z'^2}&-f_y-\frac{f_yY'^2}{Z'^2}&\frac{f_yX'Y'}{Z'^2}&\frac{f_yX'}{Z'}\end{bmatrix}
$$
描述了重投影误差关于相机位姿李代数的一阶变化关系

负号，因为这是由于误差是由观测值减预测值定义的

![image-20241128094945938](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059575.png)

该定义方式是：平移在前，旋转在后，如果旋转在前，平移在后时，只要把这个矩阵的前三列与后三列对调即可

### 优化函数分类

主要分为以下几个，我们对其功能做了详细介绍

~~~C++
/*
* @brief 仅优化位姿，不优化地图点，用于跟踪过程
* @param pFrame 普通帧
* @return 内点数量
*/
int Optimizer::PoseOptimization(Frame *pFrame)
/*
* @brief 局部建图线程中局部地图优化
* @param pKF 关键帧
* @param pbStopFlag 是否停止优化的标志
* @param pMap 局部地图
* @note 由局部建图线程调用,对局部地图进行优化的函数
*/
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map*pMap)
/**
* @brief 闭环时固定地图点进行Sim3优化
* @param[in] pKF1 当前帧
* @param[in] pKF2 闭环候选帧
* @param[in] vpMatches1 两个关键帧之间的匹配关系
* @param[in] g2oS12 两个关键帧间的Sim3变换，方向是从2到1
* @param[in] th2 卡方检验是否为误差边用到的阈值
* @param[in] bFixScale 是否优化尺度，单目进行尺度优化，双目/RGB-D不进行尺度优化
* @return int 优化之后匹配点中内点的个数
*/
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *>
&vpMatches1,g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
/**
* @brief 闭环时本质图优化，仅优化所有关键帧位姿，不优化地图点
* @param pMap 全局地图
* @param pLoopKF 闭环匹配上的关键帧
* @param pCurKF 当前关键帧
* @param NonCorrectedSim3 未经过Sim3传播调整过的关键帧位姿
* @param CorrectedSim3 经过Sim3传播调整过的关键帧位姿
* @param LoopConnections 因闭环时地图点调整而新生成的边
*/
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame*
pCurKF,const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,const LoopClosing::KeyFrameAndPose &CorrectedSim3,const map<KeyFrame *, set<KeyFrame *> >&LoopConnections, const bool &bFixScale)
~~~



### 局部BA中边的解析

对应函数 Optimizer::LocalBundleAdjustment

**EdgeSE3ProjectXYZ误差计算**

~~~C++
// 误差 = 观测 - 投影
_error = obs-cam_project(v1->estimate().map(v2->estimate()));
//其中：
// 相机2世界坐标系下的三维点坐标
v2->estimate()
//map 函数是用R，t把某个坐标系下三维点变换到另一个坐标系下
Vector3d map(const Vector3d & xyz) const
{
	return _r*xyz + _t;
}
// 把相机2地图点用相机1变换矩阵变换到相机1坐标系下
v1->estimate().map(v2->estimate())
// 反投影到图像坐标系
Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
    Vector2d proj = project2d(trans_xyz);
    Vector2d res;
    res[0] = proj[0]*fx + cx;
    res[1] = proj[1]*fy + cy;
    return res;
}
~~~



### Sim3优化中边的解析

对应函数 Optimizer::OptimizeSim3

**EdgeSim3ProjectXYZ的误差计算**

~~~C++
// 误差 = 观测 - 投影
_error = obs-v1->cam_map1(project(v1->estimate().map(v2->estimate())));
//其中：
// 相机2坐标系下的三维点坐标
v2->estimate()
//map 函数是用sim变换(r,t,s)把某个坐标系下三维变换到另一个坐标系下，定义是
Vector3 map (const Vector3& xyz) const {
	return s*(r*xyz) + t;
}
// 用V1估计的Sim12 变换把V2代表的相机2坐标系下三维点变换到相机1坐标系下
v1->estimate().map(v2->estimate())
// project 函数是把三维坐标归一化
Vector2d project(const Vector3d& v)
{
    Vector2d res;
    res(0) = v(0)/v(2);
    res(1) = v(1)/v(2);
    return res;
}
// cam_map1是用内参转化为图像坐标
Vector2d cam_map1(const Vector2d & v) const
{
    Vector2d res;
    res[0] = v[0]*_focal_length1[0] + _principle_point1[0];
    res[1] = v[1]*_focal_length1[1] + _principle_point1[1];
    return res;
}
~~~



![image-20241128100628607](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059576.png)



**EdgeInverseSim3ProjectXYZ** **的误差计算**

~~~C++
// 误差 = 观测 - 投影
_error = obs-v1->cam_map2(project(v1->estimate().inverse().map(v2->estimate())));
// 相机1坐标系下的三维点坐标（和上面不同）
v2->estimate()
// 用V1估计的Sim12 变换的逆（也就是Sim21）把V2代表的相机1坐标系下三维点变换到相机2坐标系下
v1->estimate().map(v2->estimate())
// 后面类似
~~~





## CMake理论与实践

### 什么是CMake？

CMake 是"Cross platform MAke"的缩写

• 一个开源的跨平台自动化建构系统，用来管理程序构建，不相依于特定编译器

• 需要编写CMakeList.txt 文件来定制整个编译流程（需要一定学习时间）

• 可以自动化编译源代码、创建库、生成可执行二进制文件等

• 为开发者节省大量时间，工程实践必备

Write once, run everywhere

CMake 不再使你在构建项目时郁闷地想自杀了 -- 一位开发者

学习资料：官网：www.cmake.org 、《CMake practice》 、《learning CMake》



### CMake有什么优缺点？

优点：

• 开源，使用类 BSD 许可发布

• 跨平台使用，根据目标用户的平台进一步生成所需的本地化 Makefile 和工程文件，如 Unix的Makefile

或Windows 的 Visual Studio 工程

• 能够管理大型项目，比如OpenCV、Caffe、MySql Server

• 自动化构建编译，CMake 构建项目效率非常高

注意：

• 需要根据CMake 专用语言和语法来自己编写CMakeLists.txt 文件

• Cmake 支持很多语言： C、C++、Java 等

• 如项目已经有非常完备的工程管理工具，并且不存在维护问题，没必要迁移到CMake

### CMake如何安装？

Windows下：按提示无脑安装

Linux下

• apt 安装【推荐，够用】 但是Ubuntu源里版本可能比较低

~~~shell
sudo apt-get install cmake
sudo apt-get install cmake-gui
~~~



源码编译【需要最新版本时】 ， 解压后执行

~~~shell
./bootstrap
make –j2
sudo make install
cmake --version
~~~



![image-20241128100950533](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059577.png)





### CMake 到底多好用？

举个栗子：OpenCV 在Windows下的配置方法

毛星云的博客里OpenCV配置方法

• 需要手动添加环境变量、 项目中手动添加包含路径、项目中手动添加库路径、项目中手动添加链接库

名、Debug 和Release下配置不同、 和OpenCV版本相关、构建好的项目不能直接移植到其他平台，给

别人用代码成本也太高

太繁琐！问题多，易出错！刚开始就想放弃了！

CMake 配置方法

• CMake 一键配置、以上所有东西自动关联，并且和OpenCV版本无关、跨平台移植，效率高

无脑安装！超简单！用过的都说好！学习劲头更足了！

### CMake使用注意事项

CMakeLists.txt文件

• CMake 构建专用定义文件，文件名严格区分大小写

• 工程存在多个目录，可以每个目录都放一个CMakeLists.txt文件 

• 工程存在多个目录，也可以只用一个CMakeLists.txt文件管理 

CMake指令

• 不区分大小写，可以全用大写，全用小写，甚至大小写混合，自己统一风格即可

~~~cmake
add_executable(hello main.cpp)
ADD_EXECUTABLE(hello main.cpp)
~~~



参数和变量

• 严格大小写相关。名称中只能用字母、数字、下划线、破折号

• 用${}来引用变量

• 参数之间使用空格进行间隔

### CMake常用指令介绍

```cmake
cmake_minimum_required
```





• 指定要求最小的cmake版本，如果版本小于该要求，程序终止

```cmake
project(test)
```

• 设置当前项目名称为test

```cmake
CMAKE_BUILD_TYPE
```



> [!NOTE]
>
> Debug: 调试模式，输出调试信息，不做优化
>
> Release: 发布模式，没有调试信息，全优化
>
> RelWithDebInfo::类似Release, 但包括调试信息
>
> MinSizeRel: 一种特殊的Release模式，会特别优化库的大小

```cmake
CMAKE_CXX_FLAGS
```

• 编译CXX的设置标志，比如 –std=c++11, -Wall, -O3（优化，使用向量化、CPU流水线，cache等提高代

码速度）

• 编译过程中输出警告（warnings）：set(CMAKE_CXX_FLAGS "-Wall")

• 追加，不会丢失之前的定义：set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

```cmake
include_directories
```

• 指定头文件的搜索路径，编译器查找相应头文件

• 举例：文件main.cpp中使用到路径 `/usr/local/include/opencv/cv.h` 中这个文件

• `CMakeLists.txt` 中添加 `include_directories(/usr/local/include)`

• 使用时`main.cpp`前写上 `#include “opencv/cv.h"` 即可

```cmake
set (variable value)
```

• 用变量代替值

• set (SRC_LST main.cpp other.cpp) 表示定义SRC_LST代替后面的两个cpp文件

```cmake
add_executable(hello main.cpp)
```

• 用指定的源文件为工程添加可执行文件

• 工程会用main.cpp生成一个文件名为 hello 的可执行文件

```cmake
add_library(libname STATIC/SHARED sources)
```

• 将指定的源文件生成链接库文件。STATIC 为静态链接库，SHARED 为共享链接库

```cmake
target_link_libraries (target library1 library2 ...)
```

• 为库或二进制可执行文件添加库链接，要用在add_executable之后。

• 例子如下：

• target_link_libraries (myProject libhello.a)

```cmake
add_dependencies (target-name depend)
```

• 为上层target添加依赖，一般不用

• 若只有一个targets有依赖关系，一般选择使用 target_link_libraries

• 如果两个targets有依赖关系，并且依赖库也是通过编译源码产生的。这时候用该指令可以在编译上层

target时，自动检查下层依赖库是否已经生成

```cmake
add_subdirectory(source_dir)
```

• 向当前工程添加存放源文件的子目录，目录可以是绝对路径或相对路径

```cmake
aux_source_directory( dir varname)
```

• 在目录下查找所有源文件

```cmake
message(mode "message text" )
```

• 打印输出信息，mode包括FATAL_ERROR、WARNING、STATUS、DEBUG等

• message(STATUS “Set debug mode")

一些预定义好的指令

> [!NOTE]
>
> PROJECT_NAME：项目名称，与project( xxx) 一致
>
> PROJECT_SOURCE_DIR：即内含 project() 指令的 CMakeLists 所在的文件夹
>
> EXECUTABLE_OUTPUT_PATH：可执行文件输出路径
>
> LIBRARY_OUTPUT_PATH ：库文件输出路径
>
> CMAKE_BINARY_DIR：默认是build文件夹所在的绝对路径
>
> CMAKE_SOURCE_DIR：源文件所在的绝对路径

```cmake
find_package(package version EXACT/QUIET/REQUIRED)
```

• 功能：采用两种模式（ FindXXX.cmake和XXXConfig.cmake ）搜索外部库

• 示例：find_package( OpenCV 3.4 REQUIRED )

• version：指定查找库的版本号。EXACT：要求该版本号必须精确匹配。QUIET：禁掉没有找到时的警

告信息。REQUIRED选项表示如果包没有找到的话，CMake的过程会终止，并输出警告信息。

• 搜索有两种模式：

• Module模式：搜索CMAKE_MODULE_PATH指定路径下的FindXXX.cmake文件，执行该文件从而找到

XXX库。其中，具体查找库并给XXX_INCLUDE_DIRS和XXX_LIBRARIES两个变量赋值的操作由

FindXXX.cmake模块完成。

• Config模式：搜索XXX_DIR指定路径下的XXXConfig.cmake文件从而找到XXX库。其中具体查找库并给

XXX_INCLUDE_DIRS和XXX_LIBRARIES两个变量赋值的操作由XXXConfig.cmake模块完成。

• 两种模式看起来似乎差不多，不过cmake默认采取Module模式，如果Module模式未找到库，才会采

取Config模式。

• 如果XXX_DIR路径下找不到XXXConfig.cmake文件，则会找/usr/local/lib/cmake/XXX/中的

XXXConfig.cmake文件。总之，Config模式是一个备选策略。通常，库安装时会拷贝一份

XXXConfig.cmake到系统目录中，因此在没有显式指定搜索路径时也可以顺利找到。

• 若XXX安装时没有安装到系统目录，则无法自动找到XXXConfig.cmake，需要在CMakeLists.txt最前面

添加XXX的搜索路径。

• set(XXX_DIR /home/cxl/projects/OpenCV3.1/build) #添加OpenCV的搜索路径

• 当find_package找到一个库的时候，以下变量会自动初始化:

> [!NOTE]
>
> <NAME>_FOUND : 显示是否找到库的标记
>
> <NAME>_INCLUDE_DIRS 或 <NAME>_INCLUDES : 头文件路径
>
> <NAME>_LIBRARIES 或 <NAME>_LIBRARIES 或 <NAME>_LIBS : 库文件
>
> <NAME>_DEFINITIONS：

![image-20241128103158518](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059578.png)

```cmake
list
```

• 列表操作（读、搜索、修改、排序）

• 追加例子：LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

```cmake
If, elseif, endif
```

• 判断语句，使用和C语言一致

```cmake
foreach
```



• 循环指令

![image-20241128103302088](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059579.png)

### CMake如何查询指令？

https://cmake.org/cmake/help/latest/genindex.html#

![image-20241128103337978](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059580.png)



### 静态库和共享库

静态库

• 原理：在编译时将源代码复制到程序中，运行时不用库文件依旧可以运行。

• 优点：运行已有代码，运行时不用再用库；无需加载库，运行更快

• 缺点：占用更多的空间和磁盘；静态库升级，需要重新编译程序

共享库（常用）

• 原理：编译时仅仅是记录用哪一个库里面的哪一个符号，不复制相关代码

• 优点：不复制代码，占用空间小；多个程序可以同时调用一个库；升级方便，无需重新编译

• 缺点：程序运行需要加载库，耗费一定时间

|         | 静态库 | 共享库 |
| ------- | ------ | ------ |
| Windows | lib    | .dll   |
| Linux   | .a     | .so    |
| Max OS  | .a     | dylib  |

### 如何安装库？

一般安装库流程,以Pangolin为例：

```shell
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=yourdirectory ..
make –j4
sudo make install
```

• cmake ..(注意,..代表上一级目录)

• make install 默认安装位置 /usr/bin

• make clean：可对构建结果进行清理

### 如何使用库？

当编译一个需要使用第三方库的软件时，我们需要知道：

• 去哪儿找头文件 .h

• 去哪儿找库文件 (.so/.dll/.lib/.dylib/…)

• 需要链接的库文件的名字

• 比如需要一个第三方库 curl，不使用find命令的话， CMakeLists.txt 需要指定头文件目录和库文件：

```cmake
include_directiories(/usr/include/curl)
target_link_libraries(myprogram yourpath/curl.so)
```

• 使用cmake的Modules目录下的FindCURL.cmake，就很简单了，相应的CMakeLists.txt 文件：

```cmake
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIR})
target_link_libraries(curltest ${CURL_LIBRARY})
```



## C++多线程理论与实践

### 并发 和 并行 的区别？

网上很多解释，鱼龙混杂。并行指的是程序运行时的状态，就是同时运行的意思。并发指的是程序的结构，这个程序程序同时执行多个独立的任务就说这个程序是并发的，实际上，这句话应当表述成“这个程序采用了支持并发的设计”。我们后面讲的都是代码结构，都是指并发。

### 理解并发

单核CPU如何产生“并发”

• 单核CPU：某一个时刻只能执行一个任务，由操作系统调度，每秒钟进行多次“任务切换”,来实现并发的

假象（不是真正的并发），切换任务时要保存变量的状态、执行进度等，存在时间开销；

• 人脑就是单核运算结构

• 一个人很难做到：一手画圆，一手画方

• 人为什么可以一边走路，一边看手机？

• 专心依次做多件事 和 同时做多件事情 哪个效率高？

• 富士康流水线为什么效率高？

多核CPU

• 双核，4核，8核，10核，能够实现真正的并行执行多个任务（硬件并发）

• 如何查看自己电脑CPU核心数目？

![image-20241128103850285](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059581.png)

**概念理解**

可执行程序

• Windows下扩展名为 exe

• Linux下为bin文件

进程

• 进程就是运行起来的可执行程序• 任务管理器查看

线程

• 进程可以包含多个线程

• 主线程：从main函数开始，main函数执行完，主线程结束，进程结束

• 其他线程：需要我们自己创建，入口可以是函数、类、lambda表达式

• 进程是否执行完毕的标志是：主线程是否执行完，如果主线程执行完毕了，就代表整个进程执行完

了，一般来说，此时如果其他子线程还没有执行完，也会被强行终止

**任务管理器**

![image-20241128103902363](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059582.png)

### 进程 vs. 线程

• 线程在进程下行进（单纯的车厢无法运行）

• 一个进程可以包含多个线程（一辆火车可以有多 个车厢）

• 不同进程间数据很难共享（一辆火车上的乘客很 难换到另外一辆火车，比如站点换乘）

• 同一进程下不同线程间数据很易共享（A车厢换 到B车厢很容易）

• 进程要比线程消耗更多的计算机资源（采用多列火车相比多个车厢更耗资源）

• 进程间不会相互影响，一个线程挂掉将导致整个进程挂掉（一列火车不会影响到另外一列火车，但是

如果一列火车上中间的一节车厢着火了，将 影响到所有车厢）

• 进程使用的内存地址可以上锁，即一个线程使用某些共享内存时，其他线程必须等它结束，才能 使用

这一块内存。（比如火车上的洗手间）－"互斥锁"

进程≈火车，线程≈车厢

![image-20241128103941151](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059583.png)

![image-20241128103947733](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059584.png)

**并发**

实现并发的手段

• 通过多个进程实现并发

• 在单独的进程中，写代码创建除了主线程之外的其他线程来实现并发：多线程多进程并发

• 比如SLAM采集数据（图像，其他传感器）一个进程，处理数据（跟踪、建图）一个进程多线程并发

• 一个进程中的所有线程共享地址空间（共享内存），全局变量、全局内存、全局引用都可以在线程之

间传递，所以多线程开销远远小于多进程

• 多进程并发核多线程并发可以混合使用，但建议优先考虑多线程技术

• 本课程中只讲多线程并发技术

### 多线程的应用

几乎所有的大型系统都需要多线程。

火车票购票软件

• 火车站窗口购票、代收点购票、网络购票SLAM系统

• 比如ORB-SLAM2：跟踪（主线程）、局部建图线程、闭环线程

### 构建子线程

熟悉thread, join, detach

thread 类成员函数

![image-20241128104055776](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059585.png)

**join**

![image-20241128104102614](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059586.png)

**关于join的思考**

谁调用了这个函数？

调用了这个函数的线程对象，就是putThread，一定要等这个线程的活干完了，这个join()函数才能得到返回。

在什么线程环境下调用了这个函数？

必须要等线程方法执行完毕后才能返回，那必然是阻塞调用线程的，也就是说如果一个线程对象putThread 在一个线程环境 main 调用了这个函数，那么这个线程环境 main就会被阻塞，直到这个线程对象putThread 在构造时传入的方法执行完毕后，才能继续往下走。

**应用案例**

ORB-SLAM3中的应用案例

![image-20241128104147481](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059588.png)

std::ref ：为了解决在线程的创建等过程的值拷贝问题

**实践总结**

main函数开始后自动开启主线程

进程id： this_thread::get_id()可以查看线程id

thread类可以创建线程：创建后自动开启线程，线程入口可以是函数、类、lambda表达式

join

• 意为汇合，子线程和主线程汇合

• 阻塞主线程并等待子线程执行完，当子线程执行完毕，join()就执行完毕

detach

• 分离，主线程不再与子线程汇合，不再等待子线程

joinable

• 判断是否可以成功使用join()或者detach()，可以的话返回true，否则返回false

应用：基本复杂一些的系统都会用到多线程，是绝对绕不开的

开发多线程程序：一个是实力的体现，一个是商用的必须需求

不同线程是独立的话效率很高（边走边听歌），否则不一定高（边吃饭边唱歌）

C++11增加了多线程跨平台移植，减少了开发工作量

多线程有一定难度，注意数据共享问题

### 多线程数据共享

**什么是数据共享？**

什么数据是安全的？只读数据是安全的

什么数据是不安全的？ 同一个地址数据有读有写，如果不做处理，容易冲突崩溃

例子

• 火车票订票，线上线下不能同时定一个座位

• 两个人不能同时上一个厕所



**互斥量** **mutex**

![image-20241128104244981](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059589.png)

互斥量就是个类对象，可以理解为一把锁，多个线程同一时间只有一个线程能加锁或解锁

• 例子：厕所

什么数据是不安全的？

• 同一个地址数据有读有写，如果不做处理，容易冲突崩溃

lock, unlock 要成对使用，必须对应，否则出错

使用注意事项

• 只保护需要保护的数据，控制好范围，少了达不到效果，多了影响效率。根据需要和经验确定

• 例子：本应厕所加锁，结果整个办公室加锁

**lock_guard**

内部构造时相当于执行了lock，析构时相当于执行unlock

简单但不如lock()和unlock()灵活，通过 大括号来实现，控制生命周期

![image-20241128104257136](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059590.png)

**unique_lock**

std::unique_lock要比std::lock_guard功能更多，有更多的成员函数，更灵活

但是更灵活的代价是占用空间相对更大一点且相对更慢一点学习这些目的

![image-20241128104311025](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059591.png)

• 自己会灵活运用最好， 如果做不到，至少知道这些功能，能看懂别人代码

![image-20241128104320998](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059592.png)

**std::adopt_lock**

adopt 通过，采取，表示mutex在此之前已经被上锁，不需要再 lock了 。注意：必须提前lock

![image-20241128104338366](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059593.png)

**std::unique_lock::try_lock/std::try_to_lock**

尝试去加锁，如果没有锁定成功，会立即返回，不会产生阻塞

前提：不能提前lock();

应用： 防止其他的线程锁定mutex太长时间，导致本线程一直阻塞

![image-20241128104355887](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059594.png)

**std::defer_lock**

defer 延迟

功能：初始化了一个没有加锁的mutex

应用：不给它加锁的目的是以后可以调用unique_lock的一些方法

前提：不能提前lock

![image-20241128104405090](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059595.png)

**std::unique_lock::release**

![image-20241128104417107](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059596.png)

### 多线程中死锁问题

死锁至少有两个互斥量

死锁产生原因

• 线程A执行时，这个线程先锁mutex1，并且锁成功了，然后去锁mutex2的时候，出现了上下文切换。

• 线程B执行，这个线程先锁mutex2，因为mutex2没有被锁，即mutex2可以被锁成功，然后线程B要去

锁mutex1。

• 此时，死锁产生了，A锁着mutex1，需要锁mutex2，B锁着mutex2，需要锁mutex1，

两个线程没办法继续运行下去。。。

死锁的一般解决方案

• 只要保证多个互斥量上锁的顺序一样就不会造成死锁

**忠告**

学习这些目的：自己会灵活运用最好，如果做不到，至少知道这些功能

能看懂别人代码

掌握常用的多线程编程方法，不常用的了解能看懂即可

不追求高级技巧、写出稳健的代码最重要

### ORB-SLAM2中的多线程代码解析

**ORB-SLAM2的线程**

有几个线程？主线程是什么？子线程是什么？

• System::System()

线程间如何调度？

• Tracking::CreateNewKeyFrame()

• mpLocalMapper->InsertKeyFrame();

• mpLoopCloser->InsertKeyFrame();

多线程加速单目初始化

![image-20241128104545435](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059597.png)

**ORB-SLAM2的互斥锁示例**

Local mapping线程内

• unique_lock lock(mMutexNewKFs)

Loop closure线程内

• unique_lock lock(mMutexLoopQueue)

• unique_lock lock(mMutexGBA)

多线程间互斥锁

• mMutexMapUpdate

![image-20241128104627676](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059598.png)

## 汇总：基于ORB-SLAM2的改进代码

### 点线特征相关

添加了线特征。从3D密集SLAM进行表面重建的增量3D线段提取

[Add line feature based ORB-SLAM2]([atlas-jj/ORB_Line_SLAM: line feature based SLAM, modified based on the famous ORB-SLAM2](https://github.com/atlas-jj/ORB_Line_SLAM))

![image-20241128104749773](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059599.png)

![image-20241128104754736](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059600.png)

RGB-D模式下添加了点线融合

[RGBD-SLAM with Point and Line Features, developed based on ORB_SLAM2]([maxee1900/RGBD-PL-SLAM: RGBD SLAM with Point and Line Feature. This project is developed based on ORB-SALM.](https://github.com/maxee1900/RGBD-PL-SLAM))

![image-20241128104848414](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059601.png)

[ORB-SLAM2_with_line, Monocular ORB-SLAM with Line Features]([lanyouzibetty/ORB-SLAM2_with_line: Monocular ORB-SLAM with Line Features.](https://github.com/lanyouzibetty/ORB-SLAM2_with_line))

双目点线融合，在弱纹理环境中传统点特征方法失效的情况下拥有较高的运行鲁棒性。

[PL-SLAM: a Stereo SLAM System through the Combination of Points and Line Segments]([rubengooj/pl-slam: This code contains an algorithm to compute stereo visual SLAM by using both point and line segment features.](https://github.com/rubengooj/pl-slam))

### 用新的特征点替代ORB

使用了一种更好的特征选择方法

[GF-ORB-SLAM2, Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization]([ivalab/gf_orb_slam2: Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization Capabilities](https://github.com/ivalab/gf_orb_slam2))

![image-20241128105114397](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059602.png)

用SuperPoint 替代ORB来进行特征提取

[SuperPoint-SLAM]([KinglittleQ/SuperPoint_SLAM: SuperPoint + ORB_SLAM2](https://github.com/KinglittleQ/SuperPoint_SLAM))

![image-20241128105126312](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059603.png)

![image-20241128105214912](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059604.png)



设计的GCNv2具有与ORB功能相同的描述符格式，可以替代关键点提取，精度更高，适合在嵌入式低功

耗平台上运行

[GCNv2: Efficient Correspondence Prediction for Real-Time SLAM]([jiexiong2016/GCNv2_SLAM: Real-time SLAM system with deep features](https://github.com/jiexiong2016/GCNv2_SLAM))

![image-20241128105304835](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059605.png)

### 直接法代替特征点法

1.使用SVO中直接法来跟踪代替耗时的特征点提取匹配，在保持同样精度的情况下，是原始ORB-SLAM2

速度的3倍

[ORB-YGZ-SLAM, average 3x speed up and keep almost same accuracy v.s. ORB-SLAM2, use direct tracking in SVO to accelerate the feature matching]([gaoxiang12/ORB-YGZ-SLAM](https://github.com/gaoxiang12/ORB-YGZ-SLAM))

### 融合其他传感器

1.双目VIO版本，加入了LK光流和滑动窗口BA优化

[YGZ-stereo-inertial SLAM, LK optical flow + sliding window bundle adjustment](https://github.com/gaoxiang12/ygz-stereo-inertial)

2.京胖实现的VI-ORB-SLAM2

[VIORB, An implementation of Visual Inertial ORBSLAM based on ORB-SLAM2](https://github.com/jingpang/LearnVIORB)

3.支持鱼眼，不需要rectify和裁剪输入图

[Fisheye-ORB-SLAM, A real-time robust monocular visual SLAM system based on ORB-SLAM for fisheye cameras, without rectifying or cropping the input images](https://github.com/lsyads/fisheye-ORB-SLAM)

### 地图相关

添加保存和导入地图功能

[Osmap, Save and load orb-slam2 maps](https://github.com/AlejandroSilvestri/osmap)

[ORB_SLAM2 with map load/save function](https://github.com/Jiankai-Sun/ORB_SLAM2)

添加了地图可视化

[Viewer for maps from ORB-SLAM2 Osmap](https://github.com/AlejandroSilvestri/Osmap-viewer)

高翔实现的添加稠密点云地图

[ORBSLAM2_with_pointcloud_map](https://github.com/gaoxiang12/ORBSLAM2_with_pointcloud_map)

在高翔基础上添加了稠密闭环地图

[ORB-SLAM2_RGBD_DENSE_MAP, modified from Xiang Gao's "ORB_SLAM2_modified". It is added a dense loopclosing map model](https://github.com/tiantiandabaojian/ORB-SLAM2_RGBD_DENSE_MAP)

### 动态环境

适合动态环境，增加了动态物体检测和背景修复的能力

[DynaSLAM, is a SLAM system robust in dynamic environments for monocular, stereo and RGB-D setups](https://github.com/BertaBescos/DynaSLAM)

![image-20241128105833885](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059606.png)

用YOLO来做动态环境的检测

[YOLO Dynamic ORB_SLAM](https://github.com/bijustin/YOLO-DynaSLAM)

![image-20241128105920478](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281059607.png)

### 语义相关

动态语义SLAM 目标检测+VSLAM+光流/多视角几何动态物体检测+octomap地图+目标数据库

[ORB_SLAM2_SSD_Semantic, 动态语义SLAM 目标检测+VSLAM+光流/多视角几何动态物体检测+octomap地图+目标数据库](https://github.com/Ewenwan/ORB_SLAM2_SSD_Semantic)

![image-20241128110053108](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116151.png)

用YOLO v3的语义信息来增加跟踪性能

[TE-ORB_SLAM2, Tracking Enhanced ORB-SLAM2](https://github.com/Eralien/TE-ORB_SLAM2)

![image-20241128110123371](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116152.png)

通过手持RGB-D相机进行SLAM，ORB-SLAM2作为后端，用PSPNet做语义预测并将语义融入octomap

[Semantic SLAM，Real time semantic slam in ROS with a hand held RGB-D camera](https://github.com/floatlazer/semantic_slam)

[orb-slam2 with semantic labelling](https://github.com/qixuxiang/orb-slam2_with_semantic_labelling)

![image-20241128110301978](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116153.png)

用深度学习的场景理解来增强传统特征检测方法，基于贝叶斯SegNet 和ORB-SLAM2，用于长时间定位。

[SIVO: Semantically Informed Visual Odometry and Mapping](https://github.com/navganti/SIVO)

![image-20241128110731822](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116154.png)

## ORB-SLAM2最新改进论文

### GCNv2

GCNv2：Efficient Correspondence Prediction for Real-Time SLAM

paper: https://ieeexplore.ieee.org/abstract/document/8758836

code: https://github.com/jiexiong2016/GCNv2_SLAM



**摘要**：

在本文中，我们提出了一个基于深度学习的网络，GCNv2，用于生成关键点和描述符。GCNv2是在我

们之前的方法GCN的基础上建立的，GCN是一种为三维射影几何训练的网络。GCNv2的设计采用了二进

制描述符向量作为ORB的特征，因此它可以在ORB- SLAM等系统中轻松替换ORB。GCNv2大大提高了只

能运行在桌面硬件上的GCN的计算效率。我们展示了使用GCNv2特性的改进版ORB-SLAM如何在嵌入式

低功耗平台Jetson TX2上运行。实验结果表明，GCNv2算法保持了与GCN算法基本相同的精度，具有足

够的鲁棒性，可以用于无人机的控制。

**主要贡献**：

1.GCNv2保持了GCN的准确性，与相关的基于深度学习的特征提取方法和经典方法相比，在运动估计方

面提供了显著的改进。

2.与需要桌面GPU进行实时推理的GCN相比，GCNv2的推理可以在嵌入式低功耗硬件（如Jetson TX2）

上运行。

3.作者设计的GCNv2具有与ORB功能相同的描述符格式，以便它可以作为SLAM系统（如ORB-SLAM 2或

SVO2中的关键点提取程序）中的替代项。

4.作者用GCN-SLAM1在一架真实的无人机上进行控制，证明了我们工作的有效性和鲁棒性，并表明它能

处理ORB-SLAM2失效的情况。

**整体框架**：

![image-20241128110818375](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116155.png)

GCNv2网络结构。橙色的块代表卷积层，下面的数字是对应的通道数量。在每组块之后应用ELU非线

性。符号I除以某个数字表示当前特征图的分辨率，例如I/4表示一个像素是输入图像中的4x4像素。与

GCN一样，GCNv2同时预测关键点和描述符，即网络输出关键点信任度的概率图和描述符的密集特征

图。GCNv2在低分辨率下进行预测，而不是原始分辨率的，并且只使用单一的图像。

![image-20241128110830299](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116156.png)

GCN-SLAM与ORB extractor对比，左边是原始的ORB-SLAM2关键点提取过程的插图，右边是GCNv2的

方法。GCN-SLAM中的关键点提取相对简单，这在很大程度上是因为它依赖于2D卷积和矩阵乘法，而这

些处理都交给了GPU。根据流程可以发现，其实就是整个代替了ORB的检测、描述子计算过程。

**实验结果**

1. 特征跟踪精度

   ![image-20241128110842294](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116157.png)

2. 回环检测精度

3. ![image-20241128110902587](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116158.png)

   ![image-20241128110917995](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116159.png)

4. 匹配效率

   ![image-20241128110925549](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116160.png)

5. 建图效果

   ![image-20241128110932058](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116161.png)

**总结**：

GCN在视觉跟踪方面比现有的深度学习和经典方法有更好的性能。然而，由于GCN的计算需求和使用多

个图像帧，它不能以一种有效的方式直接部署到实时SLAM系统中。在本文中，我们提出了一个更小、更

高效的GCN版本，称为GCNv2，很容易适应现有的SLAM系统，从而解决了这些问题。实验结果表明，

GCNv2可以有效地用于当前基于特征的SLAM系统，以实现最先进的跟踪性能。

### PL-SLAM

PL-SLAM: a Stereo SLAM System through the Combination of Points and Line Segments

paper: https://ieeexplore.ieee.org/abstract/document/8680013

code: https://github.com/rubengooj/pl-slam

**摘要**：

传统的双目视觉SLAM方法依赖于点特征来估计相机的轨迹并建立环境地图。然而，在低纹理环境中，通

常很难找到足够数量的可靠点特征，因此，这种算法的性能会下降。本文提出了PL-SLAM，一种结合了

点和线段的双目视觉SLAM系统，可以在更广泛的场景下稳定地工作，特别是在图像中点特征稀缺或分布

不均匀的情况下。PL-SLAM在过程的所有实例中都利用了点和线段:视觉里程计、关键帧选择、集束调整

等。我们还通过一种新颖的词袋方法提供了一个循环闭环过程，该方法利用了两种特性的组合描述能

力。此外，生成的地图在三维元素上更加丰富多样，可以用来推断有价值的、高层次的场景结构，如平

面、空地、地平面等。

![image-20241128110953800](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116162.png)

本文待解决的问题描述如上图所示。低纹理环境对基于传统关键点的基于特征的SLAM系统具有挑战性。

相比之下，线段通常在人造环境中很常见，除了改进的相机定位外，构建的地图更丰富，因为它们填充

了更多有意义的元素(三维线段) (a) *lt-easy*. (b) *euroc/V2-01-easy*. (c) *euroc/V1-01-easy*. (d) Map from

(c)。

**主要贡献**：

1.第一个实时开源的同时使用点和线分割特征深度SLAM系统，并且在弱纹理环境中传统点特征方法失效

的情况下拥有较高的运行鲁棒性。

2.一种新的同时处理点和线特征调整关键字位姿。

3.一种扩展BoW方法同时考虑点和线特征分割提高闭环检测

**整体框架**：

![image-20241128111022776](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116163.png)

本文提出的PL-SLAM系统的总体结构如图2所示。该结构基于ORB-SLAM首先提出的方案，并实现了三个

不同的线程: 视觉里程计、局部建图和环路闭合。这种高效的分布允许对VO模块进行持续跟踪，而只有

当一个新的关键帧被插入时，局部建图和环路闭合才会在后台处理。本文的提案也采用了一些ORB

SLAM思想作为开发点线SLAM系统的基础。

**实验结果**：



1. 精度比较

   ![image-20241128111044505](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116164.png)

2. 性能比较

   ![image-20241128111049774](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116165.png)

3. 轨迹和地图

   ![image-20241128111057931](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116166.png)

地图(黑色)包括点和线段，以及用PL-SLAM从室外环境中获得的轨迹(蓝色)序列KITTI-07。地图显示了某些区域(如A区域)的噪声测量值，以及来自环境的结构线，如建筑物的部分(如B区域)。

![image-20241128111115971](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116167.png)

用PL-SLAM(绿色)从KITTI数据集估计的轨迹(真值，蓝色)。(a)在序列KITTI-00中估算的轨迹，可以发现大

量的闭环。(b)序列KITTI-08没有呈现任何环路闭合，因此沿轨迹的漂移没有被纠正。(c)最后，KITTI-07

序列在轨道的初始部分和最终部分之间呈现一个环路闭合。

**结论**：

本文提出了一种新的基于关键点和线段特征相结合的双目视觉SLAM系统， 创造了PL-SLAM，贡献了一

个强大和多功能的系统，能够在所有类型的环境下工作，包括低纹理的环境，同时产生有几何意义的地

图。为此，本文开发了第一个实时运行的开源SLAM系统，同时使用关键点和线段特征。该方案基于BA

解决方案，无缝地处理不同类型特性的组合。此外，作者还扩展了地点识别BoW方法，用于同时使用点

和线段的情况，以增强闭环过程。该方法在EuRoC MAV或KITTI等知名数据集上进行了测试，以及在具

有挑战性的低纹理场景中的一系列双目图像。在这些实验中，作者将基于点系统和线系统的PL-SLAM与

ORB-SLAM2进行了比较，在大多数数据集序列的鲁棒性方面取得了优异的性能，同时仍然是实时操作

的。

### 语义特征选择

Network Uncertainty Informed Semantic Feature Selection for Visual SLAM

paper: https://ieeexplore.ieee.org/abstract/document/8781616

code: https://github.com/navganti/SIVO

**摘要**：

为了便于使用视觉SLAM进行长期定位，良好的特征选择有助于确保参考点在长时间内保持不变，算法的

运行时间和存储复杂度保持一致。作者提出了一种基于信息理论的视觉SLAM特征选择方法SIVO

(semantic Informed Visual Odometry and Mapping)，该方法将语义分割和神经网络不确定性引入到

特征选择过程中。我们的算法选择在当前状态熵和状态联合熵之间的香农熵最大减少的点，前提是利用

贝叶斯神经网络特征的分类熵加入新的特征。每一个被选择的特征都显著降低了车辆状态的不确定性，

并多次被检测为静态对象(建筑物、交通标志等)，且具有较高的置信度。这种选择策略生成稀疏地图，可

以促进长期定位。总的来说，在KITTI数据集上的实验结果，SIVO与ORB SLAM2方法的性能相当，同时

将地图大小减少了近70%。

**主要内容**：

![image-20241128111204550](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116168.png)

本文是在Active search for real-time vision和Covariance recovery from a square root information

matrix for data association两篇论文的基础上改进的，参考论文使用信息熵判断观测数据是否用于更新

估计的状态量，如果在新观测数据下待估计变量的协方差的秩相比之前数据条件下待估计变量的协方差

的秩是否下降了一定阈值，是则选择该观测数据（特征点）参与跟踪和优化，本文在此基础上将语义分

割的不确定性融入到信息熵的计算中，在选择具有。图1说明了视觉SLAM算法使用的典型特征。最好的

参考点很可能位于建筑物和标志上。这些都是有用的长期参考，因为它们只会在施工或破坏的情况下被

修改。相比之下，车辆的特性可能会在一小时内消失，树叶将不再随着季节的变化而出现。深度学习的

出现导致了场景理解方面的快速进步，允许将上下文整合到SLAM中，并解决了我们的最后一个标准。我

们现在可以根据对典型静态对象和动态对象的上下文理解来决定哪些特性更稳定。

**实验结果**：

1. 误差和地图点数量的比较

   ![image-20241128111212106](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116169.png)

2. 轨迹比较

   ![image-20241128111225929](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116170.png)

   

**总结**：

提出了一种新的视觉SLAM特征选择算法SIVO，融合了神经网络的不确定性和信息理论的视觉SLAM方

法。SIVO在KITTI序列09上的表现优于ORB SLAM2，并且在剩余10条轨迹中的6条上表现相当好，同时

平均移除69.4%的地图点。虽然将语义信息整合到SLAM算法中并不新鲜，但这是第一个直接考虑网络不

确定性的SLAM信息理论方法算法。在贝叶斯神经网络特征分类熵的基础上加入新特征，选取能显著降低

当前状态熵和联合状态熵之间的香农熵的点。每个选择的特征显著降低了车辆状态的不确定性，并多次

被检测为高置信度的静态对象，生成稀疏地图，有利于长期定位。

### DynaSLAM

DynaSLAM: Tracking, Mapping, and Inpainting in Dynamic Scenes

paper: https://ieeexplore.ieee.org/document/8421015

code: https://github.com/BertaBescos/DynaSLAM

**摘要**：

在这篇论文中，我们提出了DynaSLAM，一种基于ORB-SLAM2的视觉SLAM系统，增加了动态物体检测

和背景修复的能力。DynaSLAM对于单目、双目和深度相机在动态环境中具有非常高的鲁棒性。我们能

够检测移动的物体通过多视点几何、深度学习或者是两者的结合。拥有场景的静态地图能够修复被动态

物体遮挡的图像帧的背景。我们使用公开的单目、双目和深度相机数据集来评估我们的系统。我们研究

几种精度/速度的权衡的影响，来评估我们提出的方法的限制。在高度动态的场景中，DynaSLAM比标准

的视觉SLAM展现出更高的精度。并且也建立场景中静态部分的地图，这在真实世界中长期的应用是必要

的。

**主要贡献**：

1．提出了基于ORB-SLAM2的视觉SLAM系统，通过增加运动分割方法使得其在单目、立体、RGB-D相

机的动态环境中均具有稳健性。

2．通过对因动态物体遮挡而缺失的部分背景进行修复，生成一个静态场景地图。

**整体框架**：

![image-20241128111302774](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116171.png)

DynaSLAM在RGB-D上的结果图如上图流程所示。

![image-20241128111322830](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116172.png)

图中黑色实线是使用单目或双目相机时系统的工作流程，黑色虚线则是使用RGB-D相机时的处理流程。

而红色虚线表示“跟踪和建图”环节在工作时与稀疏地图之间的数据交换（利用稀疏地图进行跟踪，同时

不断更新地图）。对于RGB-D相机而言，将RGB-D数据传入到CNN网络中对有先验动态性质的物体如行

人和车辆进行逐像素的分割。作者使用多视几何在两方面提升动态内容的分割效果。首先作者对CNN输

出的动态物体的分割结果进行修缮；其次，将在大多数时间中保持静止的、新出现的动态对象进行标

注。对于单目和双目相机，则直接将图像传入CNN中进行分割，将具有先验动态信息的物体分割出去，

仅使用剩下的图像进行跟踪和建图处理。

![image-20241128111336265](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116173.png)

使用多视图几何(左)，深度学习(中)，以及几何和学习相结合的方法(右)检测和分割动态对象。请注意，

(a)不能识别出桌子后面的人，(b)不能分割出人所携带的书，两者结合(c)效果最好

**实验结果**：

1. 跟踪精度

   ![image-20241128111346737](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116174.png)

2. 轨迹对比

   ![image-20241128111403104](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116175.png)

   ![image-20241128111412188](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116176.png)



​	作者还将DynaSLAM(N+G)中RGB-D分支与RGB-D ORB-SLAM2进行了对比。可以发现在高动态场景中，

本文的方法优于ORB-SLAM2，并且误差与RGB-D ORB-SLAM2在静态场景中的误差相近。而在低动态场

景中，要稍微差一些。

**结论**：

 

本文提出了一种视觉SLAM系统，该系统建立在ORB-SLAM的基础上，增加了一种运动分割方法，使其

在单目、双目和RGB-D相机的动态环境中具有鲁棒性。该系统准确地跟踪相机，并创建一个静态的，因

此可重用的场景地图。在RGB-D的情况下，DynaSLAM能够获得不含动态内容且被遮挡背景被着色的合

成RGB帧，以及它们对应的合成深度帧，这对虚拟现实应用可能非常有用。与目前的技术水平相比，

DynaSLAM在大多数情况下达到了最高的准确性。在TUM动态对象数据集中，DynaSLAM是目前最好的

RGB-D SLAM解决方案。在单目情况下，该方案的精度与ORB-SLAM相似，但获得的静态地图的场景较

早的初始化。在kitti数据集中，dynasty的精度略低于单目和双目ORB-SLAM，除非动态对象代表场景的

重要部分。 然而，DynaSLAM估计的地图只包含结构对象，因此可以在长期应用中重用。这项工作的未

来扩展可能包括，除其他外，实时性能，基于RGB的运动检测器，或通过使用更精细的修复技术。

### 点线特征

Robust Visual SLAM with Point and Line Features

paper: https://ieeexplore.ieee.org/abstract/document/8205991

**摘要**：

本文提出了使用异构点线特征的slam系统，继承了ORB-SLAM，包括双目匹配、帧追踪、局部地图、回

环检测以及基于点线的BA。使用最少的参数对线特征采用标准正交表示，推导了线特征重投影误差的雅

克比矩阵，改进了实验结果。因为使用线特征能够提供更多的几何约束，传统的方法只使用了点特征，

对光照变化以及位置歧义较为敏感。但是线特征也有两个问题要解决：首先是空间中的线参数太多，图

优化时计算量增加，空间中的线只有四个自由度，但是通常被表示成6个自由度（端点表示以及Plucker

坐标）；其次，由于参数过多，大多数线特征法都采用数值计算雅克比，而本文中推导了了雅克比矩

阵，提高精确度。

**主要贡献**：

1.为了增强数据关联的鲁棒性，提出了一种改进的线特征提取与匹配方法。

2.在视觉SLAM的后端，使用标准正交(最小)表示来参数化直线，并解析计算相应的雅可比矩阵。

3.设计并实现了一个完整的基于点和线特征的视觉SLAM系统，包括立体匹配、帧跟踪、局部建图、线和

点特征的集束调整以及基于点和线的环路检测。

**整体框架**：

![image-20241128111500243](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116177.png)

与orbslam2不同之处：

1. 跟踪

校正过的双目图像作为输入，四个并行线程用于提取左右相机中的特征点（ORB特征）。LSD检测线特

征，使用LBD描述子。随后两个线程用于双目匹配，所有的特征都被区分为双目或者单目特征，如图所

示

2. 局部地图

新的关键帧增加是，当前帧和其他帧的共视信息将被更新，局部地图三角化更多的点和线。移除外点，

删除冗余帧。相机位姿和路标使用BA进行优化。由于3D线的端点对优化没有影响，但是对匹配和可视化

很有用，因此在优化后系统仍然维护3D线的端点。通过反投影2D线到当前帧，并进行修剪。（与

SLSLAM类似）

3. 回环检测以及全局BA

   视觉词袋预先离线训练，用到点特征和线特征，使用ORB特征和LBD特征建立词典，两个单词向量的相

似度度量见公式（26）。使用时间一致性完善了匹配的关系，回环时使用SE(3)变换（采用Ransac策

略）计算出变换矩阵，如果失败，采用双目视觉中的匹配直线，最后计算的位姿用于纠正回环。再计算

全局BA。



**实验结果**：

1. 25次蒙特卡罗试验的平均结果

   ![image-20241128111550914](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116178.png)

2. 回路闭合前的漂移度

   ![image-20241128111555559](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116179.png)

3. ORB-SLAM、PLSVO和PL-SLAM在KITTI数据集上的结果

   ![image-20241128111601513](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411281116180.png)

**结论**：

为了提高视觉SLAM的准确性和鲁棒性，作者提出了一种基于点和线特征的图形化方法。在优化过程中，

空间线采用标准正交表示，是最紧凑的解耦形式。并导出了重投影误差相对于线参数的雅可比矩阵，取

得了较好的性能。实验证明，融合这两种特征可以在合成和真实场景中产生更强的鲁棒性估计。
