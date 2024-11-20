# SLAM定义

(**S**imultaneous **L**ocalization **A**nd **M**apping)

- **移动**设备
- **未知**环境**未知**地点
- 在运动过程通过传感器观测**定位**自身位置和姿态，在根据自身位姿进行增量式的**地图构建**

**例子**

一个人去未知的地方逛，在自己逛了一圈的过程中，观测各种建筑，标志，获悉自身位置和大致地图

# SLAM的主要目的：定位

- 定位 = location + pose (位姿)

# 内部传感器

- 移动智能体自带
  - 相机
  - IMU
  - 激光雷达
  - GPS
  - 码盘
  - $\cdots$
- 优缺点
  - 可用于未知环境
  - 通用
  - 定位算法复杂

**<img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116155137987.png" alt="image-20241116155137987" style="zoom: 67%;" />**



# 外部传感器

- 环境中提前布设
  - 二维码
  - 导轨
  - 反光柱
- 问题
  - 相对可靠
  - 约束太多
  - 不通用
  - 不智能

# 激光雷达传感器

- 硬件

  - 精度高
  - 测量范围大（车上搭载有个几百米）
  - 受环境影响小
  - 研究充分
  - 体积大、功耗大
  - 价格昂贵（好几万）
  - 信息量有限

  <img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116160413708.png" alt="image-20241116160413708" style="zoom:33%;" />

  - 稀疏点云
  - 无色彩信息
  - 信息量有限

  <img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116160700850.png" alt="image-20241116160700850" style="zoom:50%;" />

  - 机械激光雷达运动畸变

  <img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116160910283.png" alt="image-20241116160910283" style="zoom:50%;" />

  # 视觉传感器

  <img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116161024977.png" alt="image-20241116161024977" style="zoom:50%;" />

  - 便宜
  - 体积小
  - 信息丰富
  - 计算量大
  - 对环境假设强
  - 易受干扰

<img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116161734444.png" alt="image-20241116161734444" style="zoom: 50%;" />     

# 视觉SLAM框架

- 传感器数据处理：多源数据同步，数据对齐与读取
- 视觉里程计（VO）:估计短期图像之间的运动（轨迹？）
- 后端优化：滤波（估计？）、图优化；优化地图点坐标和位姿
- 回环检测：判断是否回到曾经去过的地方，用于抑制累积误差
- 建立地图：稀疏的地图点、稠密的点云、网格地图等

<img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116202447786.png" alt="image-20241116202447786" style="zoom:50%;" />

ORB-SLAM2算法流程图 



<img src="F:\Documents\GitHub\hwj-s-study-notes\slam\slam14讲\assets\image-20241116203922065.png" alt="image-20241116203922065" style="zoom:50%;" />

# SLAM优秀开源方案 

视觉(惯导)SLAM ORB SLAM2 (单/双目/RGB-D) https://github.com/raulmur/ORB_SLAM2 

DSO (单目) https://github.com/JakobEngel/dso 

InfiniTAM v3 (RGB-D) https://github.com/victorprad/InfiniTAM 

VINS-Fusion (单双目+IMU)  https://github.com/HKUST-Aerial-Robotics/VINS-Fusion 

OKVIS (单双目+IMU)  https://github.com/ethz-asl/okvis 

ORB SLAM3 (单双目+IMU/ RGB-D)  https://github.com/UZ-SLAMLab/ORB_SLAM3 

激光(惯导)SLAM LeGO-LOAM (LIDAR)  https://github.com/RobustFieldAutonomyLab/LeGO-LOAM 

LIO-SAM (LIDAR + IMU)  https://github.com/TixiaoShan/LIO-SAM 

Cartographer (LIDAR + IMU) https://github.com/googlecartographer/cartographer 

多传感器融合 LVI-SAM (LIDAR +单目+ IMU)  https://github.com/TixiaoShan/LVI-SAM 

R3LIVE (LIDAR +单目+ IMU)  https://github.com/hku-mars/r3live



尽量做热门的项目，比如ORB-SLAM
