## traditional methods

### 4PCS配准

**1 原理**

并非全共线的共面四点a，b，c，d，定义了两个独立的比率r1和r2，其在仿射变化中是**不变且唯一**的。现在给定一个具有n个点的点集Q，以及两个由点P得到的仿射不变的比率r1，r2，对每一对点q1，q2⊂ Q，计算他们的中间点
$$
e_1=q_1+r_1(q_2-q_1)\\e_2=q_1+r_2(q_2-q_1)
$$
若任意两对这样的点，一对由 r1计算得到的中间点和另一对由 r2计算得到的中间点在允许范围内一致，那么可以认为这两对点可能是 P中基础点的仿射对应点。将四点转化应用到全局点云转化,计算点云的匹配重叠度，若达到设置的阈值，则完成点云粗配准。![图片](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202411210819736.webp)

```
pcl::registration::FPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> fpcs;
fpcs.setInputSource(source_cloud);  
fpcs.setInputTarget(target_cloud);  // 目标点云
fpcs.setApproxOverlap(0.7);         // 设置源和目标之间的近似重叠度。
fpcs.setDelta(0.01);                // 设置常数因子delta，用于对内部计算的参数进行加权
fpcs.setNumberOfSamples(100);       // 设置验证配准效果时要使用的采样点数量
```

### K-4PCS配准

**1 步骤**

K-4PCS方法主要分为两个步骤: 

(1)利用**VoxelGrid滤波器**对点云Q进行下采样，然后使用标准方法进行3D关键点检测。

(2)通过4PCS算法使用**关键点集合**而非原始点云进行数据的匹配，**降低了搜索点集的规模，提高了运算效率**。

**2 核心代码**

```
pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
kfpcs.setInputSource(source);  // 源点云
kfpcs.setInputTarget(target);  // 目标点云
kfpcs.setApproxOverlap(0.7);   // 源和目标之间的近似重叠。
kfpcs.setLambda(0.5);          // 平移矩阵的加权系数。
kfpcs.setDelta(0.002, false);  // 配准后源点云和目标点云之间的距离
kfpcs.setNumberOfThreads(6);   // OpenMP多线程加速的线程数
kfpcs.setNumberOfSamples(200); // 配准时要使用的随机采样点数量
pcl::PointCloud<pcl::PointXYZ>::Ptr kpcs(new pcl::PointCloud<pcl::PointXYZ>);
kfpcs.align(*kpcs);
```