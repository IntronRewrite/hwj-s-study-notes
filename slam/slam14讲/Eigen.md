# Eigen库基本用法

Eigen是一个高效、灵活的C++线性代数库，适用于矩阵和向量的操作。本文档介绍了Eigen库的一些基本用法。

## 矩阵声明与初始化

### 固定大小的矩阵

#### 声明一个3x3的整数矩阵，并将其初始化为单位矩阵

```
Eigen::Matrix3i matrix_33;

matrix_33.setIdentity();
```



#### 声明一个4x4的双精度浮点数矩阵，并将其随机初始化

```
Eigen::Matrix4d matrix_44;

matrix_44.setRandom();
```



#### 使用`<<`运算符初始化矩阵

```
Eigen::Matrix3d matrix_33_alt;

matrix_33_alt << 1, 0, 0,

​         0, 1, 0,

​         0, 0, 1;
```



### 动态大小的矩阵

#### 声明一个动态大小的矩阵，并将其初始化为零矩阵

```
Eigen::MatrixXd matrix_dynamic = Eigen::MatrixXd::Zero(3, 3);
```



## 向量声明与初始化

### 固定大小的向量

#### 声明一个3维的双精度浮点数向量，并将其初始化为特定值

```
Eigen::Vector3d vector_3d;

vector_3d << 1, 2, 3;
```



### 动态大小的向量

#### 声明一个动态大小的向量，并将其初始化为随机值

```
Eigen::VectorXd vector_dynamic(3);
vector_dynamic.setRandom();



Eigen::VectorXd vector_dynamic(3);

vector_dynamic.setRandom();
```



## 矩阵操作

### 访问矩阵元素

可以使用`()`运算符访问矩阵中的元素：

```
for (int i = 0; i < 2; ++i) {

  for (int j = 0; j < 3; ++j) {

​    std::cout << matrix_23(i, j) << "\t";

  }

  std::cout << std::endl;

}
```



### 矩阵和向量相乘

矩阵和向量相乘实际上是矩阵和矩阵相乘：

```
v_3d << 3, 2, 1;

vd_3d << 4, 5, 6;

// 显式转换类型

Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;

std::cout << result << std::endl;

Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;

std::cout << result2 << std::endl;
```



### 矩阵运算

#### 矩阵求逆

```
Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;

matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);

Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;

v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

clock_t time_stt = clock(); // 计时

Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;

std::cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << std::endl;
```



#### 矩阵分解

通常用矩阵分解来求解，例如QR分解，速度会快很多：

```
time_stt = clock();

x = matrix_NN.colPivHouseholderQr().solve(v_Nd);

std::cout << "time use in QR decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << std::endl;
```



## 特征值和特征向量

```
Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(matrix_NN);

std::cout << "Eigen values = \n" << eigen_solver.eigenvalues() << std::endl;

std::cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << std::endl;
```



## 总结

本文档介绍了Eigen库的一些基本用法，包括矩阵和向量的声明、矩阵操作、矩阵运算以及特征值和特征向量的计算。Eigen库提供了丰富的功能，可以满足大多数线性代数计算的需求。



以下是一些关于Eigen库的练习题，帮助你更好地理解和掌握Eigen库的基本用法：

### 练习题 1：矩阵初始化和基本操作

1. 声明一个3x3的整数矩阵，并将其初始化为单位矩阵。

2. 声明一个4x4的浮点数矩阵，并将其初始化为随机数。

3. 将一个2x2的矩阵初始化为以下值：

   1 2

   3 4

4. 计算并输出上述2x2矩阵的转置矩阵。

### 练习题 2：矩阵和向量运算

1. 声明一个3x1的向量，并将其初始化为 `[1, 2, 3]`。

2. 声明一个3x3的矩阵，并将其初始化为以下值：

   1 0 0

   0 1 0

   0 0 1

3. 计算并输出上述矩阵与向量的乘积。

### 练习题 3：矩阵分解

1. 声明一个3x3的矩阵，并将其初始化为随机数。
2. 使用LU分解求解线性方程组 `Ax = b`，其中 `A` 是上述矩阵，`b` 是一个3x1的向量，初始化为 `[1, 2, 3]`。
3. 输出求解结果。

### 练习题 4：特征值和特征向量

1. 声明一个3x3的对称矩阵，并将其初始化为以下值：

   4 1 1

   1 2 3

   1 3 6

2. 计算并输出该矩阵的特征值和特征向量。

### 练习题 5：矩阵求逆

1. 声明一个4x4的矩阵，并将其初始化为随机数。
2. 计算并输出该矩阵的逆矩阵。
3. 验证逆矩阵的正确性，即计算 `A * A_inv` 是否等于单位矩阵。

### 练习题 6：动态大小的矩阵

1. 声明一个动态大小的矩阵，并将其初始化为5x5的随机数矩阵。
2. 计算并输出该矩阵的行列式。

### 练习题 7：矩阵块操作

1. 声明一个6x6的矩阵，并将其初始化为随机数。
2. 提取并输出该矩阵的左上角3x3子矩阵。
3. 将该矩阵的右下角3x3子矩阵设置为单位矩阵。

### 练习题 8：矩阵和向量的组合操作

1. 声明一个3x3的矩阵和一个3x1的向量，并将它们初始化为随机数。
2. 计算并输出矩阵与向量的点积和叉积。

### 练习题 9：矩阵的基本统计

1. 声明一个4x4的矩阵，并将其初始化为随机数。
2. 计算并输出该矩阵的均值、标准差、最小值和最大值。

### 练习题 10：矩阵的稀疏表示

1. 声明一个5x5的稀疏矩阵，并将其初始化为以下值：

   1 0 0 0 2

   0 3 0 0 0

   0 0 4 0 0

   0 0 0 5 0

   6 0 0 0 0

2. 计算并输出该稀疏矩阵的转置矩阵。

这些练习题涵盖了Eigen库的基本操作、矩阵和向量运算、矩阵分解、特征值和特征向量计算、矩阵求逆、动态大小的矩阵、矩阵块操作、矩阵和向量的组合操作、矩阵的基本统计以及稀疏矩阵的操作。通过完成这些练习，你将能够更好地理解和掌握Eigen库的使用。