### 服务器的配置

**如何查看显卡信息**

~~~cmd
nvidia-smi
~~~

**结果**：

~~~
(base) user7@ubuntu:~/Hwj$ nvidia-smi
Wed Oct 16 09:12:26 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:18:00.0 Off |                  Off |
| 46%   40C    P2              63W / 450W |  17466MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off | 00000000:51:00.0 Off |                  Off |
| 47%   38C    P2              57W / 450W |  12562MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090        Off | 00000000:8A:00.0 Off |                  Off |
| 45%   35C    P2              54W / 450W |  12890MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce RTX 4090        Off | 00000000:C3:00.0 Off |                  Off |
| 46%   39C    P2              57W / 450W |  12818MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1378307      C   python                                     1132MiB |
|    0   N/A  N/A   1381232      C   python                                     7096MiB |
|    0   N/A  N/A   1396404      C   python                                     4708MiB |
|    0   N/A  N/A   1420290      C   python                                     4506MiB |
|    1   N/A  N/A   1378307      C   python                                      406MiB |
|    1   N/A  N/A   1381232      C   python                                     5764MiB |
|    1   N/A  N/A   1396404      C   python                                     3296MiB |
|    1   N/A  N/A   1420290      C   python                                     3072MiB |
|    2   N/A  N/A   1378307      C   python                                      456MiB |
|    2   N/A  N/A   1381232      C   python                                     5794MiB |
|    2   N/A  N/A   1396404      C   python                                     3288MiB |
|    2   N/A  N/A   1420290      C   python                                     3328MiB |
|    3   N/A  N/A   1378307      C   python                                      406MiB |
|    3   N/A  N/A   1381232      C   python                                     5818MiB |
|    3   N/A  N/A   1396404      C   python                                     3284MiB |
|    3   N/A  N/A   1420290      C   python                                     3286MiB |
+---------------------------------------------------------------------------------------+

~~~

`nvidia-smi` 输出解释

`nvidia-smi` 是 NVIDIA 提供的一个命令行工具，用于监控和管理 NVIDIA GPU。以下是输出的详细解释：

**顶部信息**

~~~
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
~~~

- **日期和时间**：显示命令执行的日期和时间。
- **NVIDIA-SMI 版本**：显示 `nvidia-smi` 工具的版本。
- **驱动程序版本**：显示当前安装的 NVIDIA 驱动程序版本。
- **CUDA 版本**：显示当前支持的 CUDA 版本。

**GPU 信息**

每个 GPU 的信息如下：

~~~
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:18:00.0 Off |                  Off |
| 46%   40C    P2              63W / 450W |  17466MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off | 00000000:51:00.0 Off |                  Off |
| 47%   38C    P2              57W / 450W |  12562MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090        Off | 00000000:8A:00.0 Off |                  Off |
| 45%   35C    P2              54W / 450W |  12890MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce RTX 4090        Off | 00000000:C3:00.0 Off |                  Off |
| 46%   39C    P2              57W / 450W |  12818MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
~~~

- **GPU**：GPU 的编号。
- **Name**：GPU 的名称。
- **Persistence-M**：持久模式状态（On/Off）。
- **Bus-Id**：GPU 的总线 ID。
- **Disp.A**：显示器附加状态（On/Off）。
- **Volatile Uncorr. ECC**：易失性不可纠正的 ECC 错误状态（On/Off）。
- **Fan**：风扇速度百分比。
- **Temp**：GPU 温度。
- **Perf**：性能状态。
- **Pwr:Usage/Cap**：当前功耗和最大功耗。
- **Memory-Usage**：当前使用的显存和总显存。
- **GPU-Util**：GPU 利用率。
- **Compute M.**：计算模式。
- **MIG M.**：多实例 GPU 模式（N/A 表示不适用）。

**进程信息**

~~~
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1378307      C   python                                     1132MiB |
|    0   N/A  N/A   1381232      C   python                                     7096MiB |
|    0   N/A  N/A   1396404      C   python                                     4708MiB |
|    0   N/A  N/A   1420290      C   python                                     4506MiB |
|    1   N/A  N/A   1378307      C   python                                      406MiB |
|    1   N/A  N/A   1381232      C   python                                     5764MiB |
|    1   N/A  N/A   1396404      C   python                                     3296MiB |
|    1   N/A  N/A   1420290      C   python                                     3072MiB |
|    2   N/A  N/A   1378307      C   python                                      456MiB |
|    2   N/A  N/A   1381232      C   python                                     5794MiB |
|    2   N/A  N/A   1396404      C   python                                     3288MiB |
|    2   N/A  N/A   1420290      C   python                                     3328MiB |
|    3   N/A  N/A   1378307      C   python                                      406MiB |
|    3   N/A  N/A   1381232      C   python                                     5818MiB |
|    3   N/A  N/A   1396404      C   python                                     3284MiB |
|    3   N/A  N/A   1420290      C   python                                     3286MiB |
+---------------------------------------------------------------------------------------+
~~~

- **GPU**：GPU 的编号。
- **GI**：GPU 实例 ID（N/A 表示不适用）。
- **CI**：计算实例 ID（N/A 表示不适用）。
- **PID**：进程 ID。
- **Type**：进程类型（C 表示计算进程）。
- **Process name**：进程名称。
- **GPU Memory Usage**：进程使用的 GPU 内存。

通过这些信息，你可以监控每个 GPU 的使用情况，包括温度、功耗、显存使用情况以及正在运行的进程。



~~~python
nvcc -version
~~~

~~~
(base) user7@ubuntu:~/Hwj$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
~~~

**详细解释**

1. **nvcc: NVIDIA (R) Cuda compiler driver**：
   - 这是 `nvcc` 工具的名称和描述。`nvcc` 是 NVIDIA 的 CUDA 编译器驱动程序，用于编译 CUDA 代码。
2. **Copyright (c) 2005-2022 NVIDIA Corporation**：
   - 版权信息，表明该工具由 NVIDIA 公司开发和维护。
3. **Built on Wed_Jun__8_16:49:14_PDT_2022**：
   - 构建日期和时间，表明这个版本的 `nvcc` 是在 2022 年 6 月 8 日下午 4:49（太平洋夏令时间）构建的。
4. **Cuda compilation tools, release 11.7, V11.7.99**：
   - CUDA 编译工具的版本信息：
     - **release 11.7**：表示这是 CUDA 11.7 版本。
     - **V11.7.99**：表示具体的版本号是 11.7.99。
5. **Build cuda_11.7.r11.7/compiler.31442593_0**：
   - 构建信息，包含内部版本号和构建标识符：
     - **cuda_11.7**：表示这是 CUDA 11.7 版本。
     - **r11.7**：表示这是 11.7 版本的一个修订版。
     - **compiler.31442593_0**：这是内部构建标识符，用于标识具体的编译器版本。

### 总结

这段输出表明你当前安装的 CUDA 编译器工具链是 CUDA 11.7 版本，具体版本号是 11.7.99，构建于 2022 年 6 月 8 日。这个版本的 `nvcc` 工具用于编译 CUDA 代码，并且与 CUDA 11.7 工具包一起使用。

如果你需要使用更高版本的 CUDA（例如 CUDA 12.2），你需要下载并安装相应版本的 CUDA 工具包，并确保环境变量指向新的 CUDA 安装路径。



**nvcc-v 是11.7，nivdia-smi是12.2 ？**



这是因为 `nvcc` 和 `nvidia-smi` 显示的 CUDA 版本信息有所不同：

- `nvcc --version` 显示的是 CUDA 编译器驱动程序的版本。这是你在编译 CUDA 程序时使用的工具链版本。
- `nvidia-smi` 显示的是 NVIDIA 驱动程序支持的 CUDA 运行时版本。这是你的 GPU 驱动程序支持的最高 CUDA 版本。

**解释**

1. **`nvcc --version`**：
   - 显示的是你安装的 CUDA 工具包的版本。在你的情况下，显示的是 CUDA 11.7。
2. **`nvidia-smi`**：
   - 显示的是你的 找不到 Python 打包工具,安装打包工具NVIDIA 驱动程序支持的最高 CUDA 版本。在你的情况下，显示的是 CUDA 12.2。



~~~cmd
# env python == 3.9
conda create -n h_pytorch python=3.9

# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
~~~

[pycharm连接远程服务器，代码成功运行，但一些基本python属性和函数会报红线（例如print）解决方案](https://www.cnblogs.com/ZZG-GANGAN/p/17764206.html)

~~~
export PATH=/home/extend1/user7/anaconda3/bin:$PATH
export PATH=/usr/local/cuda-11.7/bin:$PATH
export CPATH=/usr/local/cuda-11.7/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export TMPDIR=/home/extend2/user7/tmp
~~~

1. **服务器端开启 jupyter notebook**

~~~cmd
jupyter notebook --no-browser
~~~



注：复制倒数第二行的 token，这个 token 就是远程访问的密码，同时记下端口号8888。

2. **PC端端口映射**
在 PC 端做一个端口映射，即通过 ssh 隧道来将服务器端的8888端口号映射到本地（PC端）的某个端口（如1234）：

~~~cmd
ssh -L 1234:localhost:8888 user7@192.168.213.222
~~~


接着输入服务器的访问密码`277316389`，这时就可以在PC端的浏览器通过

http://localhost:1234直接访问服务器上的 jupyter notebook 了。



**vscode ssh 连接服务器**

**<u>登录服务器</u>**

1. **输入 SSH 命令**

~~~shell
ssh -p 22 user7@192.168.213.222
~~~

2. **输入密码并登录**

~~~shell
277316389
~~~

**<u>免密登录服务器</u>**

1. **生成 SSH 密钥对**

~~~shell
ssh-keygen -t rsa -C "weijiehe@sdust.edu.cn"
~~~



2. **发送公钥到远程服务器**

~~~shell
ssh-copy-id -p 22 user7@192.168.213.222
~~~

[【PyTorch】n卡驱动、CUDA Toolkit、cuDNN全解安装教程_nvidia-cuda-toolkit-CSDN博客](https://blog.csdn.net/qq_50791664/article/details/135881801)

[手把手教你使用anaconda安装pytorch环境（适合新手）-CSDN博客](https://blog.csdn.net/qq_54575112/article/details/128491493)

[通过ssh远程使用服务器jupyter notebook（并维持服务器端口转发）_ssh 转发jupyter-CSDN博客](https://blog.csdn.net/qq_34769162/article/details/107947034)

[本地使用 jupyter notebook 连接远程服务器_jupyter notebook连接服务器-CSDN博客](https://blog.csdn.net/u012856866/article/details/124822961?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-124822961-blog-124648150.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.1&utm_relevant_index=1)

[SSH + VS Code远程连接：从账号密码到免密登录的攻略_vscode ssh 使用密码登录远程-CSDN博客](https://blog.csdn.net/dragon_0505/article/details/142446655)

[解决Visual Studio Code 更新后一直卡在下载vscode-server问题的方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/655289233)

[2.1. 数据操作 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_preliminaries/ndarray.html)



### pandas

Pandas 是一个用于数据操作和分析的强大 Python 库。它提供了高效的数据结构和数据分析工具，特别适用于处理表格数据（类似于电子表格或 SQL 表）。

**核心数据结构**

1. **Series**：一维数组，类似于 Python 的列表或 NumPy 的一维数组。它可以包含任何数据类型，并且每个元素都有一个标签（索引）。
2. **DataFrame**：二维表格数据结构，类似于电子表格或 SQL 表。它由多个 Series 组成，每个 Series 代表一列。

**主要功能**

- **数据加载**：从各种文件格式（如 CSV、Excel、SQL 数据库等）加载数据。
- **数据清洗**：处理缺失值、重复数据、数据转换等。
- **数据操作**：过滤、选择、分组、聚合、合并等。
- **数据分析**：统计分析、时间序列分析等。
- **数据可视化**：与 Matplotlib 和 Seaborn 集成，生成各种图表。

**示例代码**

以下是一些常见的 Pandas 操作示例：

**导入 Pandas**

```python
import pandas as pd
```

**创建 Series**

```python
\# 创建一个 Series

s = pd.Series([1, 3, 5, 7, 9])

print(s)
```

**创建 DataFrame**

```python
\# 创建一个 DataFrame

data = {

  'Name': ['Alice', 'Bob', 'Charlie'],

  'Age': [25, 30, 35],

  'City': ['New York', 'Los Angeles', 'Chicago']

}

df = pd.DataFrame(data)

print(df)
```

**从 CSV 文件加载数据**

```python
\# 从 CSV 文件加载数据

df = pd.read_csv('data.csv')

print(df.head()) # 查看前五行数据
```

**数据选择和过滤**

```python
\# 选择一列

print(df['Name'])

\# 选择多列

print(df[['Name', 'Age']])

\# 过滤数据

print(df[df['Age'] > 30])
```

**数据分组和聚合**

```python
\# 按 City 分组并计算平均年龄

grouped = df.groupby('City')['Age'].mean()
```

print(grouped)

**处理缺失值**

```python
\# 填充缺失值

df.fillna(0, inplace=True)

\# 删除包含缺失值的行

df.dropna(inplace=True)
```

**数据可视化**

```python
import matplotlib.pyplot as plt
\# 绘制柱状图
df['Age'].plot(kind='bar')
plt.show()
```

**结论**

Pandas 是一个功能强大的数据操作和分析库，提供了丰富的数据结构和工具，适用于各种数据处理任务。通过 Pandas，你可以轻松地加载、清洗、操作和分析数据，并生成各种可视化图表。



### !cd h和%cd 

在 Jupyter Notebook 中，直接使用 `!cd` 命令无法持久更改工作目录，因为每个 `!` 命令都会在一个新的子 shell 中运行。要在 Jupyter Notebook 中持久更改工作目录，可以使用 `%cd` 魔法命令。

**使用 `%cd` 魔法命令**

`%cd` 是一个 IPython 魔法命令，可以在 Jupyter Notebook 中持久更改工作目录。

**示例**

\# 显示当前工作目录

```shell
!pwd
\# 更改工作目录
%cd ~/Hwj/chapter2
\# 再次显示当前工作目录，确认更改
!pwd
```

**解释**

1. `!pwd`：显示当前工作目录。
2. `%cd ~/Hwj/chapter2`：更改工作目录到 `~/Hwj/chapter2`。
3. `!pwd`：再次显示当前工作目录，确认更改。

**结论**

在 Jupyter Notebook 中，使用 `%cd` 魔法命令可以持久更改工作目录，而不是使用 `!cd`。这样可以确保工作目录的更改在整个 Notebook 会话中保持有效。



### zip()

`zip()` 是 Python 内置的一个函数，用于将多个可迭代对象（如列表、元组等）打包成一个迭代器。这个迭代器生成的元素是由来自每个可迭代对象的对应元素组成的元组。

### 主要功能

- **将多个可迭代对象打包成一个迭代器**：`zip()` 可以将多个可迭代对象的元素逐一配对，生成一个由元组组成的迭代器。
- **处理不同长度的可迭代对象**：如果传入的可迭代对象长度不同，`zip()` 会以最短的可迭代对象为准，截断多余的元素。

### 语法

zip(*iterables)

- `*iterables`：一个或多个可迭代对象。

### 示例

**示例 1：基本用法**

```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
zipped = zip(list1, list2)
print(list(zipped)) # 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
```



**示例 2：处理不同长度的可迭代对象**

```python
list1 = [1, 2, 3]
list2 = ['a', 'b']
zipped = zip(list1, list2)
print(list(zipped)) # 输出: [(1, 'a'), (2, 'b')]
```



**示例 3：解压缩**

`zip()` 的逆操作是使用 `*` 运算符进行解压缩。

```python
zipped = [(1, 'a'), (2, 'b'), (3, 'c')]
list1, list2 = zip(*zipped)
print(list1) # 输出: (1, 2, 3)
print(list2) # 输出: ('a', 'b', 'c')
```



#### 示例 4：与 `for` 循环结合

```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

for i, j in zip(list1, list2):
  print(i, j)

输出：
1 a
2 b
3 c
```



**结论**

`zip()` 是一个非常有用的函数，可以将多个可迭代对象打包成一个迭代器，方便地进行并行迭代。它在处理不同长度的可迭代对象时会自动截断，以最短的可迭代对象为准。此外，`zip()` 还可以与 `for` 循环结合使用，方便地解包元素。



### get_dummies()



[`pd.get_dummies`](vscode-file://vscode-app/e:/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 是 Pandas 提供的一个函数，用于将分类变量（categorical variables）转换为哑变量（dummy variables），也称为独热编码（one-hot encoding）。这在机器学习和数据预处理中非常常见，因为大多数机器学习算法不能直接处理分类数据。

**主要功能**

- **将分类变量转换为哑变量**：每个分类值都会被转换为一个新的列，列的值为 0 或 1，表示是否属于该分类。
- **处理缺失值**：通过设置 `dummy_na=True`，可以将缺失值也作为一个分类值进行处理。

**语法**

pd.get_dummies(data, dummy_na=False)

- [`data`](vscode-file://vscode-app/e:/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)：要进行独热编码的 DataFrame 或 Series。
- `dummy_na`：布尔值，是否将缺失值也作为一个分类值进行处理。默认为 `False`。



## 数据处理流程

~~~csv
NumRooms,Space,Alley,Ssd,Price
NA,100,Pave,shanke,127500
2.0,NA,NA,NA,106000
4.0,200,NA,NA,178100
NA,NA,NA,NA,140000
~~~



1. 获取csv路径

~~~python
data_file = os.path.join('.','data','house_test2')
~~~

2. 读取csv文件

~~~python
data = pd.read_csv(data_file)
~~~

3. 拆分数据为DataFrame

~~~python
input1，input2,outputs = data.iloc[:,0:2].data.iloc[:,2:],data.iloc[:4]
~~~

4. 处理数值型缺值

~~~python
input1 = input1.fillna(input1.mean())
~~~

5. 非数值转换独热编码

~~~python
input2 = pd.get_dummies(input2,dummy_na = True)
~~~

6. 拼接DataFrame,并存储回csv

~~~python
inputs = pd.concat([input1,input2,outputs],axis = 1)
inputs.to_csv(data_path,index = False)
~~~

7.转换为张量格式





### 自动计算梯度

~~~python
import torch

x = torch.arange(4.0)
y = torch.dot(x,x)
~~~

1. 标记张量为需要计算梯度

~~~python
x.require_grad_(True)
x.grad#返回梯度
~~~

2. 计算梯度

~~~python
y.backward()
~~~

**详细解释 `backward()` 方法**

- `backward()` 方法用于计算当前张量的梯度。
- 这个方法会从当前张量开始，沿着计算图向后传播，计算每个叶子节点（即 `requires_grad=True` 的张量）的梯度。
- 计算的梯度会累积到每个叶子节点的 `.grad` 属性中。

**计算图**

- 当你对张量进行操作时，PyTorch 会自动构建一个计算图。
- 这个计算图记录了每个操作及其输入和输出张量。
- 当你调用 `backward()` 方法时，PyTorch 会沿着计算图向后传播，计算每个叶子节点的梯度。
