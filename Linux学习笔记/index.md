# Linux（服务器操作系统)

[toc]

# 第一章

## 1 操作系统概述

### 1.1 操作系统作用

调度和管理硬件

### 1.2 常见操作系统

PC端：Windows11，Linux, MacOS,

移动端：Andeoid，IOS，HarmonyOS



## 2 Linux初识

**Linux创始人：林纳斯.托瓦兹，1991**

**Linux内核**

<u>系统组成：</u>

- Linux系统内核（开源）
- 系统级应用程序

<u>功能:</u>

- 内核功能：调度cpu，调度内存，调度文件系统，调度通讯网络，调度IO等（调度硬件）
- 系统及应用程序：可理解为系统自带程序，可供用户快速上手操作系统，如：文件管理器，任务管理器，图片查看，音乐播放等。

**Linux发行版**

内核免费且开源，也代表了：

- 任何人都可以获得并修改内核，并自行集成系统级程序
- 提供了内核+系统级程序的完整封装，称之为Linux发行版

![image-20240910102428768](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150853888.png)

常用：CentOS,Ubantu



## 3 虚拟机介绍

### 3.1 什么是虚拟机？ 

虚拟的硬件+操作系统=虚拟的电脑

### 3.2 为什么用虚拟机？

获取Linux系统



## 4 Vmare Workstation 的安装

![image-20240910103813070](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150853029.png)

下载地址:https://www.vmware.com/cn/products/workstation-pro.html



## 5 远程连接Linux系统

### 5.1 操作系统的图形化、命令行2种操作模式

- 图形化操作是指使用操作系统附带的图形化页面，以图形化的窗口形式获得操作反馈，从而对操作系统进行操作、使用
- 命令行操作是指使用各种命令，以文字字符的形式获得操作反馈从而对操作系统进行操作、使用

### 5.2 理解为什么使用命令行操作Linux系统

- Linux从诞生至今，在图形化页面的优化上，并未重点发力。所以Linux操作系统的图形化页面:不好用、不稳定
- 在开发中，使用命令行形式，效率更高，更加直观，并且资源占用低，程序运行更稳定。

### 5.3 掌握使用FinalShell软件连接Linux操作系统

- 内容的复制、粘贴跨越VMware不方便
- 文件的上传、下载跨越VMware不方便
- 也就是和Linux系统的各类交互，跨越VMware不方便

我们可以通过第三方软件，FinalShell，远程连接到Linux操作系统之上。并通过Finalshell去操作Linux系统。

<u>Finalshell的下载地址为:</u>

[FinalShell SSH工具,服务器管理,远程桌面加速软件,支持Windows,macOS,Linux,版本4.5.6,更新日期2024.8.27 - FinalShell官网 (hostbuf.com)](https://www.hostbuf.com/t/988.html)



## 6 WSL

WSL作为Windows10系统带来的全新特性，正在逐步颠开发人员既有的选择。

- 传统方式获取Linux操作系统环境，是安装完整的虚拟机，如VMware
- 使用WSL，可以以非常轻量化的方式，得到Linux系统环境

目前，开发者正在逐步抛弃以虚拟机的形式获取Linux系统环境，而在逐步拥抱WSL环境。

WSL:Windows Subsystem for Linux，是用于Windows系统之上的Linux子系统
作用很简单，可以在Windows系统中获得Linux系统环境，并完全**直连计算机硬件**，无需通过虚拟机虚拟硬件



## 7 虚拟机快照的制作和还原

在学习阶段我们无法避免的可能损坏Linux操作系统。
如果损坏的话，重新安装一个Linux操作系统就会十分麻烦。



VMware虚拟机(Workstation和Funsion)支持为虚拟机制作快照。通过快照将当前虚拟机的状态保存下来,在以后可以通过快照恢复虚拟机到保存的状态

### 7.1 在VMware Workstation Pro中制作快照



![image-20240910152550450](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854471.png)

![image-20240910152332844](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854431.png)

### 7.2 **在VMware Workstation Pro中还原快照**

![image-20240910152700462](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854478.png)

# 第二章

## 1 Linux目录结构

### 1.1 Linux的目录结构

Linux没有盘符这个概念,只有一个根目 / 所有文件都在它下面

### 1.2 Linux路径的描述方式

- 
  在Linux系统中，路径之间的层级关系，使用:/来表示

- 在windows系统中，路径之间的层级关系，使用:\来表示

  

## 2 Linux命令入门

### 2.1 什么是命令、命令行

- 命令:即Linux操作指令,是系统内置的程序，可以以字符化的形式去使用
- 命令行:即Linux终端,可以提供字符化的操作页面供命令执行

### 2.2 Linux命令的通用格式

~~~ shell
command [-option][parameter]
~~~

- 命令本体，即命令本身
- 可选选项，控制命令的行为细节
- 可选参数，控制命令的指向目标

## 3 ls命令入门

~~~shell
ls [-a -l -h][Linux路径]
~~~

### 3.1 ls命令作用

在命令行中，以平铺的形式,展示当前工作目录(默认HOME目录)下的内容(文件或文件夹)

![image-20240910163943444](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854508.png)

### 3.2 HOME目录

每一个用户在Linux系统的专属目录，默认在:/home/用户名

### 3.3 当前工作目录

Linux命令行在执行命令的时候,需要一个工作目录,打开命令行程序(终端)默认设置工作目录在用户的HOME目录

## 4 ls命令的参数和选项

### 4.1 ls命令的参数的作用

可以指定要查看的文件夹(目录)的内容,如果不给定参数,就查看当前工作目录的内容

### 4.2 ls命令的选项:

- -a 选项，可以展示出隐藏的内容，以 . 开头的文件或文件夹默认被隐藏，需要-a才能显示出来



![image-20240910164851493](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854540.png)

- -l 选项，以列表的形式展示内容,并展示更多细节



![image-20240910164927385](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854571.png)

- -h 选项，需要和-l选项搭配使用，以更加人性化的方式显示文件的大小单位

![image-20240910165445940](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854601.png)

### 4.3 命令的选项组合使用

命令的选项是可以组合使用的，比如:ls -lah,等同于ls -a- l -h



## 5 cd/pwd命令

### 5.1 cd命令的作用

cd命令来自英文:Change Directorycd命令可以切换当前工作目录，语法是:

~~~shell
cd[Linux路径]
~~~

- 没有选项，只有参数，表示目标路径
- 使用参数切换到指定路径
- 不使用参数，切换工作目录到当前用户的HOME

### 5.2 pwd命令的作用

- pwd命令来自英文:Print Work Directory

- pwd命令，没有选项，没有参数，直接使用即可
- 作用是:输出当前所在的工作目录



## 6 相对命令、绝对命令

### 6.1 相对路径和绝对路径

- 绝对路径:以根目录做起点，描述路径的方式，路径以/开头
- 相对路径:以当前目录做起点，描述路径的方式，路径不需以/开头如无特殊需求

### 6.2 特殊路径符

- . 表示当前目录，比如cd.或cd./Desktop
- .. 表示上一级目录，比如:cd.. 或 cd../.
- ~ 表示用户的HOME目录，比如:cd\~或cd\~/Desktop



## 7 mkdir命令

### 7.1 mkdir命令的语法和功能

- mkdir用以创建新的目录(文件夹)
- 语法:

~~~shell
mkdir[-p]Linux路径
~~~

- 参数**必填**，表示要创建的目录的路径，相对、绝对、特殊路径符都可以使用

### 7.2 -p选项的作用

**可选**，表示自动创建不存在的父目录，适用于创建**连续多层**级的目录

## 8 touch、cat、more命令

### 8.1 touch命令

- 用于创建一个新的文件
- 语法:

~~~shell
touch Linux路径
~~~

- 参数必填，表示要创建的文件的路径，相对、绝对、特殊路径符都可以使用

### 8.2 cat命令

- 用于查看文件内容
- 语法:

~~~shell
cat Linux路径
~~~

- 参数必填，表示要查看的文件的路径，相对、绝对、特殊路径符都可以使用

### 8.3 more命令

- 用于查看文件内容，可翻页查看
- 语法: 

~~~shell
more Linux路径
~~~

- 参数必填，表示要查看的文件的路径，相对、绝对、特殊路径符都可以使用使用空格进行翻页，使用q退出查看

## 9 cp、mv、rm命令

### 9.1 cp命令

- 用于复制文件或文件夹

- 语法:

  ~~~shell
  cp[-r]参数1 参数2
  ~~~

- -r选项，可选，用于复制文件夹使用，表示递归

- 参数1，Linux路径，表示被复制的文件或文件夹

- 参数2，Linux路径，表示要复制去的地方

### 9.2 mv命令

- 用于移动或重命名文件/文件夹

- 语法:

  ~~~shell
  mv 参数1 参数2
  ~~~

- 参数1，Linux路径，表示被移动的文件或文件夹

- 参数2，Linux路径，表示要移动去的地方，如果目标不存在，则进行改名

### 9.3 rm命令

- 用于删除文件或文件夹

- 语法:

  ~~~shell
  rm[-r-f]参数1 参数2参数N
  ~~~

- -r选项，可选，文件夹删除

- -f选项，可选，用于强制删除(不提示，一般用于root用户)参数，表示被删除的文件或文件夹路径，支持多个，空格隔开参数也支持**通配符 ***，用以做模糊匹配



## 10 which,find命令

### 10.1 which命令

- 查找命令的程序文件

- 语法:

  ~~~shell
  which 要查找的命令
  ~~~

- 无需选项，只需要参数表示查找哪个命令

### 10.2 find命令

- 用于查找指定的文件

- 按文件名查找:

  ~~~shell
  find 起始路径 -name“被查找文件名
  ~~~

  - 支持通配符

- 按文件大小查找:

  ~~~shell
  find 起始路径-size +| -n[kMG]
  ~~~



## 11 grep,wc,管道符

### 11.1 grep命令

- 从文件中通过关键字过滤文件行

- 语法:

  ~~~shell
  grep[-n]关键字 文件路径
  ~~~

- 选项-n，可选，表示在结果中显示匹配的行的行号，

- 参数，关键字，必填，表示过滤的关键字，建议使用””将关键字包围起来

- 参数，文件路径，必填，表示要过滤内容的文件路径，**可作为管道符的输入**

### 11.2 wc命令

- 命令统计文件的行数、单词数量、字节数、字符数等

- 语法:

  ~~~shell
  wc[-c-m-1-w]文件路径
  ~~~

- 不带选项默认统计:行数、单词数、字节数

- -c字节数、-m字符数、-l行数、-w单词数

- 参数，被统计的文件路径，可作为**管道符的输入**

### 11.3 管道符|

将管道符左边的结果，作为右边命令的输入



## 12 echo,tail,重定向符

### 12.1 echo

- 可以使用echo命令在命令行内输出指定内容
- 语法:

~~~shell
echo 输出的内容
~~~

- 无需选项，只有一个参数，表示要输出的内容，复杂内容可以用””包围

### 12.2 反引号符

- 被`包围的内容，会被作为命令执行，而非普通字符

### 12.3 重定向符

- 将符号左边的结果，输出到右边指定的文件中去


- `>`，表示覆盖输出
- `>>`，表示追加输出

### 12.4 tail

- 使用tail命令，可以查看文件尾部内容，跟踪文件的最新更改

- 语法:

  ~~~shell
  tail[-f -num]Linux路径
  ~~~

- 参数，Linux路径，表示被跟踪的文件路径

- 选项，-f，表示持续跟踪

- 选项,-num，表示，查看尾部多少行，不填默认10行



## 13 vi编辑器

vi\vim是visual interface的简称,是Linux中最经典的文本编辑器

同图形化界面中的文本编辑器一样，vi是命令行下对文本文件进行编辑的绝佳选择。

vim 是 vi 的加强版本，兼容 vi 的所有指令，不仅能编辑文本，而且还具有 shelll程序编辑的功能，可以不同颜色的字体来辨别语法的正确性，极大方便了程序的设计和编辑性。



<center class="half">
    <img src="./assets/image-20240914164528923.png" width="500"/>
    <img src="./assets/image-20240914165106041.png" width="500" height="310"/>
</center>

### 13.1 三种工作模式

- 命令模式，默认的模式，可以通过键盘快捷键控制文件内容
- 输入模式，通过命令模式进入，可以输入内容进行编辑，按esc退回命令模式
- 底线命令模式，通过命令模式进入，可以对文件进行保存、关闭等操作

![image-20240914164922955](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854626.png)

### 13.2 命令模式快捷键

![image-20221027215841573](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/2022/10/27/20221027215841.png)

![image-20221027215846581](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/2022/10/27/20221027215846.png)

![image-20221027215849668](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/2022/10/27/20221027215849.png)

### 13.3 底线命令快捷键

![image-20221027215858967](https://image-set.oss-cn-zhangjiakou.aliyuncs.com/img-out/2022/10/27/20221027215858.png)



# 第三章

## 1 Linux的root用户

### 1.1 root用户

root用户拥有最大的系统操作权限,而普通用户在许多地方的权限是受限的。

演示:

- 使用普通用户在根目录下创建文件夹

![image-20240914200227002](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854656.png)

- 切换到root用户后,继续尝试

![image-20240914200244081](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854685.png)

- 普通用户的权限，一般在其HOME目录内是不受限的
- 一旦出了HOME目录,大多数地方,普通用户仅有只读和执行权限,无修改权限

### 1.2 su命令

- 可以切换用户，语法:

  ~~~shell
  su[-][用户名]
  ~~~

- 表示切换后加载环境变量，建议带上

- 用户可以省略,省略默认切换到root

### 1.3 sudo命令

- 可以让一条普通命令带有root权限,语法:

  ~~~shell
  sudo 其它命令
  ~~~

- 需要以root用户执行visudo命令,增加配置方可让普通用户有sudo命令的执行权限

**为普通用户配置sudo认证**

- 切换到root用户，执行visudo命令,会自动通过vi编辑器打开:/etc/sudoers

- 在文件的最后添加:

![image-20240914201618540](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854711.png)

- 其中最后的NOPASSWD:ALL表示使用sudo命令，无需输入密码

- 最后通过 wq 保存



## 2 用户和用户组

### 2.1 Linux用户管理模式

- Linux可以支持多用户、多用户组、用户加入多个组
- Linux权限管控的单元是用户级别和用户组级别

### 2.2 用户、用户组相关管理命令

- groupadd添加组、groupdel删除组
- useradd添加用户、userdel删除用户
- usermod修改用户组、id命令查看用户信息
- getent passwd查看系统全部用户信息
- getent group查看系统全部组信息



## 3 查看权限控制信息

### 3.1 认知权限信息

![image-20240915215118290](F:\Documents\GitHub\hwj-s-study-notes\Linux\assets\image-20240915215118290.png)

- 序号1,表示文件、文件夹的权限控制信息
- 序号2,表示文件、文件夹所属用户
- 序号3,表示文件、文件夹所属用户组

权限细节总共分为10个槽位

<img src="F:\Documents\GitHub\hwj-s-study-notes\Linux\assets\image-20240915215250586.png" alt="image-20240915215250586" style="zoom:150%;" />

举例:`drwxr-xr-x`,表示：

- 这是一个文件夹，首字母d表示
- 所属用户(右上角图序号2)的权限是:有r有w有x，rwx
- 所属用户组(右上角图序号3)的权限是:有r无w有x,r-x(-表示无此权限)
- 其它用户的权限是:有r无w有x，r-x

### 3.2 rwx

- r表示读权限
- w表示写权限
- x表示执行权限

针对文件、文件夹的不同，rwx的含义有细微差别

- r,针对文件可以查看文件内容
  - 针对文件夹，可以查看文件夹内容，如`ls`命令

- w,针对文件表示可以修改此文件
  - 针对文件夹，可以在文件夹内:创建、删除、改名等操作
- x,针对文件表示可以将文件作为程序执行
  - 针对文件夹，表示可以更改工作目录到此文件夹，即`cd`进入



## 4 修改权限控制 - chmod

### 4.1 chmod命令

我们可以使用chmod命令,修改文件、文件夹的权限信息。

<font color = red>注意,只有文件、文件夹的所属用户或root用户可以修改。</font>

语法:`chmod [-R] 权限 文件或文件夹`

- 选项:-R,对文件夹内的**全部内容**应用同样的操作



示例:

- chmod u=rwx,g=rx,o=x hello.txt,将文件权限修改为:rwxr-x--x·
  - 其中:u表示user所属用户权限,g表示group组权限,o表示other其它用户权限
- chmod -R u=rwx,g=rx,o=xtest,将文件夹test以及文件夹内全部内容权限设置为:rwxr-x--x

### 4.2 权限的数字序号

权限可以用3位数字来代表,第一位数字表示用户权限,第二位表示用户组权限,第三位表示其它用户权限数字的细节如下:r记为4，w记为2,x记为1,可以有:

| 序号 |        权限        |
| :--: | :----------------: |
|  0   | 无任何权限，即---  |
|  1   | 仅有x权限， 即--x  |
|  2   |  仅有w权限，即-w-  |
|  3   | 有w和x权限，即 -wx |
|  4   |  仅有r权限，即r--  |
|  5   | 有r和x权限，即r-x  |
|  6   | 有r和w权限，即 rw- |
|  7   | 有全部权限，即rwx  |

所以751表示:`rwx(7)r-x(5)--x(1)`



<u>案例</u>

- 将hello.txt的权限修改为:r-x--xr-x,数字序号为:

`chmod 515 hello.txt`

- 将hello.txt的权限修改为:-wx-w-rw-，数字序号为

`chmod 326 hello.txt`



## 5 修改权限控制 - chown

### 5.1 chown命令

使用chown命令,可以修改文件、文件夹的所属用户和用户组

<font color = red>普通用户无法修改所属为其它用户或组，所以此命令只适用于root用户执行</font>

语法:`chown[-R][用户][:][用户组]文件或文件夹`

- 选项，-R,同chmod,对文件夹内全部内容应用相同规则
- 选项，用户,修改所属用户
- 选项，用户组，修改所属用户组
- :用于分隔用户和用户组



示例:

- chown root hello.txt,将hello.txt所属用户修改为root
- chown :root hello.txt,将hello.txt所属用户组修改为root
- chown root:itheima hello.txt,将hello.txt所属用户修改为root,用户组修改为itheima
- chown -Rroottest,将文件夹test的所属用户修改为root并对文件夹内全部内容应用同样规则

# 第四章

## 1 各类小技巧快捷键



- ctrl + c 强制停止
- ctrl + d 退出登出
- history  查看历史命令
- ! 命令前缀,自动匹配上一个命令
- ctrl + r ,搜索历史命令
- ctrl + a | e ,光标移动到命令开始或结束
- ctrl + | → ,左右跳单词
- ctrl + l 或  clear 命念清屏



## 2 软件安装

### 2.1 在CentOs系统中，使用yum命令联网管理软件安装

yum语法:

~~~shell
yum[-y][install|remove|search]软件名称
~~~

### 2.2 在Ubuntu系统中,使用apt命令联网管理软件安装

apt语法:

~~~shell
apt[-y][install|remove|search]软件名称
~~~

卡住不动？**更新镜像源**

- 备份原来的 `CentOS-Base.repo`：

~~~shell
cp /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak
~~~

- 编辑 `/etc/yum.repos.d/CentOS-Base.repo` 文件：

~~~shell
vim /etc/yum.repos.d/CentOS-Base.repo
~~~

- 将内容替换为以下阿里云的镜像源：

~~~
[base]
name=CentOS-$releasever - Base
baseurl=http://mirrors.aliyun.com/centos/$releasever/os/$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7

[updates]
name=CentOS-$releasever - Updates
baseurl=http://mirrors.aliyun.com/centos/$releasever/updates/$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7

[extras]
name=CentOS-$releasever - Extras
baseurl=http://mirrors.aliyun.com/centos/$releasever/extras/$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7
~~~

- 在修改源之后，清理缓存并更新 YUM：

~~~shell
yum clean all
yum makecache
~~~

- 然后重新尝试安装 `wget`：

```shell
yum install wget
```



## 3 systemctl命令控制软件的启动和关闭

3.1 systemctl命令的作用
可以控制软件(服务)的启动、关闭、开机自启动

- 系统内置服务均可被systemctl控制
- 第三方软件，如果自动注册了可以被systemctl控制
- 第三方软件，如果没有自动注册，可以手动注册

3.2 语法

~~~shell
systemctl start | stop | status | enable | disable 服务名
~~~



## 4 软连接

### 4.1 什么是软连接?

可以将文件、文件夹链接到其它位置
链接只是一个指向，并不是物理移动，类似Windows系统的快捷方式

### 4.2 软连接的使用语法

~~~shell
ln -s 参数1 参数2
~~~

- -s选项，创建软连接
- 参数1:被链接的文件或文件夹
- 参数2:要链接去的目的地

![image-20240914215120169](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854739.png)



## 5 日期和时区

### 5.1 date命令的作用和用法

- date命令可以查看日期时间，并可以格式化显示形式以及做日期计算
- 语法:

~~~shell
date[-d][+格式化字符串]
~~~

- %Y 年
- %y 年份后两位数字(00.99).
- %m 月份
- %d 日
- %H 小时
- %M 分钟
- %S秒
- %s自1970-01-01 00:00:00 UTC 到现在的秒数

### 5.2 如何修改Linux时区

~~~shell
rm -f /etc/localtime
ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
~~~

### 5.3 ntp的作用

可以自动联网同步时间,也可以通过`ntpdate -untp.aliyun.com`手动校准时间



## 6 IP地址和主机名

### 6.1 什么是IP地址,有什么作用?

- IP地址是联网计算机的网络地址，用于在网络中进行定位
- 格式是:a.b.c.d,其中abcd是0~255的数字
- 特殊IP有:127.0.0.1,本地回环IP,表示本机。
- 0.0.0.0:也可表示本机,也可以在一些白名单中表示任意IP

### 6.2 什么是主机名?

- 主机名就是主机的名称，用于标识一个计算机

### 6.3 什么是域名解析(主机名映射)

- 可以通过主机名找到对应计算机的IP地址，这就是主机名映射(域名解析)
- 先通过系统本地记录查找（host），如果找不到就联网去公开DNS服务器去查找

![image-20240914223745051](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854769.png)

## 7 配置Linux固定IP地址

### 7.1 为什么需要固定IP？

当前我们虚拟机的Linux操作系统，其IP地址是通过DHCP服务获取的。

DHCP:动态获取IP地址,即每次重启设备后都会获取一次，可能导致IP地址频繁变更

原因1:办公电脑IP地址变化无所谓,但是我们要远程连接到Linux系统,如果IP地址经常变化我们就要频繁修改适配很麻烦

原因2:在刚刚我们配置了虚拟机P地址和主机名的映射,如果IP频繁更改,我们也需要频繁更新映射关系

综上所述，我们需要IP地址固定下来，不要变化了。

### 7.2 在VMware Workstation中配置固定IP

配置固定IP需要2个大步骤:

- 在VMware Workstation(或Fusion)中配置IP地址网关和网段(IP地址的范围）
- 在Linux系统中手动修改配置文件，固定IP



![image-20240914232552824](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854798.png)

![image-20240914232629171](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854829.png)

![image-20240914232735188](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854852.png)

- 使用vim编辑/etc/sysconfig/network-scripts/ifcfg-ens33文件,填入如下内容

~~~
TYPE="Ethernet"
PROXY_METHOD="none"
BROWSER_ONLY="no"
BOOTPROTO="static"
DEFROUTE="yes"
IPV4_FAILURE_FATAL="no"
IPV6INIT="yes"
IPV6_AUTOCONF="yes"
IPV6_DEFROUTE="yes"
IPV6_FAILURE_FATAL="no"
IPV6_ADDR_GEN_MODE="stable-privacy"
NAME="ens33"
UUID="fe2bd086-14bc-4f40-83ab-c50ef3586f13"
DEVICE="ens33"
ONBOOT="yes"
IPADDR="192.168.88.130"
NETMASK="255.255.255.0"
GATEWAY="192.168.88.2"
DNS1="192.168.88.2"
~~~



![image-20240914232802538](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854884.png)

- 执行：

~~~shell
systemctl restart network
~~~



## 8 网络请求和下载

### 8.1 ping命令

- 通过ping命令,检查指定的网络服务器是否是可联通状态
- 语法:

~~~shell
ping[-c num]ip或主机名
~~~

- 选项:-c,检查的次数,不使用-c选项,将无限次数持续检查
- 参数:ip或主机名，被检查的服务器的ip地址或主机名地址

示例：

- 检查baidu.com是否联通

![image-20240915000122397](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854916.png)

- 检查baidu.com是否联通，并检查3次

![image-20240915001053780](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854952.png)

### 8.2 wget命令

- wget是非交互式的文件下载器，可以在命令行内下载网络文件
- 语法:

```shell
wget [-b] url
```

- 选项:-b,可选,后台下载,会将日志写入到当前工作目录的wget-log文件
- 参数:url, 下载链接

示例：

![image-20240915001522504](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854982.png)

- 
  在后台下裁: wget -b http://archive.apache.org/dist/hadoop/common/hadoop-3.3.0/hadoop-3.3.0.tar.gz

- 通过tail命令可以监控后台下载进度:tail -f wget-log


<font color=red>注意:无论下载是否完成,都会生成要下载的文件,如果下载未完成,请及时清理未完成的不可用文件。</font>

### 8.3 curl命令

- curl可以发送http网络请求，可用于:下载文件、获取信息等

- 语法:

```shell
cur1 [-0] ur1
```

- 选项:-0，用于下载文件，当url是下载链接时,可以使用此选项保存文件
- 参数:url,要发起请求的网络地址

示例：

- 向cip.cc发起网络请求:curl cip.cc

![image-20240915004216336](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854010.png)

- 向python.itheima.com发起网络请求:curl python.itheima.com

直接展示html

- 通过curl下载hadoop-3.3.0安装包: curl -0 http://archive.apache.org/dist/hadoop/common/hadoop.3.3.0/hadoop-3.3.0.tar.gz

- 
  ![image-20240915004515031](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150854039.png)



## 9 端口

### 9.1 什么是端口？

端口，是设备与外界通讯交流的出入口。端口可以分为:物理端口和虚拟端口两类

- 物理端口:又可称之为接口,是可见的端口,如USB接口，RJ45网口,HDMI端口等
- 虚拟端口:是指计算机内部的端口,是不可见的,是用来操作系统和外部进行交互使用的

![image-20240915081758436](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150818067.png)

计算机程序之间的通讯,通过IP只能锁定计算机,但是无法锁定具体的程序。通过端口可以锁定计算机上具体的程序,确保程序之间进行沟通



<font color =red>IP地址相当于小区地址,在小区内可以有许多住户(程序)而门牌号(端口)就是各个住户(程序)的联系地址</font>

### 9.2 端口的划分

Linux系统是一个超大号小区,可以支持65535个端口,这6万多个端口分为3类进行使用:

- 公认端口:1~1023,通常用于一些系统内置或知名程序的预留使用,如SSH服务的22端口,HTTPS服务的443端口非特殊需要，不要占用这个范围的端口
- 注册端口:1024~49151,通常可以随意使用,用于松散的绑定一些程序\服务
- 动态端口:49152~65535,通常不会固定绑定程序,而是当程序对外进行网络链接时,用于临时使用。

![image-20240915082143381](https://admin-hwj.oss-cn-beijing.aliyuncs.com/202409150821130.png)

### 9.3 查看端口占用

- `nmap IP地址`,查看指定IP的对外暴露端口
- `netstat -anp | grep xxx`,查看本机指定端口号的占用情况



## 10 进程

### 10.1 什么是进程？

进程是指程序在操作系统内运行后被注册为系统内的一个进程，并拥有独立的进程ID(进程号)

### 10.2 管理进程的命令

<u>ps命令</u>

功能：查看进程信息

语法：`ps -ef`，查看全部进程信息，可以搭配grep做过滤：`ps -ef | grep xxx`



<u>kill命令</u>

语法:：`ki11 [-9] 进程ID`
选项：-9,表示强制关闭进程。不使用此选项会向进程发送信号要求其关闭,但是否关闭看进程自身的处理机制。



## 11 主机状态监控

### 11. 1 top命令

功能：查看主机运行状态

语法：`top`，查看基础信息

- 可用选项

| 选项 |                             功能                             |
| :--: | :----------------------------------------------------------: |
|  -p  |                     只显示某个进程的信息                     |
|  -d  |                    设置刷新时间，默认是5s                    |
|  -c  |             显示产生进程的完整命令，默认是进程名             |
|  -n  |        指定刷新次数，比如 top -n 3，刷新输出3次后退出        |
|  -b  | 以非交互非全屏模式运行，以批次的方式执行top，一般配合-n指定输出几次统计信息，将输出重定向到指定文件，比如 top  -b -n 3 > /tmp/top.tmp |
|  -i  |           不显示任何闲置(idle)或无用(zombie)的进程           |
|  -u  |                    查找特定用户启动的进程                    |

- top交互式选项

| 按键 |                             功能                             |
| :--: | :----------------------------------------------------------: |
| h键  |                   按下h键，会显示帮助画面                    |
| c键  | 按下c键，会显示产生进程的完整命令，等同于-c参数，再次按下c键，变为默认显示 |
| f键  |               按下f键，可以选择需要展示的项目                |
| M键  |              按下M键，根据驻留内存大小(RES)排序              |
| P键  |            按下P键，根据CPU使用百分比大小进行排序            |
| T键  |              按下T键，根据时间/累计时间进行排序              |
| E键  |                按下E键，切换顶部内存显示单位                 |
| e键  |                按下e键，切换进程内存显示单位                 |
| l键  |           按下l键，切换显示平均负载和启动时间信息            |
| i键  | 按下i键，不显示闲置或无用的进程，等同于-i参数，再次按下，变为默认显示i键 |
| t键  |                 按下t键，切换显示CPU状态信息                 |
| m键  |                  按下m键，切换显示内存信息                   |

示例：

- top:命令名称,14:39:58:当前系统时间,up6min:启动了6分钟,2users:2个用户登录,load:1、5、15分钟负载

![**6d0fe5aa-f081-4404-b2d0-2c4bdfc3068b**](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151839190.png)

- Tasks:175个进程,1running:1个进程子在运行,174sleeping: 174个进程睡眠,0个停止进程,0个僵尸进程

![image-20240915161328575](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151839191.png)

- %Cpu(5):CPU使用率,S:用户CPU使用率,sv:系统CPU使用率,ni:高优先级进程占用CPU时间百分比,id:空闲CPU率,wa: 0等待CPU占用率,hi: CPU硬件中断率，si:CPU软件中断率，st:强制等待占用CPU率

![image-20240915161347731](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151839192.png)

- Kib Mem:物理内存,total: 总量,free:盛闲,used:使用,buff/cache:buff和cache占用
- Kibswap:虚拟内存(交换空间),total:总量,free:空闲,used: 使用,buff/cache:buff和cache占用

![image-20240915161538033](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151839193.png)

![image-20240915161736053](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151839194.png)

- PID:进程id
- USER:进程所属用户
- PR:进程优先级，越小越高
- N1:负值表示高优先级，正表示低优先级
- VIRT:进程使用虚拟内存，单位KB
- RES:进程使用物理内存，单位KB
- SHR:进程使用共享内存，单位KB
- S:进程状态（S休眠，R运行,Z僵死状态，N负数优先级，空闲状态）
- %CPU:进程占用CPU率
- %MEM:进程占用内存率
- TIME+:进程使用CPU时间总计，单位10毫秒
- COMMAND:进程的命令或名称或程序件路径

### 11.2 df命令

查看磁盘使用率

### 11.3 iostat命令

查看磁盘速率等信息

### 11.4 sar命令

`sar -n DEV`

查看网络情况



## 12 环境变量

### 12.1 什么是环境变量？

环境变量是操作系统(Windows、Linux、Mac)在运行的时候,记录的一些关键性信息,用以辅助系统运行。环境变量是一种KeyValue型结构。

### 12.2 env命令

在Linux系统中执行:`env`命令即可查看当前系统中记录的环境变量

### 12.3 \$符号

在Linux系统中,\$符号被用于取”变量”的值。环境变量记录的信息，除了给操作系统自己使用外,如果我们想要取用,也可以使用。取得环境变量的值就可以通过语法:`$环境变量名`来取得
比如:`echo $PATH`
就可以取得`PATH`这个环境变量的值,并通过`echo`语句输出出来

![image-20240915191059547](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151917655.png)

又或者:`echo ${PATH}ABC`
当和其它内容混合在一起的时候，可以通过来标注取的变量是谁

![image-20240915191152199](https://admin-hwj.oss-cn-beijing.aliyuncs.com/img/202409151917656.png)

### 12.4 自行设置环境变量

Linux环境变量可以用户自行设置,其中分为:

- 临时设置，语法:`export 变量名=变量值`

- 永久生效

  - 针对当前用户生效，配置在当前用户的`~/.bashrc`文件中

  - 针对所有用户生效，配置在系统的`/etc/profile`文件中

  - 并通过语法:`source 配置文件`，进行立刻生效，或重新登录FinalShell生效



## 13 Linux文件到上传和下载

### 13.1 如何使用Finalshell对Linux系统进行上传下载操作?

直接拖拽（切换root用户）

### 13.2 rz、sz命令

通过 `yum -yum install lrzsz`可以安装此命令

`rz` 进行文件上传
`sz` 文件进行文件下载



## 14 压缩和解压

市面上有非常多的压缩格式

- zip格式:Linux、Windows、MacOs,常用7zip:Windows系统常用
- rar:Windows系统常用
- tar: Linux、MacOs常用
- gzip:Linux、MacOs常用

### 14.1 tar命令

- .tar,称之为tarbal,归档文件,即**简单**的将文件**组装**到一个.tar的文件内,并没有太多文件体积的减少,仅仅是简单的封装
- .9z,也常见为.tar.gz,gzip格式压缩文件,即使用gzip压缩算法将文件压缩到一个文件内,可以极大的**减少压缩后的体积**

语法:`tar [-c -v -x -f -z -C]参数1 参数2...参数N`

- -c,创建压缩文件,用于压缩模式
- -v,显示压缩、解压过程,用于查看进度
- -x,解压模式
- -f,要创建的文件,或要解压的文件,-f选项必须在所有选项中位置处于最后一个
- -z,gzip模式,不使用-z就是普通的tarball格式
- -C,选择解压的目的地，用于解压模式



<u>tar的常用组合为:</u>

- `tar -cvf test.tar 1.txt 2.txt 3.txt`

将1.txt 2.txt 3.txt压缩到test.tar文件内

- `tar -zcvf test.tar.gz 1.txt 2.txt 3.txt`

将1.txt 2.txt 3.txt压缩到test.tar.gz文件内,使用gzip模式

> [!CAUTION]
>
> `-z`选项如果使用的话，一般处于选项位第一个
> `-f`选项<font color = red>必须</font>在选项位最后一个



<u>常用的tar解压组合有</u>

- `tar -xvf test.tar`

解压test.tar,将文件解压至当前目录

- `tar -xvf test.tar -C /home/itheima`

解压test.tar,将文件解压至指定目录(/home/itheima)

- `tar -zxvf test.tar.gz -C /home/itheima`

以Gzip模式解压test.tar.gz,将文件解压至指定目录(/home/itheima)

> [!CAUTION]
>
> 注意:
> `-f`选项，必须在选项组合体的最后一位
> `-z`选项，建议在开头位置
>
> `-C`选项单独使用,和解压所需的其它参数分开

### 14.2 zip 命令压缩文件

可以使用zip命令,压缩文件为zip压缩包



语法:zip[-r]参数1 参数2 ..· 参数N

- -r,被压缩的包含文件夹的时候,需要使用-r选项,和rm、cp等命令的-r效果一致



示例:

- zip test.zip a.txt b.txt c.txt

将a.txt b.txt c.txt压缩到test.zip文件内

- zip -r test.zip test itheima a.txt
- 将test、itheima两个文件夹和a.txt文件，压缩到test.zip文件内



### 14.3 unzip 命令解压文件

使用unzip命令，可以方便的解压zip压缩包



语法: `unzip [-d] 参数`

- -d,指定要解压去的位置，同tar的 -C 选项
- 参数，被解压的zip压缩包文件

示例:

- unzip test.zip,将test.zip解压到当前目录
- unzip test.zip -d /home/itheima,将test.zip解压到指定文件夹内(/home/itheima)

***

***

完结撒花！

# 第五章 在Linux上部署各类软件（略）
