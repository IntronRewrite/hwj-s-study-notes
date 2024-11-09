# 1463: 蓝桥杯基础练习VIP-Sine之舞

#### 没什么好说的，递归输出An，Sn，再把An嵌入Sn

**代码如下：**

~~~python
N = int(input())
def s(n):
    global N
    if n == N:
        if n!=1:
            print('(',end='')
        a(1,N-n+1)
        print('+{}'.format(n),end='')
        if n!=1:
            print(')',end='')
        return
    if n == 1:
        s(n+1)
        a(1,N-n+1)
        print('+{}'.format(n),end='')
    else:
        print('(',end='')
        s(n+1)
        a(1,N-n+1)
        print('+{}'.format(n),end='')
        print(')',end='')
    
def a(n_a,E):
    global N
    if n_a == 1:
        print('sin(1',end='')
        if n_a != E:
            a(n_a+1,E)
        print(')',end='')
    else:
        if n_a%2 == 0:
            print('-',end='')
        else:
            print('+',end='')
        print('sin({}'.format(n_a),end='')
        if n_a != E:
            a(n_a+1,E)
        print(')',end='')
s(1)
~~~

#### 1029: [编程入门]自定义函数处理素数(python)

~~~python
def is_prime(n):
    for i in range(2, int(n**0.5) + 1):
        if n%i == 0:
            print('not prime')
            return
    print('prime')
n = int(input())
is_prime(n)
~~~



#### 1031: [编程入门]自定义函数之字符串反转

~~~python
def main():
    s = input()
    s = list(s)
    s.reverse()
    s = ''.join(s)
    print(s)

if __name__ == '__main__':
    main()
~~~

#### 无需多言



~~~python
def main():
    print(input() + input())
    
if __name__ == '__main__':
    main()
~~~



####  1033: [编程入门]自定义函数之字符提取

​	可以采用关键字`in`来查询指定字符是否存在于指定字符串中，如果字符串中存在指定字符则返回True，如果不存在则返回False。

~~~python
def main():
    s = input()
    for i in s:
        if i in 'aeiou':
            print(i,end='')
    
if __name__ == '__main__':
    main()
~~~



[题目1034:[编程入门]自定义函数之数字分离](https://www.dotcpp.com/oj/problem1034.html?sid=18058228&lang=6#editor)

使用‘ ’.join()连接即可

~~~python
print(' '.join(input()))
~~~



[题目1045:[编程入门]自定义函数之整数处理](https://www.dotcpp.com/oj/problem1045.html?sid=18059704&lang=6#editor)

~~~python
a = list(map(int,input().split()))
def swap(a):
    min_num = min(a)
    min_index = a.index(min_num )
    tmp = a[0]
    a[0] = min_num
    a[min_index] = tmp

    max_num = max(a)
    max_index = a.index(max_num)
    tmp = a[-1]
    a[-1] = max_num
    a[max_index] = tmp
    
swap(a)
print(' '.join(map(str,a)))
~~~



把末尾的插入`a[0]`,再把末尾的删除

~~~python
n = int(input())
a = list(map(int,input().split()))
m = int(input())
def move(a,m):
    for i in range(m):
        a.insert(0,a[-1])
        del a[-1]
        
    print(' '.join(map(str,a)))
move(a,m)
~~~



从索引`m-1`开始切片即可

~~~pythjon
input()
s = input()
m = int(input())
print(s[m-1:])
~~~





~~~python
def is_lun(y):
    if y%100 == 0:
        if y%400 == 0:
            return True
        else:
            return False
    else:
        if y%4 == 0:
            return True
        else:
            return False
a = [0,31,59,90,120,151,181,212,243,273,304,334]
b = [0,31,60,91,121,152,182,213,244,274,305,335]
y,m,d = map(int,input().split())
if is_lun(y):
    day = b[m-1] + d
else:
    day = a[m-1] + d
print(day)
~~~



对python来说还是很好实现的

~~~python
for i in range(int(input())):
    print(','.join(map(str,[i for i in input().split()])))
~~~





[1067:二级C语言-分段函数](https://www.dotcpp.com/oj/problem1067.html?sid=18062507&lang=6#editor) 

~~~python
def f(x):
    if x < 0:
        return abs(x)
    elif x < 2:
        return (x + 1)**0.5
    elif x < 4:
        return (x + 2)**3
    else:
        return 2 * x + 5

x = float(input())
print('{:.2f}'.format(f(x)))
~~~

Python的默认最大递归深度（通常为1000）。当输入的x值较大时，函数f(x)会进行大量的递归调用，导致递归深度超过限制。
**解决方法**

- 增加递归深度限制：可以通过sys.setrecursionlimit()来增加递归深度限制，但这只是权宜之计，不推荐用于生产环境。
- 使用迭代：将递归改为迭代，这样可以避免递归深度限制的问题。

~~~python
def f(x):
##    if x == 1:
##        return 10
##    else:
##        return f(x-1) + 2
    if x == 1:
        return 10
    else:
        tmp =10
        for i in range(2,x+1):
            tmp += 2
        return tmp
            

x = int(input())
print('{}'.format(f(x)))
~~~

