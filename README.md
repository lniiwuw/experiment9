## 实验9 BP神经网络实验

### 一、实验描述

#### 1. 实验目的

（1）通过Python语言编程实现BP网络逼近标准正弦函数,来加深对BP网络的了解和认识；

（2）理解信号的正向传播和误差的反向传递过程。

#### 2. 实验内容

要求用Python语言实现三层BP前馈网络。

### 二、涉及知识点

BP算法的基本思想是把学习过程分为两个阶段：

1. 第一阶段是信号的正向传播过程;输入信息通过输入层、隐层逐层处理并计算每个单元的实际输出值；
2. 第二阶段是误差的反向传递过程;若在输入层未能得到期望的输出值,则逐层递归的计算实际输出和期望输出的差值(即误差),以便根据此差值调节权值。这种过程不断迭代,最后使得信号误差达到允许或规定的范围之内。

### 三、实验步骤

1. 系统给出该实验部分代码，需要用户补全
2. 用户补全代码示例：

```python
# 反向更新
###########开始1#############
for l in range(len(a) - 2, 0, -1):
  deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
  deltas.reverse()

for i in range(len(self.weights)):
  ayer = np.atleast_2d(a[i])
  delta = np.atleast_2d(deltas[i])
  self.weights[i] += learning_rate * layer.T.dot(delta)
###########结束1#############
```

3. 用户点击运行，系统返回运行结果给用户，如下：

测试输入：

输出：

`success` 

### 四、实验代码

见`src/main.py`文件