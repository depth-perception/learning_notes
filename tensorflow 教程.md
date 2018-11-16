# tensorflow 教程

# 1.处理结构

##  计算图纸 

Tensorflow 首先要定义神经网络的结构, 然后再把数据放入结构当中去运算和 training.

[![处理结构](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/tensorflow/1_4_1.png)

(动图效果请点击[这里](https://www.tensorflow.org/images/tensors_flowing.gif))

因为TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图, 然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算. 节点（Nodes）在图中表示数学操作,图中的线（edges）则表示在节点间相互联系的**多维数据数组**, 即张量（tensor). 训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来.

## Tensor 张量意义 

**张量（Tensor)**:

- 张量有多种. 零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 `[1]`
- 一阶张量为 向量 (vector), 比如 一维的 `[1, 2, 3]`
- 二阶张量为 矩阵 (matrix), 比如 二维的 `[[1, 2, 3],[4, 5, 6],[7, 8, 9]]`
- 以此类推, 还有 三阶 三维的 …

# 2.例子

Tensorflow 是非常重视结构的, 我们得建立好了神经网络的结构, 才能将数字放进去, 运行这个结构.

这个例子简单的阐述了 tensorflow 当中如何用代码来运行我们搭建的结构.

## 创建数据 

首先, 我们这次需要加载 tensorflow 和 numpy 两个模块, 并且使用 numpy 来创建我们的数据.

```
import tensorflow as tf
import numpy as np

# create data
#在tensorflow中大部分的数据是按照float32的形式存储的。
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
```

接着, 我们用 `tf.Variable` 来创建描述 `y` 的参数. 我们可以把 `y_data = x_data*0.1 + 0.3` 想象成 `y=Weights * x + biases`, 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.

## 搭建模型 

```
#[1]表示向量
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
```

## 计算误差 

接着就是计算 `y` 和 `y_data` 的误差:

```
loss = tf.reduce_mean(tf.square(y-y_data))
```

## 传播误差 

反向传递误差的工作就教给`optimizer`了, 我们使用的误差传递方法是梯度下降法: `Gradient Descent` 让后我们使用 `optimizer` 来进行参数的更新.

```
#0.5表示学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
```

## 训练 

到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 在使用这个结构之前, 我们必须先初始化所有之前定义的`Variable`, 所以这一步是很重要的!

```
#设置了变量以后先初始化
init = tf.global_variables_initializer()  # 替换成这样就好
```

接着,我们再创建会话 `Session`. 我们会在下一节中详细讲解 Session. 我们用 `Session` 来执行 `init` 初始化步骤. 并且, 用 `Session` 来 `run` 每一次 training 的数据. 逐步提升神经网络的预测准确性.

```
sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
```

# 3.Session 会话控制

## 简单运用 

欢迎回来！这一次我们会讲到 Tensorflow 中的 `Session`, `Session` 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 `session.run()` 可以获得你要得知的运算结果, 或者是你所要运算的部分.

首先，我们这次需要加载 Tensorflow ，然后建立两个 `matrix` ,输出两个 `matrix` 矩阵相乘的结果。

```
import tensorflow as tf

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)
```

因为 `product` 不是直接计算的步骤, 所以我们会要使用 `Session` 来激活 `product` 并得到计算结果. 有两种形式使用会话控制 `Session` 。

```
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
```

# 4.Variable 变量

## 简单运用 

这节课我们学习如何在 Tensorflow 中使用 `Variable` .

在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。

定义语法： `state = tf.Variable()`

```
import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)
```

如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 `init = tf.initialize_all_variables()` .

到这里变量还是没有被激活，需要再在 `sess` 里, `sess.run(init)` , 激活 `init` 这一步.

```
# 如果定义 Variable, 就一定要 initialize

init = tf.global_variables_initializer()  # 替换成这样就好
 
# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```

注意：直接 `print(state)` 不起作用！！

一定要把 `sess` 的指针指向 `state` 再进行 `print` 才能得到想要的结果！

以上就是我们今天所学的 `Variable` 打开模式，欢迎继续学习下一章 ———— Tensorflow 中的 `Placeholder`。

# 5.Placeholder 传入值

## 简单运用 

这一次我们会讲到 Tensorflow 中的 `placeholder` , `placeholder` 是 Tensorflow 中的占位符，**暂时储存变量**.

Tensorflow 如果想要从外部传入data, 那就需要用到 `tf.placeholder()`, 然后以这种形式传输数据 `sess.run(***, feed_dict={input: **})`.

示例：

```
#先搭框架，在输数据
import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
#如果需要设定结构为2x2，则输出：input1 =tf.placeholder(tf.float32，[2,2])

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)
```

接下来, 传值的工作交给了 `sess.run()` , 需要传入的值放在了`feed_dict={}` 并一一对应每一个 `input`. `placeholder` 与 `feed_dict={}` 是绑定在一起出现的。

```
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]
```

# 6.什么是激励函数 (Activation Function)

## 非线性方程 

我们为什么要使用激励函数? 用简单的语句来概括. 就是因为, 现实并没有我们想象的那么美好, 它是残酷多变的. 哈哈, 开个玩笑, 不过激励函数也就是为了解决我们日常生活中 不能用线性方程所概括的问题. 好了,我知道你的问题来了. 什么是线性方程 (linear function)?

[![激励函数 (Activation Function)](https://morvanzhou.github.io/static/results/ML-intro/active1.png)](https://morvanzhou.github.io/static/results/ML-intro/active1.png)

说到线性方程, 我们不得不提到另外一种方程, 非线性方程 (nonlinear function). 我们假设, 女生长得越漂亮, 越多男生爱. 这就可以被当做一个线性问题. 但是如果我们假设这个场景是发生在校园里. 校园里的男生数是有限的, 女生再漂亮, 也不可能会有无穷多的男生喜欢她. 所以这就变成了一个非线性问题.再说..女生也不可能是无穷漂亮的. 这个问题我们以后有时间私下讨论.

[![激励函数 (Activation Function)](https://morvanzhou.github.io/static/results/ML-intro/active2.png)](https://morvanzhou.github.io/static/results/ML-intro/active2.png)

然后我们就可以来讨论如何在神经网络中达成我们描述非线性的任务了. 我们可以把整个网络简化成这样一个式子. Y = Wx, W 就是我们要求的参数, y 是预测值, x 是输入值. 用这个式子, 我们很容易就能描述刚刚的那个线性问题, 因为 W 求出来可以是一个固定的数. 不过这似乎并不能让这条直线变得扭起来 , 激励函数见状, 拔刀相助, 站出来说道: “让我来掰弯它!”.

## 激励函数 

[![激励函数 (Activation Function)](https://morvanzhou.github.io/static/results/ML-intro/active3.png)](https://morvanzhou.github.io/static/results/ML-intro/active3.png)

这里的 AF 就是指的激励函数. 激励函数拿出自己最擅长的”掰弯利器”, 套在了原函数上 用力一扭, 原来的 Wx 结果就被扭弯了.

其实这个 AF, 掰弯利器, 也不是什么触不可及的东西. 它其实就是另外一个非线性函数. 比如说relu, sigmoid, tanh. 将这些掰弯利器嵌套在原有的结果之上, 强行把原有的线性结果给扭曲了. 使得输出结果 y 也有了非线性的特征. 举个例子, 比如我使用了 relu 这个掰弯利器, 如果此时 Wx 的结果是1, y 还将是1, 不过 Wx 为-1的时候, y 不再是-1, 而会是0.

你甚至可以创造自己的激励函数来处理自己的问题, 不过要确保的是这些激励函数必须是可以微分的, 因为在 backpropagation 误差反向传递的时候, 只有这些可微分的激励函数才能把误差传递回去.

## 常用选择 

[![激励函数 (Activation Function)](https://morvanzhou.github.io/static/results/ML-intro/active4.png)](https://morvanzhou.github.io/static/results/ML-intro/active4.png)

想要恰当使用这些激励函数, 还是有窍门的. 比如当你的神经网络层只有两三层, 不是很多的时候, 对于隐藏层, 使用任意的激励函数, 随便掰弯是可以的, 不会有特别大的影响. 不过, 当你使用特别多层的神经网络, 在掰弯的时候, 玩玩不得随意选择利器. 因为这会涉及到梯度爆炸, 梯度消失的问题. 因为时间的关系, 我们可能会在以后来具体谈谈这个问题.

最后我们说说, 在具体的例子中, 我们默认首选的激励函数是哪些. 在少量层结构中, 我们可以尝试很多种不同的激励函数. 在卷积神经网络 Convolutional neural networks 的卷积层中, 推荐的激励函数是 relu. 在循环神经网络中 recurrent neural networks, 推荐的是 tanh 或者是 relu (这个具体怎么选, 我会在以后 循环神经网络的介绍中在详细讲解).

## 7.添加层 def add_layer()

## 定义 add_layer() 

在 Tensorflow 里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.

神经层里常见的参数通常有`weights`、`biases`和激励函数。

首先，我们需要导入`tensorflow`模块。

```
import tensorflow as tf
```

然后定义添加神经层的函数`def add_layer()`,它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是`None`。

```
def add_layer(inputs, in_size, out_size, activation_function=None):    
```

接下来，我们开始定义`weights`和`biases`。

因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的`weights`为一个`in_size`行, `out_size`列的随机变量矩阵。

```
Weights = tf.Variable(tf.random_normal([in_size, out_size]))
```

在机器学习中，`biases`的推荐值不为0，所以我们这里是在0向量的基础上又加了`0.1`。

```
biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
```

下面，我们定义`Wx_plus_b`, 即神经网络未激活的值。其中，`tf.matmul()`是矩阵的乘法。

```
Wx_plus_b = tf.matmul(inputs, Weights) + biases
```

当`activation_function`——激励函数为`None`时，输出就是当前的预测值——`Wx_plus_b`，不为`None`时，就把`Wx_plus_b`传到`activation_function()`函数中得到输出。

```
if activation_function is None:
        outputs = Wx_plus_b
else:
        outputs = activation_function(Wx_plus_b)
```

最后，返回输出，添加一个神经层的函数——`def add_layer()`就定义好了。

```
return outputs
```

# 8.建造神经网络

## add_layer 功能 

首先，我们导入本次所需的模块。

```
import tensorflow as tf
import numpy as np
```

构造添加一个神经层的函数。（在上次课程中有详细介绍）

```
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
```

## 导入数据 

构建所需的数据。 这里的`x_data`和`y_data`并不是严格的一元二次函数的关系，因为我们多加了一个`noise`,这样看起来会更像真实情况。

```
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
#这里表示加入的噪声和x_data.shape一样的格式
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
```

利用占位符定义我们所需的神经网络的输入。 `tf.placeholder()`就是代表占位符，这里的`None`代表无论输入有多少都可以，因为输入只有一个特征，所以这里是`1`。

```
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
```

接下来，我们就可以开始定义神经层了。 通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元； 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

## 搭建网络 

下面，我们开始定义隐藏层,利用之前的`add_layer()`函数，这里使用 Tensorflow 自带的激励函数`tf.nn.relu`。

```
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
```

接着，定义输出层。此时的输入就是隐藏层的输出——`l1`，输入有10层（隐藏层的输出层），输出有1层。

```
prediction = add_layer(l1, 10, 1, activation_function=None)
```

计算预测值`prediction`和真实值的误差，对二者差的平方求和再取平均。

```
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)，reduction_indices=[1]))
```

接下来，是很关键的一步，如何让机器学习提升它的准确率。`tf.train.GradientDescentOptimizer()`中的值通常都小于1，这里取的是`0.1`，代表以`0.1`的效率来最小化误差`loss`。

```
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```

使用变量时，都要对它进行初始化，这是必不可少的。

```
init = tf.global_variables_initializer()  # 替换成这样就好
```

定义`Session`，并用 `Session` 来执行 `init` 初始化步骤。 （注意：在`tensorflow`中，只有`session.run()`才会执行我们定义的运算。）

```
sess = tf.Session()
sess.run(init)
```

## 训练 

下面，让机器开始学习。

比如这里，我们让机器学习1000次。机器学习的内容是`train_step`, 用 `Session` 来 `run`每一次 training 的数据，逐步提升神经网络的预测准确性。 (注意：当运算要用到`placeholder`时，就需要`feed_dict`这个字典来指定输入。)

```
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
```

每50步我们输出一下机器学习的误差。

```
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
```

在电脑上运行本次代码的结果为：

[![例子3 建造神经网络](https://morvanzhou.github.io/static/results/tensorflow/3_2_1.png)](https://morvanzhou.github.io/static/results/tensorflow/3_2_1.png)

通过上图可以看出，误差在逐渐减小，这说明机器学习是有积极的效果的。



# 9.结果可视化

## matplotlib 可视化 

构建图形，用散点图描述真实数据之间的关系。 （注意：`plt.ion()`用于连续显示。）

```
# plot the real data
fig = plt.figure()
#若要连续画图，则需要添加这个
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
#这个会暂停，如果要一直显示，则加入上一行
plt.show()
```

散点图的结果为：

[![例子3 结果可视化](https://morvanzhou.github.io/static/results/tensorflow/3_3_1.png)](https://morvanzhou.github.io/static/results/tensorflow/3_3_1.png)

接下来，我们来显示预测数据。

每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。

```
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        #try的功能是忽略第一条线
        try:
            #抹除上一条线
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
```

最后，机器学习的结果为：

[![例子3 结果可视化](https://morvanzhou.github.io/static/results/tensorflow/3_3_2.png)](https://morvanzhou.github.io/static/results/tensorflow/3_3_2.png)

# 10.加速神经网络训练 (Speed Up Training)



包括以下几种模式:

- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- RMSProp
- Adam

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup1.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup1.png)

越复杂的神经网络 , 越多的数据 , 我们需要在训练神经网络的过程上花费的时间也就越多. 原因很简单, 就是因为计算量太大了. 可是往往有时候为了解决复杂的问题, 复杂的结构和大数据又是不能避免的, 所以我们需要寻找一些方法, 让神经网络聪明起来, 快起来.

## Stochastic Gradient Descent (SGD) 

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup2.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup2.png)

所以, 最基础的方法就是 SGD 啦, 想像红色方块是我们要训练的 data, 如果用普通的训练方法, 就需要重复不断的把整套数据放入神经网络 NN训练, 这样消耗的计算资源会很大.

我们换一种思路, 如果把这些数据拆分成小批小批的, 然后再分批不断放入 NN 中计算, 这就是我们常说的 SGD 的正确打开方式了. 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.如果运用上了 SGD, 你还是嫌训练速度慢, 那怎么办?

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup3.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup3.png)

没问题, 事实证明, SGD 并不是最快速的训练方法, 红色的线是 SGD, 但它到达学习目标的时间是在这些方法中最长的一种. 我们还有很多其他的途径来加速训练.

## Momentum 更新方法 

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup4.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup4.png)

大多数其他途径是在更新神经网络参数那一步上动动手脚. 传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 这种方法可能会让学习过程曲折无比, 看起来像 喝醉的人回家时, 摇摇晃晃走了很多弯路.

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup5.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup5.png)

所以我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走, 走的弯路也变少了. 这就是 Momentum 参数更新. 另外一种加速方法叫AdaGrad.

## AdaGrad 更新方法 

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup6.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup6.png)

这种方法是在学习率上面动手脚, 使得每一个参数更新都会有自己与众不同的学习率, 他的作用和 momentum 类似, 不过不是给喝醉酒的人安排另一个下坡, 而是给他一双不好走路的鞋子, 使得他一摇晃着走路就脚疼, 鞋子成为了走弯路的阻力, 逼着他往前直着走. 他的数学形式是这样的. 接下来又有什么方法呢? 如果把下坡和不好走路的鞋子合并起来, 是不是更好呢? 没错, 这样我们就有了 RMSProp 更新方法.

## RMSProp 更新方法 

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup7.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup7.png)

有了 momentum 的惯性原则 , 加上 adagrad 的对错误方向的阻力, 我们就能合并成这样. 让 RMSProp同时具备他们两种方法的优势. 不过细心的同学们肯定看出来了, 似乎在 RMSProp 中少了些什么. 原来是我们还没把 Momentum合并完全, RMSProp 还缺少了 momentum 中的 这一部分. 所以, 我们在 Adam 方法中补上了这种想法.

## Adam 更新方法 

[![加速神经网络训练 (Speed Up Training)](https://morvanzhou.github.io/static/results/ML-intro/speedup8.png)](https://morvanzhou.github.io/static/results/ML-intro/speedup8.png)

计算m 时有 momentum 下坡的属性, 计算 v 时有 adagrad 阻力的属性, 然后再更新参数时 把 m 和 V 都考虑进去. 实验证明, 大多数时候, 使用 adam 都能又快又好的达到目标, 迅速收敛. 所以说, 在加速神经网络训练的时候, 一个下坡, 一双破鞋子, 功不可没.

# 11.优化器 optimizer

学习资料:

- 各种 Optimizer 的对比 [链接](http://cs231n.github.io/neural-networks-3/)(英文)
- 机器学习-简介系列 [Optimizer](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-06-speed-up-learning/)
- Tensorflow 的可用 optimizer [链接](https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html)
- 为 TF 2017 打造的[新版可视化教学代码](https://github.com/MorvanZhou/Tensorflow-Tutorial)

## 各种不同的优化器 

本次课程，我们会讲到`Tensorflow`里面的优化器。

Tensorflow 中的优化器会有很多不同的种类。最基本, 也是最常用的一种就是`GradientDescentOptimizer`。

在Google搜索中输入“tensorflow optimizer”可以看到`Tensorflow`提供了7种优化器：[链接](https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html)

[![优化器 optimizer](https://morvanzhou.github.io/static/results/tensorflow/3_4_1.png)](https://morvanzhou.github.io/static/results/tensorflow/3_4_1.png)

# 12.Tensorboard 可视化好帮手 1

好，我们开始吧。

这次我们会介绍如何可视化神经网络。因为很多时候我们都是做好了一个神经网络，但是没有一个图像可以展示给大家看。这一节会介绍一个TensorFlow的可视化工具 — tensorboard :) 通过使用这个工具我们可以很直观的看到整个神经网络的结构、框架。 以前几节的代码为例：[相关代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf14_tensorboard/for_you_to_practice.py) 通过tensorflow的工具大致可以看到，今天要显示的神经网络差不多是这样子的

[![Tensorboard 可视化好帮手 1](https://morvanzhou.github.io/static/results/tensorflow/4_1_1.png)](https://morvanzhou.github.io/static/results/tensorflow/4_1_1.png)

同时我们也可以展开看每个layer中的一些具体的结构：

[![Tensorboard 可视化好帮手 1](https://morvanzhou.github.io/static/results/tensorflow/4_1_2.png)](https://morvanzhou.github.io/static/results/tensorflow/4_1_2.png)

好，通过阅读代码和之前的图片我们大概知道了此处是有一个输入层（inputs），一个隐含层（layer），还有一个输出层（output） 现在可以看看如何进行可视化.

## 搭建图纸 

首先从 `Input` 开始：

```
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
```

对于input我们进行如下修改： 首先，可以为`xs`指定名称为`x_in`:

```
xs= tf.placeholder(tf.float32, [None, 1],name='x_in')
```

然后再次对`ys`指定名称`y_in`:

```
ys= tf.placeholder(tf.loat32, [None, 1],name='y_in')
```

这里指定的名称将来会在可视化的图层`inputs`中显示出来

使用`with tf.name_scope('inputs')`可以将`xs`和`ys`包含进来，形成一个大的图层，图层的名字就是`with tf.name_scope()`方法里的参数。

```
with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
```

接下来开始编辑`layer` ， 请看编辑前的程序片段 ：

```
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs
```

这里的名字应该叫layer, 下面是编辑后的:

```
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        Weights= tf.Variable(tf.random_normal([in_size, out_size]))
        # and so on...
```

在定义完大的框架`layer`之后，同时也需要定义每一个’框架‘里面的小部件：(Weights biases 和 activation function): 现在现对 `Weights` 定义： 定义的方法同上，可以使用`tf.name.scope()`方法，同时也可以在`Weights`中指定名称`W`。 即为：

```
    def add_layer(inputs, in_size, out_size, activation_function=None):
	#define layer name
    with tf.name_scope('layer'):
        #define weights name 
        with tf.name_scope('weights'):
            Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
        #and so on......
```

接着继续定义`biases` ， 定义方式同上。

```
def add_layer(inputs, in_size, out_size, activation_function=None):
    #define layer name
    with tf.name_scope('layer'):
        #define weights name 
        with tf.name_scope('weights')
            Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
        # define biase
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # and so on....
```

`activation_function` 的话，可以暂时忽略。因为当你自己选择用 tensorflow 中的激励函数（activation function）的时候，tensorflow会默认添加名称。 最终，layer形式如下：

```
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(
            tf.random_normal([in_size, out_size]), 
            name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(
            tf.zeros([1, out_size]) + 0.1, 
            name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(
            tf.matmul(inputs, Weights), 
            biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs
```

效果如下：（有没有看见刚才定义layer里面的“内部构件”呢？）

[![Tensorboard 可视化好帮手 1](https://morvanzhou.github.io/static/results/tensorflow/4_1_4.png)](https://morvanzhou.github.io/static/results/tensorflow/4_1_4.png)

最后编辑`loss`部分：将`with tf.name_scope()`添加在`loss`上方，并为它起名为`loss`

```
# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
    tf.reduce_sum(
    tf.square(ys - prediction),
    eduction_indices=[1]
    ))
```

这句话就是“绘制” loss了， 如下：

[![Tensorboard 可视化好帮手 1](https://morvanzhou.github.io/static/results/tensorflow/4_1_5.png)](https://morvanzhou.github.io/static/results/tensorflow/4_1_5.png)

使用`with tf.name_scope()`再次对`train_step`部分进行编辑,如下：

```
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```

我们需要使用 `tf.summary.FileWriter()` (`tf.train.SummaryWriter()` 这种方式已经在 tf >= 0.12 版本中摒弃) 将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览。 这个方法中的第二个参数需要使用`sess.graph` ， 因此我们需要把这句话放在获取`session`的后面。 这里的`graph`是将前面定义的框架信息收集起来，然后放在`logs/`目录下面。

```
sess = tf.Session() # get session
# tf.train.SummaryWriter soon be deprecated, use following
writer = tf.summary.FileWriter("logs/", sess.graph)
```

最后在你的terminal（终端）中 ，使用以下命令

```
tensorboard --logdir logs
```

同时将终端中输出的网址复制到浏览器中，便可以看到之前定义的视图框架了。

tensorboard 还有很多其他的参数，希望大家可以多多了解, 可以使用 `tensorboard --help` 查看tensorboard的详细参数 最终的[全部代码在这里](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf14_tensorboard/full_code.py)

## 可能会遇到的问题 

(1) 而且与 tensorboard 兼容的浏览器是 “**Google Chrome**”. 使用其他的浏览器不保证所有内容都能正常显示.

(2) 同时注意, 如果使用 **http://0.0.0.0:6006** 网址打不开的朋友们, 请使用 **http://localhost:6006**, 大多数朋友都是这个问题.

(3) 请确保你的 tensorboard 指令是在你的 logs 文件根目录执行的. 如果在其他目录下, 比如 `Desktop` 等, 可能不会成功看到图. 比如在下面这个目录, 你要 cd 到 `project` 这个地方执行 `/project > tensorboard --logdir logs`

```
- project
   - logs
   model.py
   env.py
```

(4) 讨论区的朋友使用 anaconda 下的 python3.5 的虚拟环境, 如果你输入 tensorboard 的指令, 出现报错: `"tensorboard" is not recognized as an internal or external command...`

解决方法的关键就是需要激活TensorFlow. 管理员模式打开 Anaconda Prompt, 输入 activate tensorflow, 接着按照上面的流程执行 tensorboard 指令.

# 13.Tensorboard 可视化好帮手 2

## 要点 

**注意:** 本节内容会用到浏览器, 而且与 tensorboard 兼容的浏览器是 “Google Chrome”. 使用其他的浏览器不保证所有内容都能正常显示.

上一篇讲到了 如何可视化TesorBorad整个神经网络结构的过程。 其实tensorboard还可以可视化训练过程( biase变化过程) , 这节重点讲一下可视化训练过程的图标是如何做的 。请看下图, 这是如何做到的呢？

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_1.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_1.png)

在histograms里面我们还可以看到更多的layers的变化:

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_2.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_2.png)

（P.S. 灰猫使用的 tensorflow v1.1 显示的效果可能和视频中的不太一样， 但是 tensorboard 的使用方法的是一样的。）

这里还有一个events , 在这次练习中我们会把 整个训练过程中的误差值（loss）在event里面显示出来, 甚至你可以显示更多你想要显示的东西.

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_3.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_3.png)

好了, 开始练习吧, 本节内容包括:

## 制作输入源 

由于这节我们观察训练过程中神经网络的变化, 所以首先要添一些模拟数据. Python 的 numpy 工具包可以帮助我们制造一些模拟数据. 所以我们先导入这个工具包:

```
import tensorflow as tf
import numpy as np
```

然后借助 np 中的 `np.linespace()` 产生随机的数字, 同时为了模拟更加真实我们会添加一些噪声, 这些噪声是通过 `np.random.normal()` 随机产生的.

```
 ## make up some data
 x_data= np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
 noise=  np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
 y_data= np.square(x_data) -0.5+ noise
```

输入源的问题解决之后, 我们开始制作对`Weights`和`biases`的变化图表吧. 我们期望可以做到如下的效果, 那么首先从 layer1/weight 做起吧

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_4.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_4.png)

这个效果是如何做到的呢,请看下一个标题

## 在 layer 中为 Weights, biases 设置变化图表 

通过上图的观察我们发现每个 layer 后面有有一个数字: layer1 和layer2

于是我们在 `add_layer()` 方法中添加一个参数 `n_layer`,用来标识层数, 并且用变量 `layer_name`代表其每层的名名称, 代码如下:

```
def add_layer(
    inputs , 
    in_size, 
    out_size,
    n_layer, 
    activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer  ## define a new var
    ## and so on ……
```

接下来,我们层中的`Weights`设置变化图, tensorflow中提供了`tf.histogram_summary()`方法,用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量

```
def add_layer(inputs , 
            in_size, 
            out_size,n_layer, 
            activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
              tf.histogram_summary(layer_name+'/weights',Weights)   # tensorflow 0.12 以下版的
              tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12
    ##and so no ……
```

同样的方法我们对`biases`进行绘制图标:

```
with tf.name_scope('biases'):
    biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
    # tf.histogram_summary(layer_name+'/biase',biases)   # tensorflow 0.12 以下版的
    tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12
```

至于`activation_function` 可以不绘制. 我们对output 使用同样的方法:

```
# tf.histogram_summary(layer_name+'/outputs',outputs) # tensorflow 0.12 以下版本
tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12
```

最终经过我们的修改 , `addlayer()`方法成为如下的样子:

```
def add_layer(inputs , 
              in_size, 
              out_size,n_layer, 
              activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
              # tf.histogram_summary(layer_name+'/weights',Weights)
              tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12

         with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
              # tf.histogram_summary(layer_name+'/biase',biases)
              tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12

         with tf.name_scope('Wx_plus_b'):
              Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)

         if activation_function is None:
            outputs=Wx_plus_b
         else:
            outputs= activation_function(Wx_plus_b)

         # tf.histogram_summary(layer_name+'/outputs',outputs)
         tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12

    return outputs
```

修改之后的名称会显示在每个tensorboard中每个图表的上方显示, 如下图所示:

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_5.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_5.png)

由于我们对`addlayer` 添加了一个参数, 所以修改之前调用`addlayer()`函数的地方. 对此处进行修改:

```
# add hidden layer
l1= add_layer(xs, 1, 10 ,  activation_function=tf.nn.relu)
# add output  layer
prediction= add_layer(l1, 10, 1,  activation_function=None)
```

添加`n_layer`参数后, 修改成为 :

```
# add hidden layer
l1= add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output  layer
prediction= add_layer(l1, 10, 1, n_layer=2, activation_function=None)
```

## 设置loss的变化图 

`Loss` 的变化图和之前设置的方法略有不同. loss是在tesnorBorad 的event下面的, 这是由于我们使用的是`tf.scalar_summary()` 方法.

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_6.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_6.png)

观看loss的变化比较重要. 当你的loss呈下降的趋势,说明你的神经网络训练是有效果的.

修改后的代码片段如下：

```
with tf.name_scope('loss'):
     loss= tf.reduce_mean(tf.reduce_sum(
              tf.square(ys- prediction), reduction_indices=[1]))
     # tf.scalar_summary('loss',loss) # tensorflow < 0.12
     tf.summary.scalar('loss', loss) # tensorflow >= 0.12
```

## 给所有训练图合并 

接下来， 开始合并打包。 `tf.merge_all_summaries()` 方法会对我们所有的 `summaries` 合并到一起. 因此在原有代码片段中添加：

```
sess= tf.Session()

# merged= tf.merge_all_summaries()    # tensorflow < 0.12
merged = tf.summary.merge_all() # tensorflow >= 0.12

# writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
writer = tf.summary.FileWriter("logs/", sess.graph) # tensorflow >=0.12

# sess.run(tf.initialize_all_variables()) # tf.initialize_all_variables() # tf 马上就要废弃这种写法
sess.run(tf.global_variables_initializer())  # 替换成这样就好
```

## 训练数据 

假定给出了`x_data,y_data`并且训练1000次.

```
for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
```

以上这些仅仅可以记录很绘制出训练的图表， 但是不会记录训练的数据。 为了较为直观显示训练过程中每个参数的变化，我们每隔上50次就记录一次结果 , 同时我们也应注意, merged 也是需要run 才能发挥作用的,所以在for循环中写下：

```
if i%50 == 0:
    rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
    writer.add_summary(rs, i)
```

最后修改后的片段如下：

```
for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
   if i%50 == 0:
      rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
      writer.add_summary(rs, i)
```

## 在 tensorboard 中查看效果 

程序运行完毕之后, 会产生logs目录 , 使用命令 `tensorboard --logdir logs`

**注意:** 本节内容会用到浏览器, 而且与 tensorboard 兼容的浏览器是 “Google Chrome”. 使用其他的浏览器不保证所有内容都能正常显示.

**同时注意, 如果使用 http://0.0.0.0:6006 或者 tensorboard 中显示的网址打不开的朋友们, 请使用 http://localhost:6006, 大多数朋友都是这个问题.**

会有如下输出:

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_7.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_7.png)

将输出中显示的URL地址粘贴到浏览器中便可以查看. 最终的效果如下:

[![Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/static/results/tensorflow/4_2_8.png)](https://morvanzhou.github.io/static/results/tensorflow/4_2_8.png)



# 14.Classification 分类学习

## MNIST 数据 

首先准备数据（MNIST库）

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

MNIST库是手写体数字库，差不多是这样子的

[![Classification 分类学习](https://morvanzhou.github.io/static/results/tensorflow/5_01_1.png)](https://morvanzhou.github.io/static/results/tensorflow/5_01_1.png)

数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。

## 搭建网络 

```
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
```

每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。

```
ys = tf.placeholder(tf.float32, [None, 10])
```

调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层。

```
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
```

其中输入数据是784个特征，输出数据是10个特征，激励采用softmax函数，网络结构图是这样子的

[![Classification 分类学习](https://morvanzhou.github.io/static/results/tensorflow/5_01_2.png)](https://morvanzhou.github.io/static/results/tensorflow/5_01_2.png)

## Cross entropy loss 

loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。

```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
reduction_indices=[1])) # loss
```

train方法（最优化算法）采用梯度下降法。

```
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
# tf.initialize_all_variables() 这种写法马上就要被废弃
# 替换成下面的写法:
sess.run(tf.global_variables_initializer())
```

## 训练 

现在开始train，每次只取100张图片，免得数据太多训练太慢。

```
batch_xs, batch_ys = mnist.train.next_batch(100)
sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
```

每训练50次输出一下预测精度

```
if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
```

# 15.过拟合

## 过于自负 

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting1.png)

在细说之前, 我们先用实际生活中的一个例子来比喻一下过拟合现象. 说白了, 就是机器学习模型于自信. 已经到了自负的阶段了. 那自负的坏处, 大家也知道, 就是在自己的小圈子里表现非凡, 不过在现实的大圈子里却往往处处碰壁. 所以在这个简介里, 我们把自负和过拟合画上等号.

## 回归分类的过拟合 

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting2.png)

机器学习模型的自负又表现在哪些方面呢. 这里是一些数据. 如果要你画一条线来描述这些数据, 大多数人都会这么画. 对, 这条线也是我们希望机器也能学出来的一条用来总结这些数据的线. 这时蓝线与数据的总误差可能是10. 可是有时候, 机器过于纠结这误差值, 他想把误差减到更小, 来完成他对这一批数据的学习使命. 所以, 他学到的可能会变成这样 . 它几乎经过了每一个数据点, 这样, 误差值会更小 . 可是误差越小就真的好吗? 看来我们的模型还是太天真了. 当我拿这个模型运用在现实中的时候, 他的自负就体现出来. 小二, 来一打现实数据 . 这时, 之前误差大的蓝线误差基本保持不变 .误差小的 红线误差值突然飙高 , 自负的红线再也骄傲不起来, 因为他不能成功的表达除了训练数据以外的其他数据. 这就叫做过拟合. Overfitting.

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting3.png)

那么在分类问题当中. 过拟合的分割线可能是这样, 小二, 再上一打数据 . 我们明显看出, 有两个黄色的数据并没有被很好的分隔开来. 这也是过拟合在作怪.好了, 既然我们时不时会遇到过拟合问题, 那解决的方法有那些呢.

## 解决方法 

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting4.png)

方法一: 增加数据量, 大部分过拟合产生的原因是因为数据量太少了. 如果我们有成千上万的数据, 红线也会慢慢被拉直, 变得没那么扭曲 . 方法二:

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting5.png)

运用正规化. L1, l2 regularization等等, 这些方法适用于大多数的机器学习, 包括神经网络. 他们的做法大同小异, 我们简化机器学习的关键公式为 y=Wx . W为机器需要学习到的各种参数. 在过拟合中, W 的值往往变化得特别大或特别小. 为了不让W变化太大, 我们在计算误差上做些手脚. 原始的 cost 误差是这样计算, cost = 预测值-真实值的平方. 如果 W 变得太大, 我们就让 cost 也跟着变大, 变成一种惩罚机制. 所以我们把 W 自己考虑进来. 这里 abs 是绝对值. 这一种形式的 正规化, 叫做 l1 正规化. L2 正规化和 l1 类似, 只是绝对值换成了平方. 其他的l3, l4 也都是换成了立方和4次方等等. 形式类似. 用这些方法,我们就能保证让学出来的线条不会过于扭曲.

[![过拟合 (Overfitting)](https://morvanzhou.github.io/static/img/description/loading.gif)](https://morvanzhou.github.io/static/results/ML-intro/overfitting6.png)

还有一种专门用在神经网络的正规化的方法, 叫作 dropout. 在训练的时候, 我们随机忽略掉一些神经元和神经联结 , 是这个神经网络变得”不完整”. 用一个不完整的神经网络训练一次.

到第二次再随机忽略另一些, 变成另一个不完整的神经网络. 有了这些随机 drop 掉的规则, 我们可以想象其实每次训练的时候, 我们都让每一次预测结果都不会依赖于其中某部分特定的神经元. 像l1, l2正规化一样, 过度依赖的 W , 也就是训练参数的数值会很大, l1, l2会惩罚这些大的 参数. Dropout 的做法是从根本上让神经网络没机会过度依赖.

## 16.Dropout 解决 overfitting

## 要定 

Overfitting 也被称为过度学习，过度拟合。 它是机器学习中常见的问题。 举个Classification（分类）的例子。

[![Dropout 解决 overfitting](https://morvanzhou.github.io/static/results/tensorflow/5_02_1.png)](https://morvanzhou.github.io/static/results/tensorflow/5_02_1.png)

图中黑色曲线是正常模型，绿色曲线就是overfitting模型。尽管绿色曲线很精确的区分了所有的训练数据，但是并没有描述数据的整体特征，对新测试数据的适应性较差。

举个Regression (回归)的例子，

[![Dropout 解决 overfitting](https://morvanzhou.github.io/static/results/tensorflow/5_02_2.png)](https://morvanzhou.github.io/static/results/tensorflow/5_02_2.png)

第三条曲线存在overfitting问题，尽管它经过了所有的训练点，但是不能很好的反应数据的趋势，预测能力严重不足。 TensorFlow提供了强大的dropout方法来解决overfitting问题。

## 建立 dropout 层 

本次内容需要使用一下 sklearn 数据库当中的数据, 没有安装 sklearn 的同学可以参考一下[这个教程](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/1-2-install/) 安装一下. 然后 `import` 以下模块.

```
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
```

```
keep_prob = tf.placeholder(tf.float32)
...
...
Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
```

这里的`keep_prob`是保留概率，即我们要保留的结果所占比例，它作为一个`placeholder`，在`run`时传入， 当`keep_prob=1`的时候，相当于100%保留，也就是dropout没有起作用。 下面我们分析一下程序结构，首先准备数据，

```
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
```

其中`X_train`是训练数据, `X_test`是测试数据。 然后添加隐含层和输出层

```
# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)
```

loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零。

```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
```

train方法（最优化算法）采用梯度下降法。

```
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

## 训练 

最后开始train，总共训练500次。

```
#在此处添加dropout。
sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
#sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
```

## 可视化结果 

训练中`keep_prob=1`时，就可以暴露出overfitting问题。`keep_prob=0.5`时，`dropout`就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。

当`keep_prob=1`时，模型对训练数据的适应性优于测试数据，存在overfitting，输出如下： 红线是 `train` 的误差, 蓝线是 `test` 的误差.

[![Dropout 解决 overfitting](https://morvanzhou.github.io/static/results/tensorflow/5_02_3.png)](https://morvanzhou.github.io/static/results/tensorflow/5_02_3.png)

当`keep_prob=0.5`时效果好了很多，输出如下：

[![Dropout 解决 overfitting](https://morvanzhou.github.io/static/results/tensorflow/5_02_4.png)](https://morvanzhou.github.io/static/results/tensorflow/5_02_4.png)

程序中用到了Tensorboard输出结果，可以参考前面教程:

- [可视化好助手 Tensorboard 1](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/)
- [可视化好助手 Tensorboard 2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-2-tensorboard2/)



# 17.CNN

## 定义卷积层的 weight bias 

首先我们导入

```
import tensorflow as tf
```

采用的数据集依然是`tensorflow`里面的`mnist`数据集

我们需要先导入它

```
from tensorflow.examples.tutorials.mnist import input_data
```

本次课程代码用到的数据集就是来自于它

```
mnist=input_data.read_data_sets('MNIST_data',one_hot=true)
```

接着呢，我们定义`Weight`变量，输入`shape`，返回变量的参数。其中我们使用`tf.truncted_normal`产生随机变量来进行初始化:

```
def weight_variable(shape): 
	inital=tf.truncted_normal(shape,stddev=0.1)
	return tf.Variable(initial)
```

同样的定义`biase`变量，输入`shape` ,返回变量的一些参数。其中我们使用`tf.constant`常量函数来进行初始化:

```
def bias_variable(shape): 
	initial=tf.constant(0.1,shape=shape) 
	return tf.Variable(initial)
```

定义卷积，`tf.nn.conv2d`函数是`tensoflow`里面的二维的卷积函数，`x`是图片的所有参数，`W`是此卷积层的权重，然后定义步长`strides=[1,1,1,1]`值，`strides[0]`和`strides[3]`的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是`SAME`。

```
def conv2d(x,W):
    #两种paddeing，SAME会保持大小不变，另一种valid会发生变化
	return tf.nn.conv2d(x,W,strides=[1,1,1,1]，padding='SAME') 
```

## 定义 pooling 

接着定义池化`pooling`，为了得到更多的图片信息，padding时我们选的是一次一步，也就是`strides[1]=strides[2]=1`，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用`pooling`来稀疏化参数，也就是卷积神经网络中所谓的下采样层。`pooling `有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化`tf.max_pool()`。

池化的核函数大小为2x2，因此`ksize=[1,2,2,1]`，步长为2，因此`strides=[1,2,2,1]`:

```
#第一个和第三个参数的1是默认值
def max_poo_2x2(x): 
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])
```

# 18.cnn

这一次我们一层层的加上了不同的 layer. 分别是:

1. convolutional layer1 + max pooling;
2. convolutional layer2 + max pooling;
3. fully connected layer1 + dropout;
4. fully connected layer2 to prediction.

我们利用上节课定义好的函数来构建我们的网络

## 图片处理 

首先呢，我们定义一下输入的`placeholder`

```
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
```

我们还定义了`dropout`的`placeholder`，它是解决过拟合的有效手段

```
keep_prob=tf.placeholder(tf.float32)
```

接着呢，我们需要处理我们的`xs`，把`xs`的形状变成`[-1,28,28,1]`，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。

```
x_image=tf.reshape(xs,[-1,28,28,1])
```

## 建立卷积层 

接着我们定义第一层卷积,先定义本层的`Weight`,本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap

```
W_conv1=weight_variable([5,5,1,32])
```

接着定义`bias`，它的大小是32个长度，因此我们传入它的`shape`为`[32]`

```
b_conv1=bias_variable([32])
```

定义好了`Weight`和`bias`，我们就可以定义卷积神经网络的第一个卷积层`h_conv1=conv2d(x_image,W_conv1)+b_conv1`,同时我们对`h_conv1`进行非线性处理，也就是激活函数来处理喽，这里我们用的是`tf.nn.relu`（修正线性单元）来处理，要注意的是，因为采用了`SAME`的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32

```
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
```

最后我们再进行`pooling`的处理就ok啦，经过`pooling`的处理，输出大小就变为了14x14x32

```
h_pool=max_pool_2x2(h_conv1)
```

接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64

```
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
```

接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64

```
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
```

最后也是一个pooling处理，输出大小为7x7x64

```
h_pool2=max_pool_2x2(h_conv2)
```

## 建立全连接层 

好的，接下来我们定义我们的 fully connected layer,

进入全连接层时, 我们通过`tf.reshape()`将`h_pool2`的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.

```
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) 
```

此时`weight_variable`的`shape`输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024

```
W_fc1=weight_variable([7*7*64,1024]) 
b_fc1=bias_variable([1024])
```

然后将展平后的`h_pool2_flat`与本层的`W_fc1`相乘（注意这个时候不是卷积了）

```
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
```

如果我们考虑过拟合问题，可以加一个dropout的处理

```
h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)
```

接下来我们就可以进行最后一层的构建了，好激动啊, 输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值

```
W_fc2=weight_variable([1024,10]) b_fc2=bias_variable([10])
```

然后呢我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类

```
prediction=tf.nn.softmax(tf.matmul(h_fc1_dropt,W_fc2),b_fc2)
```

## 选优化方法 

接着呢我们利用交叉熵损失函数来定义我们的cost function

```
cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices=[1]))
```

我们用`tf.train.AdamOptimizer()`作为我们的优化器进行优化，使我们的`cross_entropy`最小

```
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

接着呢就是和之前视频讲的一样喽 定义`Session`

```
sess=tf.Session()
```

初始化变量

```
# tf.initialize_all_variables() 这种写法马上就要被废弃
# 替换成下面的写法:
sess.run(tf.global_variables_initializer())
```

好啦接着就是训练数据啦，我们假定训练`1000`步，每`50`步输出一下准确率， 注意`sess.run()`时记得要用`feed_dict`给我们的众多 `placeholder` 喂数据哦.

# 19.网络保存

## 保存 

`import`所需的模块, 然后建立神经网络当中的 `W` 和 `b`, 并初始化变量.

```
import tensorflow as tf
import numpy as np

## Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()
```

保存时, 首先要建立一个 `tf.train.Saver()` 用来保存, 提取变量. 再创建一个名为`my_net`的文件夹, 用这个 `saver` 来保存变量到这个目录 `"my_net/save_net.ckpt"`.

```
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)

"""    
Save to path:  my_net/save_net.ckpt
"""
```

## 提取 

提取时, 先建立零时的`W` 和 `b`容器. 找到文件目录, 并用`saver.restore()`我们放在这个目录的变量.

```
# 先建立 W, b 的容器
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

"""
weights: [[ 1.  2.  3.]
          [ 3.  4.  5.]]
biases: [[ 1.  2.  3.]]
"""
```

# 20.scope 命名方法

## tf.name_scope() 

在 Tensorflow 当中有两种途径生成变量 variable, 一种是 `tf.get_variable()`, 另一种是 `tf.Variable()`. 如果在 `tf.name_scope()` 的框架下使用这两种方式, 结果会如下.

```
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]
```

可以看出使用 `tf.Variable()` 定义的时候, 虽然 `name` 都一样, 但是为了不重复变量名, Tensorflow 输出的变量名并不是一样的. 所以, 本质上 `var2`, `var21`, `var22` 并不是一样的变量. 而另一方面, 使用`tf.get_variable()`定义的变量不会被`tf.name_scope()`当中的名字所影响.

## tf.variable_scope() 

如果想要达到重复利用变量的效果, 我们就要使用 `tf.variable_scope()`, 并搭配 `tf.get_variable()`这种方式产生和提取变量. 不像 `tf.Variable()` 每次都会产生新的变量, `tf.get_variable()` 如果遇到了同样名字的变量时, 它会单纯的提取这个同样名字的变量(避免产生新变量). 而在重复使用的时候, 一定要在代码中强调 `scope.reuse_variables()`, 否则系统将会报错, 以为你只是单纯的不小心重复使用到了一个变量.

```
with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3',)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]
    
```

# 21.什么是批标准化 (Batch Normalization)

## 普通数据标准化 

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB1.png)](https://morvanzhou.github.io/static/results/ML-intro/NB1.png)

Batch Normalization, 批标准化, 和普通的数据标准化类似, 是将分散的数据统一的一种做法, 也是优化神经网络的一种方法. 在之前 Normalization 的简介视频中我们一提到, 具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律.

## 每层都做标准化 

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB2.png)](https://morvanzhou.github.io/static/results/ML-intro/NB2.png)

在神经网络中, 数据分布对训练会产生影响. 比如某个神经元 x 的值为1, 某个 Weights 的初始值为 0.1, 这样后一层神经元计算结果就是 Wx = 0.1; 又或者 x = 20, 这样 Wx 的结果就为 2. 现在还不能看出什么问题, 但是, 当我们加上一层激励函数, 激活这个 Wx 值的时候, 问题就来了. 如果使用 像 tanh 的激励函数, Wx 的激活值就变成了 ~0.1 和 ~1, 接近于 1 的部已经处在了 激励函数的饱和阶段, 也就是如果 x 无论再怎么扩大, tanh 激励函数输出值也还是 接近1. 换句话说, 神经网络在初始阶段已经不对那些比较大的 x 特征范围 敏感了. 这样很糟糕, 想象我轻轻拍自己的感觉和重重打自己的感觉居然没什么差别, 这就证明我的感官系统失效了. 当然我们是可以用之前提到的对数据做 normalization 预处理, 使得输入的 x 变化范围不会太大, 让输入值经过激励函数的敏感部分. 但刚刚这个不敏感问题不仅仅发生在神经网络的输入层, 而且在隐藏层中也经常会发生.

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB3.png)](https://morvanzhou.github.io/static/results/ML-intro/NB3.png)

只是时候 x 换到了隐藏层当中, 我们能不能对隐藏层的输入结果进行像之前那样的normalization 处理呢? 答案是可以的, 因为大牛们发明了一种技术, 叫做 batch normalization, 正是处理这种情况.

## BN 添加位置 

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB4.png)](https://morvanzhou.github.io/static/results/ML-intro/NB4.png)

Batch normalization 的 batch 是批数据, 把数据分成小批小批进行 stochastic gradient descent. 而且在每批数据进行前向传递 forward propagation 的时候, 对每一层都进行 normalization 的处理,

## BN 效果 

Batch normalization 也可以被看做一个层面. 在一层层的添加神经网络的时候, 我们先有数据 X, 再添加全连接层, 全连接层的计算结果会经过 激励函数 成为下一层的输入, 接着重复之前的操作. Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间.

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB5.png)](https://morvanzhou.github.io/static/results/ML-intro/NB5.png)

之前说过, 计算结果在进入激励函数前的值很重要, 如果我们不单单看一个值, 我们可以说, 计算结果值的分布对于激励函数很重要. 对于数据值大多分布在这个区间的数据, 才能进行更有效的传递. 对比这两个在激活之前的值的分布. 上者没有进行 normalization, 下者进行了 normalization, 这样当然是下者能够更有效地利用 tanh 进行非线性化的过程.

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB6.png)](https://morvanzhou.github.io/static/results/ML-intro/NB6.png)

没有 normalize 的数据 使用 tanh 激活以后, 激活值大部分都分布到了饱和阶段, 也就是大部分的激活值不是-1, 就是1, 而 normalize 以后, 大部分的激活值在每个分布区间都还有存在. 再将这个激活后的分布传递到下一层神经网络进行后续计算, 每个区间都有分布的这一种对于神经网络就会更加有价值. Batch normalization 不仅仅 normalize 了一下数据, 他还进行了反 normalize 的手续. 为什么要这样呢?

## BN 算法 

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB7.png)](https://morvanzhou.github.io/static/results/ML-intro/NB7.png)

我们引入一些 batch normalization 的公式. 这三步就是我们在刚刚一直说的 normalization 工序, 但是公式的后面还有一个反向操作, 将 normalize 后的数据再扩展和平移. 原来这是为了让神经网络自己去学着使用和修改这个扩展参数 gamma, 和 平移参数 β, 这样神经网络就能自己慢慢琢磨出前面的 normalization 操作到底有没有起到优化的作用, 如果没有起到作用, 我就使用 gamma 和 belt 来抵消一些 normalization 的操作.

[![批标准化 (Batch Normalization)](https://morvanzhou.github.io/static/results/ML-intro/NB8.png)](https://morvanzhou.github.io/static/results/ML-intro/NB8.png)

最后我们来看看一张神经网络训练到最后, 代表了每层输出值的结果的分布图. 这样我们就能一眼看出 Batch normalization 的功效啦. 让每一层的值在有效的范围内传递下去.



## 23.Batch Normalization 批标准化

## 什么是 Batch Normalization 

请参考我制作的 [Batch normalization 简介视频](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/) Batch normalization 是一种解决深度神经网络层数太多, 而没办法有效前向传递(forward propagate)的问题. 因为每一层的输出值都会有不同的 均值(mean) 和 方差(deviation), 所以输出数据的分布也不一样, 如下图, 从左到右是每一层的输入数据分布, 上排的没有 Batch normalization, 下排的有 Batch normalization.

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_01.png)](https://morvanzhou.github.io/static/results/tensorflow/5_13_01.png)

我们以前说过, 为了更有效的学习数据, 我们会对数据预处理, 进行 normalization (请参考我制作的 [为什么要特征标准化](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-02-normalization/)). 而现在请想象, 我们可以把 “每层输出的值” 都看成 “后面一层所接收的数据”. 对每层都进行一次 normalization 会不会更好呢? 这就是 Batch normalization 方法的由来.

## 搭建网络 

输入需要的模块和定义网络的结构

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


ACTIVATION = tf.nn.relu # 每一层都使用 relu 
N_LAYERS = 7            # 一共7层隐藏层
N_HIDDEN_UNITS = 30     # 每个层隐藏层有 30 个神经元
```

使用 `build_net()` 功能搭建神经网络:

```
def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None):
        # 添加层功能
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    fix_seed(1)

    layers_inputs = [xs]    # 记录每层的 input

    # loop 建立所有层
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
        )
        layers_inputs.append(output)    # 把 output 加入记录

    # 建立 output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]
```

## 创建数据 

创造数据并可视化数据:

```
x_data = np.linspace(-7, 10, 500)[:, np.newaxis]
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

# 可视化 input data
plt.scatter(x_data, y_data)
plt.show()
```

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_02.png)](https://morvanzhou.github.io/static/results/tensorflow/5_13_02.png)

## Batch Normalization 代码 

为了实现 Batch Normalization, 我们要对每一层的代码进行修改, 给 `built_net` 和 `add_layer`都加上 `norm` 参数, 表示是否是 Batch Normalization 层:

```
def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
```

然后每层的 `Wx_plus_b` 需要进行一次 batch normalize 的步骤, 这样输出到 `activation` 的 `Wx_plus_b` 就已经被 `normalize` 过了:

```
if norm:    # 判断书否是 BN 层
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_b,
        axes=[0],   # 想要 normalize 的维度, [0] 代表 batch 维度
                    # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
    )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
    # 上面那一步, 在做如下事情:
    # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
    # Wx_plus_b = Wx_plus_b * scale + shift
    # 如果你已经看不懂了, 请去我最上面学习资料里的链接 (我制作的 Batch normalization 简介视频)
```

如果你是使用 batch 进行每次的更新, 那每个 batch 的 mean/var 都会不同, 所以我们可以使用 moving average 的方法记录并慢慢改进 mean/var 的值. 然后将修改提升后的 mean/var 放入 `tf.nn.batch_normalization()`. 而且在 test 阶段, 我们就可以直接调用最后一次修改的 mean/var 值进行测试, 而不是采用 test 时的 fc_mean/fc_var.

```
# 对这句进行扩充, 修改前:
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

# 修改后:
ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度
def mean_var_with_update():
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)
mean, var = mean_var_with_update()      # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var

# 将修改后的 mean / var 放入下面的公式
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
```

那如何确定我们是在 train 阶段还是在 test 阶段呢, 我们可以修改上面的算法, 想办法传入 `on_train` 参数, 你也可以把 `on_train` 定义成全局变量. (**注意: github 的代码中没有这一段, 想做 test 的同学们需要自己修改**)

```
# 修改前:
mean, var = mean_var_with_update() 

# 修改后:
mean, var = tf.cond(on_train,    # on_train 的值是 True/False
                    mean_var_with_update,   # 如果是 True, 更新 mean/var
                    lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                        ema.average(fc_mean), 
                        ema.average(fc_var)
                        )    
                    )
```

同样, 我们也可以在输入数据 `xs` 时, 给它做一个 normalization, **同样, 如果是最 batch data 来训练的话, 要重复上述的记录修改 mean/var 的步骤**:

```
    if norm:
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        xs = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)
```

然后我们把在建立网络的循环中的这一步加入 `norm` 这个参数:

```
output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
            norm,           # normalize before activation
        )
```

## 对比有无 BN 

搭建两个神经网络, 一个没有 BN, 一个有 BN:

```
xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = built_net(xs, ys, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True) # with BN
```

训练神经网络:

代码中的 `plot_his()` 不会在这里讲解, 请自己在[全套代码中查看](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf23_BN/tf23_BN.py).

```
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 记录两种网络的 cost 变化
cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7, 3))
for i in range(251):
    if i % 50 == 0:
        # 每层在 activation 之前计算结果值的分布
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)

    sess.run(train_op, feed_dict={xs: x_data, ys: y_data})
    sess.run(train_op_norm, feed_dict={xs: x_data, ys: y_data})
    if i % record_step == 0:
        # 记录 cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()
plt.show()
```

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_03.gif)](https://morvanzhou.github.io/static/results/tensorflow/5_13_03.gif)

可以看出, 没有用 BN 的时候, 每层的值迅速全部都变为 0, 也可以说, 所有的神经元都已经死了. 而有 BN, `relu` 过后, 每层的值都能有一个比较好的分布效果, 大部分神经元都还活着. (看不懂了? 没问题, 再去看一遍[我制作的 Batch normalization 简介视频](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/)).

Relu 激励函数的图在这里:

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_04.png)](https://morvanzhou.github.io/static/results/tensorflow/5_13_04.png)

我们也看看使用 relu cost 的对比:

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_05.png)](https://morvanzhou.github.io/static/results/tensorflow/5_13_05.png)

因为没有使用 NB 的网络, 大部分神经元都死了, 所以连误差曲线都没了.

如果使用不同的 `ACTIVATION` 会怎么样呢? 不如把 `relu` 换成 `tanh`:

```
ACTIVATION = tf.nn.tanh
```

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_06.gif)](https://morvanzhou.github.io/static/results/tensorflow/5_13_06.gif)

可以看出, 没有 NB, 每层的值迅速全部都饱和, 都跑去了 -1/1 这个饱和区间, 有 NB, 即使前一层因变得相对饱和, 但是后面几层的值都被 normalize 到有效的不饱和区间内计算. 确保了一个活的神经网络.

tanh 激励函数的图在这里:

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_07.gif)](https://morvanzhou.github.io/static/results/tensorflow/5_13_07.gif)

最后我们看一下使用 tanh 的误差对比:

[![Batch Normalization 批标准化](https://morvanzhou.github.io/static/results/tensorflow/5_13_08.png)](https://morvanzhou.github.io/static/results/tensorflow/5_13_08.png)