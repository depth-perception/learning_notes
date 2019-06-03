# GAN笔记

[TOC]

## **1.基本概念**

GAN有两个部分，生成器和判别器。

### **生成器**

生成器的基本概念其实很简单，输入一个向量，通过一个NN，输出一个高维向量（可以是图片，文字...）通常Input向量的每一个维度都代表着一些特征。

![img](https://pic3.zhimg.com/80/v2-7ab035aea1fac4491fc9c4264b2bb6e2_hd.jpg)

### **判别器**

同时呢GAN还有一个部分，叫做“discriminator”(判别器），它的输入是你想产生的东西（其实就是生成器产生的output），比如一张图片，或者一段语音...它的输出是一个标量，这个标量代表的是这个Input的质量如何，这个数字越大，表示这个输入越真实。

![img](https://pic2.zhimg.com/80/v2-3e9716ac9d914be3e5771210f08bbfe5_hd.jpg)

### **生成器和判别器的关系**

其实就是生成器生成一个东西，输入到判别器中，然后由判别器来判断这个输入是真实的数据还是机器生成的，如果没有骗过判别器，那么生成器继续进化，输出第二代Output，再输入判别器，判别器同时也在进化，对生成器的output有了更严格的要求。这样生成器和判别器不断进化，他们的关系有点像一个竞争的关系，所以有了“生成对抗网络(adversarial)”的名字的由来。

```python
'''
其实就是老师和学生的关系：比如一开始画一张鸟，学生根据自己了解到的鸟的一些特征，画了一个鸟的图像，即生成器做的事情，画出来的图让老师检查，老师发现学生画出的鸟缺乏一些特征，比如翅膀，便给了学生低分，即判别器做的事情，学生知道了画鸟需要翅膀上有所修饰，便根据翅膀的特征，画了更接近真实鸟的画，这个过程生成器在进化，而此时老师也在进化，他了解到了画鸟需要具备新的特征，因此判别器也在进化，如此过程对抗循环，直到老师无法判别画出的鸟是真还是假。
'''
```



### **GAN算法流程简述**

1. 初始化generator和discriminator

2. 每一次迭代过程中：

3. 1. 固定generator， 只更新discriminator的参数。从你准备的数据集中随机选择一些，再从generator的output中选择一些，现在等  于discriminator有两种input。接下来， discriminator的学习目标是, 如果输入是来自于真实数据集，则给高分；如果是generator产生的数据，则给低分，可以把它当做一个回归问题。
   2. 接下来，固定住discriminator的参数, 更新generator。将一个向量输入generator， 得到一个output， 将output扔进discriminator, 然后会得到一个分数，这一阶段discriminator的参数已经固定住了，generator需要调整自己的参数使得这个output的分数越大越好。

按这个过程听起来好像有两个网络，而实际过程中，generator和discriminator是同一个网络，只不过网络中间的某一层hidden-layer的输出是一个图片（或者语音，取决于你的数据集）。在训练的时候也是固定一部分hidden-layer，调其余的hidden-layer。当然这里的目标是让output越大越好，所以做的不是常规的梯度下降，而是gradient ascent， 当然其实是类似的。

### **GAN算法-具体操作**

上面用通俗的语言解释了GAN的算法流程，现在将其正式化：

初始化 ![\theta_d ](https://www.zhihu.com/equation?tex=%5Ctheta_d+) for D( discriminator) ， ![\theta_g](https://www.zhihu.com/equation?tex=%5Ctheta_g) for G( generator)

在每次迭代中：

1. 从数据集 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 中sample出m个样本点 ![\{x^1, x^2...x^m\}](https://www.zhihu.com/equation?tex=%5C%7Bx%5E1%2C+x%5E2...x%5Em%5C%7D) ，这个m也是一个超参数，需要自己去调
2. 从一个分布(可以是高斯，正态..., 这个不重要)中sample出m个向量 ![\{z^1,z^2,..,z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2Cz%5E2%2C..%2Cz%5Em%5C%7D)
3. 将第2步中的z作为输入，获得m个生成的数据 ![\{\check{x}^1,\check{x}^2...\check{x}^m\}, \check{x}^i= G(z^i)](https://www.zhihu.com/equation?tex=%5C%7B%5Ccheck%7Bx%7D%5E1%2C%5Ccheck%7Bx%7D%5E2...%5Ccheck%7Bx%7D%5Em%5C%7D%2C+%5Ccheck%7Bx%7D%5Ei%3D+G%28z%5Ei%29)
4. 更新discriminator的参数 ![\theta_d](https://www.zhihu.com/equation?tex=%5Ctheta_d) 来最大化 ![\check{V}](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D) , 我们要使得 ![\check{V}](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D) 越大越好，那么下式中就要使得 ![D(\check{x}^i) ](https://www.zhihu.com/equation?tex=D%28%5Ccheck%7Bx%7D%5Ei%29+) 越小越好，也就是去压低generator的分数，会发现discriminator其实就是一个二元分类器:

- ![Maximize (\check{V} = \frac{1}{m}\sum_{i=1}^mlogD(x^i ) + \frac{1}{m}\sum_{i=1}^mlog(1-D(\check{x}^i)))](https://www.zhihu.com/equation?tex=Maximize+%28%5Ccheck%7BV%7D+%3D+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5EmlogD%28x%5Ei+%29+%2B+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emlog%281-D%28%5Ccheck%7Bx%7D%5Ei%29%29%29)
- ![\theta_d \leftarrow\theta_d + \eta \nabla\check{V}(\theta_d)](https://www.zhihu.com/equation?tex=%5Ctheta_d+%5Cleftarrow%5Ctheta_d+%2B+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_d%29) ( ![\eta ](https://www.zhihu.com/equation?tex=%5Ceta+) 也是超参数，需要自己调）

1~4步是在训练discriminator, 通常**discriminator的参数可以多更新几次**

\5. 从一个分布中sample出m个向量 ![\{z^1,z^2,..,z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2Cz%5E2%2C..%2Cz%5Em%5C%7D)注意这些sample不需要和步骤2中的保持一致。

\6. 更新generator的参数![\theta_g](https://www.zhihu.com/equation?tex=%5Ctheta_g) 来最小化:

- ![\check{V} = \frac{1}{m}\sum_{i=1}^mlog(1- (D(G(z^i ))) ](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D+%3D+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emlog%281-+%28D%28G%28z%5Ei+%29%29%29+)
- ![\theta_g \leftarrow\theta_g - \eta \nabla\check{V}(\theta_g)](https://www.zhihu.com/equation?tex=%5Ctheta_g+%5Cleftarrow%5Ctheta_g+-+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_g%29)

5~6步是在训练generator，通常在训练generator的过程中，**generator的参数最好不要变化得太大，可以少update几次**。



**对于generator来说，希望其生成的图片经过判别器时能越大越好，而对于discrimator来说，希望generator生成的图片在经过判别器时能越小越好。**



$********************************************************$

## **2.GAN背后的数学理论**

考虑一下，GAN到底生成的是什么呢？比如说，假如我们想要生成一些人脸图，实际上，我们是想找到一个分布，从这个分布内sample出来的图片，像是人脸，而不属于这个distribution的分布，生成的就不是人脸。而GAN要做的就是找到这个distribution。

![img](https://pic3.zhimg.com/80/v2-9807e30da8f358096f43e4dfc223dee2_hd.jpg)

在GAN出生之前，我们怎么做这个事情呢？

之前用的是Maximum Likelihood Estimation，最大似然估计来做生成的，我们先看一下最大似然估计做的事情。

### **从最大似然估计讲起**

最大似然估计的理念是，假如说我们的数据集的分布是 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) ，我们定义一个分布 ![P_{G}(x;\theta)](https://www.zhihu.com/equation?tex=P_%7BG%7D%28x%3B%5Ctheta%29) ,我们想要找到一组参数 ![\theta ](https://www.zhihu.com/equation?tex=%5Ctheta+) ,使得 ![P_{G}(x;\theta)](https://www.zhihu.com/equation?tex=P_%7BG%7D%28x%3B%5Ctheta%29) 越接近 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 越好。比如说，加入 ![P_{G}(x;\theta)](https://www.zhihu.com/equation?tex=P_%7BG%7D%28x%3B%5Ctheta%29) 如果是一个高斯混合模型，那么 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) 就是均值和方差。

具体怎么操作呢

1. 从 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 中sample出 ![\{x^1, x^2, x^3,..,x^m\}](https://www.zhihu.com/equation?tex=%5C%7Bx%5E1%2C+x%5E2%2C+x%5E3%2C..%2Cx%5Em%5C%7D)
2. 对每一个sample出来的x， 我们都可以计算它的likelihood，也就是给定一组参数 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) ,我们就能够知道 ![P_G (x;\theta)](https://www.zhihu.com/equation?tex=P_G+%28x%3B%5Ctheta%29) 长什么样，然后我们就可以计算出在这个分布里面sample出某一个x的几率。
3. 我们把在某个分布可以产生 ![x_i](https://www.zhihu.com/equation?tex=x_i) 的likelihood乘起来，可以得到总的likelihood： ![L=\Pi^m_{i=1}P_G(x^i;\theta)](https://www.zhihu.com/equation?tex=L%3D%5CPi%5Em_%7Bi%3D1%7DP_G%28x%5Ei%3B%5Ctheta%29) , 我们要找到一组 ![\theta^*](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A) , 可以最大化 ![L](https://www.zhihu.com/equation?tex=L)

### **最大似然估计的另一种解释：Minimize KL Divergence**

前面我们已经解释过，我们要找到一组 ![\theta^*](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A) ,使得![\theta^*=argmax_{\theta}\Pi_{i=1}^{m}P_G(x_i;\theta)](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A%3Dargmax_%7B%5Ctheta%7D%5CPi_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x_i%3B%5Ctheta%29) ，我们对其最一些变换, 加一个Log，再把Log乘进去：

![\theta^*=argmax_{\theta}\Pi_{i=1}^{m}P_G(x_i;\theta) =argmax_{\theta}log\Pi_{i=1}^{m}P_G(x_i;\theta) \\ = argmax_{\theta}\sum_{i=1}^{m}logP_G(x_i;\theta)](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A%3Dargmax_%7B%5Ctheta%7D%5CPi_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x_i%3B%5Ctheta%29+%3Dargmax_%7B%5Ctheta%7Dlog%5CPi_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x_i%3B%5Ctheta%29+%5C%5C+%3D+argmax_%7B%5Ctheta%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7DlogP_G%28x_i%3B%5Ctheta%29)

其中, ![\sum_{i=1}^m](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5Em) 就相当于我从 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 中sample出m个样本出来。我们把式子写成：

![\theta^* \approx argmax_\theta E_{x\sim P_{data}[logP_g(x;\theta)]} \\](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A+%5Capprox+argmax_%5Ctheta+E_%7Bx%5Csim+P_%7Bdata%7D%5BlogP_g%28x%3B%5Ctheta%29%5D%7D+%5C%5C)

接下来我们把 ![E_{x\sim P_{data}}](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+P_%7Bdata%7D%7D) 这一项展开，做一个积分，得到：

![\theta^* = argmax_{\theta}\int_{x}P_{data}(x)logP_G(x;\theta)dx \\](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A+%3D+argmax_%7B%5Ctheta%7D%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29logP_G%28x%3B%5Ctheta%29dx+%5C%5C)

接下来我们在上个式子的基础上加一项和 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) 无关的项：

![\theta^* = argmax_{\theta}\int_{x}P_{data}(x)logP_G(x;\theta)dx - \int_{x}P_{data}(x)logP_{data}(x)dx\\](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A+%3D+argmax_%7B%5Ctheta%7D%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29logP_G%28x%3B%5Ctheta%29dx+-+%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29logP_%7Bdata%7D%28x%29dx%5C%5C)

为什么要加上这一项看上去没用的项呢？因为加上这一项后，会发现这就是一个 ![P_{data}, P_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2C+P_G) 的KL divergence。数学上KL divergence使用来衡量两个分布的差异程度的，那么现在我们的目标就是找一组 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) 来最小化 ![P_{data}, P_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2C+P_G) 的KL divegence：

![\theta^*=argmin_\theta KL(P_{data}||P_G) \\](https://www.zhihu.com/equation?tex=%5Ctheta%5E%2A%3Dargmin_%5Ctheta+KL%28P_%7Bdata%7D%7C%7CP_G%29+%5C%5C)

所以机器学习中的最大似然估计，其实就是最小化我们要寻找的目标分布 ![P_G](https://www.zhihu.com/equation?tex=P_G) 与 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 的KL divergence

那么现在来考虑一个问题，如何定义 ![P_G](https://www.zhihu.com/equation?tex=P_G) 呢？

### **Generator**

过去如果使用最大似然估计，采用高斯混合模型定义 ![P_G](https://www.zhihu.com/equation?tex=P_G) ,生成的图片会非常模糊。而现在我们用的Generator的方法，是从一个简单的分布（比如正态分布）中sample出样本，然后扔进一个network(即generator)，然后得到输出，把这些输出统统集合起来，我们会得到一个distribution, 这个distribution就是我们要找的 ![P_G](https://www.zhihu.com/equation?tex=P_G) ，而我们的目标是使得 ![P_G](https://www.zhihu.com/equation?tex=P_G) 与 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 越接近越好。

![img](https://pic3.zhimg.com/80/v2-87ace001edf0cf811e0b4562639dd2c6_hd.jpg)

优化目标是最小化 ![P_G, P_{data}](https://www.zhihu.com/equation?tex=P_G%2C+P_%7Bdata%7D) 之间的差异：

![G^* = argmin_G Div(P_G, P_{data}) \\](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_G+Div%28P_G%2C+P_%7Bdata%7D%29+%5C%5C)

那么怎么计算两个分布的差异呢？ ![P_{G}, P_{data}](https://www.zhihu.com/equation?tex=P_%7BG%7D%2C+P_%7Bdata%7D) 的公式我们都是不知道的，怎么算呢？这里就要引出Discriminator了：

### **Discriminator**

虽然我们不知道 ![P_G](https://www.zhihu.com/equation?tex=P_G) 和 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 的公式，但是我们可以从这两个分布中sample出一些样本出来。对于 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 来说，我们从给定的数据集中sample出一些样本就可以了。对于 ![P_G](https://www.zhihu.com/equation?tex=P_G) 来说，我们随机sample一些向量，扔到Generator里面，然后generator会输出一些图片，这就是从 ![P_G](https://www.zhihu.com/equation?tex=P_G) 里面sample了。现在我们有了这些sample， 透过discriminator，我们可以计算 ![P_{G}](https://www.zhihu.com/equation?tex=P_%7BG%7D) 和 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 的converge。Discriminator的目标函数是：

![V(G, D) = E_{x\sim P_{data}}[logD(x)] + E_{x\sim P_{G}}[log(1-D(x))] \\ D^{*} = argmax_DV(D,G) \\](https://www.zhihu.com/equation?tex=V%28G%2C+D%29+%3D+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BlogD%28x%29%5D+%2B+E_%7Bx%5Csim+P_%7BG%7D%7D%5Blog%281-D%28x%29%29%5D+%5C%5C+D%5E%7B%2A%7D+%3D+argmax_DV%28D%2CG%29+%5C%5C)

我们要最大化 ![V(D,G)](https://www.zhihu.com/equation?tex=V%28D%2CG%29) 。这个目标函数的意思是，假设x是从 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 里面sample出来的，那么希望 ![logD(x)](https://www.zhihu.com/equation?tex=logD%28x%29) 越大越好。如果是从 ![P_G](https://www.zhihu.com/equation?tex=P_G) 里面sample出来的，就希望它的值越小越好，其实这个公式与逻辑回归的损失是一样的。而训练出来的 ![max_DV(D,G)](https://www.zhihu.com/equation?tex=max_DV%28D%2CG%29) ,其实相当于JS divergence。直观的理解是，如果discriminator无法区分 ![P_G](https://www.zhihu.com/equation?tex=P_G) 和 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) ,那么它就没法将 ![V(D,G)](https://www.zhihu.com/equation?tex=V%28D%2CG%29) 的值调的很大，这就说明两个分布的差异比较小。接下来从数学上去解释这个结论。

给定Generator, 我们要找到能最大化目标函数 ![V(D,G)](https://www.zhihu.com/equation?tex=V%28D%2CG%29) 的 ![D^*](https://www.zhihu.com/equation?tex=D%5E%2A) :

![\begin{align*} V &= E_{x\sim P_{data}}[logD(x)]  + E_{x\sim P_{G}}[log(1-D(x))] \\ &= \int_{x}P_{data}(x)logD(x)dx + \int_{x}P_G(x)log(1-D(x))dx \\ &= \int_{x}[P_{data}(x)logD(x) + P_G(x)log(1-D(x))]dx \end{align*} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+V+%26%3D+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BlogD%28x%29%5D++%2B+E_%7Bx%5Csim+P_%7BG%7D%7D%5Blog%281-D%28x%29%29%5D+%5C%5C+%26%3D+%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29logD%28x%29dx+%2B+%5Cint_%7Bx%7DP_G%28x%29log%281-D%28x%29%29dx+%5C%5C+%26%3D+%5Cint_%7Bx%7D%5BP_%7Bdata%7D%28x%29logD%28x%29+%2B+P_G%28x%29log%281-D%28x%29%29%5Ddx+%5Cend%7Balign%2A%7D+%5C%5C)

现在我们把积分里面的这一项拿出来看：

![P_{data}(x)logD(x) + P_G(x)log(1-D(x)) \\](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29logD%28x%29+%2B+P_G%28x%29log%281-D%28x%29%29+%5C%5C)

我们想要找到一组参数 ![D^*](https://www.zhihu.com/equation?tex=D%5E%2A) ,让这一项最大。我们把这个式子简写一下，将 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 用a表示， ![P_{G}](https://www.zhihu.com/equation?tex=P_%7BG%7D) 用b表示，那么上式可写为：

![f(D) = alog(D) + blog(1-D) \\](https://www.zhihu.com/equation?tex=f%28D%29+%3D+alog%28D%29+%2B+blog%281-D%29+%5C%5C)

我们对其求导，有:

![\frac{df(D)}{dD} = a *\frac{1}{D} + b*\frac{1}{1-D}*(-1) \\](https://www.zhihu.com/equation?tex=%5Cfrac%7Bdf%28D%29%7D%7BdD%7D+%3D+a+%2A%5Cfrac%7B1%7D%7BD%7D+%2B+b%2A%5Cfrac%7B1%7D%7B1-D%7D%2A%28-1%29+%5C%5C)

我们令这个求导的结果为0，则得到：

![D^*= \frac{a}{a+b} \\](https://www.zhihu.com/equation?tex=D%5E%2A%3D+%5Cfrac%7Ba%7D%7Ba%2Bb%7D+%5C%5C)

将a, b代表的内容代回去，则是：

![D^*= \frac{P_{data}(x)}{P_{data}(x)+P_G(x)} \\](https://www.zhihu.com/equation?tex=D%5E%2A%3D+%5Cfrac%7BP_%7Bdata%7D%28x%29%7D%7BP_%7Bdata%7D%28x%29%2BP_G%28x%29%7D+%5C%5C)

我们求出了这个D，把它代到 ![V(G,D^*)](https://www.zhihu.com/equation?tex=V%28G%2CD%5E%2A%29) 里面，然后将分子分母同时除以2，然后提出来（这一步是为了之后方便化简）,之后可以将其化简成Jensen-Shannon divergence(某一种计算分部差异的公式）的形式：

![\begin{align*} max_DV(G,D) &= V(G,D*) \\  &= E_{x\sim P_{data}}[log\frac{P_{data}(x)}{P_{data}(x)+P_G(x)}]  + E_{x\sim P_{G}}[log\frac{P_G(x)}{P_{data}(x)+P_G(x)}]  \\ &= \int_{x}P_{data}(x)log\frac{P_{data}(x)}{P_{data}(x)+P_G(x)}dx + \int_{x}P_G(x)log\frac{P_G(x)}{P_{data}(x)+P_G(x)}dx \\ &= -2log2 + \int_{x}P_{data}(x)log\frac{P_{data}(x)}{(P_{data}(x)+P_G(x))/2}dx + \int_{x}P_G(x)log\frac{P_G(x)}{(P_{data}(x)+P_G(x))/2}dx \\ &=-2log2 + KL(P_{data}||\frac{P_{data}+P_G}{2}) + KL(P_{G}||\frac{P_{data+P_G}}{2})\\ &=-2log2 + 2JSD(P_{data}||P_G) \end{align*} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+max_DV%28G%2CD%29+%26%3D+V%28G%2CD%2A%29+%5C%5C++%26%3D+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5Blog%5Cfrac%7BP_%7Bdata%7D%28x%29%7D%7BP_%7Bdata%7D%28x%29%2BP_G%28x%29%7D%5D++%2B+E_%7Bx%5Csim+P_%7BG%7D%7D%5Blog%5Cfrac%7BP_G%28x%29%7D%7BP_%7Bdata%7D%28x%29%2BP_G%28x%29%7D%5D++%5C%5C+%26%3D+%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29log%5Cfrac%7BP_%7Bdata%7D%28x%29%7D%7BP_%7Bdata%7D%28x%29%2BP_G%28x%29%7Ddx+%2B+%5Cint_%7Bx%7DP_G%28x%29log%5Cfrac%7BP_G%28x%29%7D%7BP_%7Bdata%7D%28x%29%2BP_G%28x%29%7Ddx+%5C%5C+%26%3D+-2log2+%2B+%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29log%5Cfrac%7BP_%7Bdata%7D%28x%29%7D%7B%28P_%7Bdata%7D%28x%29%2BP_G%28x%29%29%2F2%7Ddx+%2B+%5Cint_%7Bx%7DP_G%28x%29log%5Cfrac%7BP_G%28x%29%7D%7B%28P_%7Bdata%7D%28x%29%2BP_G%28x%29%29%2F2%7Ddx+%5C%5C+%26%3D-2log2+%2B+KL%28P_%7Bdata%7D%7C%7C%5Cfrac%7BP_%7Bdata%7D%2BP_G%7D%7B2%7D%29+%2B+KL%28P_%7BG%7D%7C%7C%5Cfrac%7BP_%7Bdata%2BP_G%7D%7D%7B2%7D%29%5C%5C+%26%3D-2log2+%2B+2JSD%28P_%7Bdata%7D%7C%7CP_G%29+%5Cend%7Balign%2A%7D+%5C%5C)

通过这一系列的化简，我们可以知道，最大化 ![V(G,D*)](https://www.zhihu.com/equation?tex=V%28G%2CD%2A%29) ，其实就是求解分布 ![P_{data},P_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2CP_G) 的JS divergence。所以当去训练一个distriminator，就是通过 ![P_{data}, P_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2C+P_G) sample出来的样本去求这两个分布的差异。

我们从头整理一下，我们的目标是找到一个 ![G^*](https://www.zhihu.com/equation?tex=G%5E%2A) ,去最小化 ![P_{G}, P_{data}](https://www.zhihu.com/equation?tex=P_%7BG%7D%2C+P_%7Bdata%7D) 的差异。也就是：

![G^* = argmin_GDiv(P_G, P_{data}) \\](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_GDiv%28P_G%2C+P_%7Bdata%7D%29+%5C%5C)

而这个divergence我们没有办法直接去算，我们不知道 ![P_G, P_{data}](https://www.zhihu.com/equation?tex=P_G%2C+P_%7Bdata%7D) 的公式具体是什么。于是我们通过一个discriminator来计算两个分布间的差异：

![D^*=argmax_DV(D,G) \\](https://www.zhihu.com/equation?tex=D%5E%2A%3Dargmax_DV%28D%2CG%29+%5C%5C)

那么我们的优化目标就变为:

![G^* = argmin_Gmax_DV(G,D) \\](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_Gmax_DV%28G%2CD%29+%5C%5C)

这个看起来很复杂，其实直观理解一下，如下图，我们假设已经把Generator固定住了，图片的曲线表示，红点表示固定住G后的 ![max_DV(G,D)](https://www.zhihu.com/equation?tex=max_DV%28G%2CD%29) , 也就是 ![P_G](https://www.zhihu.com/equation?tex=P_G) 和 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 的差异。而我们的目标是最小化这个差异，所以下图的三个网络中， ![G_3](https://www.zhihu.com/equation?tex=G_3) 是最优秀的。

![img](https://pic4.zhimg.com/80/v2-dfe9f82dbb409cabcd59447567215f8b_hd.jpg)

上面确定了我们需要解决的问题。那么如何求解这个问题呢？过程如下：

- 给定 ![G_0](https://www.zhihu.com/equation?tex=G_0)
- 找到可以使得 ![V(G_0, D)](https://www.zhihu.com/equation?tex=V%28G_0%2C+D%29) 最大的 ![D^*](https://www.zhihu.com/equation?tex=D%5E%2A) ，这个过程可以用梯度上升来求解。 ![V(G_0,D_0^*)](https://www.zhihu.com/equation?tex=V%28G_0%2CD_0%5E%2A%29) 就是在求 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 和 ![P_{G_0}(x)](https://www.zhihu.com/equation?tex=P_%7BG_0%7D%28x%29) 的JS divergence（前面已经证明过了）
- 找到 ![max_DV(G,D)](https://www.zhihu.com/equation?tex=max_DV%28G%2CD%29) 后，对 ![G^* = argmin_Gmax_DV(G,D)](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_Gmax_DV%28G%2CD%29) 求导，![\theta_G\leftarrow\theta_G - \eta \partial V(G, D_0^*)/\partial \theta_G](https://www.zhihu.com/equation?tex=%5Ctheta_G%5Cleftarrow%5Ctheta_G+-+%5Ceta+%5Cpartial+V%28G%2C+D_0%5E%2A%29%2F%5Cpartial+%5Ctheta_G) 更新参数得到 ![G_1](https://www.zhihu.com/equation?tex=G_1) , 这个过程其实是在最小化JSd ivergence
- 重新寻找 ![D_1^*](https://www.zhihu.com/equation?tex=D_1%5E%2A) 可以最大化 ![V(G_1,D)](https://www.zhihu.com/equation?tex=V%28G_1%2CD%29) 。这个过程求的是 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 和 ![P_{G_1}(x)](https://www.zhihu.com/equation?tex=P_%7BG_1%7D%28x%29) 的JS divergence
- ![\theta_G\leftarrow\theta_G - \eta \partial V(G, D_1^*)/\partial \theta_G](https://www.zhihu.com/equation?tex=%5Ctheta_G%5Cleftarrow%5Ctheta_G+-+%5Ceta+%5Cpartial+V%28G%2C+D_1%5E%2A%29%2F%5Cpartial+%5Ctheta_G) 得到 ![G_1](https://www.zhihu.com/equation?tex=G_1) , 这个过程可以看做是在最小化JS divergence，这个过程需要**注意不要让 ![G](https://www.zhihu.com/equation?tex=G) 变化得太大，尽量让 ![G_0,G_1](https://www.zhihu.com/equation?tex=G_0%2CG_1) 的差别不要太大，少量多次**。这也是训练过程中的一个tricks
- ...

$********************************************************$

## **3.有监督条件生成网络（CGAN）**

可能首先想到的办法是用传统的监督学习的方法来训练：

![img](https://pic2.zhimg.com/80/v2-289ad291500a67e77b9f3053b6b16fa1_hd.jpg)

但是但是这么做会有一个问题，会发现产生出来的Image都很模糊，为什么呢，比如说，我输入“火车”， 那么火车的图片有正面，有侧面，有不同角度，最后network尝试产生的是多张image的平均值，于是图片就会非常模糊。这里也是为什么要用到Condition GAN的原因。

首先我们回顾一下**传统的GAN**:

![img](https://pic4.zhimg.com/80/v2-3a4c28b07b1b6e500e3ee78a20aa442b_hd.jpg)

它的输入呢就是一个由一个分布sample来的向量。

而 **Conditional GAN**, 除了输入这个sample出来的vector, 还有一个c, 代表输入的文字（在下图中是'train(火车)'）

![img](https://pic4.zhimg.com/80/v2-35fac276bff9a28b3c0f1db5ea158eb3_hd.jpg)

而Discriminator也需要做一些改进，之前的传统的Discriminator，它的任务是检查照片是否真实就可以了，不用去关注图片的内容。因此在conditional GAN里面， discriminator也多了一个输入"c", 代表的是图片的内容。这个时候的discriminator不仅仅要检查图片是否真实，还需要检查c和输入的图片x是不是匹配的。

![img](https://pic4.zhimg.com/80/v2-b2ee744dcf1e4a7efb114bfc853b16ab_hd.jpg)

$也就是在生成器和判别器的输入分别加入一个值，该值可以是标签，也可以是一张图片$

### **Conditional GAN算法流程**

上面说了conditonal GAN的基本理念，这里说一下它的实现算法：

在每一次迭代中：

discriminator部分：

1. 从数据集中sample出m个正例 ![\{(c^1,x^1), (c^2, x^2)...(c^m,x^m)\}](https://www.zhihu.com/equation?tex=%5C%7B%28c%5E1%2Cx%5E1%29%2C+%28c%5E2%2C+x%5E2%29...%28c%5Em%2Cx%5Em%29%5C%7D)
2. 从一个分布中sample出m个噪声点 ![\{z^1,z^2,..,z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2Cz%5E2%2C..%2Cz%5Em%5C%7D)
3. 通过 ![\check{x}^i = G(c^i,z^i) ](https://www.zhihu.com/equation?tex=%5Ccheck%7Bx%7D%5Ei+%3D+G%28c%5Ei%2Cz%5Ei%29+) 生成数据 ![\{\check{x}^1,\check{x}^2...\check{x}^m\}](https://www.zhihu.com/equation?tex=%5C%7B%5Ccheck%7Bx%7D%5E1%2C%5Ccheck%7Bx%7D%5E2...%5Ccheck%7Bx%7D%5Em%5C%7D)
4. 再从数据集里面sample出m个样本点 ![\{\hat{x}^1, \hat{x}^2...\hat{x}^m\}](https://www.zhihu.com/equation?tex=%5C%7B%5Chat%7Bx%7D%5E1%2C+%5Chat%7Bx%7D%5E2...%5Chat%7Bx%7D%5Em%5C%7D)
5. 更新discriminator的参数 ![\theta _d](https://www.zhihu.com/equation?tex=%5Ctheta+_d) 来最大化 ![\check V](https://www.zhihu.com/equation?tex=%5Ccheck+V) :

![\check V = \frac{1}{m}\sum_{i=1} ^mlogD(c^i, x^i)+\frac{1}{m} \sum_{i=1}^mlog(1-D(c^i, \check{x}^i) )+ \frac{1}{m} \sum_{i=1}^mlog(1-D(c^i, \hat{x}^i))  \\](https://www.zhihu.com/equation?tex=%5Ccheck+V+%3D+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D+%5EmlogD%28c%5Ei%2C+x%5Ei%29%2B%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Emlog%281-D%28c%5Ei%2C+%5Ccheck%7Bx%7D%5Ei%29+%29%2B+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Emlog%281-D%28c%5Ei%2C+%5Chat%7Bx%7D%5Ei%29%29++%5C%5C)

![\theta_d \leftarrow\theta_d + \eta \nabla\check{V}(\theta_d)](https://www.zhihu.com/equation?tex=%5Ctheta_d+%5Cleftarrow%5Ctheta_d+%2B+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_d%29)

这个优化过程跟之前的传统GAN的求解过程的差别在于， 多了一项 ![\frac{1}{m} \sum_{i=1}^mlog(1-D(c^i, \hat{x}^i) ](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Emlog%281-D%28c%5Ei%2C+%5Chat%7Bx%7D%5Ei%29+) , 这个 ![\hat{x}^i](https://www.zhihu.com/equation?tex=%5Chat%7Bx%7D%5Ei) 我们前面提到了也是从数据集里面取出来的图片。其实就是说，这个 ![\check{V}](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D) ,不但要求图片真实，而且图片真实的条件下还得和输入文字匹配才能给高分，否则都给低分。

Generator部分：

\6. 从一个分布中sample出m个噪声点 ![\{z^1, z^2...z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2C+z%5E2...z%5Em%5C%7D)

\7. 从数据集中sample出m个条件 ![\{c^1, c^2...c^m\}](https://www.zhihu.com/equation?tex=%5C%7Bc%5E1%2C+c%5E2...c%5Em%5C%7D)

\8. 通过更新generator的参数 ![\theta_g](https://www.zhihu.com/equation?tex=%5Ctheta_g) 来最大化：

![\check V = \frac{1}{m} \sum_{i=1}^mlog(D(G(c^i, z^i)))  ](https://www.zhihu.com/equation?tex=%5Ccheck+V+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Emlog%28D%28G%28c%5Ei%2C+z%5Ei%29%29%29++)

![\theta_g \leftarrow\theta_g + \eta \nabla\check{V}(\theta_g)](https://www.zhihu.com/equation?tex=%5Ctheta_g+%5Cleftarrow%5Ctheta_g+%2B+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_g%29)

### **Conditional GAN的设计**

CGAN要怎么设计呢，一种比较合理的网络设计方式是，将样本输入一个network，输出这个值是否真实， 同时将这个样本和条件一起输入一个Network，得到c与x是否匹配。

![img](https://pic1.zhimg.com/80/v2-e8244896c272e383bcabe1d200e9734c_hd.jpg)

## **4.无监督条件生成网络**

假设有这么一个问题，你想做一个Generation， 将一张实物图转化为某种风格画风的画作，比如说将照片转化为梵高的油画风，那么其实在这种情况下，你是没有一对一的数据可以用的， 你只有一个照片数据集X和画作数据集Y。那么怎样才能做到这种风格迁移的事情呢？这里要提的是unsupervised conditional GAN, 无监督生成网络。

无监督生成网络要解决的问题有两个，一个是将属于X domain的数据转成Y domain， 另一个，X domain的数据内容不能丢失太多，比如说，输入一张手机拍摄的风景照，希望可以转出一张梵高风格的风景画，而不是梵高风格的人物画。其实这个跟之前提过的 [conditional GAN](https://zhuanlan.zhihu.com/p/52915933)有点像，也就是说，即需要生成指定的风格，也需要保证能对应的上内容。

Unsupervised CGAN主要有两种实现途径：

### **第一种方式：Direct Transformation**

一种方式是直接输入X domain图片，经过Generator后生成对应的Y domain的图像。这种转化input和output不能够差太多。通常只能实现较小的转化，比如改变颜色等。

![img](https://pic4.zhimg.com/80/v2-629ff2a9bd9f75dedaefedef573a4e9f_hd.jpg)

具体一点， Generator输入来自Domain X的样本，输出一个vector,那么对于Discriminator来说， 输入一张图片，他需要输出这张图片是否属于Domain Y。

![img](https://pic4.zhimg.com/80/v2-89aa757489c1cdc32a8a7f0430419287_hd.jpg)

但这样子会有一个问题，无监督学习没有一对一的关系，那么Generator可能会生成具有Domain Y的风格但是内容与真实图片完全无关的图片。怎么解决这个问题呢？

有三种解决方案：

1. 设计简单的网络，那么简单的生成器会使得输入与输出更加的相似(Tomer Galanti, et al.ICLR, 2018)

2. 如果网络很深的话，在方法1的基础上，还需要拿一个预训练好的network, 把generator的input和output丢入一个同样的encoder network, 两个网络分别输出一个vector，然后要使得这两个vector尽可能的相似，也就是加大genertor的input和output的联系

3. 第三种做法，就是大家所熟知的cycle GAN：

   

   $不能讲方法一运用到较深的网络，容易出现风格类似但是内容不同的情况，方法一只是小修小改，比如只是颜色上的一些变化$

#### **cycle GAN**

```python
'''
本质上来说，cylcegan就是在第一种方法的基础上，利用产生器，又产生一个y-x的转化，使得生成器的输入和输出一致，这非常类似于第二种方法，使用2个encoder网络，分别作用于生成器的输入和输出，使得输入和输出越接近越好
'''
```

cycle GAN的大致思想是，不仅仅训练一个X domian到 Y domain的generator, 同时还需要训练一个Y domain 到 X domain的generator, 然后要使得Input和最后产生的还原的domain X的图越像越好。所以cycle GAN的理念就是转了两次后要能转回初始输入（Jun-Yan Zhu, et al, ICCV, 2017）

![img](https://pic1.zhimg.com/80/v2-f9b9b85e656270051f17b9bec1375d9c_hd.jpg)

cycle GAN不但可以完成从X domain到Y domain的转化，也可以完成从Y domain到X domain的转化，当然原理是一样的。

![img](https://pic4.zhimg.com/80/v2-b95b7fd1ea097b963032caf0de4fe127_hd.jpg)

#### **一些有意思的cycle gan项目：**

[Aixile/chainer-cyclegan](https://link.zhihu.com/?target=https%3A//github.com/Aixile/chainer-cyclegan)

cycle GAN也有一些潜在的问题，2017年NIPS上就有一篇论文[CycleGAN, a Master of Steganography](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1712.02950) 提出来说， 

就是cyclegan在转化的过程中即x-y，产生的图会有一些不一样的特点，会将原图上的某些特点隐藏起来，例如

![img](https://pic4.zhimg.com/80/v2-449ca025d2fc948010a4580e277b2207_hd.jpg)

### **第四种方式：使用encoder和decoder的形式**

具体总结如下：

```python
'''
使用encoder和decoder的形式：这种方法比较适合复杂的风格转换

思路：训练一个domain x的encoder提取特征v，然后decoder解码，然后用domain x的判别器判断产生的是否是domian x，同理，domain y也一样。

问题：domain x和domain y特征表示不一样，有可能Vx的第一维表示眼睛，Vy的第一维度表示嘴巴，无法直接转，具体解决办法如下：

方法1.couple GAN：让domain x 和 domain y的共享encoder的最后几层和decoder的前几层特征，这样就是为了强行让表示的特征一致。

方法2：domain discrimator，强行判断v是来自domain x还是domain y的。

方法3：cycle consistency：先把图片放到domain x的encoder，然后通过domain y的decoder，产生的图片在放到domain y的encoder，在放到domain x的decoder，产生图片，最后让产生的这个图片与一开始输入的图片越接近越好

方法4：XGAN：先将一张图放到domain x的encoder，产生Vx，然后经过domain y的decoder，产生图片，然后放到domain x的encoder，然后产生vy，目的是让Vx和Vy越接近越好


'''
```

前面说的是输入Input直接生成output的方式，当然前面也提到直接拿图片去转，只能做一些简单的改变。所以通常会用另一种方式，先将input通过一个encoder，把人脸的特征抽出来, 然后再通过decoder,生成对应的Y domain的内容。这一种方式可以实现比较复杂的内容。

![img](https://pic2.zhimg.com/80/v2-eddd17401e1fc42aa2aa6d819c439431_hd.jpg)

具体的操作是，建立一个Domain X 的encoder和decoder, 并且在后面接一个Domain X的discriminator, 判断输出是否属于x domain,要能够把domain X的图片解回来, 这里可以看做是一个VAE的网络。同时训练一个Domain Y的encoder和decoder和判断Y domain的discriminator。训练好两个网络后，将domain x输入encoder X, 得到一个向量，再将这个向量输入Decoder Y, 得到由domain X转化的domain Y的图片，这个经由encoder X得到的向量，可以把它看作是脸部抽出的特征。

![img](https://pic4.zhimg.com/80/v2-5c4ada5022352a3768ae17413b4dbb57_hd.jpg)

这种办法可以做到domain X到domain Y的转化，但是有一个问题，

```python
 '''
 X domain和Y domain是分开训练的，无法保证domain x的图像经过encoder提取的特征，经过domain y 的encoder的解码后会变成对应的特征，例如上一层经过domain x 后，产生的特征，第一维可能是眼睛，但是对于下一层，第一维可能是嘴巴。
 '''
```

下面介绍几种不同的网络：

**Couple GAN/ UNIT**

一种办法是使得**encoder X和encoder Y共用最后几层参数， decoder X和decoder Y共用前几层参数**。是通过参数共享来加强domain X和domain Y的对应关系，这就是Couple GAN和UNIT的思想。

![img](https://pic2.zhimg.com/80/v2-a980063f510f304cefb03691de566361_hd.jpg)

另外一种办法是增加一个Domain Discriminator对encoder输出的向量进行判断，这个Discriminator需要判断这个向量是来自X domain还是Y domain。这种方式会让encoder X和encoder Y解出来的向量尽量保持一致。

![img](https://pic1.zhimg.com/80/v2-1bf574ba8836dfbcff52512c68f1050c_hd.jpg)

#### **ComboGAN:cycle consistency**

另外一种方式是cycle consistency,原理是先把图片放到domain x的encoder，然后通过domain y的decoder，产生的图片在放到domain y的encoder，在放到domain x的decoder，产生图片，最后让产生的这个图片与一开始输入的图片越接近越好

![img](https://pic2.zhimg.com/80/v2-6a34ee7d01206a08fec6db16416022a1_hd.jpg)

#### **XGAN: semantic Consistency**

还有一种思想是semantic Consistency，也就是XGAN

把一张图片丢进encoder X,用生成vector经过Y domain的decoder， 然后再经过Y domain的encoder, 得到的向量与由原始图像经过X domian的encoder的向量越接近越好。

前面提到的cycle consistency，算的是Image之间的区别，可能更偏表象一些，而这里算的是semantic(语义）上的区别。

![img](https://pic1.zhimg.com/80/v2-7367a2734ce4dc9bf4f0449ef6502588_hd.jpg)



## **5.F-GAN**

```python
'''
F-GAN的主要意思就是提出一个分布距离度量函数，该度量函数是用来求解两个分布之间的差异的，该分布函数展开就和判别器的损失函数一致了，所以理论上来说，更换不同的f的表达式，都可以进行度量两个分布之间的差异。
'''
```

在开始讲fGAN之前，需要先补充两个基础知识，f-divergence和Fenchel Conjugate。为什么要讲它们，后面会提到。

### **f-divergence：通用的divergence模型**

P和Q是两个分布， ![p(x),q(x)](https://www.zhihu.com/equation?tex=p%28x%29%2Cq%28x%29) 是样本x从分布P sample出来的概率和从分布Q sample出来的概率。那么对于任意f：

![D_f(P||Q) = \int_{x}q(x)f(\frac{p(x)}{q(x)})dx \\](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%3D+%5Cint_%7Bx%7Dq%28x%29f%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29dx+%5C%5C)

如果f同时满足两个条件：![f(1)=0](https://www.zhihu.com/equation?tex=f%281%29%3D0) ,且f是convex的，那么 ![D_f(P||Q)](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29) 就是一个divergence。 ![D_f (P||Q)](https://www.zhihu.com/equation?tex=D_f+%28P%7C%7CQ%29) 衡量的是P, Q的差异。为什么这么说呢？我们看看以下几种情况：

如果 对于所有的x，都有![p(x)=q(x)](https://www.zhihu.com/equation?tex=p%28x%29%3Dq%28x%29) ，则 ![D_f(P||Q) = \int_{x}q(x)f(1)dx = 0](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%3D+%5Cint_%7Bx%7Dq%28x%29f%281%29dx+%3D+0)

而对于其它情况，由于f是convex的，上式可以写成

![D_f(P||Q) = \int_{x}q(x)f(\frac{p(x)}{q(x)})dx \geq f(\int_xq(x)\frac{p(x)}{q(x)}dx) =f(\int_xp(x)dx)=f(1)=0](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%3D+%5Cint_%7Bx%7Dq%28x%29f%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29dx+%5Cgeq+f%28%5Cint_xq%28x%29%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7Ddx%29+%3Df%28%5Cint_xp%28x%29dx%29%3Df%281%29%3D0)

只要p(x)和q(x)有差异，那么 ![D_f(P||Q)](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29) 就是大于零的。我们看几个例子：

![f(x) = xlogx \\ D_f(P||Q) = \int_xq(x)\frac{p(x)}{q(x)}log(\frac{p(x)}{q(x)})dx=\int_xp(x)log(\frac{p(x)}{q(x)})dx ](https://www.zhihu.com/equation?tex=f%28x%29+%3D+xlogx+%5C%5C+D_f%28P%7C%7CQ%29+%3D+%5Cint_xq%28x%29%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7Dlog%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29dx%3D%5Cint_xp%28x%29log%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29dx+)

上式是KL divergence

![f(x) = -logx \\ D_f(P||Q) = \int_xq(x)(-log(\frac{p(x)}{q(x)}))dx = \int_xq(x)log(\frac{q(x)}{p(x)})dx](https://www.zhihu.com/equation?tex=f%28x%29+%3D+-logx+%5C%5C+D_f%28P%7C%7CQ%29+%3D+%5Cint_xq%28x%29%28-log%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29%29dx+%3D+%5Cint_xq%28x%29log%28%5Cfrac%7Bq%28x%29%7D%7Bp%28x%29%7D%29dx)

上面这个式子是Reverse KL divergence

![f(x) = (x-1)^2\\ D_f(P||Q) = \int_xq(x)(\frac{p(x)}{q(x)}-1)^2dx = \int_x \frac{(p(x)-q(x))^2}{q(x)}dx \\](https://www.zhihu.com/equation?tex=f%28x%29+%3D+%28x-1%29%5E2%5C%5C+D_f%28P%7C%7CQ%29+%3D+%5Cint_xq%28x%29%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D-1%29%5E2dx+%3D+%5Cint_x+%5Cfrac%7B%28p%28x%29-q%28x%29%29%5E2%7D%7Bq%28x%29%7Ddx+%5C%5C)

上面的是熟悉的Chi Square

所以f divergence是不是很神奇！

### **Fenchel Conjugate共轭函数**

每一个凸函数都有一个共轭函数(conjugate function)，记为 ![f^*](https://www.zhihu.com/equation?tex=f%5E%2A) ,长这样：

![f^*(t) = max_{x\epsilon dom(f)}{\{xt-f(x)\}} \\](https://www.zhihu.com/equation?tex=f%5E%2A%28t%29+%3D+max_%7Bx%5Cepsilon+dom%28f%29%7D%7B%5C%7Bxt-f%28x%29%5C%7D%7D+%5C%5C)

这是什么意思呢，就是说带一个值t到 ![f^*](https://www.zhihu.com/equation?tex=f%5E%2A) 里面，穷举所有的x，看看哪个x可以使得 ![f^*](https://www.zhihu.com/equation?tex=f%5E%2A) 最大，比如说， 我们计算一个 ![f*(t_1)](https://www.zhihu.com/equation?tex=f%2A%28t_1%29)![f*(t_1) = max_{x\epsilon dom(f)}\{xt_1-f(x)\} \\ x_1t_1 - f(x_1) \\ x_2t_1 = f(x_2) \\ x_3t_1 = f(x_3) \\](https://www.zhihu.com/equation?tex=f%2A%28t_1%29+%3D+max_%7Bx%5Cepsilon+dom%28f%29%7D%5C%7Bxt_1-f%28x%29%5C%7D+%5C%5C+x_1t_1+-+f%28x_1%29+%5C%5C+x_2t_1+%3D+f%28x_2%29+%5C%5C+x_3t_1+%3D+f%28x_3%29+%5C%5C)

假如说上面式子里面最大的是 ![x_1t_1 - f(x_1)](https://www.zhihu.com/equation?tex=x_1t_1+-+f%28x_1%29) ,那么 ![f^*(t_1) = x_1t_1 - f(x_1)](https://www.zhihu.com/equation?tex=f%5E%2A%28t_1%29+%3D+x_1t_1+-+f%28x_1%29) , 我们如果把 ![f^*(t) ](https://www.zhihu.com/equation?tex=f%5E%2A%28t%29+) 的图像画出来，会发现 ![f^*(t)](https://www.zhihu.com/equation?tex=f%5E%2A%28t%29) 也是一个凸函数，此处省略证明。对于共轭函数，还有一个性质，就是共轭函数是相互的，也就是对每一对共轭函数来说，有：

![f*(t) = max_{x\epsilon dom(f)}\{xt-f(x)\} \leftrightarrow f*(x) = max_{t\epsilon dom(f)}\{xt-f^*(t)\} \\](https://www.zhihu.com/equation?tex=f%2A%28t%29+%3D+max_%7Bx%5Cepsilon+dom%28f%29%7D%5C%7Bxt-f%28x%29%5C%7D+%5Cleftrightarrow+f%2A%28x%29+%3D+max_%7Bt%5Cepsilon+dom%28f%29%7D%5C%7Bxt-f%5E%2A%28t%29%5C%7D+%5C%5C)

### **Connection with GAN**

上面的内容到底和GAN有什么关系呢？

我们假设有一个divergence:

![D_f(P||Q) = \int_xq(x)f(\frac{p(x)}{q(x)})dx  = \int_xq(x)(max_{t\epsilon dom(f^*)}\{\frac{p(x)}{q(x)}t-f^*(t)\})dx  ](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%3D+%5Cint_xq%28x%29f%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%29dx++%3D+%5Cint_xq%28x%29%28max_%7Bt%5Cepsilon+dom%28f%5E%2A%29%7D%5C%7B%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7Dt-f%5E%2A%28t%29%5C%7D%29dx++)

接下来，我们学习一个function D，这个D的输入是x， 输出是t，我们将t用D(x)替代，同时我们去掉max的概念，改为 ![\geq](https://www.zhihu.com/equation?tex=%5Cgeq) ，那么上式可改写为：

![D_f(P||Q)\geq\int_xq(x)(\frac{p(x)}{q(x)}D(x) - f^*(D(x))dx\\ = \int_xp(x)D(x)dx - \int_xq(x)f^*(D(x))dx\\](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29%5Cgeq%5Cint_xq%28x%29%28%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7DD%28x%29+-+f%5E%2A%28D%28x%29%29dx%5C%5C+%3D+%5Cint_xp%28x%29D%28x%29dx+-+%5Cint_xq%28x%29f%5E%2A%28D%28x%29%29dx%5C%5C)

那么其实相当于：

![D_f(P||Q) \approx max_D\int_xp(x)D(x)dx - \int_xq(x)f^*(D(x))dx \\](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%5Capprox+max_D%5Cint_xp%28x%29D%28x%29dx+-+%5Cint_xq%28x%29f%5E%2A%28D%28x%29%29dx+%5C%5C)

上面这个公式，我们把它改写一下：

![D_f(P||Q) = max_D\{E_{x\sim P}[D(x)] - E_{x\sim Q}[f^*(D(x))]\} \\](https://www.zhihu.com/equation?tex=D_f%28P%7C%7CQ%29+%3D+max_D%5C%7BE_%7Bx%5Csim+P%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+Q%7D%5Bf%5E%2A%28D%28x%29%29%5D%5C%7D+%5C%5C)

我们令 ![P=P_{data} , Q=P_G](https://www.zhihu.com/equation?tex=P%3DP_%7Bdata%7D+%2C+Q%3DP_G) ,那么 ![P_{data}, p_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2C+p_G) 的f-divergence就可以写成：

![D_f(P_{data}||P_G) = max_D\{E_{x\sim P_{data}}[D(x)\ - E_{x \sim P_G}[f^*(D(x))]\} \\](https://www.zhihu.com/equation?tex=D_f%28P_%7Bdata%7D%7C%7CP_G%29+%3D+max_D%5C%7BE_%7Bx%5Csim+P_%7Bdata%7D%7D%5BD%28x%29%5C+-+E_%7Bx+%5Csim+P_G%7D%5Bf%5E%2A%28D%28x%29%29%5D%5C%7D+%5C%5C)

这个 ![f^*](https://www.zhihu.com/equation?tex=f%5E%2A) 取决于f-divergence是什么。

这个式子怎么看起来好像GAN需要minimize的目标呢？GAN的训练目标是 ![G^* = argmin_GD_f(P_{data}||P_G)](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_GD_f%28P_%7Bdata%7D%7C%7CP_G%29) ,我们把GAN的训练目标展开，就是：

![G^* = argmin_Gmax_D\{E_{x\sim P_{data}}D[(x)] - E_{x\sim P_G}[f^*(D(x))]\} \\ =argmin_Gmax_DV(G,D)](https://www.zhihu.com/equation?tex=G%5E%2A+%3D+argmin_Gmax_D%5C%7BE_%7Bx%5Csim+P_%7Bdata%7D%7DD%5B%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5Bf%5E%2A%28D%28x%29%29%5D%5C%7D+%5C%5C+%3Dargmin_Gmax_DV%28G%2CD%29)

所以你可以选用不同的f-divergence,优化不同的divergence,论文里面给了个清单，你可以自己选：

![img](https://pic1.zhimg.com/80/v2-f7ab77645beb057f82f8905e6721e408_hd.jpg)

那么使用不同的divergence会有什么用处吗？它可能能够用于解决GAN在训练过程中会出现的一些问题（众所周知GAN难以训练）：



```python
'''
GAN训练过程中出现的问题：
mode collapse:迭代很多代以后，产生的数据分布越来越小，直接的结果就是图像越来越像，可能原因是f函数选择的不好
解决办法：需要多少张，就训练多少个生成器，这样每个生成器生成的图片都是一致的，彼此不会太像。
Mode Dropping：原始分布有两个比较集中的波峰，而GAN有可能把分布集中在其中一个波峰，而抛弃掉了另一个
解决办法：更换f函数

'''
```



### **GAN训练过程中可能产生的问题：**

**Mode Collapse**

这个概念是GAN难以训练的原因之一，它指的是GAN产生的样本单一，认为满足某一分布的结果为True,其余为False。如下图，原始数据分布的范围要比GAN训练的结果大得多。从而导致generator训练出来的结果可能都差不多，图片差异性不大。

![img](https://pic1.zhimg.com/80/v2-9b7127f39235be364189094037a3b434_hd.jpg)

**Mode Dropping**

这个问题从字面上也好理解，假设原始分布有两个比较集中的波峰，而GAN有可能把分布集中在其中一个波峰，而抛弃掉了另一个，如下图：

![img](https://pic4.zhimg.com/80/v2-4bfbc925f27761e536e99abdf2b8f387_hd.jpg)

为什么会有这样的结果呢，一个猜测是divergence选得不好，选择不同的divergence，最后generator得到的distribution会不一样。如下图，minimize KL divergence和reverse KL divergence的时候，最后得到的分布是不一样的， 前者容易导致模糊的问题，后者则可能导致mode dropping。如果你觉得在训练过程中出现的mode collapse或者mode dropping是由于divergence导致的，你可以通过尝试更换 ![f^*](https://www.zhihu.com/equation?tex=f%5E%2A) 来实验。当然不一定说就一定有效果，这里只是提供一种可能的猜测。

![img](https://pic2.zhimg.com/80/v2-bb365d68f336aa0c01b34da0f20ea3d1_hd.jpg)

对于mode collapse到底有没有一些更加通用的解决办法呢？你可以用ensemble的方法，其实就是训练多个Generator，然后在使用的时候随机挑一个generator来生成结果，当然是一个很流氓的招数，看你用到哪里了...

![img](https://pic1.zhimg.com/80/v2-e7a06ef3687500e2fae78a23a36b31d8_hd.jpg)

当然还有一些别的解决mode collapse的方法，可以参照 [GAN 的Mode collapse](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_38322426/article/details/82462897)

## **6.WGAN, WGAN-GP与EBGAN**

### **WGAN**

WGAN到底在原始GAN的基础上做了什么事情呢？这里先列出它的优化点，下面会具体解释原因：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不再取log
- weight clipping: 对参数w做一个限制，如果发现w大于某个常数c，就令w = c; 如果w<-c , w = -c，等于说将参数限制在某一个范围内。(目的是为了限制discriminator不要变化得太剧烈，尽量平滑）
- 不要用基于动量的优化算法（momentum、Adam等），是一个tricks

在一一描述这些改动的意义之前，先探讨一下原始GAN的难题：

### **为什么GAN难以训练？**

我们在深入了解WGAN为什么可以提升GAN的表现之前，先解释一个问题，为什么原始的GAN难以训练？

在 [生成对抗网络(GAN) 背后的数学理论](https://zhuanlan.zhihu.com/p/54096381) 中提到，GAN的本质是在优化 ![P_G,P_{data}](https://www.zhihu.com/equation?tex=P_G%2CP_%7Bdata%7D) 的JS divergence。如果使用JS divergence的话，由于它的特性，假设两个distributions没有重合，算出来的就是log2。下图中三个分布，我们能看出 ![G_2](https://www.zhihu.com/equation?tex=G_2) 比 ![G_1](https://www.zhihu.com/equation?tex=G_1) 是要更贴近 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) ,但是用 JS divergence来算的话，两者其实没啥区别。对于Generator来说， ![P_{G_0}, P_{G_1}](https://www.zhihu.com/equation?tex=P_%7BG_0%7D%2C+P_%7BG_1%7D) 是一样差的，generator不会把 ![P_{G_0}](https://www.zhihu.com/equation?tex=P_%7BG_0%7D) 迭代成 ![P_{G_1}](https://www.zhihu.com/equation?tex=P_%7BG_1%7D) , 在这里就卡住了。从直觉的角度来说，我们说GAN的discriminator本质上其实是一个逻辑回归分类器。如果两个分布没有重叠的部分，那么二元分类器会很容易地得到100%的accuracy，但这其实是没法证明分类器的能力了。然而在GAN的训练过程中，GAN生成的分布与原始数据分布不重叠的情况其实是大概率发生的。这也是为什么原始的GAN比较难训练的原因之一。（这一部分没有详细证明，推荐查看 [令人拍案叫绝的Wasserstein GAN - 郑华滨的文章 - 知乎](https://zhuanlan.zhihu.com/p/25071913) 这篇博客，写得很好）

![img](https://pic1.zhimg.com/80/v2-6fc57d63b73c1b37ac3a285e2c66e000_hd.jpg)

怎么解决上述问题呢？肯定是不能再用JS divergence了。而今天的主角WGAN，它使用了一种更好的衡量分部差异性的方式，也就是earth mover's distance

### **Earth Mover's Distance**

**在**[fGAN : 一种通用的GAN框架](https://zhuanlan.zhihu.com/p/54909858)这篇文章里面，提到说在构建GAN的模型的时候，不一定要用JS divergence, 你可以用任何一种divergence来构建GAN。那么在WGAN里面，用到的是Earth Mover's Distance（EMD）来替代原始GAN的JS divergence。EMD直译过来是推土机距离，这个其实就是说把分布X变为分布Y所需要付出的最小代价。这个问题会有很多种方案，每一个方案我们称之为一个“moving plan”，对于单个moving plan ![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma) 来说，计算分布X与分布Y每一个配对的元素的距离，总和距离是：

![B(\gamma) = \sum_{x_p,x_q}\gamma(x_p,x_q)||x_p - x_q||\\](https://www.zhihu.com/equation?tex=B%28%5Cgamma%29+%3D+%5Csum_%7Bx_p%2Cx_q%7D%5Cgamma%28x_p%2Cx_q%29%7C%7Cx_p+-+x_q%7C%7C%5C%5C)

接下来穷举所有的 ![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma) ,看哪一个 ![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma) 对应的距离最小，这个最小的距离就是对应的Earth Mover's Distance。

![W(P,Q) = min_{\gamma \epsilon\Pi}B(\gamma) \\](https://www.zhihu.com/equation?tex=W%28P%2CQ%29+%3D+min_%7B%5Cgamma+%5Cepsilon%5CPi%7DB%28%5Cgamma%29+%5C%5C)

容易发现说这个EMD和别的divergence有一些不同，它不是按照一个固定的公式去代值的。它本身就是一个优化问题。

那么今天如果用JS divergence进行优化，我们前面说过，如果两个分布没有重叠，那么它们的距离一直都是log2。那么就没有办法去演化。但是EMD（也就是wasserstein distance) 就不一样，即便没有重叠，它的距离也是一直有变化的。

![img](https://pic1.zhimg.com/80/v2-34ed731b5cfe6c39895923cbfbb20218_hd.jpg)

### **WGAN**

说了EMD的概念，那么怎么样将它应用到GAN里面呢？WGAN的证明比较复杂，这里直接给出公式，我们要让 ![E_{x\sim P_{data}}[D(x)]](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BD%28x%29%5D) 越大越好， ![E_{x\sim P_{G}}[D(x)]](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+P_%7BG%7D%7D%5BD%28x%29%5D) 越小越好：

![V(G,D) = max_{D\epsilon {1-Lipschitz} \{{ E_{x\sim P_{data}}[D(x)] - E_{x\sim P_G}[D(x)]\}}} \\](https://www.zhihu.com/equation?tex=V%28G%2CD%29+%3D+max_%7BD%5Cepsilon+%7B1-Lipschitz%7D+%5C%7B%7B+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5BD%28x%29%5D%5C%7D%7D%7D+%5C%5C)

这里会发现D是属于1-Lipschitz function的，这又是个什么东西呢？它是限制discriminator必须要平滑，如果没有这个限制的话，discriminator为了满足上面的优化目标，可能会使得 ![D(x)](https://www.zhihu.com/equation?tex=D%28x%29) 变为无穷大或无穷小，这样子就没有办法收敛了。Lipschitz function就是起这么一个作用。如果把公式写出来，就是：

![||f(x_1) - f(x_2)|| \leq K ||x_1 - x_2|| \\](https://www.zhihu.com/equation?tex=%7C%7Cf%28x_1%29+-+f%28x_2%29%7C%7C+%5Cleq+K+%7C%7Cx_1+-+x_2%7C%7C+%5C%5C)

就是说input有变化的时候， output的差距不可以太大。当k=1的时候，就是我们要满足的“1-Lipschitz”, 也就是output的变化是比Input要小的。总结来说，这个条件就是限制discriminator不要变化得太剧烈。那么在实际的训练过程中是怎么实现这个限制条件的呢？那就是weight clipping--

### **Weight Clipping**

weight clipping的想法很简单，就是在原来的训练过程基础上，对参数w做一个限制，如果发现w大于某个常数c，就令w = c; 如果w<-c , w = -c，等于说将参数限制在某一个范围内。这一招看起去简单，但是的确是有效的。







### **WGAN的算法实现**

之前在[生成对抗网络GAN（一）：基本概念与算法流程](https://zhuanlan.zhihu.com/p/52233472) 里面描述过普通的GAN的算法。如果现在我们想要改成WGAN，也很简单,主要是对目标函数进行修改。原始的GAN其实用的是逻辑回归，输出的是分类。而在使用WGAN的时候，我们需要优化的是Wasserstein距离，属于回归任务。把 ![D](https://www.zhihu.com/equation?tex=D) 的sigmoid函数拿掉，同时去掉log， 这样输出的就是一个线性的结果。同时在更新梯度的时候进行weight clipping，至于优化器为什么用RMSProp、SGD不要用流行的 Adam\momentum，这个是基于实验得到较好的结果。这与本文开头所述的WGAN的算法改进是一一对应地啦。具体一点：

初始化 ![\theta_d ](https://www.zhihu.com/equation?tex=%5Ctheta_d+) for D( discriminator) ， ![\theta_g](https://www.zhihu.com/equation?tex=%5Ctheta_g) for G( generator)

在每次迭代中：

1. 从数据集 ![P_{data}(x)](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 中sample出m个样本点 ![\{x^1, x^2...x^m\}](https://www.zhihu.com/equation?tex=%5C%7Bx%5E1%2C+x%5E2...x%5Em%5C%7D) ，这个m也是一个超参数，需要自己去调
2. 从一个分布(可以是高斯，正态..., 这个不重要)中sample出m个向量 ![\{z^1,z^2,..,z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2Cz%5E2%2C..%2Cz%5Em%5C%7D)
3. 将第2步中的z作为输入，获得m个生成的数据 ![\{\check{x}^1,\check{x}^2...\check{x}^m\}, \check{x}^i= G(z^i)](https://www.zhihu.com/equation?tex=%5C%7B%5Ccheck%7Bx%7D%5E1%2C%5Ccheck%7Bx%7D%5E2...%5Ccheck%7Bx%7D%5Em%5C%7D%2C+%5Ccheck%7Bx%7D%5Ei%3D+G%28z%5Ei%29)
4. 更新discriminator的参数 ![\theta_d](https://www.zhihu.com/equation?tex=%5Ctheta_d) 来最大化 ![\check{V}](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D) ,

- ![Maximize (\check{V} = \frac{1}{m}\sum_{i=1}^mD(x^i ) + \frac{1}{m}\sum_{i=1}^mD(\check{x}^i)](https://www.zhihu.com/equation?tex=Maximize+%28%5Ccheck%7BV%7D+%3D+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5EmD%28x%5Ei+%29+%2B+%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5EmD%28%5Ccheck%7Bx%7D%5Ei%29)
- ![\theta_d \leftarrow\theta_d + \eta \nabla\check{V}(\theta_d)](https://www.zhihu.com/equation?tex=%5Ctheta_d+%5Cleftarrow%5Ctheta_d+%2B+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_d%29) ，在这个过程中需要使用前面提到的Weight clipping的技术

1~4步是在训练discriminator, 通常discriminator的参数可以多更新几次

\5. 从一个分布中sample出m个向量 ![\{z^1,z^2,..,z^m\}](https://www.zhihu.com/equation?tex=%5C%7Bz%5E1%2Cz%5E2%2C..%2Cz%5Em%5C%7D)注意这些sample不需要和步骤2中的保持一致。

\6. 更新generator的参数![\theta_g](https://www.zhihu.com/equation?tex=%5Ctheta_g) 来最小化:

- ![\check{V} = -\frac{1}{m}\sum_{i=1}^mD(G(z^i )) ](https://www.zhihu.com/equation?tex=%5Ccheck%7BV%7D+%3D+-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5EmD%28G%28z%5Ei+%29%29+)
- ![\theta_g \leftarrow\theta_g - \eta \nabla\check{V}(\theta_g)](https://www.zhihu.com/equation?tex=%5Ctheta_g+%5Cleftarrow%5Ctheta_g+-+%5Ceta+%5Cnabla%5Ccheck%7BV%7D%28%5Ctheta_g%29)

5~6步是在训练generator，通常在训练generator的过程中，generator的参数最好不要变化得太大，可以少update几次

### **其余的优化算法**

WGAN的主题内容已经介绍完了，这里提一下其它的一些对GAN的提升与进展~

### **Improved WGAN(WGAN-GP)**

上面的weight clipping是原始WGAN的技术，但是使用这个技术存在一些限制， [详见 生成式对抗网络GAN有哪些最新的发展，可以实际应用到哪些场景中？ - 郑华滨的回答](https://www.zhihu.com/question/52602529/answer/158727900)中有详细描述 ，于是有了improved WGAN, 其中使用了另一种办法实现 ![D\epsilon 1 - Lipschitz](https://www.zhihu.com/equation?tex=D%5Cepsilon+1+-+Lipschitz) 的限制。

针对

![V(G,D) = max_{D\epsilon {1-Lipschitz} \{{ E_{x\sim P_{data}}[D(x)] - E_{x\sim P_G}[D(x)]\}}} \\](https://www.zhihu.com/equation?tex=V%28G%2CD%29+%3D+max_%7BD%5Cepsilon+%7B1-Lipschitz%7D+%5C%7B%7B+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5BD%28x%29%5D%5C%7D%7D%7D+%5C%5C)

中的 ![D\epsilon 1 - Lipschitz](https://www.zhihu.com/equation?tex=D%5Cepsilon+1+-+Lipschitz) 部分，其实等价于 ![|| \bigtriangledown_x D(x)|| \leq 1](https://www.zhihu.com/equation?tex=%7C%7C+%5Cbigtriangledown_x+D%28x%29%7C%7C+%5Cleq+1) ，在Improved WGAN里面，用了这样一种办法来实现这个限制：

![V(G,D) \approx max_D\{E_{x\sim P_{data}[D(x)] - E_{x\sim P_G}[D(x)] - \lambda \int_x max(0, ||\bigtriangledown_xD(x)|| - 1) dx\}} \\](https://www.zhihu.com/equation?tex=V%28G%2CD%29+%5Capprox+max_D%5C%7BE_%7Bx%5Csim+P_%7Bdata%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5BD%28x%29%5D+-+%5Clambda+%5Cint_x+max%280%2C+%7C%7C%5Cbigtriangledown_xD%28x%29%7C%7C+-+1%29+dx%5C%7D%7D+%5C%5C)

上述公式的意思是，对于所有的x，令 ![||\bigtriangledown_xD(x)|| \leq 1](https://www.zhihu.com/equation?tex=%7C%7C%5Cbigtriangledown_xD%28x%29%7C%7C+%5Cleq+1) 。但是实际应用中我们只有一部分sample出来的x，而没有所有的x, 所以将上式改写为：

![V(G,D) \approx max_D\{E_{x\sim P_{data}[D(x)] - E_{x\sim P_G}[D(x)] - \lambda E_{x\sim P_{penalty}}max(0, ||\bigtriangledown_xD(x)|| - 1) dx\}} \\](https://www.zhihu.com/equation?tex=V%28G%2CD%29+%5Capprox+max_D%5C%7BE_%7Bx%5Csim+P_%7Bdata%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5BD%28x%29%5D+-+%5Clambda+E_%7Bx%5Csim+P_%7Bpenalty%7D%7Dmax%280%2C+%7C%7C%5Cbigtriangledown_xD%28x%29%7C%7C+-+1%29+dx%5C%7D%7D+%5C%5C)

就是说对某一个分布 ![P_{penalty}](https://www.zhihu.com/equation?tex=P_%7Bpenalty%7D) sample出的x, 都令 ![||\bigtriangledown_xD(x)|| \leq 1](https://www.zhihu.com/equation?tex=%7C%7C%5Cbigtriangledown_xD%28x%29%7C%7C+%5Cleq+1) 。那么这个penalty分布是啥呢？

如下图，假设从 ![P_{data}, P_G](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%2C+P_G) 中分别sample出一个x, 两个sample的连线上，随机sample一点，这个连线上sample的点就是penalty，如下图中的蓝色部分。为什么要对 ![P_{penalty}](https://www.zhihu.com/equation?tex=P_%7Bpenalty%7D)这个区域进行限制呢？一种可能的解释是我们训练的目的其实是希望可以将 ![P_G](https://www.zhihu.com/equation?tex=P_G) 的分布往 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D)靠拢，所以在它们之间的 ![P_{penalty}](https://www.zhihu.com/equation?tex=P_%7Bpenalty%7D) 这个区域会影响到最后的结果，这也是为什么WGAN优化的是 ![P_{penalty}](https://www.zhihu.com/equation?tex=P_%7Bpenalty%7D)。不过最近有新的观点说到其实应该优化 ![P_{data}](https://www.zhihu.com/equation?tex=P_%7Bdata%7D) 的区域而不是 ![P_{penalty}](https://www.zhihu.com/equation?tex=P_%7Bpenalty%7D) 。这里说的是原始版本的WGAN

![img](https://pic3.zhimg.com/80/v2-7ec29bf238fd4ef74a6ab27324472a96_hd.jpg)

而实际上在训练过程中，用到的是下面这个式子：

![V(G,D) \approx max_D\{E_{x\sim P_{data}[D(x)] - E_{x\sim P_G}[D(x)] - \lambda E_{x\sim P_{penalty}}(||\bigtriangledown_xD(x)|| - 1)^2 \}} \\](https://www.zhihu.com/equation?tex=V%28G%2CD%29+%5Capprox+max_D%5C%7BE_%7Bx%5Csim+P_%7Bdata%7D%5BD%28x%29%5D+-+E_%7Bx%5Csim+P_G%7D%5BD%28x%29%5D+-+%5Clambda+E_%7Bx%5Csim+P_%7Bpenalty%7D%7D%28%7C%7C%5Cbigtriangledown_xD%28x%29%7C%7C+-+1%29%5E2+%5C%7D%7D+%5C%5C)

使用WGAN-GP的效果会明显好过weight clipping,见下图：

![img](https://pic3.zhimg.com/80/v2-51b8d41f30555cb17924ef3ba7c2cf9e_hd.jpg)

### **Spectrum Norm**

18年的ICLR论文[Spectral Normalization for Generative Adversarial Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.05957) 中提到另一种GAN的优化方法——Spectrum Norm， 简单地说是对每层网络的参数除以该层的谱范数来满足Lipschitz=1的约束，该技术被称为Spectrum Norm(谱归一化）

### **Energy-based GAN(EBGAN)**

这个也是improved GAN的一个优化版本，这里简单提一下它的思想。主要是对discriminator进行了修改，在discriminator中是用了一个auto-encoder,具体来说，如下图所示：discriminator的output算的是auto-encoder把input的图片编码解码回来后的结果与原始输入图片的差值作为discriminator的输出。这个技术的好处是discriminator可以使用真实的图片进行预训练，而传统的GAN必须要等generator生成图片后才可以对discriminator训练，所以discriminator无法预训练。所以EBGAN的优点是discriminator一开始就可以比较强，不用等待generator迭代到很棒棒后自己才能变强。

![img](https://pic3.zhimg.com/80/v2-f1d4e5148c49efbb2326a09206208436_hd.jpg)

## **7.InfoGAN的通俗解释**

在[生成对抗网络(GAN) 背后的数学理论](https://zhuanlan.zhihu.com/p/54096381) 提到，generator和discriminator的对抗学习，它的目标其实是得到一个与real data分布一致的fake data分布。

但是由于generator的输入是一个连续的噪声信号，并且没有任何约束，导致GAN将z的具体维度与output的语义特征对应起来，可解释性很差。

它的原理很简单，在info GAN里面，把输入向量z分成两部分，c和z'。c可以理解为可解释的隐变量，而z可以理解为不可压缩的噪声。希望通过约束c与output的关系，使得c的维度对应output的语义特征，以手写数字为例，比如笔画粗细，倾斜度等。

为了引入c，坐着通过互信息的方式来对c进行约束，也可以理解成自编码的过程。具体的操作是，generator的output，经过一个分类器，看是否能够得到c。其实可以看成一个anto-encoder的反过程。其余的discriminator与常规的GAN是一样的。

![img](https://pic1.zhimg.com/80/v2-b85a31bbe8ed2b42a3ad11a707720674_hd.jpg)

在实际过程中，classifier和discriminator会共享参数，只有最后一层是不一样的，classifier输出的是一个vector, discriminator输出的是一个标量。

从损失函数的角度来看，infoGAN的损失函数变为：

![min_Gmax_DV_I(D,G) = V(D,G) - \lambda I(c;G(z,c)) \\](https://www.zhihu.com/equation?tex=min_Gmax_DV_I%28D%2CG%29+%3D+V%28D%2CG%29+-+%5Clambda+I%28c%3BG%28z%2Cc%29%29+%5C%5C)

相比起原始的GAN，多了一项 ![\lambda I(c;G(z,c)) ](https://www.zhihu.com/equation?tex=%5Clambda+I%28c%3BG%28z%2Cc%29%29+) ,这一项代表的就是c与generator的output的互信息。这一项越大，表示c与output越相关。

为什么info GAN是有效的，直观的理解就是，如果c的每一个维度对Output都有明确的影响，那么classifier就可以根据x返回原来的c。如果c对output没有明显的影响，那么classifier也没法返回原来的c。

下面是info GAN的结果。改变categorical变量可以生成不同的数字，改变continuous变量可以改变倾斜度和笔画粗细。

![img](https://pic3.zhimg.com/80/v2-3cda6e503b5fbef0811a9e7c58d0fb16_hd.jpg)



## **8.VAE-GAN和BIGAN**

如果使用auto-encoder来做生成，普遍存在的问题是会得到非常模糊的结果；如果使用原始GAN，训练就会不太稳定。基于auto-encoder的GAN希望通过GAN来强化auto-encoder，能够使得生成清晰的结果，同时也能生成出不一样的output。这中间的代表就是VAE-GAN和BiGAN

### **VAE-GAN**

VAE-GAN的原理是用GAN来强化VAE，VAE本身就是一个auto-encoder的变形。

![img](https://pic4.zhimg.com/80/v2-de8953c3e71bf5f7402014036a4d6f07_hd.jpg)

在原来VAE的基础上加一个discriminator，看看output的image越真实越好。如果只是做VAE，那么图片会很模糊。加上discriminator后迫使output越真实越好。从GAN的角度来看，在train GAN的时候，generator从来没见过真正的image长什么样，如果通过auto-encoder的架构，generator不仅仅要骗过discriminator，它见过真实的图片长什么样，所以VAE GAN学起来会更稳一点。

![img](https://pic4.zhimg.com/80/v2-265412f5a2875aa2004c1b7b9e4f1d4f_hd.jpg)

在VAE-GAN中，各个部件的优化目标如下：

encoder:minimize reconstruction error,同时希望encode的向量z越接近真实越好。

Generator(同时也是decoder): Minimize reconstruction error, 同时 cheat discriminator

Discriminator: 区分真实图片和generator生成的图片。

### **算法流程**

```python
'''
注意：在decoder（generator）中输入的两个Z，第一个encoder x的，第二个是随机分布的。
生成器产生的图片有2个，1个是第一个z产生的，一个的第二个z产生的。
'''
```

我们再来看看具体的算法流程：

1. 初始化encoder, generator(decoder), discriminator
2. 在每次迭代中：

- 从数据集中sample出M个样本 ![x^1, x^2,..., x^M](https://www.zhihu.com/equation?tex=x%5E1%2C+x%5E2%2C...%2C+x%5EM)
- 从encoder中生成M个向量 ![\tilde{z}^1,\tilde{z}^2...\tilde{z}^M](https://www.zhihu.com/equation?tex=%5Ctilde%7Bz%7D%5E1%2C%5Ctilde%7Bz%7D%5E2...%5Ctilde%7Bz%7D%5EM) ， ![\tilde{z}^i = En(x^i)](https://www.zhihu.com/equation?tex=%5Ctilde%7Bz%7D%5Ei+%3D+En%28x%5Ei%29)
- 从Generator中生成M个output ![\tilde{x}^1,\tilde{x}^2,...,\tilde{x}^M](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%5E1%2C%5Ctilde%7Bx%7D%5E2%2C...%2C%5Ctilde%7Bx%7D%5EM) , ![\tilde{x}^i = Generator(\tilde{z}^i)](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%5Ei+%3D+Generator%28%5Ctilde%7Bz%7D%5Ei%29)
- 从一个分布 ![P(z)](https://www.zhihu.com/equation?tex=P%28z%29) 中sample出M个向量 ![\hat{z}^1,\hat{z}^2,...,\hat{z}^M,](https://www.zhihu.com/equation?tex=%5Chat%7Bz%7D%5E1%2C%5Chat%7Bz%7D%5E2%2C...%2C%5Chat%7Bz%7D%5EM%2C)
- 从Generator中生成M个output ![\hat{x}^1,\hat{x}^2,...,\hat{x}^M,](https://www.zhihu.com/equation?tex=%5Chat%7Bx%7D%5E1%2C%5Chat%7Bx%7D%5E2%2C...%2C%5Chat%7Bx%7D%5EM%2C)![\hat{x}^i = Generator(\hat{z}^i)](https://www.zhihu.com/equation?tex=%5Chat%7Bx%7D%5Ei+%3D+Generator%28%5Chat%7Bz%7D%5Ei%29)
- **优化Encode**r:更新参数来减小 ![||\tilde{x}^i - x^i||](https://www.zhihu.com/equation?tex=%7C%7C%5Ctilde%7Bx%7D%5Ei+-+x%5Ei%7C%7C) 和 ![KL(P(\tilde{z}^i|x^i)||P(z))](https://www.zhihu.com/equation?tex=KL%28P%28%5Ctilde%7Bz%7D%5Ei%7Cx%5Ei%29%7C%7CP%28z%29%29) ，就是说希望可以把decoder(也就是generator）产生的输出和原始输入越接近越好，同时希望中间的 ![z](https://www.zhihu.com/equation?tex=z) 与normal distribution越接近越好。
- **优化Generator：** 更新参数来减小![||\tilde{x}^i - x^i||](https://www.zhihu.com/equation?tex=%7C%7C%5Ctilde%7Bx%7D%5Ei+-+x%5Ei%7C%7C)，同时让 ![Discriminator(\tilde{x}^i),Discriminator(\hat{x}^i)](https://www.zhihu.com/equation?tex=Discriminator%28%5Ctilde%7Bx%7D%5Ei%29%2CDiscriminator%28%5Chat%7Bx%7D%5Ei%29) 越大越好。也就是要骗过discriminator。
- **优化Discriminator**: 更新参数增加 ![Discriminator(x^i)](https://www.zhihu.com/equation?tex=Discriminator%28x%5Ei%29) ,减小 ![Discriminator(\tilde{x}^i), Discriminator(\hat{x}^i)](https://www.zhihu.com/equation?tex=Discriminator%28%5Ctilde%7Bx%7D%5Ei%29%2C+Discriminator%28%5Chat%7Bx%7D%5Ei%29)

那么VAE-GAN是修改了auto-encoder，另一种BiGAN也是修改了auto encoder, 看看它的原理：

### **BiGAN**

在BiGAN中，Encoder和Decoder分开了，对于Encoder，输入一张图片，得到一个vector， =对于decoder（也就是Generator），从一个normal distribution中随机sample一个vector， 输入Decoder得到图片。然后对于Discriminator, 我们同时投喂图片和其配对的z，让discriminator去判断这是来自于encoder还是decoder。Bi-GAN为什么有效呢，它其实就是在拟合生成原始数据集的分布P和Generator所拟合的分布Q的divergences。虽然Encoder和Decoder没有直接接在一起，但透过Discriminator可以让他们形成理想的auto-encoder

![img](https://pic1.zhimg.com/80/v2-9933eef8dff9713cb5cfa77f99eb3754_hd.jpg)

Bi-GAN的算法流程如下：

1. 初始化encoder, generator(decoder), discriminator
2. 在每次迭代中：

- 从数据集中sample出M个样本 ![x^1, x^2,..., x^M](https://www.zhihu.com/equation?tex=x%5E1%2C+x%5E2%2C...%2C+x%5EM)
- 从encoder中生成M个向量 ![\tilde{z}^1,\tilde{z}^2...\tilde{z}^M](https://www.zhihu.com/equation?tex=%5Ctilde%7Bz%7D%5E1%2C%5Ctilde%7Bz%7D%5E2...%5Ctilde%7Bz%7D%5EM) ， ![\tilde{z}^i = En(x^i)](https://www.zhihu.com/equation?tex=%5Ctilde%7Bz%7D%5Ei+%3D+En%28x%5Ei%29)
- 从一个prior ![P(z)](https://www.zhihu.com/equation?tex=P%28z%29) 中sample出M个向量 ![{z}^1,{z}^2,...,{z}^M](https://www.zhihu.com/equation?tex=%7Bz%7D%5E1%2C%7Bz%7D%5E2%2C...%2C%7Bz%7D%5EM)
- 从Generator中生成M个output ![\tilde{x}^1,\tilde{x}^2,...,\tilde{x}^M](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%5E1%2C%5Ctilde%7Bx%7D%5E2%2C...%2C%5Ctilde%7Bx%7D%5EM) , ![\tilde{x}^i =Encoder(\tilde{z}^i)](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%5Ei+%3DEncoder%28%5Ctilde%7Bz%7D%5Ei%29)
- **优化Discriminator**: 更新参数增加 ![Discriminator(x^i, \tilde{z}^i)](https://www.zhihu.com/equation?tex=Discriminator%28x%5Ei%2C+%5Ctilde%7Bz%7D%5Ei%29) ,减小 ![Discriminator(\tilde{x}^i,z^i)](https://www.zhihu.com/equation?tex=Discriminator%28%5Ctilde%7Bx%7D%5Ei%2Cz%5Ei%29)
- **优化Encoder和Decoder**: 更新Encoder和Decoder来减小 ![Discriminator(x^i, \tilde{z}^i)](https://www.zhihu.com/equation?tex=Discriminator%28x%5Ei%2C+%5Ctilde%7Bz%7D%5Ei%29) ,增大 ![Discriminator(\tilde{x}^i,z^i)](https://www.zhihu.com/equation?tex=Discriminator%28%5Ctilde%7Bx%7D%5Ei%2Cz%5Ei%29),也就是Encoder和Decoder联手来骗过Discriminator。

