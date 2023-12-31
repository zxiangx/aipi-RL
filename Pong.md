# atari游戏：Pong

[toc]

Taxi项目的实现可以采用sarsa方法，很轻松就可以收敛，让智能体在大部分情况下都可以作出最优的策略。由于Taxi的动作和状态都是离散的，所以学习起来非常快，效果也非常明显，因此难度较低。

而从Taxi项目到atari游戏，则是一个巨大的跨越:dizzy_face:。我在完成atari游戏Pong的收敛过程中遇到了很多的困难，在下面分享一下学习的过程:joy::

## 选择算法：DQN

在对atari游戏毫无了解，对以图像为输入的强化学习毫无经验的情况下，由于《动手学强化学习》中DQN算法中提了一句学习atari游戏（利用卷积神经网络），所以我就选择了DQN算法来搭建智能体。

回顾一下DQN算法的主要过程：首次利用了**经验回放池**，并且还使用目标网络来计算**TD目标**，主要通过计算loss来对深度Q网络进行反向传播梯度下降：
$$
L=\frac{1}{2N}\sum^{N}_{i=1}\Big(Q_\omega(s_0,a_0)-\big(r(s_0,a_0)+\gamma\max_aQ_{\omega^-}(s_1,a)\big)\Big)^2
$$

## 遇到的很多问题

### 网络的搭建方式

#### 选择卷积网络

虽然学习资料上直接给出了一种卷积神经网络搭建方式，但是我还是先采取了别的网络搭建方式。我在网上主要看到了两种搭建方式，**一种是和前面一模一样的网络，还有一种是利用全连接网络**。卷积神经网络，对于完全没有神经网络基础的我是一个大难题:dizzy_face:。所以我花费了大量的时间对卷积网络（Conv2d函数）进行了解，才大致弄清楚了其中各个参数的原理以及卷积神经网络的主要工作原理，即对一个多维张量进行一定程度上的修剪变成另一个易于处理的张量，并且Conv2d函数还有很多奇奇怪怪的参数。

**首先我采用了卷积网络的形式**（非资料直接给出的，而是自己尝试）。想要很好的利用卷积网络，首先要对图片进行预处理。我在网上找到的一种预处理方式，是手工将图片加工成一定规格的像素张量，然后处理一下通道数，我看到的版本是将图片处理成80×80规格的，然后我就按照这个规格去设计网络了。我调整了各种参数，调整了卷积核的大小，两个方向的步长，图片处理的通道数，还进行了很多卷积层的增添和删减，目的就是想要卷积网络跑快一点。结果就是，修改了半天对速度的影响几乎没有，还是一样慢。

**然后我又尝试了采用之前提到的全连接网络**，简单设计了一个，还是非常非常慢，原因是一个图片实在太大了，全连接网络的结点太多了，所以并没有带来什么优化（说不定完全不如卷积网络，因为时间问题我没有花太多时间尝试全连接网络）。

#### 求助baselines

所以还是回到了卷积网络，再想想怎么设计。我看到了资料中提了一句：**同时把几帧图片传入网络**，这样可以让智能体感知到环境的变化，从而更好的学习。那么应该怎么把几帧同时传入网络呢？我先采用的办法是：把4帧图片打包成一个整体，然后每次都把这个整体传入网络。但考虑到卷积神经网络非常“挑剔”，即必须规定传入的网络的形式是NCHW，即卷积层传入的张量的维数必须是4维或者3维，所以这个“打包”就较难处理。所以我每次传入一个“打包”好的图片时，先通过一个全连接网络，将4帧图片线性组成一个单帧图片，然后进行后面的卷积层操作。

于是我就直接设计了一个in=4，out=1的连接层Linear，结果是直接报错，因为网络把传入的张量中的每一个数都看作了一个结点，而不会把一帧看作一个结点。最后，我硬是设置了一个4-1网络，然后每次从中获得具体的参数，然后直接对打包中的各个图片进行加权得到了一个新的图片。然后程序就可以跑了，一调试，发现：这个连接层的参数从来没变过。

也就是说搞了这一层和没搞差不多，参数完全靠初始化，可能是因为网络中一些奇怪的地方没有required_grad=True，导致没法对这个参数进行传播:dizzy_face:。搞了半天线性组合没啥效果，我又尝试了一下把图片横着拼起来，然后又调了一波参数，发现效果还是很差。

最后，折腾了几天还是回到网上求助，然后发现了有一个baselines文件，里面放了很多的处理环境的函数，可以直接用来构造一个可以同时处理多帧、有效预处理图片，还将多帧图片作为状态的环境！所以我直接采用了baselines现成的函数来构造环境，效果非常不错，而且还和前面学习资料上给出的卷积神经网络相对应，所以我又采用了资料上的网络，即：

```python
class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 7 * 7 * 64)
        x = F.relu(self.fc4(x))
        return self.head(x)
```

### 图片格式

在我使用baselines的函数来构建环境的时候，我发现了一个大问题：输出的图片总是把通道数放在最后，图片的长宽放在中间。但是Conv2d函数非常挑剔，必须要求输入的图片是把通道数放在第2，长宽放在最后，所以必须找点办法把图片的轴进行变换。

在最开始，对于环境step得到的state，我都是进行一些复杂的列表操作，去列表，套列表，搞了一通才把那个一维的通道数放到了第一位（不考虑batch_size）。但是现在通道数不太一样了，所以就需要用一些更好的方法来解决这个问题，最后我的解决方法是：利用permute函数（一种处理张量规格的函数），将张量的轴进行变换将原本的NHWC格式变成了NCHW格式传入卷积层操作。

### $\epsilon$退火

由于刚开始智能体学习的时候需要不断尝试各种动作，所以贪婪策略需要在一开始具备较大的随机性，之后再逐渐减退转而采取习得的最佳动作。所以需要在过程中不断让$\epsilon$降低。

但是就这样一个简单的一句代码，加进去之后效果如下：

![rubbish_pong](RL_learn_png\rubbish_pong.png)

也就是说，跑完了之后效果和随机打的效果差不多，速度还慢的很（用我自己电脑跑的）。

最后查了半天，发现我犯了一个非常愚蠢的错误：每次迭代时降低的是用来初始化的$\epsilon$，而这个$\epsilon$在用来初始化之后就没用了，也就是说智能体里面的$\epsilon$一直没有改变，都是1！！！！所以上面的结果就是随机打的结果（悲）。

然后换成了改变agent里面的$\epsilon$，效果一下子就出来了。

### 调整超参数

超参数一大堆，要怎么选择合适的参数？而且最关键的是，设一套超参数，验证的成本巨大！跑一套下来要花好几个小时，而且是租一块最好的4090（贵:broken_heart:）。所以我尽量跑最少的次数，来获得最优的超参数（主要是学习率，batch_size和$\epsilon$退火速率）。

我一共设置了6套数据，从晚上开始跑到早上，一共跑了5套半，成本10块左右：

![7hours](Pong_pic\7hours.png)

我设计的数据如下，尽量兼顾了学习率、batch_size和退火速率的简单探究：

```python
lr_batchsize = [(1e-4, 64, 1e-5), (2e-4, 64, 1e-5), (5e-4, 64, 1e-5),
                (2e-4, 32, 1e-5), (2e-4, 64, 1e-6), (2e-4, 64, 3e-5)]#前三验证学习率，4验证batch，后2验证eps退火率
```

其中分别为学习率、batch_size和退火率。跑出来的图片我放到了文件里面，得到的结论是：

学习率2e-4偏高，1e-4较好，5e-4等于没学习。batch_size，64就可以了，退火速率1e-6等于没退火，1e-5又稍微慢了点，所以我早上又试着跑了一个退火率为1.5e-5的超参数。上面跑的6组中，表现最好的是第一组（因为最稳定，虽然其他的跑出来学习速度快一点但是后面不太稳定）。表现结果如下：（注：figa表示没有平滑处理，figb有平滑处理，下面展示的是第一组的平滑处理）：

![figb0](Pong_pic\figb0.png)

虽然稳定，但是跑了1000次也才学习到了-10左右，速度太慢了，于是我调整了退火率后又跑了一遍，结果如下（3hours）：

![3hours](Pong_pic\3hours.png)

![figb_x](Pong_pic\figb_x.png)

可以看出把退火率调高了一点，学习速度快了很多，跑差不多400就达到了前面1000才达到的-10。但前面学习的不错，后面的效果就比较拉了。。。进步不大，而且不太稳定，估计是退火率又偏大了，然后学习率也偏小了一点点，但是我不想再去试了，跑这一个就要花上3hours和7￥的成本:dizzy_face:

## 总结

这个atari的Pong游戏智能体搭建算是勉强完成了，但是还有很多的不足，例如程序跑的速度很慢，收敛效果也不是很好，可能是因为超参数的原因，也有可能是因为我的算法复杂度高，一些地方浪费了较多时间（就像之前跑决策树），还有很多可以优化的地方。而且我仅仅采用了DQN算法来实现智能体的搭建，其他的算法还没有尝试，并且DQN算法的优化算法也没有使用。但时间有限，而且验证一套超参数的成本巨大，就不再尝试用别的方法或者超参数再进行尝试:joy:。

在第二次测试中，我学习到了很多有关于RL的知识，具体为各种RL算法，以及自己上手初步构建了一些智能体，并尝试着让智能体收敛，这样的实践让我对构建智能体需要的各种工具（代码组成部分）、算法的原理有了更加深入的了解。并且在学习用到网络的算法时，还回头把第一次测试没学的**利用pytorch构建神经网络**的部分学了一遍:joy:，也算是没有落下第一次测试的部分。

RL邻域对我有着独特的吸引力，因为我一直很好奇为什么机器人可以轻易做到一些人类做不到的事，在某些邻域甚至可以战胜人类，并且某些时候还比人类更有创造力。通过RL的学习，我逐渐对训练有素的机器人背后的原理有了初步认识，即机器人是通过不断的尝试，从尝试的结果不断学习更新自己的策略，从而不断进步。并且还尝试自己构建了智能体，让其对某方面进行学习并且取得进步。总之，ai$\pi$的这两次测试，让我学到了很多有关于ai的知识，也为我将来的ai学习打下了基础。（还提高了搜索能力
