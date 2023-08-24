# Taxi环境作业

具体代码见文件夹Taxi.py。
[toc]
## 测试结果

我选择的算法是在线策略算法Sarsa，即next_action选取的策略和当前的action当时被选择的策略相同，均为$\epsilon$-贪婪策略（不过发现当$\epsilon=0$时收敛效果更好）。而策略相同也可以直接把next_action赋值给当前的action进行下去。

代码总体分为3个部分：

1. class Sarsa的定义。Sarsa类中包含了3个函数：初始化函数__init__，采取动作的函数take_action，更新Q网络的函数update。

2. 主循环，走序列。即while not done，注意将每一条走完的序列最终得到的奖励存入一个列表便于绘图。

3. 把奖励列表画出来。

具体运行收敛结果如下：

![before_rewardshaping](RL_learn_png\before_rewardshaping.png)

## 算法中的细节（探索和利用）

$\epsilon$-贪婪策略就是一种探索与利用的结合。**贪婪**，指直接根据当前得到的信息得出最优的动作，并采取之，这**就是利用**。而$\epsilon$策略中以**小概率**采取贪婪策略之外的各种策略，就是一种**探索**，也即通过采取一些在当前看来并不是最优的方案，获得有关于这种方案的情况，从而根据结果来对这种方案进行评估，全面掌握各种情况可能带来的结果。

换句话说，**利用**就是对当前已经获得的信息进行合理推断，利用信息作出当前最优的选择，是智能体对自身成果的复现；而**探索**则与利用相反，不考虑当前的信息，而直接在各种方案中随机挑选一个，是智能体得到当前成果之外信息的渠道。利用的作用是可以保证智能体在每个时刻都采取当前最优的方案，朝目标最快的收敛；而探索的作用就是弥补利用盲目性的不足，对“答案”以外的情况加以分析，从而可能突破盲目利用的局限而找到全局更优解。

## Reward shaping技巧

reward shaping技巧实际上就是在智能体的学习过程中加入人为的指导，从而使得智能体更快得到最优解。在Taxi环境中，人为的指导就是：要不断朝着靠近目标的方向前进。加入了这个指导，智能体的学习速度得到加快。

但实际上，这样的方法不一定对，甚至可能偏离了最佳路线，所以对于有局限的人为指导需要控制其用量。例如对于Taxi环境，如果朝着目标不断前进了，万一遇到一堵墙怎么办？一味得朝目标靠近，反而得不偿失，所以得使用得当。

我的reward shaping方法是：在利用时序差分更新Q值的时候，在更新值中加入一个**状态潜力**，即离目标越近的状态潜力越大。在更新值中加入两个状态的潜力差值，得到新的更新方式：
$$
Q(s_0,a_0)+=\alpha\big(r+\gamma Q(s_1,a_1)-Q(s_0,a_0)+\gamma \phi(s_1)-\phi(s_0)\big)
$$
这里潜力函数$\phi$设为当前状态中车到目标横向和纵向距离之和的倒数（加入1防止分母为0），并且如果此时状态下乘客在车上，那么额外加上一个大额奖励，以鼓励车尽快接到乘客。如果此时乘客在车上，那么目标为乘客；否则目标为目的地。具体潜力函数的代码如下：

```python
def potential(self, state):
    x, y, pass_loc, dest_loc = env.unwrapped.decode(state)
    dx, dy = env.unwrapped.locs[dest_loc]
    if pass_loc == 4: # 如果乘客在车上
        return 15/(abs(x-dx)+abs(y-dy)+1)+20
    else:
        px, py = env.unwrapped.locs[pass_loc]
        return 15/(abs(x-px)+abs(y-py)+1)
```

运行的效果如下：

![after_rewardshaping](RL_learn_png\after_rewardshaping.png)

可以看出结果收敛得更快，但问题是收敛过后效果不稳定，原因正是加入的人为干扰因素。所以reward shaping技巧可以通过人的经验来告诉智能体一个大致的方向，但是最终学习的效果却并没有其本身学习的效果好，这也是reward shaping技巧的问题所在。