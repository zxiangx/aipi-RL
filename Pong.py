import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
from baselines.common.atari_wrappers import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from learning_code import rl_utils
# 以车杆模型为例的DQN算法
    
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done)) # s,a,r',s',done

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size) # 得到一个列表，里面放了部分的(s,a,r',s',done)
        state, action, reward, next_state, done = zip(*transitions)# 利用zip，将列表中的(s,a,r',s',done)一一对应，得到5条各自的链
        return np.array(state), action, reward, np.array(next_state), done # 把链都返回

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
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
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, 
                 action_dim, #输出层维度
                 learning_rate, #学习率（用于adam优化用）
                 gamma, #折扣因子
                 epsilon, #贪婪策略参数
                 target_update, #目标网络多久更新一次
                 device):#设备
        self.action_dim = action_dim
        self.q_net = ConvolutionalQnet(self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = ConvolutionalQnet(self.action_dim).to(device)
        # 使用SGD优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        # self.sheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon: #生成一个0-1随机数，如果小于epsilon就随机取
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device) # state先变成一个张量
            state = state.permute(2,0,1)
            action = self.q_net(state).argmax().item()#得到最大的动作
        return action

    def update(self, transition_dict):# 更新函数，主要负责将损失函数的loss反向传播，然后对网络的参数进行修改
        #首先把所有东西都处理成张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        states = states.permute(0,3,1,2)
        next_states = next_states.permute(0,3,1,2)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        # self.sheduler.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络

        self.count += 1

lr = 1.1e-4
num_episodes = 1000
gamma = 0.98
epsilon = 1.0
target_update = 1000
buffer_size = 10000
minimal_size = 200
batch_size = 64
derta = 1.3e-5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lr_batchsize = [(1e-4, 64, 1e-5), (2e-4, 64, 1e-5), (5e-4, 64, 1e-5),
                (2e-4, 32, 1e-5), (2e-4, 64, 1e-6), (2e-4, 64, 3e-5)]#前三验证学习率，4验证batch，后2验证eps退火率

for k in range(6):
    lr, batch_size, derta = lr_batchsize[k]
    env_name = 'Pong-v4'
    env = gym.make(env_name)
    env = make_env(env)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(action_dim, lr, gamma, epsilon, target_update, device)#首先，生成一个智能体

    return_list = []
    full_episode = 0
    for i in range(10):#一共画10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:#画进度条
            for i_episode in range(int(num_episodes / 10)):#每个进度条都要处理多少个episode
                full_episode += 1
                episode_return = 0
                state = env.reset()#每一个序列开始都将state初始化
                done = False
                while not done:#序列没有走完
                    agent.epsilon -= derta
                    action = agent.take_action(state)#利用贪婪策略选一个动作
                    next_state, reward, done, _ = env.step(action)# 环境step函数得到信息
                    replay_buffer.add(state, action, reward, next_state, done)# 加入回放池
                    state = next_state#更新状态
                    episode_return += reward#当前序列的奖励数值（最终想要得到的结果是最终的序列奖励数最大）
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)#取出多条链
                        transition_dict = {#把链全部放到字典里面，便于参数传递
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)#智能体利用回放池中的数据进行更新
                return_list.append(episode_return)# 序列走完了，把奖励加入数组
                # if (i_episode + 1) % 10 == 0:
                # if full_episode % 100 == 0:
                #     torch.save(agent.q_net, "dqn_pong_model{}.pth".format(full_episode))
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))# 画奖励变化图像
    fig, ax = plt.subplots()
    ax.plot(episodes_list, return_list)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    ax.set_title('DQN on {}'.format(env_name))
    # plt.show()
    fig.savefig('figa{}.png'.format(k))

    mv_return = rl_utils.moving_average(return_list, 9)
    fig, ax = plt.subplots()
    ax.plot(episodes_list, mv_return)# 平滑处理
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    ax.set_title('DQN on {}'.format(env_name))
    # plt.show()
    fig.savefig('figb{}.png'.format(k))