import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from learning_code import rl_utils
import time

class Sarsa:
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.032,
                 gamma=0.9,
                 e_greed=0.0):
        self.n_action = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q_table = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    # 采用reward—shaping方法，计算状态的潜力
    def potential(self, state):
        x, y, pass_loc, dest_loc = env.unwrapped.decode(state)
        dx, dy = env.unwrapped.locs[dest_loc]
        if pass_loc == 4:# 如果乘客在车上
            return 4/(abs(x-dx)+abs(y-dy)+1)+5
        else:
            px, py = env.unwrapped.locs[pass_loc]
            return 4/(abs(x-px)+abs(y-py)+1)
        
    # 更新Q-table
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.potential(s1) - self.potential(
            s0) + self.gamma * self.Q_table[s1,a1] - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.lr * td_error  # 修正q

env = gym.make('Taxi-v3')        
np.random.seed(0)

agent = Sarsa(env.observation_space.n, env.action_space.n)
num_episodes = 8000  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

mv_return = rl_utils.moving_average(return_list, 9)
episodes_list = list(range(len(return_list)))
fig, ax = plt.subplots()
ax.plot(episodes_list, mv_return)
ax.set_xlabel('Episodes')
ax.set_ylabel('Returns')
ax.set_title('Sarsa on {}'.format('Taxi-v3'))
plt.show()
# fig.savefig('before_rewardshaping.png')

# env = gym.make('Taxi-v3', render_mode='human')

# def test_episode(env, agent):
#     state = env.reset()
#     while True:
#         agent.potential(state)
#         action = agent.take_action(state)  # greedy
#         next_state, __, done, _ = env.step(action)
#         state = next_state
#         time.sleep(0.5)
#         if done:
#             break

# for i in range(30):
#     test_episode(env,agent)