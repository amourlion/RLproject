import gymnasium as gym  # 导入Gymnasium环境库（强化学习环境的标准库）
import numpy as np       # 导入NumPy数值计算库
import torch            # 导入PyTorch深度学习框架
import torch.nn as nn   # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入函数式API
import matplotlib.pyplot as plt  # 导入matplotlib绘图库用于可视化
import matplotlib.animation as animation  # 导入动画模块
import matplotlib
import warnings
from collections import namedtuple, deque  # 导入数据结构
import random
import time
import os

# 设置中文显示和忽略警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        经验回放缓冲区，用于存储智能体的经验
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加一个经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


# OU噪声类，用于连续动作空间的探索
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        """
        Ornstein-Uhlenbeck噪声过程
        Args:
            action_dim: 动作维度
            mu: 长期均值
            theta: 回归系数
            sigma: 噪声强度
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """生成噪声样本"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# 高斯噪声类
class GaussianNoise:
    def __init__(self, action_dim, mean=0.0, std=0.2, decay=0.995):
        """
        高斯噪声，用于探索
        Args:
            action_dim: 动作维度
            mean: 噪声均值
            std: 噪声标准差
            decay: 噪声衰减率
        """
        self.action_dim = action_dim
        self.mean = mean
        self.std = std
        self.decay = decay
        self.current_std = std
    
    def sample(self):
        """生成高斯噪声"""
        noise = np.random.normal(self.mean, self.current_std, self.action_dim)
        return noise
    
    def decay_noise(self):
        """衰减噪声强度"""
        self.current_std = max(self.current_std * self.decay, 0.01)
    
    def reset(self):
        """重置噪声强度"""
        self.current_std = self.std


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Actor网络，用于策略函数近似
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 权重初始化
        # 原始DDPG论文中明确建议对最后一层使用小权重初始化，3e-3（0.003）是经过实验验证的有效值。
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出在[-1, 1]范围内
        return x


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Critic网络，用于价值函数近似
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # 状态处理层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 动作和状态融合层
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        # 这里输出是Q值，一维
        # DQN中最后一层返回所有动作的Q值: [Q(s,a0), Q(s,a1), ..., Q(s,an)]
        # 但是这里输出的是单个动作的Q值
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 权重初始化
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)  # 拼接状态特征和动作
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DDPG智能体
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, 
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 noise_type='gaussian', noise_std=0.2):
        """
        DDPG智能体
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            action_bound: 动作边界
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            tau: 软更新参数
            noise_type: 噪声类型('gaussian'或'ou')
            noise_std: 噪声强度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 硬复制权重到目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
        # 噪声
        if noise_type == 'gaussian':
            self.noise = GaussianNoise(action_dim, std=noise_std)
        else:
            self.noise = OUNoise(action_dim, sigma=noise_std)
    
    def hard_update(self, target, source):
        """硬更新目标网络"""
        # zip() 将多个可迭代对象的元素一一配对
        for target_param, param in zip(target.parameters(), source.parameters()):
            # 这里采用copy_方法深拷贝权重
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def select_action(self, state, add_noise=True):
        """选择动作"""
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            action += self.noise.sample()
        
        # 限制动作范围
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self, batch_size=256):
        """学习更新网络"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # 采样经验
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # 训练Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 训练Actor
        # Actor输出为单个动作的Q值，但是这里一个batch_size=256
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # 清零梯度
        self.actor_optimizer.zero_grad()
        # 反向传播
        actor_loss.backward()
        # 更新网络参数
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        return critic_loss.item(), actor_loss.item()


def train_ddpg(episodes=500, max_steps=200):
    """
    训练DDPG智能体
    Args:
        episodes: 训练轮数
        max_steps: 每轮最大步数
    """
    # 创建环境
    env = gym.make('Pendulum-v1', render_mode=None)
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # Pendulum-v1的动作范围是[-2, 2]
    
    print(f"环境信息:")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作边界: [-{action_bound}, {action_bound}]")
    print(f"奖励范围: 最低-16.27，最高0")
    
    # 创建智能体
    agent = DDPGAgent(state_dim, action_dim, action_bound, 
                     noise_type='gaussian', noise_std=0.3)
    
    # 训练记录
    episode_rewards = []
    episode_steps = []
    critic_losses = []
    actor_losses = []
    
    print(f"\n开始训练DDPG智能体...")
    print(f"使用高斯噪声进行探索，初始标准差: 0.3")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # 选择动作（带噪声探索）
            action = agent.select_action(state, add_noise=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习
            critic_loss, actor_loss = agent.learn()
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 记录训练信息
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        
        if critic_loss is not None:
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        
        # 噪声衰减
        agent.noise.decay_noise()
        
        # 打印训练进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Current Noise Std: {agent.noise.current_std:.3f}")
    
    env.close()
    
    # 绘制训练结果
    plot_training_results(episode_rewards, critic_losses, actor_losses)
    
    return agent, episode_rewards


def plot_training_results(episode_rewards, critic_losses, actor_losses):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('每回合奖励')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True)
    
    # 平滑奖励曲线
    window = 50
    if len(episode_rewards) >= window:
        smoothed_rewards = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                          for i in range(len(episode_rewards))]
        axes[0, 1].plot(smoothed_rewards)
        axes[0, 1].set_title(f'{window}回合平滑奖励')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('平均奖励')
        axes[0, 1].grid(True)
    
    # Critic损失
    if critic_losses:
        axes[1, 0].plot(critic_losses)
        axes[1, 0].set_title('Critic损失')
        axes[1, 0].set_xlabel('更新步数')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].grid(True)
    
    # Actor损失
    if actor_losses:
        axes[1, 1].plot(actor_losses)
        axes[1, 1].set_title('Actor损失')
        axes[1, 1].set_xlabel('更新步数')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ddpg_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_agent(agent, episodes=5, render=True, save_gif=False):
    """测试训练好的智能体"""
    # 创建环境（用于渲染）
    env = gym.make('Pendulum-v1', render_mode='rgb_array' if save_gif else 'human')
    
    test_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        frames = []
        
        for step in range(200):
            # 选择动作（不加噪声）
            action = agent.select_action(state, add_noise=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if save_gif and episode < 3:  # 只保存前3个回合的GIF
                frame = env.render()
                frames.append(frame)
            elif render and not save_gif:
                env.render()
                time.sleep(0.02)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"测试回合 {episode + 1}: 奖励 = {episode_reward:.2f}, 步数 = {step_count}")
        
        # 保存GIF
        if save_gif and episode < 3:
            save_animation_gif(frames, f'ddpg_episode_{episode + 1}.gif')
    
    env.close()
    
    print(f"\n测试结果:")
    print(f"平均奖励: {np.mean(test_rewards):.2f}")
    print(f"最高奖励: {np.max(test_rewards):.2f}")
    print(f"最低奖励: {np.min(test_rewards):.2f}")
    
    return test_rewards


def save_animation_gif(frames, filename, fps=30):
    """保存动画为GIF"""
    if not frames:
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('DDPG Pendulum 控制')
    
    im = ax.imshow(frames[0])
    
    def animate(frame):
        im.set_array(frames[frame])
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                interval=1000//fps, blit=True, repeat=True)
    
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"动画已保存为: {filename}")


def main():
    """主函数"""
    print("DDPG算法实现 - Pendulum-v1环境")
    print("="*50)
    
    # 训练智能体
    agent, rewards = train_ddpg(episodes=500, max_steps=200)
    
    print("\n训练完成！")
    print("="*50)
    
    # 测试智能体
    print("开始测试智能体...")
    test_rewards = test_agent(agent, episodes=5, render=False, save_gif=True)
    
    print("\n程序执行完毕！")
    print(f"训练结果图表已保存为: ddpg_training_results.png")
    print(f"测试动画已保存为: ddpg_episode_*.gif")


if __name__ == "__main__":
    main()