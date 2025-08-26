"""
双深度Q网络(Double Deep Q-Network, DDQN)实现CartPole-v0环境
作者: 强化学习初学者
日期: 2025年8月26日

DDQN是DQN的改进版本，解决了DQN中存在的过高估计问题：
1. DQN使用同一个网络来选择动作和评估动作，容易导致Q值过高估计
2. DDQN使用两个网络：主网络(policy_net)选择动作，目标网络(target_net)评估动作
3. 这种分离使得Q值估计更加准确，训练更稳定
"""

import gymnasium as gym  # 导入Gymnasium环境库（强化学习环境的标准库）
import numpy as np       # 导入NumPy数值计算库
import torch            # 导入PyTorch深度学习框架
import torch.nn as nn   # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入函数式API
import matplotlib.pyplot as plt  # 导入matplotlib绘图库用于可视化
import matplotlib
import warnings
from collections import namedtuple, deque  # 导入数据结构
import random

# 设置中文显示和忽略警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ============================
# 1. 配置参数类
# ============================
class Config:
    """
    配置类：包含所有超参数设置
    这些参数对训练效果有重要影响，初学者可以通过调整这些参数来观察效果
    """
    def __init__(self):
        # 环境参数
        self.env_name = 'CartPole-v0'  # 环境名称：倒立摆问题
        
        # 训练参数
        self.max_episodes = 500       # 最大训练回合数
        self.max_steps = 500         # 每回合最大步数
        
        # 网络参数
        self.hidden_dim = 128        # 神经网络隐藏层维度
        self.lr = 0.001             # 学习率：控制网络权重更新速度
        
        # DQN特有参数
        self.batch_size = 32        # 批次大小：每次从经验回放中采样的经验数量
        self.gamma = 0.99           # 折扣因子：未来奖励的重要性权重(0-1)
        self.epsilon_start = 1.0    # ε-贪婪策略起始值：初期完全随机探索
        self.epsilon_end = 0.01     # ε-贪婪策略最终值：后期主要利用已学知识
        self.epsilon_decay = 0.995  # ε衰减率：控制从探索到利用的转换速度
        
        # DDQN特有参数
        self.target_update = 10     # 目标网络更新频率：每10个回合更新一次目标网络
        self.memory_size = 10000    # 经验回放缓冲区大小
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. 经验回放缓冲区
# ============================
# 定义经验元组：存储一次交互的所有信息
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    经验回放缓冲区：存储智能体与环境的交互经验
    
    为什么需要经验回放？
    1. 打破数据相关性：连续的经验往往高度相关，直接用于训练会导致网络过拟合
    2. 提高样本利用效率：一个经验可以被多次使用，提高数据利用率
    3. 稳定训练：随机采样经验使得训练更加稳定
    """
    def __init__(self, capacity):
        """
        初始化经验缓冲区
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)  # 使用双端队列，自动保持固定大小
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一次经验
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 将numpy数组转换为tensor并添加到缓冲区
        state = torch.FloatTensor(state).unsqueeze(0) if isinstance(state, np.ndarray) else state
        next_state = torch.FloatTensor(next_state).unsqueeze(0) if isinstance(next_state, np.ndarray) else next_state
        
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        Args:
            batch_size: 采样数量
        Returns:
            批次经验数据
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)

# ============================
# 3. 深度Q网络结构
# ============================
class DQN(nn.Module):
    """
    深度Q网络：用神经网络来近似Q函数
    
    Q函数Q(s,a)表示在状态s下执行动作a的期望累积奖励
    神经网络输入状态s，输出每个动作的Q值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化网络结构
        Args:
            state_dim: 状态维度（CartPole-v0中为4维：位置、速度、角度、角速度）
            action_dim: 动作维度（CartPole-v0中为2维：左推、右推） 这里输出的Q(s,a)
            hidden_dim: 隐藏层维度
        """
        super(DQN, self).__init__()
        
        # 定义全连接神经网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)    # 输入层到隐藏层1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(hidden_dim, action_dim)   # 隐藏层2到输出层
    
    def forward(self, x):
        """
        前向传播：计算Q值
        Args:
            x: 输入状态
        Returns:
            每个动作对应的Q值
        """
        x = F.relu(self.fc1(x))  # 第一层 + ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层 + ReLU激活函数
        x = self.fc3(x)          # 输出层（不使用激活函数，因为Q值可以是任意实数）
        return x

# ============================
# 4. DDQN智能体
# ============================
class DDQNAgent:
    """
    双深度Q网络智能体：实现DDQN算法的核心类
    
    DDQN相比DQN的改进：
    1. 使用主网络选择动作：a* = argmax Q_main(s', a)
    2. 使用目标网络评估动作价值：Q_target(s', a*)
    3. 这样避免了同一网络既选择又评估导致的过高估计问题
    """
    def __init__(self, state_dim, action_dim, cfg):
        """
        初始化DDQN智能体
        Args:
            state_dim: 状态维度
            action_dim: 动作维度  
            cfg: 配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        
        # 创建两个相同结构的神经网络
        self.policy_net = DQN(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)  # 主网络：用于选择动作
        self.target_net = DQN(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)  # 目标网络：用于计算目标Q值
        
        # 初始化时目标网络权重与主网络相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设置为评估模式，不参与梯度更新
        
        # 优化器：用于更新主网络权重
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(cfg.memory_size)
        
        # ε-贪婪策略参数
        self.epsilon = cfg.epsilon_start
        
        # 记录训练步数
        self.steps_done = 0
    
    def select_action(self, state):
        """
        使用ε-贪婪策略选择动作
        
        ε-贪婪策略：
        - 以ε的概率随机选择动作（探索，exploration）
        - 以1-ε的概率选择当前认为最优的动作（利用，exploitation）
        - ε会随着训练逐渐减小，从探索转向利用
        
        Args:
            state: 当前状态
        Returns:
            选择的动作
        """
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():  # 不需要计算梯度，节省内存和计算
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.cfg.device)
                q_values = self.policy_net(state_tensor)  # 计算所有动作的Q值
                return q_values.argmax().item()  # 返回Q值最大的动作索引
    
    def update(self):
        """
        更新网络权重：DDQN算法的核心
        
        DDQN更新过程：
        1. 从经验缓冲区采样一批经验
        2. 使用主网络在下一状态选择动作
        3. 使用目标网络评估该动作的价值
        4. 计算目标Q值和当前Q值的差异（TD误差）
        5. 使用梯度下降最小化TD误差
        """
        # 如果经验不足，不进行更新
        if len(self.memory) < self.cfg.batch_size:
            return
        
        # 从经验缓冲区采样
        transitions = self.memory.sample(self.cfg.batch_size)
        batch = Transition(*zip(*transitions))
        
        # 将数据转换为tensor
        state_batch = torch.cat(batch.state).to(self.cfg.device)
        action_batch = torch.LongTensor(batch.action).to(self.cfg.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.cfg.device)
        next_state_batch = torch.cat(batch.next_state).to(self.cfg.device)
        done_batch = torch.BoolTensor(batch.done).to(self.cfg.device)
        
        # 贝尔曼方程的数学表达：
        # Q(s,a) = r + γ * max_a' Q(s',a')
        # 翻译成自然语言：
        # "在状态s执行动作a的价值" = "立即奖励r" + "折扣后的下一状态最大价值"
        # 计算当前状态的Q值：Q(s_t, a_t) 需要更新的目标
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # DDQN的核心：分别使用主网络和目标网络
        with torch.no_grad():
            # 使用主网络选择下一状态的最优动作
            next_actions = self.policy_net(next_state_batch).argmax(1)
            # 使用目标网络评估选定动作的Q值, 延迟更新的旧的Q值
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # 贝尔曼方程计算目标Q值：r + γ * max_a Q_target(s', a) （如果未结束）
            # 数学上可以证明一定条件下这个过程会收敛到Q*
            # 关键理解：为什么要乘以 ~done_batch
            # 让我们分情况详细解释：
            # 情况1：如果回合未结束 (done=False, ~done=True)
            # Q(s,a) = r + γ * max_a' Q(s',a')  # 标准的贝尔曼方程
            # 情况2：如果回合已结束 (done=True, ~done=False)  
            # Q(s,a) = r + γ * 0 = r  # 因为终止状态没有下一个状态，未来奖励为0
            target_q_values = reward_batch + (self.cfg.gamma * next_q_values * ~done_batch)
            
        # 计算损失：均方误差
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播更新网络权重
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()             # 计算梯度
        
        # 梯度裁剪：防止梯度爆炸，提高训练稳定性
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        
        self.optimizer.step()       # 更新权重
        
        # 更新ε值：逐渐减少随机探索
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

# ============================
# 5. 训练函数
# ============================
def train_ddqn():
    """
    训练DDQN智能体的主函数
    """
    # 初始化配置
    cfg = Config()
    
    # 创建环境
    env = gym.make(cfg.env_name)
    env.action_space.seed(1)    # 设置动作空间随机种子
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]  # 状态维度：4（位置、速度、角度、角速度）
    action_dim = env.action_space.n             # 动作维度：2（左推、右推）
    
    print(f"环境信息：")
    print(f"- 状态维度: {state_dim}")
    print(f"- 动作维度: {action_dim}")
    print(f"- 设备: {cfg.device}")
    print("-" * 50)
    
    # 创建DDQN智能体
    agent = DDQNAgent(state_dim, action_dim, cfg)
    
    # 记录训练过程的指标
    rewards = []                    # 每回合的总奖励
    moving_average_rewards = []     # 滑动平均奖励（平滑曲线）
    ep_steps = []                   # 每回合的步数
    
    # 开始训练
    for i_episode in range(1, cfg.max_episodes + 1):
        # 重置环境，获取初始状态
        state, _ = env.reset()
        ep_reward = 0  # 当前回合累积奖励
        
        # 在当前回合中与环境交互
        for i_step in range(1, cfg.max_steps + 1):
            # 智能体选择动作
            action = agent.select_action(state)
            
            # 执行动作，获取环境反馈
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 回合是否结束
            
            # 累积奖励
            ep_reward += reward
            
            # 将经验存入回放缓冲区
            agent.memory.push(state, action, reward, next_state, done)
            
            # 状态转移
            state = next_state
            
            # 更新网络权重
            agent.update()
            
            # 如果回合结束，跳出循环
            if done:
                break
        
        # 定期更新目标网络：将主网络的权重复制到目标网络
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"回合 {i_episode}: 目标网络已更新")
        
        # 记录训练指标
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        
        # 计算滑动平均奖励（用于平滑显示训练曲线）
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            # 指数移动平均：新值权重0.1，历史值权重0.9
            moving_average_rewards.append(
                0.9 * moving_average_rewards[-1] + 0.1 * ep_reward
            )
        
        # 打印训练进度
        if i_episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])  # 最近10回合平均奖励
            print(f"回合 {i_episode:3d} | "
                  f"奖励: {int(ep_reward):3d} | "
                  f"步数: {i_step:3d} | "
                  f"平均奖励: {avg_reward:.1f} | "
                  f"探索率: {agent.epsilon:.3f}")
        
        # 判断是否已经学会（连续100回合平均奖励>=195）
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 195:
            print(f"\n🎉 在回合 {i_episode} 解决了环境！")
            print(f"最近100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
            break
    
    # 关闭环境
    env.close()
    
    # 返回训练结果
    return rewards, moving_average_rewards, ep_steps

# ============================
# 6. 结果可视化
# ============================
def plot_results(rewards, moving_average_rewards, ep_steps):
    """
    绘制训练结果图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DDQN训练结果分析', fontsize=16)
    
    # 奖励曲线
    axes[0, 0].plot(rewards, alpha=0.6, color='blue', label='每回合奖励')
    axes[0, 0].plot(moving_average_rewards, color='red', linewidth=2, label='滑动平均奖励')
    axes[0, 0].set_xlabel('训练回合')
    axes[0, 0].set_ylabel('累积奖励')
    axes[0, 0].set_title('奖励学习曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 步数曲线
    axes[0, 1].plot(ep_steps, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('训练回合')
    axes[0, 1].set_ylabel('每回合步数')
    axes[0, 1].set_title('每回合生存步数')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励分布直方图
    axes[1, 0].hist(rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('累积奖励')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('奖励分布直方图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 训练进度分析
    if len(rewards) >= 100:
        # 计算滑动窗口平均奖励
        window_size = 100
        avg_rewards = []
        for i in range(window_size - 1, len(rewards)):
            avg_rewards.append(np.mean(rewards[i - window_size + 1:i + 1]))
        
        axes[1, 1].plot(range(window_size, len(rewards) + 1), avg_rewards, 
                       color='orange', linewidth=2)
        axes[1, 1].axhline(y=195, color='red', linestyle='--', 
                          label='解决阈值(195)')
        axes[1, 1].set_xlabel('训练回合')
        axes[1, 1].set_ylabel('100回合平均奖励')
        axes[1, 1].set_title('学习进度（100回合滑动平均）')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存图片
    plt.savefig('ddqn_training_results.png', dpi=300, bbox_inches='tight')
    print("训练结果图表已保存为 'ddqn_training_results.png'")

# ============================
# 7. 主函数
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 双深度Q网络(DDQN) - CartPole-v0 训练开始")
    print("=" * 60)
    
    # 开始训练
    rewards, moving_average_rewards, ep_steps = train_ddqn()
    
    print("\n" + "=" * 60)
    print("📊 训练完成，开始分析结果...")
    print("=" * 60)
    
    # 打印统计信息
    print(f"\n📈 训练统计信息：")
    print(f"- 总回合数: {len(rewards)}")
    print(f"- 最高单回合奖励: {max(rewards)}")
    print(f"- 平均奖励: {np.mean(rewards):.2f}")
    print(f"- 最后100回合平均奖励: {np.mean(rewards[-100:]):.2f}")
    
    # 绘制结果
    plot_results(rewards, moving_average_rewards, ep_steps)
