import gymnasium as gym  # 导入Gymnasium环境库（强化学习环境的标准库）
import numpy as np       # 导入NumPy数值计算库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库用于可视化
import matplotlib
import warnings

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class Config:
    """配置类：存储强化学习训练的超参数"""
    def __init__(self):
        self.train_eps = 1000  # 训练episodes数（一个episode代表一次完整的游戏过程）
        self.policy_lr = 0.1    # 学习率（控制Q值更新的步长，越大学得越快但可能不稳定）
        self.gamma = 0.9        # 折扣因子（控制对未来奖励的重视程度，0-1之间）
        self.algorithm = "BOTH"  # 选择算法："Q-LEARNING", "SARSA", 或 "BOTH"(对比两种算法)
        self.epsilon = 0.1      # ε-贪婪策略中的探索概率

cfg = Config()  # 创建配置实例

class CliffWalkingWrapper(gym.Wrapper):
    """
    环境包装器：用于适配新版本的Gymnasium API
    CliffWalking是一个经典的网格世界强化学习问题：
    - 智能体需要从起点走到终点
    - 路径中有悬崖，掉下去会受到大的负奖励
    - 目标是找到安全且高效的路径
    """
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):  # 重置环境到初始状态
        state, _ = self.env.reset(**kwargs)  # gymnasium 返回 (state, info) 元组
        return state  # 只返回状态，保持兼容性
    
    def step(self, action):
        """执行动作并返回环境反馈"""
        return self.env.step(action)

class QLearning:
    """
    Q-Learning算法实现
    Q-Learning是一种经典的强化学习算法，属于时序差分(TD)学习方法
    
    核心思想：
    1. 维护一个Q表，记录在每个状态下执行每个动作的价值(Q值)
    2. 使用ε-贪婪策略平衡探索(exploration)和利用(exploitation)
    3. 通过贝尔曼方程更新Q值：Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        """
        初始化Q-Learning算法
        
        参数:
        - state_dim: 状态空间大小（环境中总共有多少个不同的状态）
        - action_dim: 动作空间大小（每个状态下可以执行多少种不同的动作）
        - learning_rate: 学习率α，控制每次更新的步长
        - gamma: 折扣因子γ，控制对未来奖励的重视程度
        - epsilon: ε-贪婪策略中的探索概率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # 初始化Q表：所有状态-动作对的Q值都设为0
        self.q_table = np.zeros((state_dim, action_dim))
    
    def choose_action(self, state):
        """
        根据ε-贪婪策略选择动作
        
        ε-贪婪策略：
        - 以ε的概率随机选择动作（探索，exploration）
        - 以(1-ε)的概率选择当前Q值最大的动作（利用，exploitation）
        
        这样可以在学习过程中平衡探索新策略和利用已知的好策略
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择动作 
            # 返回值0~3
            return np.random.choice(self.action_dim)
        else:
            # 利用：选择Q值最大的动作（贪婪动作） 
            # 返回q_table最大值的下标
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        使用Q-Learning公式更新Q表
        
        Q-Learning更新公式：
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        其中：
        - s: 当前状态
        - a: 执行的动作  
        - r: 获得的奖励
        - s': 下一个状态
        - α: 学习率
        - γ: 折扣因子
        """
        if done:
            # 如果游戏结束，目标值就是当前奖励（没有未来奖励）
            target = reward
        else:
            # 否则，目标值 = 当前奖励 + 折扣后的下一状态最大Q值
            # 这里是只计算下一个状态的Q值，叫做1步时序差分
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # 更新Q值：朝着目标值的方向调整
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

class SARSA:
    """
    SARSA算法实现
    SARSA(State-Action-Reward-State-Action)是一种在策略(on-policy)的时序差分学习算法
    
    核心思想：
    1. 维护一个Q表，记录在每个状态下执行每个动作的价值(Q值)
    2. 使用ε-贪婪策略选择动作
    3. 使用贝尔曼方程更新Q值：Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
    
    与Q-Learning的区别：
    - Q-Learning使用max(Q(s',a'))进行更新（离策略，off-policy）
    - SARSA使用实际选择的动作Q(s',a')进行更新（在策略，on-policy）
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        """
        初始化SARSA算法
        
        参数:
        - state_dim: 状态空间大小
        - action_dim: 动作空间大小  
        - learning_rate: 学习率α
        - gamma: 折扣因子γ
        - epsilon: ε-贪婪策略中的探索概率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # 初始化Q表：所有状态-动作对的Q值都设为0
        self.q_table = np.zeros((state_dim, action_dim))
    
    def choose_action(self, state):
        """
        根据ε-贪婪策略选择动作
        
        ε-贪婪策略：
        - 以ε的概率随机选择动作（探索，exploration）
        - 以(1-ε)的概率选择当前Q值最大的动作（利用，exploitation）
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择动作 
            return np.random.choice(self.action_dim)
        else:
            # 利用：选择Q值最大的动作（贪婪动作） 
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        使用SARSA公式更新Q表
        
        SARSA更新公式：
        Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        
        其中：
        - s: 当前状态
        - a: 执行的动作  
        - r: 获得的奖励
        - s': 下一个状态
        - a': 在下一个状态实际选择的动作
        - α: 学习率
        - γ: 折扣因子
        
        注意：SARSA使用实际选择的下一个动作a'，而Q-Learning使用最大Q值对应的动作
        """
        if done:
            # 如果游戏结束，目标值就是当前奖励（没有未来奖励）
            target = reward
        else:
            # 否则，目标值 = 当前奖励 + 折扣后的下一状态实际选择动作的Q值
            target = reward + self.gamma * self.q_table[next_state][next_action]
        
        # 更新Q值：朝着目标值的方向调整
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

'''初始化环境'''  
# 创建CliffWalking环境：一个4x12的网格世界，智能体需要避开悬崖到达目标
env = gym.make("CliffWalking-v0", render_mode=None)  
env = CliffWalkingWrapper(env)  # 使用包装器适配API

def train_agent(algorithm_name):
    """
    训练指定算法的智能体
    
    参数:
    - algorithm_name: "Q-LEARNING" 或 "SARSA"
    
    返回:
    - agent: 训练好的智能体
    - rewards: 训练过程中的奖励记录
    - ema_rewards: 指数移动平均奖励记录
    """
    print(f"\n开始{algorithm_name}训练...")
    print("环境信息：")
    print(f"- 状态空间大小: {env.observation_space.n}")
    print(f"- 动作空间大小: {env.action_space.n}")
    print(f"- 训练episodes: {cfg.train_eps}")
    print("-" * 50)
    
    # 根据算法选择创建智能体
    if algorithm_name == "Q-LEARNING":
        agent = QLearning(
            state_dim=env.observation_space.n,    # 状态空间大小（48个状态：4x12网格）
            action_dim=env.action_space.n,        # 动作空间大小（4个动作：上下左右）
            learning_rate=cfg.policy_lr,          # 学习率
            gamma=cfg.gamma,                      # 折扣因子
            epsilon=cfg.epsilon                   # 探索概率
        )
    elif algorithm_name == "SARSA":
        agent = SARSA(
            state_dim=env.observation_space.n,    
            action_dim=env.action_space.n,        
            learning_rate=cfg.policy_lr,          
            gamma=cfg.gamma,                      
            epsilon=cfg.epsilon                   
        )
    else:
        raise ValueError(f"未知算法: {algorithm_name}")
    
    # 用于记录训练过程的列表
    rewards = []     # 记录每个episode的总奖励
    ema_rewards = []  # 记录指数移动平均奖励（用于平滑曲线观察训练趋势）
    
    # 开始训练循环
    for i_ep in range(cfg.train_eps):  # 训练指定数量的episodes
        ep_reward = 0  # 记录当前episode的累积奖励
        state = env.reset()  # 重置环境到初始状态，开始新的一个episode
        
        if algorithm_name == "Q-LEARNING":
            # Q-Learning训练循环
            while True:
                # 1. 根据当前状态和ε-贪婪策略选择动作
                action = agent.choose_action(state)  
                
                # 2. 在环境中执行动作，获得反馈
                next_state, reward, terminated, truncated, _ = env.step(action)  
                if truncated:
                    raise ValueError("truncated is true!!")
                done = terminated or truncated  # 判断episode是否结束
                
                # 3. 使用Q-Learning算法更新Q表
                agent.update(state, action, reward, next_state, done)  
                
                # 4. 更新状态和累积奖励
                state = next_state  # 转移到下一个状态
                ep_reward += reward  # 累加奖励
                
                # 5. 如果episode结束，跳出循环
                if done:
                    break
        
        elif algorithm_name == "SARSA":
            # SARSA训练循环
            # 选择初始动作
            action = agent.choose_action(state)
            
            while True:
                # 1. 在环境中执行动作，获得反馈
                next_state, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise ValueError("truncated is true!!")
                done = terminated or truncated  # 判断episode是否结束
                
                ep_reward += reward  # 累加奖励
                
                if done:
                    # 如果episode结束，只需要更新当前状态-动作对
                    agent.update(state, action, reward, next_state, None, done)
                    break
                else:
                    # 2. 为下一个状态选择动作
                    next_action = agent.choose_action(next_state)
                    
                    # 3. 使用SARSA算法更新Q表
                    agent.update(state, action, reward, next_state, next_action, done)
                    
                    # 4. 更新状态和动作
                    state = next_state
                    action = next_action
        
        # 记录本episode的奖励
        rewards.append(ep_reward)
        
        # 计算移动平均奖励（用指数移动平均平滑曲线）
        if ema_rewards:
            ema_rewards.append(ema_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            ema_rewards.append(ep_reward)

        # 每100个episodes打印一次进度
        if (i_ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])  # 最近100个episodes的平均奖励
            print(f"Episode {i_ep+1}/{cfg.train_eps}: 最近100episodes平均奖励: {avg_reward:.1f}")

    print(f"\n{algorithm_name}训练完成！")
    print(f"最终100个episodes平均奖励: {np.mean(rewards[-100:]):.1f}")
    print(f"最佳episode奖励: {max(rewards):.1f}")
    print(f"最差episode奖励: {min(rewards):.1f}")
    
    return agent, rewards, ema_rewards

# 训练选定的算法
if cfg.algorithm == "BOTH":
    # 如果选择对比两种算法
    print("=== 开始对比Q-Learning和SARSA算法 ===")
    
    # 训练Q-Learning
    q_agent, q_rewards, q_ema_rewards = train_agent("Q-LEARNING")
    
    # 训练SARSA
    sarsa_agent, sarsa_rewards, sarsa_ema_rewards = train_agent("SARSA")
    
    # 使用Q-Learning的结果作为主要结果（用于后续可视化）
    agent = q_agent
    rewards = q_rewards
    ema_rewards = q_ema_rewards
    
else:
    # 训练单一算法
    agent, rewards, ema_rewards = train_agent(cfg.algorithm)

# 可视化训练结果
def plot_results(algorithm_name="Q-LEARNING", compare_results=None):
    """
    绘制训练过程的奖励曲线
    
    参数:
    - algorithm_name: 主要算法名称
    - compare_results: 可选的对比结果，格式为 {"算法名": {"rewards": [...], "ema_rewards": [...]}}
    """
    plt.figure(figsize=(15, 10))
    
    if compare_results is None:
        # 单算法可视化
        # 创建子图
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
        plt.plot(ema_rewards, color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.title(f'{algorithm_name} 训练过程中的奖励变化')
        plt.legend(['原始奖励', '移动平均奖励'])
        plt.grid(True, alpha=0.3)
        
        # 最近1000个episodes的详细视图
        plt.subplot(2, 2, 2)
        recent_rewards = rewards[-1000:] if len(rewards) >= 1000 else rewards
        recent_ma = ema_rewards[-1000:] if len(ema_rewards) >= 1000 else ema_rewards
        plt.plot(recent_rewards, alpha=0.6, color='blue', linewidth=0.5)
        plt.plot(recent_ma, color='red', linewidth=2)
        plt.xlabel('Episode (最近1000个)')
        plt.ylabel('奖励')
        plt.title(f'{algorithm_name} 最近1000个Episodes的奖励')
        plt.legend(['原始奖励', '移动平均奖励'])
        plt.grid(True, alpha=0.3)
        
        # 奖励分布直方图
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('奖励值')
        plt.ylabel('频次')
        plt.title(f'{algorithm_name} 奖励分布直方图')
        plt.grid(True, alpha=0.3)
        
        # 学习进度（每100个episodes的平均奖励）
        plt.subplot(2, 2, 4)
        if len(rewards) >= 100:
            progress_rewards = []
            progress_episodes = []
            for i in range(100, len(rewards) + 1, 100):
                progress_rewards.append(np.mean(rewards[i-100:i]))
                progress_episodes.append(i)
            plt.plot(progress_episodes, progress_rewards, 'o-', color='green', linewidth=2, markersize=6)
            plt.xlabel('Episode')
            plt.ylabel('平均奖励 (每100episodes)')
            plt.title(f'{algorithm_name} 学习进度')
            plt.grid(True, alpha=0.3)
            
    else:
        # 算法对比可视化
        colors = ['blue', 'red', 'green', 'orange']
        
        # 对比移动平均奖励
        plt.subplot(2, 2, 1)
        plt.plot(ema_rewards, color=colors[0], linewidth=2, label=algorithm_name)
        for i, (alg_name, results) in enumerate(compare_results.items(), 1):
            plt.plot(results["ema_rewards"], color=colors[i], linewidth=2, label=alg_name)
        plt.xlabel('Episode')
        plt.ylabel('移动平均奖励')
        plt.title('算法对比：移动平均奖励')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习进度对比
        plt.subplot(2, 2, 2)
        if len(rewards) >= 100:
            # 主算法的学习进度
            progress_rewards = []
            progress_episodes = []
            for i in range(100, len(rewards) + 1, 100):
                progress_rewards.append(np.mean(rewards[i-100:i]))
                progress_episodes.append(i)
            plt.plot(progress_episodes, progress_rewards, 'o-', color=colors[0], linewidth=2, markersize=4, label=algorithm_name)
            
            # 对比算法的学习进度
            for i, (alg_name, results) in enumerate(compare_results.items(), 1):
                alg_rewards = results["rewards"]
                if len(alg_rewards) >= 100:
                    alg_progress_rewards = []
                    for j in range(100, len(alg_rewards) + 1, 100):
                        alg_progress_rewards.append(np.mean(alg_rewards[j-100:j]))
                    plt.plot(progress_episodes[:len(alg_progress_rewards)], alg_progress_rewards, 'o-', 
                            color=colors[i], linewidth=2, markersize=4, label=alg_name)
        
        plt.xlabel('Episode')
        plt.ylabel('平均奖励 (每100episodes)')
        plt.title('算法对比：学习进度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最终性能对比（箱线图）
        plt.subplot(2, 2, 3)
        final_performances = [rewards[-100:]]  # 主算法最后100个episodes
        labels = [algorithm_name]
        
        for alg_name, results in compare_results.items():
            final_performances.append(results["rewards"][-100:])
            labels.append(alg_name)
        
        plt.boxplot(final_performances, labels=labels)
        plt.ylabel('奖励')
        plt.title('算法对比：最终性能分布 (最后100 episodes)')
        plt.grid(True, alpha=0.3)
        
        # 收敛速度对比
        plt.subplot(2, 2, 4)
        # 计算达到一定性能水平的episode数
        target_performance = -20  # 设定一个目标性能水平
        
        convergence_episodes = []
        alg_names_conv = []
        
        # 主算法
        for i, reward in enumerate(ema_rewards):
            if reward >= target_performance:
                convergence_episodes.append(i)
                alg_names_conv.append(algorithm_name)
                break
        else:
            convergence_episodes.append(len(ema_rewards))
            alg_names_conv.append(algorithm_name)
        
        # 对比算法
        for alg_name, results in compare_results.items():
            for i, reward in enumerate(results["ema_rewards"]):
                if reward >= target_performance:
                    convergence_episodes.append(i)
                    alg_names_conv.append(alg_name)
                    break
            else:
                convergence_episodes.append(len(results["ema_rewards"]))
                alg_names_conv.append(alg_name)
        
        plt.bar(alg_names_conv, convergence_episodes, color=colors[:len(alg_names_conv)])
        plt.ylabel('收敛所需Episodes')
        plt.title(f'算法对比：收敛速度 (目标性能: {target_performance})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'training_results.png' if compare_results is None else 'algorithm_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_q_table():
    """可视化Q表，显示学到的策略"""
    # CliffWalking是4x12的网格，重新整理Q表用于可视化
    policy = np.argmax(agent.q_table, axis=1).reshape(4, 12)
    q_values = np.max(agent.q_table, axis=1).reshape(4, 12)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 可视化学到的策略
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}  # CliffWalking的动作映射
    
    im1 = ax1.imshow(q_values, cmap='viridis', aspect='auto')
    ax1.set_title('Q值热力图')
    ax1.set_xlabel('列位置')
    ax1.set_ylabel('行位置')
    
    # 在每个格子中添加策略箭头
    for i in range(4):
        for j in range(12):
            if not (i == 3 and 1 <= j <= 10):  # 避开悬崖区域
                arrow = action_arrows[policy[i, j]]
                ax1.text(j, i, arrow, ha='center', va='center', 
                        color='white' if q_values[i, j] < q_values.mean() else 'black',
                        fontsize=12, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Q值')
    
    # 显示悬崖和目标位置
    cliff_map = np.zeros((4, 12))
    cliff_map[3, 1:11] = -1  # 悬崖位置
    cliff_map[3, 0] = 0.5    # 起点
    cliff_map[3, 11] = 1     # 终点
    
    im2 = ax2.imshow(cliff_map, cmap='RdYlGn', aspect='auto')
    ax2.set_title('环境布局')
    ax2.set_xlabel('列位置')
    ax2.set_ylabel('行位置')
    
    # 标记特殊位置
    ax2.text(0, 3, 'S', ha='center', va='center', fontsize=16, fontweight='bold', color='blue')  # 起点
    ax2.text(11, 3, 'G', ha='center', va='center', fontsize=16, fontweight='bold', color='green')  # 终点
    for j in range(1, 11):
        ax2.text(j, 3, '×', ha='center', va='center', fontsize=16, fontweight='bold', color='red')  # 悬崖
    
    plt.colorbar(im2, ax=ax2, label='区域类型')
    plt.tight_layout()
    plt.savefig('q_table_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_trained_agent():
    """测试训练好的智能体"""
    print("\n测试训练好的智能体...")
    test_episodes = 10
    test_rewards = []
    
    # 临时关闭探索（设置epsilon为0）
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # 纯贪婪策略，不探索
    
    for ep in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        path = [state]  # 记录路径
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            path.append(state)
            
            if done:
                break
        
        test_rewards.append(total_reward)
        print(f"测试Episode {ep+1}: 奖励={total_reward:.1f}, 步数={steps}")
    
    # 恢复原来的epsilon
    agent.epsilon = original_epsilon
    
    print(f"\n测试结果:")
    print(f"平均奖励: {np.mean(test_rewards):.1f}")
    print(f"奖励标准差: {np.std(test_rewards):.1f}")
    
    return test_rewards

# 执行可视化和测试
print("\n生成可视化图表...")
if cfg.algorithm == "BOTH":
    # 对比两种算法
    compare_results = {
        "SARSA": {
            "rewards": sarsa_rewards,
            "ema_rewards": sarsa_ema_rewards
        }
    }
    plot_results("Q-LEARNING", compare_results)
    print("算法对比图表已保存为 'algorithm_comparison.png'")
    
    # 打印对比统计信息
    print("\n=== 算法性能对比 ===")
    q_final_avg = np.mean(q_rewards[-100:])
    sarsa_final_avg = np.mean(sarsa_rewards[-100:])
    
    print(f"Q-Learning 最后100episodes平均奖励: {q_final_avg:.2f}")
    print(f"SARSA 最后100episodes平均奖励: {sarsa_final_avg:.2f}")
    print(f"性能差异: {abs(q_final_avg - sarsa_final_avg):.2f}")
    
    if q_final_avg > sarsa_final_avg:
        print("Q-Learning 性能更好")
    elif sarsa_final_avg > q_final_avg:
        print("SARSA 性能更好")
    else:
        print("两种算法性能相近")

else:
    # 单算法可视化
    plot_results(cfg.algorithm)
    print(f"{cfg.algorithm}训练结果图表已保存为 'training_results.png'")

print("\n生成Q表可视化...")
visualize_q_table() 
print("Q表可视化已保存为 'q_table_visualization.png'")

print("\n开始测试智能体性能...")
test_results = test_trained_agent()
