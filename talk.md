# Pong RL 项目 — 工作记录

## 项目目标

复现 Andrej Karpathy 2016 年博文 [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)，并在此基础上进行性能优化和算法实验。

---

## 工作时间线

**Day 1**（2026-03-27，20:25 ~ 凌晨）：复现 + 版本迭代 + 性能分析 + PPO + 离线实验数据收集
**Day 2**（2026-03-29 ~ 03-30）：PPO 验证 + 离线实验 + Loss 函数对比实验

### 第一阶段：复现原始项目

1. 获取 Karpathy 原始 Python 2 代码，适配 Python 3 + Gymnasium API
2. 创建 `pg_pong.py`（v0 NumPy），忠于原文的 130 行纯 NumPy 实现
3. 修复 `ale_py` 环境注册问题（需要 `gym.register_envs(ale_py)`）
4. 成功运行，验证 agent 从 -21 开始慢慢学习

### 第二阶段：PyTorch 版本迭代

| 版本 | 文件 | 核心改动 | 结果 |
|------|------|---------|------|
| v0 | `pg_pong.py` | NumPy 手写 forward/backward | **+0.07 @ 14319 局** |
| v1 | `pg_pong_torch.py` | PyTorch，逐步建图（慢） | 已停（backward 占 70%） |
| v2 | `pg_pong_torch_v2.py` | **batch forward+backward** | **+0.02 @ 16221 局**（第一个破 0） |
| v3 | `pg_pong_torch_v3.py` | CPU 推理 + MPS GPU 训练 | 已停（GPU 无优势） |
| v4 | `pg_pong_v4.py` | 4 进程并行 env + GPU 训练 | 已停 |
| v4b | `pg_pong_v4b.py` | 4 进程并行 env + 纯 CPU | benchmark 最快（7.2 ep/s） |
| v5 | `pg_pong_v5.py` | frameskip=8 + 40x40 + 并行 env | 梯度爆炸崩溃（无 grad clip） |
| v6 | `pg_pong_v6.py` | NumPy 推理 + 预分配数组 | 收效不大 |
| v7 | `pg_pong_v7.py` | CNN 网络 | 推理太重，学得慢 |
| rec | `pg_pong_recorder.py` | v2 + 样本记录 | **+5.09 @ 24720 局**（全阶段样本） |

### 第三阶段：PPO

| | PG (v2) | PPO |
|--|---------|-----|
| 到 running mean > 0 | 16221 局 | **~700 局** |
| 到 running mean > 5 | 24720 局 | **408 局** |
| 样本效率提升 | - | **60 倍** |

### 第四阶段：离线实验（Loss 函数对比）

用 recorder 攒的 24700 局样本（-21 到 +5），对比 3 种 loss × 不同数据策略：

| 实验 | Loss | 数据策略 | Mean Reward |
|------|------|---------|------------|
| seq_pg | REINFORCE | 顺序 | -20.53 |
| shuf_pg | REINFORCE | 打散 | -20.60 |
| seq_bc | 行为克隆 | 顺序 | -15.03 |
| shuf_bc | 行为克隆 | 打散 | -17.30 |
| **seq_weighted** | **加权 BC** | **顺序** | **-10.50** |
| shuf_weighted | 加权 BC | 打散 | -13.70 |
| filt_bc | 行为克隆 | 过滤>-15 | -13.33 |
| **filt_weighted** | **加权 BC** | **过滤>-15** | **-11.50** |

**关键结论**：
1. **PG loss 在离线数据上完全无效**（-20.5）— on-policy loss 不能用 off-policy 数据
2. **BC loss 有效但不区分好坏**（-15 ~ -17）
3. **加权 BC 最好**（-10.5 ~ -13.7）— 好样本权重大
4. **顺序 > 打散**（-10.5 vs -13.7）— curriculum 效应实验验证
5. **过滤好样本有帮助**（-11.5 vs -15.0）

---

## 性能分析

### 速度对比（单独运行 benchmark）

| 版本 | 速度 | 说明 |
|------|------|------|
| v0 NumPy 单 env | 4.50 ep/s | 基线 |
| v2 PyTorch 单 env | 4.93 ep/s | batch 优化 |
| **v4b 多 env CPU** | **7.20 ep/s** | **最快** |
| v5 frameskip | ~10 ep/s（被挤压） | frameskip 减步数 |
| v7 CNN | 1.60 ep/s | CNN 推理重 |

### 关键发现

1. **v1 为什么慢**：逐步建计算图，backward 占 70%。系统态 35%，大量时间花在内存分配/释放
2. **GPU 对小网络无优势**：内核启动开销 > 计算收益（无论 FC 还是 CNN）
3. **真正的瓶颈是 PyTorch 推理**：cProfile 显示 `torch._C._nn.linear` 占 53%，模拟器不是瓶颈
4. **误判过 Gymnasium 包装层**：benchmark 本身就通过 Gymnasium，16000fps 已含其开销
5. **torch.compile 在 macOS MPS 上无效**：编译开销 > 收益
6. **学得越好越慢**：agent 学会接球后每局步数 200→1000+，ep/s 大幅下降
7. **v5 梯度爆炸**：无 gradient clipping，训练后期 NaN。PPO 的 clip 机制自然防止此问题

---

## 技术讨论记录

### Policy Gradient 核心原理
- 每步只做前向传播，不做 backward
- 一局结束后批量计算折扣奖励，乘以 dlogp 作为梯度信号
- 每 10 局累积梯度后 RMSProp 更新
- **网络不感知序列**，差分帧编码运动信息，每步都是独立的图片分类

### dlogp 解释
- `y - aprob`：期望动作和实际概率的差
- 网络只输出一个数 p（UP 概率），P(DOWN) = 1-p
- 先记录方向，后乘好坏（discounted reward）

### 折扣奖励
- 大部分 step reward=0，只有得分/丢分时 ±1
- `discount_rewards` 将信号往前传播，`if r[t]!=0: reset` 保证每球独立
- 越靠近得分的 step 权重越大（γ^距离）
- **原始 reward 不修改**，是新建 discounted 数组分配功劳

### On-policy vs Off-policy
- **On-policy**：数据来自当前策略，用完即弃（REINFORCE、PPO）
- **Off-policy**：数据来自旧策略或别人的策略（DQN、行为克隆）
- PG loss 假设 `E_{s~π_θ}`，off-policy 数据的状态分布不匹配，梯度有偏
- BC loss 是监督学习，不依赖状态分布假设，off-policy 可用

### Loss 函数对比
```python
# 三个 loss 本质相同：-log_prob × weight，区别只在 weight
PG:       -log π(a|s) × discounted_reward   # on-policy 折扣回报（off-policy 有偏）
BC:       -log π(a|s) × 1                   # 不看 reward，纯模仿
Weighted: -log π(a|s) × norm(episode_reward) # 归一化局得分（无偏且区分好坏）
```

### Atari 2600 / Stella 模拟器
- TIA 芯片**为 Pong 设计**：2 player + 1 ball + playfield
- **碰撞检测在渲染过程中发生**（TIA 逐像素画时检测重叠），逻辑和渲染不可分离
- 无帧缓冲（128 字节 RAM），"Racing the Beam"
- Pong ROM 仅 **2KB**，核心逻辑不到 30 行现代代码
- ALE 模拟器 ~3 万行 C++，忠实模拟整台硬件
- GPU 版模拟器（CuLE）：把整个 Stella 用 CUDA 重写，每个 GPU 线程跑一个游戏实例

### Pong 游戏机制
- 两个动作：UP 和 DOWN
- 先到 21 分赢一局，对手是硬编码 AI
- 球碰挡板根据碰撞位置改变角度，球会逐渐加速
- 25% 动作重复概率（随机性保证 rally 不会无限）

### 用户独立提出的 idea（对应经典论文）
| 用户 idea | 对应论文 | 实验验证 |
|----------|---------|---------|
| 样本可以重复用 | PPO (2017) | PPO 60 倍样本效率 |
| 只学好样本 | Prioritized Experience Replay (2015) | filt_weighted -11.5 vs seq_bc -15.0 |
| 顺序学有 curriculum 效应 | Curriculum Learning (2009) | seq_weighted -10.5 vs shuf_weighted -13.7 |
| 打散步会损害学习 | Experience Replay + DQN (2013) | 实验验证 PG loss 下确实有害 |
| 回球应该被奖励 | Reward Shaping | exp5 15645 局破 0（vs PG 16221 局） |

---

## Reading List — Pong / Atari RL 经典论文

| # | 论文 | 年份 | 引用数 | 核心贡献 |
|---|------|------|--------|---------|
| 1 | [Human-level control through deep RL](https://www.nature.com/articles/nature14236) | 2015 | ~31,000 | DQN + Experience Replay |
| 2 | [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) | 2013 | ~15,000 | 首次深度学习玩 Atari |
| 3 | [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | 2017 | ~18,000 | clip 约束复用样本 |
| 4 | [A3C: Asynchronous Methods for Deep RL](https://arxiv.org/abs/1602.01783) | 2016 | ~12,000 | 多线程并行 + Actor-Critic |
| 5 | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | 2015 | ~8,000 | 按重要性采样 |
| 6 | [Rainbow](https://arxiv.org/abs/1710.02298) | 2017 | ~4,000 | 6 个 DQN 改进组合 |
| 7 | [Decision Transformer](https://arxiv.org/abs/2106.01345) | 2021 | ~3,000 | RL → 序列建模 |
| 8 | [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) | 2009 | ~5,000 | 从易到难渐进学习 |
| 9 | [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) | 1992 | ~12,000 | Policy Gradient 原文 |
| 10 | [Karpathy: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) | 2016 | 博文 | 本项目出发点 |

**建议阅读顺序**：10 → 9 → 2 → 1 → 5 → 4 → 3 → 6 → 8 → 7

---

## 最终成绩汇总

### 在线训练（从零开始）

| 算法 | 到 running mean > 0 | 到 +5 |
|------|---------------------|-------|
| PG v0 NumPy | 14319 局 | - |
| PG v2 PyTorch | 16221 局 | - |
| PG + reward shaping | 15645 局 | - |
| PG recorder | 16875 局 | 24720 局 |
| **PPO** | **~700 局** | **408 局（恢复后）** |

### 离线训练（用 recorder 的 24700 局样本）

| Loss 类型 | 最好成绩 | 最好配置 |
|----------|---------|---------|
| PG (REINFORCE) | -20.5 | 无效 |
| BC (行为克隆) | -13.3 | 过滤好样本 |
| **Weighted BC** | **-10.5** | **顺序 + 全量** |

### 项目文件

```
pong/
├── pg_pong.py              # v0 NumPy（Karpathy 原文复现）
├── pg_pong_torch_v2.py     # v2 batch CPU（最快单 env）
├── pg_pong_v4b.py          # v4b 多 env CPU（benchmark 最快）
├── pg_pong_recorder.py     # 样本录制器
├── experiments/
│   ├── ppo/train.py        # PPO 实现
│   ├── run_all_offline.py  # 离线实验统一脚本
│   └── offline_common.py   # 共用模块（3 种 loss）
├── replay_data/            # 24700 局样本（2.2GB）
├── talk.md                 # 本文件
├── experiments.md          # 实验计划
└── README.md               # 项目说明
```
