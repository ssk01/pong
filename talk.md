# Pong RL 项目 — 工作记录

## 项目目标

复现 Andrej Karpathy 2016 年博文 [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)，并在此基础上进行性能优化和算法实验。

---

## 工作时间线

**总耗时：约 2 小时**（20:25 ~ 22:25，2026-03-27）

### 第一阶段：复现原始项目

1. 获取 Karpathy 原始 Python 2 代码，适配 Python 3 + Gymnasium API
2. 创建 `pg_pong.py`（v0 NumPy），忠于原文的 130 行纯 NumPy 实现
3. 修复 `ale_py` 环境注册问题（需要 `gym.register_envs(ale_py)`）
4. 成功运行，验证 agent 从 -21 开始慢慢学习

### 第二阶段：PyTorch 版本迭代

| 版本 | 文件 | 核心改动 | 状态 |
|------|------|---------|------|
| v0 | `pg_pong.py` | NumPy 手写 forward/backward | 运行中 |
| v1 | `pg_pong_torch.py` | PyTorch，逐步建图（慢） | 已停 |
| v2 | `pg_pong_torch_v2.py` | **batch forward+backward** | 运行中（领跑） |
| v3 | `pg_pong_torch_v3.py` | CPU 推理 + MPS GPU 训练 | 已停 |
| v4 | `pg_pong_v4.py` | 4 进程并行 env + GPU 训练 | 已停 |
| v4b | `pg_pong_v4b.py` | 4 进程并行 env + 纯 CPU | 运行中 |
| v5 | `pg_pong_v5.py` | frameskip=8 + 40x40 输入 + 并行 env | 运行中 |
| v6 | `pg_pong_v6.py` | NumPy 推理 + 预分配数组 | 已停（结论：收效不大） |
| v7 | `pg_pong_v7.py` | **CNN 网络**（Conv+FC 替代纯 FC） | 已停（学得太慢） |
| rec | `pg_pong_recorder.py` | 基于 v2 + 样本记录 | 运行中（攒样本） |

### 第三阶段：性能分析与优化

#### 速度对比（单独运行）

| 版本 | 速度 | 优化点 |
|------|------|--------|
| v0 NumPy | ~3.9 ep/s | 基线 |
| v1 per-step graph | ~0.9 ep/s | 逐步建图拖累 |
| **v2 batch CPU** | **~5.3 ep/s**（早期） | batch 消除建图开销 |
| v3 CPU+GPU | ~3.9 ep/s | GPU 对小网络无优势 |
| v4b multi-env CPU | ~4.2 ep/s | 多进程并行采样 |
| v5 frameskip | ~10 ep/s | frameskip 减少步数 |
| v6 numpy infer | ~4.0 ep/s | 消除 PyTorch 推理开销 |
| v7 CNN | ~1.6 ep/s | CNN 推理更重 |

**注意**：v2 后期因 agent 学会接球导致每局步数暴涨，速度从 5.3 降至 ~1.3 ep/s。

#### 关键发现

1. **v1 为什么慢**：逐步建计算图，backward 占 70%。系统态时间 35%，大量时间花在内存分配/释放
2. **GPU 对小网络无优势**：6400→200→1 的网络太小，GPU 内核启动开销 > 计算收益（无论 FC 还是 CNN）
3. **真正的瓶颈是 PyTorch 推理**：cProfile 显示 `torch._C._nn.linear` 占 53%，模拟器只占很小比例
4. **之前误判 Gymnasium 包装层和模拟器**：benchmark 本身就通过 Gymnasium 调用，16000fps 已含其开销。模拟器不是瓶颈
5. **Apple 统一内存**：省掉 PCIe 带宽瓶颈，但省不掉 GPU 内核启动/同步开销
6. **torch.compile 在 macOS MPS 上无效**：编译开销大于收益，反而更慢
7. **学得越好越慢**：agent 学会接球后每局步数从 ~200 涨到 ~1000+，ep/s 大幅下降

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
- 赢了强化，输了抑制

### 折扣奖励
- 大部分 step reward=0，只有得分/丢分时才有 ±1
- `discount_rewards` 将信号往前传播，`if r[t]!=0: reset` 保证每球独立
- 越靠近得分的 step 权重越大（γ^距离）
- **原始 reward 不修改**，是新建 discounted 数组分配功劳

### 反向传播时序
- 一局内不做 backward，只存 (x, h, dlogp, reward)
- 一局结束后堆成矩阵一次算完，**没有步与步之间的顺序**
- 所有步的梯度加在一起，顺序信息已编码在 discounted_reward 里

### Atari 2600 / Stella 模拟器
- TIA 芯片**为 Pong 设计**：2 player + 1 ball + playfield，1977 年 Atari 靠 Pong 起家
- **碰撞检测在渲染过程中发生**（TIA 逐像素画时检测重叠），逻辑和渲染不可分离
- 无帧缓冲（128 字节 RAM），"Racing the Beam"
- Pong ROM 仅 **2KB**（~800 条 6507 汇编指令），核心逻辑不到 30 行现代代码，50% 代码在伺候硬件
- ALE 模拟器 ~3 万行 C++，忠实模拟整台硬件
- 获取游戏状态是通过**拷贝模拟器输出的 210x160x3 像素图片**（~100KB/帧）

### Pong 游戏机制
- 两个动作：UP 和 DOWN（不动也可以但不用）
- 先到 21 分赢一局
- 对手是 ROM 里**写死的硬编码 AI**，不会学习，不会变强
- 球碰挡板会**根据碰撞位置改变角度**，碰边缘角度更大
- 球会**逐渐加速**（碰撞次数增加后）
- ALE 有 **25% 动作重复概率**（随机性保证 rally 不会无限）
- 三个机制保证 rally 必定结束：随机性 + 角度变化 + 球加速

### 离线学习 idea（用户提出）
1. **顺序学专家样本** → Behavior Cloning + 自然 Curriculum
2. **只学好样本** → Reward-weighted Regression
3. **局间乱序、局内保序** → 破坏 curriculum 看影响
4. **完全乱序** → Experience Replay (DQN)
5. **回球给正奖励** → Reward Shaping
- **用户关键洞察**：顺序学会自然形成从差到好的 curriculum（专家训练过程本身就是天然课程），乱序则 reward 均匀分布，失去渐进学习效果
- **用户关键洞察**：长 rally 中成功回球的步应该**被奖励**而不只是"少惩罚"

### 样本效率优化
- 当前 REINFORCE 每个样本只用一次就扔掉
- **PPO** 可以同一批样本重复训练 3-10 次，加 clip 约束
- PPO 比 PG 快 **5-10 倍**样本效率
- 模型结构优化：CNN 参数少 4.7 倍（27 万 vs 128 万），能捕捉空间特征，但推理更重

### PPO vs PG 预期对比（Pong）
| | PG (当前) | PPO |
|--|----------|-----|
| 到 0 分 | ~15000 局 | ~2000-3000 局 |
| 到 +20 | ~30000+ 局 | ~5000-8000 局 |
| 训练稳定性 | 可能震荡 | 平稳上升 |

---

## Reading List — Pong / Atari RL 经典论文

| # | 论文 | 年份 | 引用数 | 核心贡献 | 与本项目的关系 |
|---|------|------|--------|---------|---------------|
| 1 | [Human-level control through deep RL](https://www.nature.com/articles/nature14236) (DQN Nature) | 2015 | ~31,000 | Experience Replay + Target Network，49 个 Atari 游戏达到人类水平 | 用户独立想到了 experience replay |
| 2 | [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (DQN 原版) | 2013 | ~15,000 | 首次用深度学习玩 Atari，1 张 GPU 训练 | 开山之作 |
| 3 | [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | 2017 | ~18,000 | clip 约束复用样本，样本效率提升 5-10x | 用户想到的"样本多次更新" |
| 4 | [A3C: Asynchronous Methods for Deep RL](https://arxiv.org/abs/1602.01783) | 2016 | ~12,000 | 多线程并行训练 + Actor-Critic | 用户的多 env 并行方案 |
| 5 | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | 2015 | ~8,000 | 按 TD error 优先采样重要样本 | 用户的"只学好样本" idea |
| 6 | [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) | 2017 | ~4,000 | 6 个 DQN 改进组合 | 综合最优实践 |
| 7 | [Decision Transformer](https://arxiv.org/abs/2106.01345) | 2021 | ~3,000 | RL 变成序列建模，条件生成 | 用户的"顺序学+只学好的"思路 |
| 8 | [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) (Bengio) | 2009 | ~5,000 | 从易到难渐进式学习 | 用户发现的专家轨迹天然 curriculum |
| 9 | [REINFORCE: Simple Statistical Gradient-Following](https://link.springer.com/article/10.1007/BF00992696) (Williams) | 1992 | ~12,000 | Policy Gradient 算法原文 | 本项目的基础算法 |
| 10 | [Karpathy: Deep RL, Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) | 2016 | 博文 | 130 行 NumPy 实现 PG 打 Pong | 本项目的出发点 |

**建议阅读顺序**：10 → 9 → 2 → 1 → 5 → 4 → 3 → 6 → 8 → 7

### DQN (2013) 训练细节
- 硬件：**1 张 GPU**（Tesla K40），单机
- 训练量：5000 万帧/游戏，约 1-2 周
- Pong 成绩：+20（几乎满分碾压 AI）

### PPO (2017) 训练细节
- 硬件：单机，MPI 多进程（~8 个并行 env），原文未注明具体 GPU
- 训练量：4000 万帧
- PPO2 后来增加了 GPU 优化版本
- 现代配置：EnvPool + A100 可达 100 万 fps，几分钟训完 Pong

---

## 当前训练进度（22:25 更新）

| 版本 | 局数 | Running Mean | 实际耗时 | 状态 |
|------|------|-------------|---------|------|
| v0 NumPy | 10717 | **-7.77** | 107 min | 运行中 |
| **v2 batch CPU** | **13449** | **-6.62** | **117 min** | **领跑** |
| v4b multi-env CPU | 5291 | -18.14 | 24 min | 运行中（刚起步） |
| v5 optimized | 26099 | **-8.50** | 87 min | 运行中（单位时间最快） |
| recorder | 6104 | -17.59 | 39 min | 录制中 |

**停止条件**：任一版本 running mean > 0 自动停止所有训练（保留 recorder）

**对比图**：见 [all_compare.png](all_compare.png)（使用真实进程启动时间计算）

---

## TODO

- [ ] 单独 benchmark v6 vs v2 真实速度对比
- [ ] 实现 v8 PPO 版本（同一批样本多次更新，clip 约束）
- [ ] 实验 1: 行为克隆 — 顺序学习专家样本
- [ ] 实验 2: 只学好样本 — reward 过滤/加权
- [ ] 实验 3: 乱序学习 — 打散 episode 顺序但保持局内顺序
- [ ] 实验 4: 完全乱序 — experience replay 风格
- [ ] 实验 5: reward shaping — 回球给正奖励
- [ ] 详见 [experiments.md](experiments.md)
