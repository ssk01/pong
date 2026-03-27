# Pong RL 项目 — 工作记录

## 项目目标

复现 Andrej Karpathy 2016 年博文 [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)，并在此基础上进行性能优化和算法实验。

---

## 工作时间线

### 第一阶段：复现原始项目

1. 获取 Karpathy 原始 Python 2 代码，适配 Python 3 + Gymnasium API
2. 创建 `pg_pong.py`（v0 NumPy），忠于原文的 130 行纯 NumPy 实现
3. 修复 `ale_py` 环境注册问题（需要 `gym.register_envs(ale_py)`）
4. 成功运行，验证 agent 从 -21 开始慢慢学习

### 第二阶段：PyTorch 版本迭代

| 版本 | 文件 | 核心改动 |
|------|------|---------|
| v0 | `pg_pong.py` | NumPy 手写 forward/backward |
| v1 | `pg_pong_torch.py` | PyTorch，逐步建图（慢，已停） |
| v2 | `pg_pong_torch_v2.py` | **batch forward+backward**，消除逐步建图开销 |
| v3 | `pg_pong_torch_v3.py` | CPU 推理 + MPS GPU 训练（已停） |
| v4 | `pg_pong_v4.py` | 4 进程并行 env + GPU 训练 |
| v5 | `pg_pong_v5.py` | frameskip=8 + 40x40 输入 + 并行 env |
| v6 | `pg_pong_v6.py` | NumPy 推理 + 预分配数组，极限吞吐 |
| v7 | `pg_pong_v7.py` | **CNN 网络**（Conv+FC 替代纯 FC） |
| rec | `pg_pong_recorder.py` | 基于 v2 + 样本记录，用于离线实验 |

### 第三阶段：性能分析与优化讨论

#### 速度对比

| 版本 | 单独跑速度 | 优化点 |
|------|-----------|--------|
| v0 NumPy | ~3.9 ep/s | 基线 |
| v1 per-step graph | ~0.9 ep/s | 逐步建图拖累 |
| **v2 batch CPU** | **~5.3 ep/s** | 最快单进程 |
| v3 CPU+GPU | ~3.9 ep/s | GPU 对小网络无优势 |
| v4 parallel | ~5.0 ep/s | 多进程 IPC 有开销 |
| v5 frameskip | ~10 ep/s | frameskip 减少步数 |
| v6 numpy infer | 待单测 | 消除 PyTorch 推理开销 |
| v7 CNN | ~1.6 ep/s | CNN 推理更重 |

#### 关键发现

1. **v1 为什么慢**：逐步建计算图，backward 占 70%。系统态时间 35%，大量时间花在内存分配/释放
2. **GPU 对小网络无优势**：6400→200→1 的网络太小，GPU 内核启动开销 > 计算收益（无论 FC 还是 CNN）
3. **模拟器不是瓶颈**：ALE 裸跑 16000fps，实际只用 ~1000fps。cProfile 显示 PyTorch 推理占 53%
4. **之前误判 Gymnasium 包装层**：benchmark 本身就通过 Gymnasium 调用，16000fps 已含其开销
5. **Apple 统一内存**：省掉 PCIe 带宽瓶颈，但省不掉 GPU 内核启动/同步开销
6. **torch.compile 在 macOS MPS 上无效**：编译开销大于收益，反而更慢

---

## 技术讨论记录

### Policy Gradient 核心原理
- 每步只做前向传播，不做 backward
- 一局结束后批量计算折扣奖励，乘以 dlogp 作为梯度信号
- 每 10 局累积梯度后 RMSProp 更新
- 网络不感知序列，差分帧编码运动信息

### dlogp 解释
- `y - aprob`：期望动作和实际概率的差
- 先记录方向，后乘好坏（discounted reward）
- 赢了强化，输了抑制

### 折扣奖励
- 大部分 step reward=0，只有得分/丢分时才有 ±1
- `discount_rewards` 将信号往前传播，`if r[t]!=0: reset` 保证每球独立
- 越靠近得分的 step 权重越大

### Atari 2600 / Stella 模拟器
- TIA 芯片为 Pong 设计：2 player + 1 ball + playfield
- **碰撞检测在渲染过程中发生**（TIA 逐像素画时检测重叠）
- 无帧缓冲（128 字节 RAM），"Racing the Beam"
- Pong ROM 仅 2KB（~800 条 6507 汇编指令），核心逻辑不到 30 行现代代码
- ALE 模拟器 ~3 万行 C++，忠实模拟整台硬件

### 离线学习 idea（用户提出）
1. **顺序学专家样本** → Behavior Cloning + 自然 Curriculum
2. **只学好样本** → Reward-weighted Regression
3. **局间乱序、局内保序** → 破坏 curriculum 看影响
4. **完全乱序** → Experience Replay (DQN)
- **用户关键洞察**：顺序学会自然形成从差到好的 curriculum，乱序则 reward 均匀分布，失去渐进学习效果

### 样本效率优化
- 当前 REINFORCE 每个样本只用一次就扔掉
- **PPO** 可以同一批样本重复训练 3-10 次，加 clip 约束
- 模型结构优化：CNN 参数少 4.7 倍（27 万 vs 128 万），能捕捉空间特征

---

## 当前训练进度

| 版本 | 局数 | Running Mean | 状态 |
|------|------|-------------|------|
| v0 NumPy | 8044 | -14.31 | 快速上升中 |
| **v2 batch CPU** | **10540** | **-12.64** | **领跑** |
| v4 parallel GPU | 9655 | -13.90 | 快速上升中 |
| v5 optimized | 18742 | -14.58 | 追赶中 |
| v6 max throughput | 5308 | -17.98 | 还在爬 |
| v7 CNN | 1483 | -20.06 | 早期 |
| recorder | 499 | -20.43 | 录制中 |

**对比图**：见 [all_compare.png](all_compare.png)

---

## TODO

- [ ] 单独 benchmark v6 vs v2 真实速度对比
- [ ] 实现 v8 PPO 版本（同一批样本多次更新，clip 约束）
- [ ] 实验 1: 行为克隆 — 顺序学习专家样本
- [ ] 实验 2: 只学好样本 — reward 过滤/加权
- [ ] 实验 3: 乱序学习 — 打散 episode 顺序但保持局内顺序
- [ ] 实验 4: 完全乱序 — experience replay 风格
- [ ] 详见 [experiments.md](experiments.md)
