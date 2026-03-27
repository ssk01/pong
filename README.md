# 复现 Karpathy "Pong from Pixels"

复现 Andrej Karpathy 2016 年经典博文 [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)。

原始代码基于 Python 2 + OpenAI Gym，本项目已适配 Python 3 + Gymnasium API。

---

## 项目原理

### 1. 核心思想：策略梯度（Policy Gradient）

- **目标**：训练一个神经网络，仅通过观察游戏画面像素，学会玩 Atari Pong
- **方法**：不需要人类标注"正确动作"，而是让 agent 自己玩，赢了就强化那些动作，输了就抑制
- **公式**：`∇θ J(θ) = E[∇θ log π(a|s) · R]`，即沿"好结果"方向调整策略参数

### 2. 网络结构

```
输入: 80×80 差分帧 (当前帧 - 上一帧) → 6400维向量
  ↓
隐藏层: 200个神经元, ReLU激活
  ↓
输出: 1个神经元, Sigmoid激活 → 向上移动的概率 p
  ↓
动作: p > random() → UP(2), 否则 → DOWN(3)
```

### 3. 训练流程

```
1. 用当前策略玩若干局 Pong
2. 记录每一步的：状态 x、隐层 h、动作梯度 dlogp、奖励 r
3. 一局结束后，计算折扣奖励并标准化
4. 反向传播：好动作(赢球)梯度放大，坏动作(丢球)梯度反转
5. 每10局累积梯度后用 RMSProp 更新参数
6. 重复，策略逐渐变好
```

### 4. 关键技巧

| 技巧 | 说明 |
|------|------|
| 差分帧 | 用当前帧减去上一帧作为输入，让网络感知运动方向 |
| 折扣奖励 | γ=0.99，让早期动作也能获得远期奖励的信号 |
| 奖励标准化 | 减均值除标准差，稳定训练 |
| RMSProp | 自适应学习率优化器，比 SGD 更稳定 |

---

## 项目文件

```
pong/
├── pg_pong.py          # 主训练脚本 (Python 3 + Gymnasium)
├── requirements.txt    # 依赖清单
└── README.md           # 本文件
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 开始训练

```bash
python pg_pong.py
```

### 3. 观察训练进度

- 前几百局：agent 基本乱动，分数约 -21（全输）
- ~500 局后：开始偶尔得分，running mean 逐渐上升
- ~3000+ 局后：能与 AI 对手打得有来有回

### 4. 恢复训练

修改 `pg_pong.py` 中 `resume = True`，从 `save.p` 恢复上次训练的模型。

### 5. 可视化

修改 `pg_pong.py` 中 `render = True`，可以看到实时游戏画面（会降低训练速度）。

---

## 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| H | 200 | 隐藏层神经元数 |
| batch_size | 10 | 每多少局更新一次参数 |
| learning_rate | 1e-4 | 学习率 |
| gamma | 0.99 | 奖励折扣因子 |
| decay_rate | 0.99 | RMSProp 衰减率 |

---

## 参考

- [Karpathy 博文: Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)
- [原始代码 Gist](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
- [Gymnasium 文档](https://gymnasium.farama.org/)
