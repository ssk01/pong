# Offline RL 实验计划

基于 recorder 收集的完整训练样本，进行以下离线学习实验。

## 样本格式

每个 step 记录：
- `x`: 80x80 差分帧 (int8, -1/0/1)
- `action`: 动作标签 (1.0=UP, 0.0=DOWN)
- `reward`: 即时奖励 (0, +1, -1)
- `aprob`: 采集时模型输出的 UP 概率（用于 importance sampling）
- `discounted_reward`: 折扣奖励（预计算）

每个 episode 记录：
- `episode_id`, `total_reward`, `num_steps`

存储路径：`replay_data/`，每 100 局一个 `batch_XXXX.npz`，`index.json` 记录元数据。

---

## 实验 1: 行为克隆（顺序学习）

**思路**：按 episode 时间顺序喂给新模型学习。由于专家是从差到好逐步进步的，自然形成 curriculum — 先学基础，再学高级策略。

**对应概念**：Behavior Cloning + Curriculum Learning (Bengio 2009)

**评估**：对比从头在线训练，看收敛到 running mean > 0 需要多少 epoch。

---

## 实验 2: 只学好样本

**思路**：过滤掉 total_reward < 阈值的 episode，只学表现好的局。或按 reward 加权：好局权重大，差局权重小。

**对应概念**：Reward-weighted Regression / Offline RL with filtering (CQL, IQL)

**变体**：
- 硬过滤：只用 reward > -15 的 episode
- 软加权：weight = sigmoid(reward + 15)

**评估**：对比实验 1，看跳过垃圾样本是否加速。

---

## 实验 3: 局间乱序，局内保序

**思路**：随机打散 episode 顺序，但每局内部的 step 保持原始时序。破坏 curriculum 效应，测试从差到好的渐进学习有多重要。

**对应概念**：标准 mini-batch SGD on episodes

**评估**：对比实验 1（顺序），如果差距大说明 curriculum 效应显著。

---

## 实验 4: 完全乱序 + 跨局混合

**思路**：把所有 step 完全打散，从全局随机采样 mini-batch 训练。每个 step 已经预存了 discounted_reward，不需要局内顺序来计算。

**对应概念**：Experience Replay (DQN, Mnih et al. 2013)

**注意**：Policy Gradient 是 on-policy 的，直接用 off-policy 数据会有 bias。需要 importance sampling 修正：`weight = π_new(a|s) / π_old(a|s)`，`aprob` 字段就是为此记录的。

**评估**：对比实验 3，看完全打破时序结构的影响。

---

## 额外实验 (待定)

### 实验 5: PPO — 同一批样本多次更新

每批样本做 3-10 次梯度更新，用 clip 约束限制策略偏移。样本效率提升 3-10 倍。

### 实验 6: Decision Transformer 风格

用期望 reward 作为条件输入，训练模型在高 reward 条件下的行为。把 RL 问题转化为条件序列建模。

---

## 预期结论

| 实验 | 预期效果 | 理由 |
|------|---------|------|
| 1 顺序学 | 好 | 自然 curriculum，渐进式学习 |
| 2 只学好样本 | 可能更快 | 去噪，但丢失负样本信息 |
| 3 乱序局 | 稍差于 1 | 失去 curriculum 但保留局内结构 |
| 4 完全乱序 | 最差或需要 IS 修正 | 破坏时序 + off-policy bias |
