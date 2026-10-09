# miniStructureVLA

一个极简、面向初学者的 **Vision-Language-Action** 模型：输入图像、文本指令和机器人自身
状态，输出连续的机器人动作。这是教学项目，不是 state of the art——几百行代码，一个下午
能读完。

派生自 [keivalya/mini-vla](https://github.com/keivalya/mini-vla)（MIT, © 2025 Keivalya
Pandya），见 [LICENSE](LICENSE)。

> English version: [README.md](README.md)

## 它做什么

```
图像 ──► CNN encoder   ─┐
文本 ──► GRU encoder   ─┼──► 融合 ──► action head ──► 4 维连续动作
状态 ──► MLP encoder   ─┘
```

动作头之前的所有部分都是三个策略共用的，所以**动作头是唯一的变量**，由 YAML 配置选择：

| 动作头 | 怎么产生动作 | 参数量 |
|---|---|---|
| **MLP** | 三个 token 拼接后过一次 MLP，出一个动作，每一步重新规划 | 413k |
| **ACT** | transformer 解码器用可学习 query 一次吐出 20 个动作；一个 VAE 塑造隐变量，推理时置零 | 1.31M |
| **Diffusion** | 手写的 DDPM 从纯噪声去噪出一整段动作序列；预测 16 步，执行 8 步 | 67.6M |

任务是 Meta-World MT1，一个仿真的 Sawyer 机械臂。默认数据集是 `bin-picking-v3`：把方块从
一个箱子捡起来放进另一个箱子。

## 安装

```bash
conda create -n mini-vla python=3.10 && conda activate mini-vla
pip install -r requirements.txt
```

## 运行

所有脚本都要**从仓库根目录用模块方式**运行——`from models…` 这类导入需要它。

```bash
# 1. 用 Meta-World 自带的脚本专家采集演示数据
python -m scripts.collect_data --env-name bin-picking-v3 --camera-name corner3 \
  --episodes 100 --max-steps 200 --output-path data/bp_masked.npz

# 2. 训练。配置定义了整个实验，并且会被存进 checkpoint，
#    所以评测时不需要再给配置文件。
python -m scripts.train --config configs/mlp_bin_picking.yaml

# 3. 在环境里评测
python -m scripts.test --checkpoint checkpoints/mlp_bp.pt --episodes 50 --max-steps 250
```

`data/` 和 `checkpoints/` 都在 `.gitignore` 里，所以刚 clone 下来两者都不存在——要先采集和
训练，才能评测。

## 观测

Meta-World 的 MT1 观测是 39 维，帧堆叠：

| 切片 | 含义 |
|---|---|
| `[0:3]` / `[3]` | 末端执行器位置 / 夹爪开合 |
| `[4:18]` | 物体位姿——两个槽位，每个是（位置, 四元数） |
| `[18:36]` | 同样的这一块，上一帧的值 |
| `[36:39]` | 目标位置 |

`utils/state_masking.py` 把**特权**字段置零——物体位姿和目标位置，39 维里的 31 维——只留下
真机上靠自身传感器就能拿到的 8 维。这正是逼着视觉通路去干活的原因；`collect_data.py` 和
`test.py` 共用同一个函数，所以两者不会对"遮哪些"产生分歧。

## 目录

```
configs/    每个实验一个 YAML
envs/       MT1 的 wrapper，以及一个按类别列出全部 50 个任务的查看器
models/     编码器、三个动作头、U-Net、损失函数
scripts/    collect_data.py · train.py · test.py
utils/      状态遮罩、chunk 处理、分词器、EMA、配置加载
```

## 可以试试

- **换动作头，而不是改代码。** `configs/*.yaml` 就是整个实验面。
- **换任务。** 50 个 MT1 任务共用同一套观测和动作布局，所以换任务就是换个 `--env-name`
  加一份新数据。任务清单在 `envs/metaworld_mt1.py` 里；`python -m envs.metaworld_mt1`
  可以开个窗口先看一眼。
- **让语言真正起作用。** 这里整份数据只有一条固定指令，所以把文本编码器删掉也不会有任何
  变化。要让指令真的有活干，要么一个数据集里放多个任务，要么换一个"同一个场景下不同指令
  都成立"的 benchmark。
