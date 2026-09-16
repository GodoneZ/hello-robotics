# 前沿算法2：Fast-WAM 复现与部署

本章在 G2 Omnipicker 上复现 Fast-WAM，完成三相机数据采集、LeRobot 数据转换、8 卡全量微调、单卡推理服务和 Isaac Sim 闭环评测。模型为什么能在推理时删除未来视频生成，请先阅读同目录的[《Fast-WAM 模型原理》](./Fast-WAM%20模型原理.md)。

> 论文：*Fast-WAM: Do World Action Models Need Test-time Future Imagination?*
>
> 官方资源：[arXiv](https://arxiv.org/abs/2603.16666)｜[项目主页](https://yuantianyuan01.github.io/FastWAM/)｜[代码](https://github.com/yuantianyuan01/FastWAM)｜[模型权重](https://huggingface.co/yuanty/fastwam)

Fast-WAM 训练时联合优化视频预测和动作预测，推理时默认只编码当前观测并生成动作。本文还保留 `joint` 推理入口，用于观察同一个模型显式生成的未来三视角视频，但正式部署使用延迟更低的 `action` 模式。

## 第一部分 G2 任务与数据合同

### 1.1 任务定义

任务沿用本教程的 G2 三色物块入盒场景：桌面放置红、绿、蓝三个物块和一个空盒，机器人根据语言指令抓取指定颜色并放入盒中。

```text
Pick up the red block and place it into the empty box.
Pick up the green block and place it into the empty box.
Pick up the blue block and place it into the empty box.
```

三种颜色共享模型和数据集，由语言指令区分目标。仿真场景、自动专家、成功判据和 16 维关节接口位于 `code/task_runtime/`。

### 1.2 观测与动作空间

| 项目 | 取值 | 说明 |
|---|---|---|
| 相机 | `cam_head`、`cam_left_wrist`、`cam_right_wrist` | 三路同步 RGB，各 `240×320` |
| 状态 | 16 维 float32 | 左臂 7 + 右臂 7 + 左右夹爪 |
| 动作 | 16 维 float32 | 同一关节顺序下的绝对目标位置 |
| 采集频率 | 30 Hz | 物理仿真 120 Hz，每个动作含 4 个物理步 |
| 动作块 | 32 步 | 一次策略调用的输出形状为 `[32,16]` |
| 闭环执行 | 前 24 步 | 执行后重新观测并规划 |
| 视频块 | 9 帧 | `t+0,t+4,…,t+32` |

### 1.3 三相机如何组成模型输入

Fast-WAM 沿用官方 RoboTwin 三相机拼接方式：头部图像放在上方，两个腕部图像放在下方。

```text
        320 px
┌──────────────────┐
│                  │
│   Head Camera    │ 256 px
│                  │
├─────────┬────────┤
│  Left   │ Right  │ 128 px
│  Wrist  │ Wrist  │
└─────────┴────────┘
       总高 384 px
```

单个模型视频帧为 `320×384`（宽×高）。9 帧是9个未来时刻，每个时刻都包含完整的三个视角，并非每个相机分别生成9帧。

### 1.4 原始轨迹与 LeRobot 数据

采集脚本把每条轨迹保存为 NPZ，转换脚本再写入 LeRobot 2.1：

| NPZ 字段 | 形状/类型 | LeRobot 特征 |
|---|---|---|
| `head_image` | `[T,240,320,3]` uint8 | `observation.images.cam_head` |
| `left_image` | `[T,240,320,3]` uint8 | `observation.images.cam_left_wrist` |
| `right_image` | `[T,240,320,3]` uint8 | `observation.images.cam_right_wrist` |
| `qpos` | `[T,16]` float32 | `observation.state` |
| `command` | `[T,16]` float32 | `action` |
| `instruction` | 字符串 | `task` |
| `success` | bool | 仅成功轨迹进入训练集 |
| `raw_fps` | 30 | 数据集 `fps` |

转换前会检查轨迹长度、图像形状、频率、字段完整性和数值有限性。少于33帧的轨迹不能形成一个完整训练窗口，会被拒绝。

## 第二部分 代码结构与环境配置

### 2.1 正式教程目录

```text
前沿算法2-Fast-WAM/
├── docs/
│   ├── Fast-WAM 模型原理.md
│   ├── Fast-WAM 复现与部署.md
│   └── assets/                     # 论文插图与精选执行视频
└── code/
    ├── README.md
    ├── configs/                    # G2 数据与训练配置
    ├── task_runtime/               # G2 Isaac Sim 任务
    ├── tests/                      # 离线数据合同测试
    ├── collect_data.py
    ├── convert_dataset.py
    ├── inspect_data.py
    ├── fastwam_runtime.py
    ├── serve_policy.py
    ├── evaluate.py
    ├── summarize_latency.py
    ├── setup_env.sh
    ├── download_weights.sh
    ├── precompute_text.sh
    ├── train.sh                  # 8卡全量微调
    ├── train_lora.sh             # 单卡 LoRA 微调
    ├── lora.py                   # LoRA 注入与 int8 基座
    └── merge_lora.py             # 合并部署权重
```

官方 Fast-WAM 仓库、数据、预训练权重、训练 checkpoint 和评测输出均为运行时产物，由 `.gitignore` 排除，不随教程保存。

### 2.2 准备仿真资产与模型环境

本章已验证的环境为 Linux、NVIDIA GPU 和 Isaac Sim 5.1.0；模型端使用 Python 3.10、PyTorch 2.7.1 和 CUDA 12.8。Joint 模式如需保存预测视频，系统还需要 `ffmpeg`。

G2 USD 不随 GitHub 代码仓库分发。先在 `hello-robotics` 根目录下载本章所需的最小资产集：

```bash
python -m pip install -U huggingface_hub
hf download Datawhale/g2_assets \
  --repo-type dataset \
  --revision df7aae7e160f0eeabb2ebc88a3a396405f4a5aee \
  --include "assets/robot/G2_omnipicker/**" \
  --local-dir code
test -f code/assets/robot/G2_omnipicker/robot.usda
```

资产来源为 [Datawhale/g2_assets](https://huggingface.co/datasets/Datawhale/g2_assets)。如果已在其他位置保存完整资产，可设置 `G2_ASSETS_ROOT` 指向其 `assets` 目录。

然后进入本章代码目录，克隆并固定官方 Fast-WAM 版本：

```bash
cd sota_algorithm/g2/前沿算法2-Fast-WAM/code
git clone https://github.com/yuantianyuan01/FastWAM.git FastWAM
git -C FastWAM checkout 7faa71108368fbb3b6885649f112af607427a2d4
bash setup_env.sh
conda activate fastwam
```

`setup_env.sh` 创建或复用名为 `fastwam` 的 Python 3.10 环境，安装 PyTorch 2.7.1、Fast-WAM、LeRobot 2.1 写入器和离线测试依赖。Isaac Sim 仍使用教程现有的仿真环境，不与模型环境混装。

### 2.3 接入 G2 配置与运行目录

官方 Fast-WAM 训练脚本以 `FastWAM/` 为工作目录，只会按官方约定从 `FastWAM/configs`、`FastWAM/data` 和 `FastWAM/checkpoints` 查找配置、数据与初始权重。本教程则把 G2 任务代码和运行产物放在章节目录中，便于单独管理，因此需要在两种目录结构之间建立连接。

`link_dirs.sh` 会接入 G2 数据与训练配置、安装可选的 LoRA 扩展，并创建两个软链接：

```text
FastWAM/data        -> code/data
FastWAM/checkpoints -> code/checkpoints
```

软链接只是路径指向，不会复制数据或额外占用一份磁盘空间。脚本可重复运行；如果目标位置已存在普通目录，它会保留原目录并输出警告，不会覆盖。

训练、权重准备和文本缓存脚本都会自动调用 `link_dirs.sh`，通常无需手动执行。如需提前检查目录状态，可以单独运行：

```bash
bash link_dirs.sh
python -m pytest tests -q
```

`link_dirs.sh` 会校验 Fast-WAM commit；版本不一致时会直接停止，避免在未验证的上游接口上训练。

### 2.4 准备模型权重

```bash
conda activate fastwam
bash download_weights.sh
```

本步准备两部分权重：

- Wan2.2-TI2V-5B 视频骨干，位于 `code/weights/`；
- 由视频骨干插值得到的 Action DiT 初始化权重，位于 `code/checkpoints/`。

Action DiT 插值在 CPU 上执行，避免同时把完整视频骨干载入小显存 GPU。已有 Wan2.2 权重时可以用软链接复用。

## 第三部分 数据采集与转换

### 3.1 采集专家轨迹

先只查看采集配额：

```bash
${ISAACLAB_ROOT}/_isaac_sim/python.sh collect_data.py --plan-only
```

正式采集默认包含150条 clean 和1500条 randomized 成功轨迹，并按红、绿、蓝均衡分配：

```bash
${ISAACLAB_ROOT}/_isaac_sim/python.sh collect_data.py --headless
```

小规模验证可指定配额：

```bash
${ISAACLAB_ROOT}/_isaac_sim/python.sh collect_data.py \
  --clean-total 30 --randomized-total 60 --headless
```

采集支持断点续跑。失败轨迹会保留用于诊断，但转换时只接收成功且满足数据合同的轨迹。

### 3.2 转换并检查数据

```bash
conda activate fastwam
python convert_dataset.py
python inspect_data.py
python inspect_data.py --lerobot data/lerobot/g2_pick_block_lerobot
```

正式数据在官方读取器中的有效窗口为：

| 划分 | 样本窗口数 |
|---|---:|
| train | 377,223 |
| val | 3,927 |

单个样本的视频形状为 `[3,9,384,320]`，动作与状态分别为 `[32,16]`，文本缓存形状为 `[128,4096]`。

### 3.3 预计算文本表征

```bash
conda activate fastwam
bash precompute_text.sh
```

训练和部署均复用三条指令的 T5 context。推理服务默认读取预计算 context，避免 T5 与约6B参数的策略同时占用显存。

## 第四部分 模型训练

### 4.1 训练目标选择

Fast-WAM 始终学习动作预测，本章提供两种训练目标：

| 训练目标 | 视频损失权重 |
|---|---:|
| 加视频损失 | `lambda_video=1` |
| 不加视频损失 | `lambda_video=0` |

视频损失是训练阶段的辅助监督。开启后，模型在学习动作的同时，还通过未来帧学习场景的运动和交互变化；关闭后，模型结构、训练数据和动作目标都不变，只移除这项辅助监督，作为消融对照。

两种训练目标不需要分别维护脚本或配置文件，运行时直接通过 `model.loss.lambda_video` 参数切换。

### 4.2 全量微调

本章的正式实验采用8卡全量微调。下面依次说明参数更新范围、训练参数配置和启动方法。

#### 4.2.1 参数更新范围

全量微调会更新 MoT 和 G2 状态编码模块中的全部可训练参数：

| 模块 | 训练状态 |
|---|---|
| Video Expert | 全量更新 |
| Action Expert | 全量更新 |
| G2 proprio encoder | 全量更新 |
| 视频 VAE | 冻结 |
| T5 文本编码器 | 不加载，使用预计算文本表征 |

Video Expert 和 Action Expert 的注意力层、FFN 与归一化层都会参与训练，Action Expert 的动作输入输出层也会更新。可训练参数总数为 `6,020,761,808`。

#### 4.2.2 训练配置与启动

配置文件为 `configs/task/g2_uncond_3cam384_1e-4.yaml`，正式训练参数如下：

| 配置项 | 数值 |
|---|---:|
| GPU | 8×80GB |
| 每卡 batch | 8 |
| 梯度累积 | 2 |
| 有效全局 batch | 128 |
| 训练精度 | bf16 |
| 分布式策略 | DeepSpeed ZeRO-1 |
| 优化步 | 7,500 |
| 学习率 | `1e-4` |
| 学习率计划 | cosine |
| warmup | 375 step |
| 验证间隔 | 500 step |
| 保存间隔 | 2500 step |

正式训练前，先接入 G2 配置并运行 dryrun：

```bash
bash link_dirs.sh
cd FastWAM
python scripts/dryrun_fastwam.py task=g2_uncond_3cam384_1e-4
cd ..
```

根据4.1节选择训练目标：

```bash
# 加视频损失
bash train.sh 8 model.loss.lambda_video=1

# 不加视频损失
bash train.sh 8 model.loss.lambda_video=0
```

两种训练目标除视频损失权重外，其余训练参数完全一致。训练完成后，部署需要以下两个配套文件：

```text
FastWAM/runs/<run>/
├── checkpoints/weights/step_007500.pt
└── dataset_stats.json
```

权重和 `dataset_stats.json` 必须来自同一个 run。

### 4.3 LoRA 微调

LoRA 是没有多卡资源时的备选方案。它在双 expert 的注意力和 FFN 线性层中加入 rank-16 适配器，完整训练 G2 新增输入输出层，并用 int8 保存冻结基座以降低显存占用。

| 配置项 | 数值 |
|---|---:|
| GPU | 1×24GB |
| batch | 1 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0 |
| 冻结基座 | int8 per-output-channel |
| 优化步 | 50,000 |
| 学习率 | `1e-4` cosine |
| 验证/保存间隔 | 1000 / 5000 step |

`link_dirs.sh` 会自动接入 LoRA 模块和 trainer 补丁。选择训练目标后启动单卡训练：

```bash
# 加视频损失
bash train_lora.sh model.loss.lambda_video=1

# 不加视频损失
bash train_lora.sh model.loss.lambda_video=0
```

LoRA checkpoint 在部署前需要合并为普通 bf16 权重：

```bash
python merge_lora.py \
  --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_050000.pt \
  --output FastWAM/runs/<run>/checkpoints/weights/step_050000_merged.pt \
  --rank 16 --alpha 32
```

合并后的 checkpoint 可以直接交给 `serve_policy.py`，同时仍需使用同一 run 的 `dataset_stats.json`。LoRA 仅作为低显存训练路径，不参与后文实验指标对比。

## 第五部分 推理服务与闭环评测

### 5.1 推理模式选择

Fast-WAM 提供两种推理模式：

| 推理模式 | 参数 | 模型输出 |
|---|---|---|
| Action-only | `--inference-mode action` | 32步 Action Chunk，形状为 `[32,16]` |
| Joint | `--inference-mode joint` | 32步 Action Chunk和9帧预测视频 |

两种模式使用相同的 checkpoint 和当前三视角观测。Action-only 只计算动作；Joint 在计算动作的同时生成未来视频。后文的 Joint 可视化使用加视频损失训练的模型。

### 5.2 启动推理服务

Action-only 模式：

```bash
conda activate fastwam
python serve_policy.py \
  --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_007500.pt \
  --dataset-stats FastWAM/runs/<run>/dataset_stats.json \
  --inference-mode action \
  --measure-model-latency \
  --host 127.0.0.1 --port 8622 \
  2>&1 | tee results/latency_action.log
```

Joint 模式需要把 `--inference-mode` 改为 `joint`；如需保存预测视频，可同时加入保存参数：

```bash
python serve_policy.py \
  --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_007500.pt \
  --dataset-stats FastWAM/runs/<run>/dataset_stats.json \
  --inference-mode joint \
  --measure-model-latency \
  --save-predicted-video-dir results/predicted \
  --save-predicted-video-limit-per-prompt 20 \
  --host 127.0.0.1 --port 8623 \
  2>&1 | tee results/latency_joint.log
```

模型服务与 Isaac Sim 通过 NumPy-over-TCP 协议通信，两端可以使用不同的 Python 环境，也可以分别部署在远程 GPU 机器和本地仿真机器上。分机部署时，服务端改用 `--host 0.0.0.0`，评测端的 `--host` 填写模型服务器地址。

`--measure-model-latency` 会记录每次 Action Chunk 的模型推理时间。完成闭环评测后，丢弃前5次预热并计算算术平均值：

```bash
python summarize_latency.py results/latency_action.log \
  --mode action --warmup 5
# mode=action warmup=5 samples=49 mean_ms_per_chunk=326.8

python summarize_latency.py results/latency_joint.log \
  --mode joint --warmup 5
# mode=joint warmup=5 samples=1379 mean_ms_per_chunk=687.1
```

计时前后均执行 `torch.cuda.synchronize()`。Action-only 统计 `infer_action()`，Joint 统计包含视频生成的 `infer_joint()`；两者均不包含 RPC 通信、图像预处理、动作反归一化、仿真和动作执行。

### 5.3 闭环评测

本章采用统一的 RGB 扩大评测协议：

| 配置项 | 数值 |
|---|---:|
| 颜色 | 红、绿、蓝 |
| 每种颜色 | 50集 |
| 位置扰动 | ±0.01m |
| 单次预测 | 32步动作 |
| 单次执行 | 前24步 |
| 最大重规划 | 20次 |
| 推理期间 | 暂停仿真 |

启动任一种推理服务后，在 Isaac Sim 环境中运行：

```bash
${ISAACLAB_ROOT}/_isaac_sim/python.sh evaluate.py \
  --host 127.0.0.1 --port 8622 \
  --colors red green blue \
  --episodes-per-color 50 \
  --execute-steps 24 \
  --action-seconds 0.03333333333333333 \
  --position-noise 0.01 \
  --max-replans 20 \
  --pause-during-inference \
  --headless \
  --output results/eval_rgb50.json
```

评测端每执行24步就重新获取三视角观测并请求下一个 Action Chunk。测试 Joint 服务时，将端口改为该服务使用的端口。

### 5.4 实验结果

#### 5.4.1 试验指标

以下结果均来自 step7500，并使用5.3节的同一评测协议。

**推理模式对比：固定加视频损失**

| 推理模式 | 红色 | 绿色 | 蓝色 | 总成功率 | 平均延迟（ms/chunk） |
|---|---:|---:|---:|---:|---:|
| Action-only | 50/50 | 48/50 | 44/50 | 142/150（94.7%） | 326.8 |
| Joint | 50/50 | 46/50 | 46/50 | 142/150（94.7%） | 687.1 |

Action-only 与 Joint 的总成功率相同，Joint 的单次延迟约为 Action-only 的2.10倍。延迟按 5.2 节的方法在单张 NVIDIA H100 80GB 上测量，模型使用 BF16、10次去噪和 eager 模式。

**训练目标对比：固定 Action-only 推理**

| 训练目标 | 红色 | 绿色 | 蓝色 | 总成功率 |
|---|---:|---:|---:|---:|
| 加视频损失 | 50/50 | 48/50 | 44/50 | 142/150（94.7%） |
| 不加视频损失 | 49/50 | 46/50 | 42/50 | 137/150（91.3%） |

加视频损失的结果高3.3个百分点，但配对检验为 `p=0.267`，当前样本尚不能确认该差异具有统计显著性。

#### 5.4.2 可视化

下面是 step7500 Action-only 模式的一次成功执行。模型只输出动作，画面按照头部相机在上、两个腕部相机在下的方式排列。动画保持原始速度，完整时长为 31.3 秒。

![Fast-WAM Action-only 一次成功执行](assets/fastwam_action_only_success_blue.gif)

下面是 step7500 Joint 模式的一次成功执行。左侧为真实三视角，右侧为同一次推理生成的未来三视角。动画保持原始速度，完整时长为 31.3 秒。

![Fast-WAM Joint 真实执行与预测视频](assets/fastwam_joint_success_actual_vs_predicted.gif)
