# Fast-WAM G2 三色物块代码

本目录提供 G2 Omnipicker 三色物块入盒任务的完整 Fast-WAM 复现流程。运行时生成的数据、权重、官方源码副本和评测结果不纳入教程目录。

| 模块 | 文件 | 作用 |
|---|---|---|
| 统一配置 | `settings.py` | 路径、三相机布局、16维动作接口和时序参数 |
| 数据采集 | `collect_data.py` | 采集 RGB 三色物块专家轨迹 |
| 数据转换 | `convert_dataset.py` | 将 NPZ 转换为 LeRobot 2.1 |
| 数据检查 | `inspect_data.py` | 校验轨迹并输出数据概览 |
| 图像布局 | `image_layout.py` | 将头部和双腕相机拼成 384×320 输入 |
| 训练配置 | `configs/` | G2 数据、全量与 LoRA 配置，支持切换视频损失 |
| 环境安装 | `setup_env.sh` | 安装官方 Fast-WAM 训练环境 |
| 权重准备 | `download_weights.sh` | 准备 Wan2.2 与 Action DiT 初始化权重 |
| 文本缓存 | `precompute_text.sh` | 预计算三条任务指令的 T5 表征 |
| 全量微调 | `train.sh` | 8卡 DeepSpeed ZeRO-1 训练入口 |
| LoRA 微调 | `train_lora.sh` | 单卡 int8 基座 + rank-16 LoRA 训练入口 |
| LoRA 权重合并 | `merge_lora.py` | 生成标准部署 checkpoint |
| 推理封装 | `fastwam_runtime.py` | 三相机预处理、归一化和动作块推理 |
| 模型服务 | `serve_policy.py` | 启动动作或 joint video+action RPC 服务 |
| 闭环评测 | `evaluate.py` | 在 Isaac Sim 中执行 RGB 闭环评测 |
| 延迟统计 | `summarize_latency.py` | 统计服务日志中的单次 Action Chunk 平均推理延迟 |
| 执行控制 | `control.py` | 关节限位和可选抓取保护 |
| 仿真任务 | `task_runtime/` | G2 场景、机器人、相机和成功判据 |
| 离线测试 | `tests/test_offline.py` | 检查数据合同与三相机布局 |

运行前克隆官方仓库：

```bash
git clone https://github.com/yuantianyuan01/FastWAM.git FastWAM
git -C FastWAM checkout 7faa71108368fbb3b6885649f112af607427a2d4
bash setup_env.sh
```

G2 USD 不随代码仓库分发。在 `hello-robotics` 根目录按教程下载后，默认路径为 `code/assets/robot/G2_omnipicker/robot.usda`。

完整环境配置、训练命令、评测协议与本章实测结果见 [`../docs/Fast-WAM 世界动作模型复现与部署.md`](../docs/Fast-WAM%20世界动作模型复现与部署.md)。

如资产位于其他目录，运行前通过 `G2_ASSETS_ROOT` 指定资产根目录。
