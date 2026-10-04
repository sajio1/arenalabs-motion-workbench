# Greenwich Motion Pipelines

一个本地 repo，三条生产路线：SOMA/SMPL 全身转换、Nymeria 双腕双踝补全、Gento 轮式上身适配。Greenwich 是自研模型。本仓库整合现有调用与依赖，不重新训练模型，也不声称此次迁移修好了旧动作质量问题。

## 快速开始

Python 3.10+，完整运行需 PyTorch、MuJoCo 和本地 Prime checkpoint。全身后端目前要求 CUDA。建议独立虚拟环境：

```bash
python -m pip install -e ".[runtime,test]"
python -m greenwich_pipelines list
python -m greenwich_pipelines plan examples/gento/job.json
python -m greenwich_pipelines doctor examples/gento/job.json
python -m greenwich_pipelines run examples/gento/job.json
```

先修改例子的输入路径和 runtime 配置；`*.local.json` 不进入 Git。checkpoint 目录需要 `config.json`、`model.safetensors`。语义编码模型需提前放入 Hugging Face 缓存；运行默认离线，不自动下载或发布。示例机器人掌框架必须用实际机械元数据替换，不能把模板向量当作校准结果。

## 目录

```text
src/greenwich_pipelines/
  config.py                 # 统一 job 与相对路径解析
  cli.py / worker.py        # plan、doctor、隔离进程、状态和结果
  runtime.py                # 共享本地 Prime checkpoint
  artifacts.py              # 原始编码的结果索引和 SHA256
  pipelines/
    fullbody/               # SOMA/SMPL/BVH/canonical；100+100
    nymeria/                # 原始 Xsens / 四端点；V3 物理精修
    gento/                  # EgoDex 双腕+头参考；混合关节精修
src/alphamotion/             # 模型推理核心与机械描述器
src/greenwich_motion_sdk/    # 全身输入格式、投影、协调与几何模块
src/greenwich_umi_proof/     # 共用足底、掌框架、原生几何工具
assets/robots/gento/         # 两个已接入机器人模型、TCP 与 vendor 网格
configs/                    # runtime、机械 roster、公共策略
examples/                   # 三条 job、输入清单和可选验收接口
docs/                       # 架构、输入输出、迁移与已知边界
provenance/                  # 迁移来源和原文件哈希
tests/                      # 格式、契约、数值迁移回归
validation/                 # 本次整合验证报告
```

## 三条流程

|流程|输入|生产算法|原生输出|
|---|---|---|---|
|fullbody|SOMA BVH+bind；SMPL poses/trans+shaped skeleton；canonical NPZ|全身编码→Prime 解码→机械投影→接触 100→协调 100|Y-up/cm；q、rot6d、root_t、源动作与几何报告|
|nymeria|Xsens xdata 或明确的四端点 NPZ，含 heel/toe 接触|四端点 mask→Prime 补全→V3 q/root 联合精修，最大 1200 次|Y-up/cm；原始模型 prior、关节轨迹、接触与报告|
|gento|EgoDex HDF5，双腕、掌框架地标、camera/hip/Arm|三节点 mask→Prime→真实 TCP、躯干和滑轨时序求解|Z-up/m；MuJoCo qpos、实际 TCP、目标与报告|

只有 30 Hz。两个 100 步只属于 fullbody。Nymeria 默认 900 帧独立窗口，不承诺窗口接缝连续。Gento 是固定底盘上身适配，不包含轮式行驶规划。手指地标只用于定义腕朝向，不生成手指动作。

## 结果与 Motion Lab

每个任务有独立输出目录。`result.json` 是未来网页读取的统一索引：相对 artifact 路径、机器人、片段、帧数、坐标轴、单位、关节编码和哈希。原生 NPZ 保持原编码，不能把不同管线的 `q`/`qpos` 混读。

原始模型输出保存为 `prime_original.npz`，精修输出单独保存。任务目录已存在会拒绝运行；失败保留中间结果与 `status.json`。代码不会覆盖之前的模型 prior 或拿前一个精修结果代替 prior。

默认 `accepted=null`、`quality_status=not_evaluated`。原求解器历史报告中的 `accepted=false` 不会被整合层改成通过。可选 `acceptance: "your_module:evaluate"` 接收最终 result.json，并返回 `{accepted: bool, ...}`。未启用严格 mean/P95 指标，也没有新增早停或总 loss 门槛。结构检查、有限值检查、Gento 独立 FK 一致性检查不代表运动质量认证。

已加入本地 Motion Lab 网页：三个独立页面、共享三卡槽 3D 场景、动作拖入与替换、逐卡槽机器人选择、URDF/ZIP 上传检查。启动与边界见 [Motion Lab](docs/motion-lab.md)。

阅读顺序：[架构](docs/architecture.md) → [输入输出契约](docs/contracts.md) → [迁移范围与限制](docs/migration.md) → [验证报告](validation/README.md)。
