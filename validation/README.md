# 整合验证记录

本次验证确认目录迁移、模型调用、输入契约、保存与独立 FK 可以工作；没有重新做全量动作验收、仿真学习或 zero-shot 泛化认证。

## 自动测试

`python -m pytest -q`：34 项通过。覆盖现有 SDK 的 SMPL/SOMA/BVH/canonical 适配、source manifest、contact/surface/compact 几何，以及新增 job 相对路径、30Hz 限制、非法机械行为配置、源旋转验证、窗口尾帧、结果有限值、两份 Gento 机械模型、混合关节 FK、失败留存、禁止复用输出目录、自定义验收不得改轨迹。

`math-migration.json`：Nymeria V3 fit、Gento source/unit/metadata/FK 的函数 AST 与原源码一致。路径、入口和中间文件契约的修改单独记录在 provenance/source-files.json。这个检查不声称整个新旧系统逐位等价。

## 实际模型短样本

使用本机已有 Prime checkpoint，统一 30Hz，通过新 CLI 实际运行：

|路线|输入|机器人|保存帧数|结果|
|---|---|---|---|---|
|fullbody|真实 SOMA 步行 BVH 的开头片段|H2、G1|各 10 帧|生成 prior、最终轨迹和报告|
|nymeria|真实 Xsens 记录开头 30 个原始样本，官方修复时钟后下采样|G1|4 帧|生成 prior 和 V3 结果；达到 1200 次上限，不标质量通过|
|gento|真实 EgoDex 擦拭 HDF5 前 6 帧|Skye、Luna|各 6 帧|生成 prior、qpos、TCP 和报告|

Gento 保存 qpos 的独立 MuJoCo FK 与导出 TCP 最大差值为 0m，底盘变化为 0。所有索引中 artifact 的 SHA256 重新核验一致。各路线最终 quality_status 仍为 not_evaluated。

`smoke-results.json` 保存结构化结果。原始短样本与运行输出只在本地 runs/，不进 Git。SMPL 格式由自动测试覆盖；本次没有把原生 SMPL 数据库的全流程质量验收算作已完成。

这些片段很短，只能用来排查迁移错误，不能判断长期连续性、碰撞、跳跃动力学、完整操作精度或机器人泛化。

## 复现

```bash
python -m pip install -e ".[runtime,test]"
python -m pytest -q
python -m greenwich_pipelines doctor your_job.json
python -m greenwich_pipelines run your_job.json
```

需要提前配置本地模型、语义缓存、输入数据与机器人资源。运行已有目录会被拒绝，复现时使用新的 output。
