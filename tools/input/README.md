# 上游预测输出桥

这些是已有的源格式导出工具，不属于三条精修求解器。

- genmo_export_smpl_canonical.py：GEM-SMPL hpe_results.pt + SMPLX_NEUTRAL → 55 节点 MotionClip；需要 GEM、smplx。
- genmo_export_soma_canonical.py：GEM-X SOMA 预测和 rig → MotionClip；需要匹配的源 rig/identity 数据。
- genmo_export_foot_surface.py：不可变 GEM-X 预测的足表面几何观察；需要 GEM-X SomaLayer。

在已有感知环境中运行各脚本的 `--help`，按显式参数提供 prediction、body model 和 output。导出的 canonical NPZ 通过 fullbody source manifest 进入统一 runner。视频估计与模型下载不在这个仓库的 run 命令内。
