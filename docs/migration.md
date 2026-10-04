# 迁移范围

本次整合三条实际生产调用链。原文件、原 SHA256、迁入文件和迁移说明在 provenance/source-files.json。原目录保留，本仓库是独立副本，不依赖旧聊天目录的 Python import。

迁入：SDK 格式/几何模块、全身 staged backend、Nymeria corpus completion 与 V3 physical solver、Gento prepare/FK/arm-priority recipe、共享 Greenwich 推理源码、固定机器人元数据和所需几何。

没有迁入：云下载/上传 worker、恢复远程批量任务、HF 发布逻辑、旧动作挑选/机器人专用演示入口、V22/V3 的其他实验求解分支、独立 EgoDex 人形修法、新 Viser 网页。一次性脚本的来源清单另存 docs/legacy-script-inventory.csv，便于后续有目标地移入工具，不把未审查实验放进生产 imports。

`tools/input/` 另收录现有 GEM-SMPL、GEM-X SOMA 到 canonical MotionClip 的导出桥，以及源足表面导出工具。它们需要外部 GEM/GEM-X 环境、对应人体模型和已完成的 prediction；不在普通 job 中启动视频重建，也不隐式安装感知模型。

代码移除了个人机器默认路径和历史 refined baseline 回退。Gento 历史 source/preview 模块只提取源适配、元数据和 FK；batch_five 只提取 prepare。Nymeria 旧 latent 模块只提取 V3 实际使用的 PhysicalFK 和公共数值策略，未把旧 latent 求解器带进来。

模型 weights 与人形 XML/网格仍是显式外部资源，不进 Git；本机 local 配置被忽略。将仓库移到另一台机器须配置这些资源和依赖缓存。示例配置中的掌轴仅示意，不能替代校准。

当前边界：没有新增质量算法。保留旧版本的滑移、朝向、碰撞、root 动力学等已知风险；Gento 固定底盘，Nymeria 独立窗口。小样本迁移验证不等于任意动作/机器人的泛化验收。原 V3/Gento accepted=false 保留，不因统一目录而转为通过。

全身原始解码结果新增独立 prime_original.npz；精修数学不变。默认不启用早停、loss 上限或严格均值/P95 验收。当前采用的数值配方与后续质量改进须分别提交和验证。
