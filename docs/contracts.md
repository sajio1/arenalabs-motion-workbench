# 输入与输出契约

## Job

`schema=greenwich.job.v1`，`pipeline=fullbody|nymeria|gento`，`id` 为安全文件名。source、robots、runtime、output 路径均相对 job 文件解析。30Hz 固定；robots 不允许带 solver/loss/seed/平滑等行为参数。fullbody 可显式限制 max_seconds，默认全长。Nymeria window_frames 默认 900。Gento 保留完整记录。

## Fullbody

source 是 `alphamotion.sources.v1` 清单。SOMA 要同骨架 bind；SMPL 需要 poses/trans、names、parents、shaped rest_offsets、pelvis_offset_source_units 和 frame。源 trans+shaped J0 定义骨盆，不能替换成机器人身高。角色为 root/head/双腕/双踝，另有 ankle/toe 配对。详情见 [原有格式契约](fullbody-formats.md)。本仓库不包含视频人体重建模型；视频重建后的 SMPL/SOMA 输出从这层进入。

## Nymeria

原始 xdata 必须包含 segment_tXYZ、segment_qWXYZ、tpose_segment_tXYZ、tpose_segment_qWXYZ、timestamps_us、foot_contacts。23 段索引采用数据集固定定义，位置 m/Z-up，四元数 WXYZ。使用上游原始 clock repair，再取最近源行到均匀 30Hz；不重排空间样本。

也可提供预处理 NPZ：position_yup_cm `(T,4,3)`、semantic_rotation_yup `(T,4,3,3)`、foot_contacts `(T,4)`、fps=30。端点顺序左腕/右腕/左踝/右踝；接触顺序左 heel/toe、右 heel/toe。这里的 rotation 是明确的掌/足语义框架，不是未校准的 Xsens 段旋转。可带均匀 sample_timestamp_us；原始最近采样行时间与均匀目标时间分别保留。

机器人提供 palm_frames（vendor 局部 distal/normal）和可选 endpoints 精确关节名。缺失元数据应显式报错，不按动作猜手腕朝向。

## Gento

输入是 30Hz EgoDex HDF5 的 transforms：camera、hip、left/right Hand、MiddleFingerKnuckle、IndexFingerKnuckle、LittleFingerKnuckle、Arm。所有 transform 为 `(T,4,4)`。指根仅确定腕语义框架。

metadata 包含 model、source URDF、wrists、torso、end_effector_frames、物理关节 limits 等；路径相对 metadata 文件。实际机器人需带一个自由 chassis root 和其后的单自由度 hinge/slide 链；不支持的机械结构应扩展通用机械实现，不按机器人名字绕过。

## Result

`result.json` 使用 `greenwich.result.v1`。artifact.path 相对输出目录，可作为后续 Viser 的读取契约。kind 是 motion/model_prior/source/report。坐标、单位、joint_encoding 显式声明；fullbody/Nymeria 是 Y-up cm 与 descriptor angle slots，Gento 是 Z-up m 与 MuJoCo qpos，自由根四元数 WXYZ。保存哈希用于诊断与来源追踪，不是质量证书。

验收模块接收 result.json 的 Path，返回含布尔 accepted 的 dict。模块不得更改输出轨迹；整合层会在验收后重新核验 artifact 哈希。默认不加载验收模块。
