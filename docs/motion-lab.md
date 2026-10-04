# Motion Lab

本地 Three.js + FastAPI 网页，直接调用仓库内三条管线。没有修改模型、精修权重或生产轨迹。

## 启动

```powershell
python -m pip install -e ".[runtime,test,viewer]"
cd web
npm.cmd ci
npm.cmd run build
cd ..
python -X utf8 tools/prepare_motion_lab.py --workspace C:/Users/Eric/Documents/Codex
python -X utf8 -m greenwich_pipelines.viewer.server --config configs/viewer.local.json
```

准备脚本从已有本地工作区生成样本目录、真实模型 GLB 和逐帧回放。所依赖的历史动作、机器人资产和 checkpoint 不包含在 Git 中；换机器需要配置这些路径。私有配置、缓存、任务结果与 node_modules 均被忽略。服务只监听 127.0.0.1:8765。

## 页面与操作

|URL|数据|预设机器人|
|---|---|---|
|/locomotion|SOMA 全身动作|G1、H2|
|/localmanipulation|Nymeria 双腕双脚|G1、H2|
|/manipulation|EgoDex 上身适配|Gento Skye、Luna|

左上切换独立页面；左栏按 family 分类，可搜索，下面显示 metadata。拖动作至空卡槽添加，拖至已有卡槽替换；最多三个机器人共用一个场景。双击动作也可添加。机器人、动作和时间状态按页面保存在当前浏览器会话。

左下机器人框向右展开：未选择卡槽时设置新动作的默认机器人；选择卡槽后仅切换该卡槽机器人。点击场景机器人或卡槽标签选择卡槽。返回默认选择按钮退出卡槽选择。

时间轴统一按秒播放，逐帧步进为 30 Hz；短动作到末帧停止，整组按最长动作循环。支持速度、网格、骨架、输入轨迹和线框显示。自动旋转与动作播放独立控制；拖放或拖动视角会暂停自动旋转。

## 转换与上传

有缓存的组合直接回放；无缓存的已配置机器人组合排入单工作线程队列，通过独立 Python 进程调用原生产管线。任务状态可查询，完整输出保留在 .cache/motion-lab/jobs。当前运行完成的组合可在同一服务进程中复用；新增任务缓存索引暂不跨服务重启恢复。历史预置缓存由本地配置载入。

上传支持单个 URDF 或带 meshes 的 ZIP，解析模型、检查资源引用并生成真实缩略图。上传机器人会保存于本地注册表。URDF 能加载不代表能转换：还需要端点、TCP/掌框架、闭链等机械元数据；当前上传后默认 needs_metadata，禁止直接发起动作转换。元数据接入界面尚未实现。

回放统一物理尺度、Y-up/m。全身编码通过原生 FK 还原，根旋转做坐标基共轭；轮式 qpos 通过 MuJoCo FK 还原。不在 viewer 中补抬 root、缩放机器人或修正动作。卡槽只添加恒定水平展示偏移。

## 本次验证

- Python 全套 41 项通过；前端状态 3 项通过；TypeScript/Vite 生产构建通过。
- 浏览器实测三页面真实模型加载与播放，Nymeria 动作拖入、替换及逐卡槽换机器人；轮式拖入第三动作；SOMA 实际转换结果回放。
- 实际 SOMA/G1 慢跑转换生成 147 帧、30 Hz，完成原 100+100 步管线。
- 上传测试验证模型导出、注册表重载、ZIP 路径越界拒绝和本地绝对 mesh 引用拒绝。
- 根坐标基回归用独立 MuJoCo FK 验证位置与朝向。

这些验证证明界面与调用链工作，不代表动作质量通过。初始库包含 10 条抽样动作，其中 Nymeria、Gento 为标明版本的历史结果，行走样本仅 10 帧；所有结果均显示未做动作质量验收。未运行动力学仿真。
