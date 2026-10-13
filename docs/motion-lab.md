# arenalab Motion Lab

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

- Python 全套 42 项通过；前端状态 3 项通过；TypeScript/Vite 生产构建通过。
- 浏览器实测三页面真实模型加载与播放，Nymeria 动作拖入、替换及逐卡槽换机器人；轮式拖入第三动作；SOMA 实际转换结果回放。
- 实际 SOMA/G1 慢跑转换生成 147 帧、30 Hz，完成原 100+100 步管线。
- 上传测试验证模型导出、注册表重载、ZIP 路径越界拒绝和本地绝对 mesh 引用拒绝。
- 根坐标基回归用独立 MuJoCo FK 验证位置与朝向。

这些验证证明界面与调用链工作，不代表动作质量通过。当前库包含 37 条真实动作（31 SOMA、4 Nymeria、2 EgoDex），新增 SOMA 为源动作，首次拖入尚无缓存的机器人组合会发起转换，其中 Nymeria、Gento 为标明版本的历史结果，行走样本仅 10 帧；所有结果均显示未做动作质量验收。未运行动力学仿真。

## 品牌与元数据

界面统一英文，使用提供的 arenalab logo、papaya 橙 #FF8906 与炭黑底色。Metadata 接口支持任意 JSON 对象；面板动态渲染实际字段，嵌套对象、长列表可展开，缺失字段不显示。输入数据与机器人结果分开标示。新增源动作分类标注为 filename-derived；3 条未通过现有 FK/bind 检查的片段不加入动作库，原因保存在 .cache/motion-lab/catalog-skipped.json。没有为扩充目录放宽生产适配器检查。

## Full local library and dense browser

The sidebar now uses filename rows (29 px), no motion thumbnails or invented action names, and a desktop width of 411 px / 450 px (1.5 × previous). The scene headline and top playback control are removed. Playback remains on the bottom timeline. Orbit speed is 0.09 rad/s, twice the previous speed.

`prepare_motion_lab.py` now invokes the full local index after bootstrapping robot assets and historical cached excerpts. To rescan without rebuilding assets:

```powershell
python -X utf8 tools/index_motion_library.py --workspace C:/Users/Eric/Documents/Codex --roots D:/MyDatasets
```

No sample-count or duration cap is applied to the recording index. The current workspace contains 110 unique SOMA recordings, 10 Nymeria sequences and 594 EgoDex recordings, plus 9 explicitly marked cached excerpts. Indexing reads recording headers and stores original paths. It does not download the entire official remote datasets or validate/retarget every recording. New recording combinations convert on demand. Missing channels and unreadable files are reported in `.cache/motion-lab/library-index-report.json`; complete sources are preferred over shorter SOMA copies. SOMA families are filename-derived; EgoDex families use original task folders.

Nymeria can produce multiple independent windows. The viewer now retains all window outputs in chronological order with explicit frame boundaries; it does not smooth or claim continuity across those boundaries. Long recordings may require substantial conversion and playback memory. Native source files remain unchanged.

## Nymeria: 500 explicit source windows

The current localmanipulation page indexes 500 distinct, non-overlapping excerpts from 10 local recordings, plus 4 historical cached excerpts. New excerpts contain 450 frames at 30 Hz (15 s half-open interval; last frame 14.9667 s). The selector prioritizes 17 existing curated excerpts with verified Ego-video synchronization, then fills the remaining count by deterministic round-robin sampling. Existing narration annotations describe roughly 30-second intervals, not atomic activity boundaries: generic 15-second windows cannot be certified as complete activities. Metadata states the boundary policy explicitly. Files remain unchanged, and no robot quality filter is applied.

```powershell
python -X utf8 tools/prepare_nymeria_windows.py --count 500 --seconds 15 --stride 15
python -X utf8 tools/index_source_videos.py
```

Input windows are extracted on demand after the existing official Xsens clock correction. Positions, rotations and contact labels receive no additional spatial modification. Three actual input checks (indices 0, 249, 499) confirmed 450 finite endpoint frames and uniform 30 Hz timing; all 500 source ranges are unique and mutually non-overlapping. Robot trajectories are not precomputed for all 500 excerpts. SOMA corpus expansion and further EgoDex dataset expansion were stopped at the user's request.

## Source Ego videos

Enabling **Input traces** also displays a compact wrist telemetry panel below each slot's Ego video, including slots without source video. Two traces show left/right position error (mm); a second row shows orientation geodesic error (degrees). Current values and a cursor follow the shared 30 Hz timeline, clamping to each clip's final frame. Dashed references are 10 mm and 15 degrees, display references only, never acceptance gates. Solid 3D traces are inputs, dashed traces are achieved wrist paths; connecting segments show current-frame position error. Telemetry is recomputed from final native MuJoCo FK using Nymeria's descriptor wrist frames or Gento's declared TCP offset and mounting rotation. Missing historical endpoint contracts explicitly show unavailable metrics, not zero. It does not certify the input tracking calibration or dynamic feasibility. Segment boundaries retain unsmoothed samples.

Library presentation uses two levels: readable activity families and subclasses, retaining each original clip filename. EgoDex task identifiers are humanized and grouped by task semantics; SOMA categories remain filename-derived. Nymeria uses local dataset scenario metadata for generic session windows, and existing curated action labels only for windows with a verified source-video mapping. The category origin is shown in metadata; recording scenarios are not claimed to be per-frame annotations. Search includes both hierarchy levels.

Uploaded robot choices collapse only packages with the same canonical physical URDF and identical referenced mesh bytes. The usable upload is preferred; differing geometry, missing evidence and distinct versions remain separate. All registry records and existing slot references remain intact. To refresh the presentation metadata, run `tools/organize_motion_library.py --annotations <local annotation directory> --curated <local calibrated sources directory>` from the repository root, then restart the viewer. This tool updates catalog/registry metadata without changing motion files or solver policy.

Both manipulation pages display three numbered source-video tiles, one per scene slot. Clips start together on the shared relative timeline, regardless of their original recording offsets. Each video follows its own clip mapping. Short clips stop at their final frame together with their robot; replacement changes only that slot's source binding. Click a tile to enlarge it; pause, frame steps, timeline seeks and speed remain shared. Videos remain muted. During conversion the new video does not play against the previous robot result.

The current catalog has verified media bindings for 297 EgoDex entries (including both historical excerpts) and 17 Nymeria windows. Missing videos are explicitly marked unavailable. EgoDex copies/prefixes are matched against original camera and both wrist pose arrays, then checked against paired video frame counts and 30 Hz cadence. Nymeria uses existing verified RGB device-timestamp receipts plus Xsens timecode mappings and a per-output-frame video time lookup. No filename-only timestamp guess is accepted. This is not full video coverage of the 500 excerpts.

Source media is served only by configured clip ID, with byte-range seeking. Original mp4v videos are not Chromium compatible: the server generates a local H.264 preview cache on first use, preserving cadence and timestamps, without overwriting originals. First load may take a few seconds. Repeated slots reuse that cache. Cache path and raw source paths are omitted from the public video descriptor.

Validation: 45 Python tests pass, including metadata isolation, source-range extraction, export basis, multi-window joining and video byte-range serving. Frontend TypeScript/Vite build passes. Browser verification covered source-video playback, pause, next frame, enlargement, three slots with two different sources, and the short clip stopping at 4.9667 s while the long clips reach 14.9667 s. This verifies viewer binding and timing, not robot motion quality or physics.

Cloud deployments may set `conversion_devices` in the viewer config to GPU index strings, for example `["0", "1", "2", "3"]`. Each GPU runs one conversion subprocess with its own CUDA visibility; excess requests remain queued. The default remains one worker. This changes scheduling only; conversion policy and refinement budgets stay unchanged.
