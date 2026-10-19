# AUTO_YangGuan

独立的保守型牙冠 CAD MVP。目标不是一次做到高分临床设计，而是先快速、稳定地产生 **20–40 分、方向正确、可复核的外冠候选**。

本仓库与 `LuoDaa/CrownFlow` 独立；不导入、不修改 CrownFlow。当前前端流程为：

```text
首页
  ↓
订单 / 定位
  ↓
基台定位 → Margin → 插入轴 → PatientFrame 复核
  ↓
牙冠生成
  ↓
自动设计 / 导出
```

## 当前定位与生成策略

真实病例不会再使用旧的 `parts[1] + PCA axis + 圆形假 Margin` 快速代理。当前主路径是：

1. 根据 FDI 确定工作颌和对颌；
2. 对工作颌 connected components 做物理尺度 / 面积 / 与牙列关系分析，定位真正基台支撑；
3. 结合对颌方向和基台侧壁法向得到 rough axis；
4. 在基台表面提取 3D finish-line / Margin 工程候选；
5. 由 Margin 稳健平面法向 + 基台侧壁法向求初始插入轴；
6. 在约 ±20° 锥域内按倒凹比例、拔模严重度和偏角求最终 insertion axis；
7. 用 final axis 再 refine Margin，并再次求轴；
8. 建立右手 PatientFrame：
   - 原点 = Margin 中心；
   - `Z` = 插入轴，并定向到对颌；
   - `Y` = 牙弓外向 / 颊侧；
   - `X` = 右手牙弓切向；
   - 解剖近中侧由 FDI quadrant 映射为 `X` 的正/负方向，不再为了“近中=+X”把颊舌方向整体翻转；
9. 利用已注册的 32 牙位牙库/对颌方向规范模板长轴，再在 STL 三角面层面裁出有且只有一条完整颈环的 crown prior；若未证明环拓扑，明确进入复核而不声称成功；
10. 模板仅做温和 X/Y 尺寸适配；利用邻牙的真实咬合高度进行有界冠高调整；
11. 自动提取近中 / 远中参考 patch，做小范围 proximal grow / projection-back；
12. 根据患者坐标系裁出目标冠附近的真实对颌 ROI，只在该区域进行局部咬合干涉修整；不对整个牙弓做 Boolean；
13. 每个 HEIGHT/邻接/咬合阶段新增独立的顶点、颈缘冻结、三角翻面和变形幅度 gate；不能证明安全时退让或回滚到上一阶段；
14. 生成定向 blockout 和 variable-gap 工程 cutter（单独产物，错误不影响外冠）；
15. 输出阶段 STL、定位报告、每步 geometry_guard、来源绑定信息和工程 QC。

## 重要边界

当前自动 Margin 和 insertion axis 是 **工程自动候选**，前端必须视觉复核；它们不是临床最终确认线。**现在生成接口要求先点击“我已在 3D 核对基台、Margin 与插入轴”**，且这个确认与本次输入 STL/模板和定位结果的摘要绑定，重新定位会撤销旧确认。

**本仓库仍是工程测试平台**：已从 CrownFlow 移植 insertion-axis directional blockout + variable cement-gap 核心，可以独立生成封闭的 engineering cutter；但 PreparedSupport clinical ownership、128 点共享 Margin cervical collar、最终内外冠封闭组装与制造 QC 仍未实现完整的临床可验证链路。Cutter 不能称作 final intaglio，不能将 preview 或开放冠底当作可制造内冠，也不能将通过 CI 等同于临床验收。

## 最简单的本地测试

### Windows

```powershell
git pull
.\run_dev.ps1
```

如果 PowerShell 阻止脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dev.ps1
```

脚本会：

- 创建 / 复用 `.venv`；
- 安装 Python 后端依赖；
- 安装前端 npm 依赖；
- 启动 FastAPI `http://127.0.0.1:8000`；
- 启动 Vite 前端 `http://127.0.0.1:5173`。

浏览器打开：

```text
http://127.0.0.1:5173
```

### macOS / Linux

```bash
git pull
chmod +x run_dev.sh
./run_dev.sh
```

## 前端怎么测

### 方式 A：仓库真实测试病例

首页直接读取 `testdata/1.0测试数据/`：

- 基台测试1-牙位46
- 基台测试2-牙位36
- 基台测试3-牙位46
- 基台测试4-牙位26
- 基台测试5-牙位26

建议第一例先跑：

```text
基台测试2-牙位36
```

进入订单后不要立刻生成。先点：

```text
1. 基台定位 / Margin / 插入轴
```

左侧 3D 需要先检查四件事：

- **棕色基台**：必须是真正目标支撑，而不是邻牙 / 扫描碎片；
- **绿色 Margin**：必须围绕支撑 finish-line；
- **紫色 Z**：从 Margin 朝对颌，表示 insertion / occlusal direction；
- **蓝色 Y**：应朝牙弓外侧 / 颊侧；红色 X 是右手牙弓切向，近中侧由 FDI 自动映射。

定位可接受后，先点 **“我已在 3D 核对基台、Margin 与插入轴”**；这一步保存针对当前定位和源文件的确认。然后再点：

```text
2. 牙冠生成
```

任何新的定位结果都会重置确认；源码和文件变化不会沿用旧定位进行生成。

设计页可以开关：

- 目标颌；
- 对颌；
- 基台；
- 近中 / 远中参考 patch；
- Margin；
- XYZ / 插入轴；
- 黄色自动外冠；
- 青色工程 blockout 与 cutter（均默认关闭，非最终内冠）；
- 初始摆位、冠高适配、近中、远中与咬合后各阶段 STL（默认关闭，勾选可单独比较）。

设计页会显示每个形变阶段的 `geometry_guard` 状态与比例；`OUTER_REVIEW_REQUIRED` 表示能看工程候选，但**不代表通过制造级 QC**。

并可导出 `Outer STL`、打开 `localization.json` 和 `report.json`。工程 cutter 构建成功时还可导出 `06_cutter.stl` 并查看 `06_intaglio_report.json`，且这些文件不代表最终冠。

### 方式 B：上传自己的上下颌

1. 输入目标 FDI；
2. 选择上颌 STL；
3. 选择下颌 STL；
4. 创建订单；
5. 执行“基台定位 / Margin / 插入轴”；
6. 视觉复核；
7. 再执行“牙冠生成”。

如果工作颌里没有可分离的目标支撑 component，当前自动定位会明确失败，不会再偷偷拿第二大 component 当基台。

## 模板库

仓库包含：

```text
templates/32牙位基础解剖模板库/32位牙列-分/FDI_11.stl ... FDI_48.stl
```

模板是完整天然牙模型。当前会先：

```text
完整天然牙
→ 长轴与冠根方向分析
→ radius-vs-axis neck profile 找 cervical cut
→ 三角级连续裁切 + 单一开口颈环/绕向/组件验证
→ crown prior（不能证明时仍可留诊断候选，但必须 REVIEW_REQUIRED）
→ library arch radial 定颊侧
→ canonical frame
```

再进入患者放置。牙根不会直接参与外冠形变。

## 调试输出

每笔病例写到：

```text
work/<case_id>/output/
```

定位阶段：

```text
00_support_component.stl
00_target_jaw_component.stl
00_margin.json
00_localization.json
```

生成阶段：

```text
00_template_canonical.stl
00_neighbor_mesial.stl      # 成功提取时
00_neighbor_distal.stl      # 成功提取时
01_placed.stl
01_height_fit.stl
02_mesial.stl               # 阶段实际输出时
03_distal.stl               # 阶段实际输出时
04_occlusal.stl             # 阶段实际输出时
05_outer_candidate.stl
06_blockout.stl             # 工程 blockout 成功时
06_cutter.stl               # 工程 variable-gap cutter 成功时
06_intaglio_report.json     # W3 独立阶段报告；失败也会留存
report.json
```

因此可以明确区分失败来自：基台定位、Margin、插入轴、PatientFrame、模板规范化、邻接、对颌还是最终 QC。

## 自动测试

```bash
pip install -e ".[dev]"
pytest
```

CI 还会执行：

```bash
cd frontend
npm install
npm run build
```

测试包含 synthetic cervical/局部对颌/形变保护回归、完整牙库 FDI16/26/36/46 规范化回归，以及仓库真实 `基台测试2-牙位36` 的定位回归。CI 目前验证的是程序/几何约束，不代表整套真实病例的制造级内外冠验收。

## CLI

标准输入 CLI 仍保留：

```bash
autocrown outer \
  --template template.stl \
  --margin margin.json \
  --frame frame.json \
  --mesial mesial.stl \
  --distal distal.stl \
  --antagonist antagonist.stl \
  --out output_case
```

## Intaglio 状态

`autocrown intaglio-preview` 仍是旧的 support-normal-offset 诊断预览；真实病例的 `generate` 现在新增一条**独立的 CrownFlow-derived W3 工程路径**：

1. 使用已定位基台和 insertion axis 构造 directional shadow/height-field blockout；
2. 根据参数构造 variable cement-gap cutter，默认 `cement_gap_mm=0.06`、`cement_gap_start_mm=0.10`、5轴工程 profile；未输入 top gap 不猜值；
3. 只在 blockout 和 cutter 通过单连通、水密及绕向检查时导出 STL；
4. 若几何失败，只记录 `06_intaglio_report.json` 并删除旧 cutter；**已生成外冠不受影响**。

仍须实现并验证：临床 support ownership / gap regions、正式 Margin-owned clipping、双表面无穿插、128 点共享 cervical collar、最终 watertight crown、真实厚度与就位 QC。**工程 cutter 不是 final crown。**

移植源：`CrownFlow/backend/app/geometry/intaglio.py`，在 `src/autocrown/crownflow_intaglio_core.py` 中保留 A/B 两段核心；仅适配独立仓库的 insertion-axis 工具和参数入口。

当前阶段先把 **“基台位置正确 → 插入轴正确 → 模板方向正确 → 保守外冠稳定”** 这条链跑通。
