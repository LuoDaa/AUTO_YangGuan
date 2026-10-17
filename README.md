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
9. 完整天然牙模板先裁出 crown prior，再规范到同一 canonical frame；
10. 模板仅做温和 X/Y 尺寸适配；
11. 自动提取近中 / 远中参考 patch，做小范围 proximal grow / projection-back；
12. 只处理明显对颌干涉；
13. 输出阶段 STL、定位报告和工程 QC。

## 重要边界

当前自动 Margin 和 insertion axis 是 **工程自动候选**，前端必须视觉复核；它们不是临床最终确认线。当前正式 intaglio / 制造级 cutter 尚未接入这个独立外冠 MVP。

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

定位可接受后再点：

```text
2. 牙冠生成
```

设计页可以开关：

- 目标颌；
- 对颌；
- 基台；
- 近中 / 远中参考 patch；
- Margin；
- XYZ / 插入轴；
- 黄色自动外冠。

并可导出 `Outer STL`、打开 `localization.json` 和 `report.json`。

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
→ crown prior
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
02_mesial.stl               # 阶段实际输出时
03_distal.stl               # 阶段实际输出时
04_occlusal.stl             # 阶段实际输出时
05_outer_candidate.stl
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

测试包含 synthetic geometry regression，并包含仓库真实 `基台测试2-牙位36` 的定位回归。

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

`autocrown intaglio-preview` 目前仍是诊断预览，不是制造级 cutter。正式内冠后续需要独立完成：

1. insertion-axis directional blockout；
2. cement-gap field；
3. clinical face ownership；
4. support coverage validation；
5. Margin-owned clipping；
6. deterministic cervical collar / final assembly。

当前阶段先把 **“基台位置正确 → 插入轴正确 → 模板方向正确 → 保守外冠稳定”** 这条链跑通。
