# AUTO_YangGuan

独立的保守型牙冠 CAD MVP，目标不是一次做到高分临床设计，而是先快速、稳定地产生 **20–40 分可检查外冠候选**。

本仓库与 `LuoDaa/CrownFlow` 独立；不导入、不修改 CrownFlow。当前本地测试界面按 CrownFlow 基台快速流程重新组织：

```text
首页 → 订单 → 自动设计 / 导出
```

## 当前 1.0 外冠策略

1. 根据 FDI 自动选择基础解剖模板；
2. 对完整天然牙模板先做快速 crown-prior 裁取与规范化；
3. 从整颌病例中分离目标颌主体与基台 component；
4. 建立快速 PatientFrame / Margin 代理；
5. 模板轻微缩放并摆放；
6. 仅对明显对颌穿透做局部回退；
7. 保留模板形态，输出阶段 STL 与 QC；
8. 在 3D 前端中叠加目标颌、对颌、基台和生成外冠检查。

当前 raw-case adapter 是为了尽快验证“保形模板路线”是否值得继续，并不是最终临床定位模块。当前明确限制：

- Margin 仍是快速代理线，不是正式临床 Margin；
- 整颌自动近中/远中分牙尚未接入，所以当前病例不会做正式 proximal solver；
- 内冠还没有接入这条前端生成链；
- 输出用于研发比较，不用于临床。

## 最简单的本地测试

### Windows

```powershell
git pull
.\run_dev.ps1
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

打开首页后有两种方式：

### 方式 A：直接使用仓库内 5 个真实测试病例

首页会自动读取 `testdata/1.0测试数据/`，目前可直接点：

- 基台测试1-牙位46
- 基台测试2-牙位36
- 基台测试3-牙位46
- 基台测试4-牙位26
- 基台测试5-牙位26

点击病例后进入“订单”页，能先看到上下颌 3D 预览和自动选择的 `FDI_XX.stl` 模板。然后点击：

```text
牙冠生成
```

完成后自动进入“自动设计 / 导出”页，可以：

- 开关目标颌；
- 开关对颌；
- 开关识别出的基台 component；
- 开关黄色自动外冠；
- 查看工程 QC；
- 下载 `Outer STL`；
- 打开 `report.json`。

### 方式 B：上传自己的上下颌

首页右侧：

1. 输入目标 FDI；
2. 选择上颌 STL；
3. 选择下颌 STL；
4. 创建订单；
5. 点击“牙冠生成”。

## 模板库

仓库已经包含：

```text
templates/32牙位基础解剖模板库/32位牙列-分/FDI_11.stl ... FDI_48.stl
```

这些是完整天然牙模型，因此当前 raw-case adapter 会先将完整牙快速转换成 crown-like anatomical prior，再交给外冠 pipeline；不会直接拿整颗牙根去做患者化形变。

## 调试输出

每次前端生成都会写到：

```text
work/<case_id>/output/
```

包括：

```text
00_template_canonical.stl
00_support_component.stl
00_target_jaw_component.stl
00_margin_proxy.xyz
01_placed.stl
04_occlusal.stl        # 若对颌阶段实际输出
05_outer_candidate.stl
report.json
```

这些文件专门用来判断失败发生在：模板裁取、基台识别、摆放、对颌避让还是最终 QC。

## CLI 仍然保留

对于已经准备好 `margin.json + frame.json + 邻牙 patch` 的标准输入，仍可以直接使用：

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

`autocrown intaglio-preview` 目前仍只是诊断预览，不是制造级 cutter。正式内冠后续仍需要：

1. insertion-axis directional blockout；
2. cement-gap field；
3. clinical face ownership；
4. support coverage validation；
5. Margin-owned clipping；
6. deterministic cervical collar / final assembly。

当前阶段先验证外冠 MVP 是否能比复杂全局形变更稳定地保住模板解剖形态。
