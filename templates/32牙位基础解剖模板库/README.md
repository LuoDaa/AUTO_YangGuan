# 32 牙位基础解剖模板库

本目录保存恒牙列的基础解剖 STL 模型。独立牙位统一采用 **FDI 两位数编号**，便于在牙冠自动生成、牙位识别和 CAD 流程中直接使用。

## 目录结构

```text
32牙位基础解剖模板库/
├─ 32位牙-分/                # 32 个独立牙位模型及 2 个预备牙模型
│  ├─ FDI_11.stl
│  ├─ ...
│  ├─ FDI_48.stl
│  ├─ FDI_16_Prep.stl
│  └─ FDI_36_Prep.stl
└─ 32位牙-合并/
   └─ 完整 32 牙位基础解剖牙列.stl
```

## 文件命名规则

- `FDI_XX.stl`：牙位 `XX` 的完整解剖牙模型。
- `FDI_XX_Prep.stl`：牙位 `XX` 的预备牙模型，不代表新的牙位。
- 当前额外提供的预备牙模型：`FDI_16_Prep.stl`、`FDI_36_Prep.stl`。

## FDI 编号说明

FDI 恒牙编号由两位数字组成：

- 第一位表示象限：`1` 右上、`2` 左上、`3` 左下、`4` 右下。
- 第二位表示牙齿从中线向后的位置：`1` 中切牙，依次到 `8` 第三磨牙。
- 左右方向以患者自身方向为准。

## Universal 1–32 与 FDI 对照

原始 Dental Merlin 模型使用美国 Universal Numbering System。对应关系如下：

| 原始文件 | Universal | 当前文件 | 牙齿名称 |
|---|---:|---|---|
| `Tooth_1.stl` | 1 | `FDI_18.stl` | 右上第三磨牙 |
| `Tooth_2.stl` | 2 | `FDI_17.stl` | 右上第二磨牙 |
| `Tooth_3.stl` | 3 | `FDI_16.stl` | 右上第一磨牙 |
| `Tooth_4.stl` | 4 | `FDI_15.stl` | 右上第二前磨牙 |
| `Tooth_5.stl` | 5 | `FDI_14.stl` | 右上第一前磨牙 |
| `Tooth_6.stl` | 6 | `FDI_13.stl` | 右上犬牙 |
| `Tooth_7.stl` | 7 | `FDI_12.stl` | 右上侧切牙 |
| `Tooth_8.stl` | 8 | `FDI_11.stl` | 右上中切牙 |
| `Tooth_9.stl` | 9 | `FDI_21.stl` | 左上中切牙 |
| `Tooth_10.stl` | 10 | `FDI_22.stl` | 左上侧切牙 |
| `Tooth_11.stl` | 11 | `FDI_23.stl` | 左上犬牙 |
| `Tooth_12.stl` | 12 | `FDI_24.stl` | 左上第一前磨牙 |
| `Tooth_13.stl` | 13 | `FDI_25.stl` | 左上第二前磨牙 |
| `Tooth_14.stl` | 14 | `FDI_26.stl` | 左上第一磨牙 |
| `Tooth_15.stl` | 15 | `FDI_27.stl` | 左上第二磨牙 |
| `Tooth_16.stl` | 16 | `FDI_28.stl` | 左上第三磨牙 |
| `Tooth_17.stl` | 17 | `FDI_38.stl` | 左下第三磨牙 |
| `Tooth_18.stl` | 18 | `FDI_37.stl` | 左下第二磨牙 |
| `Tooth_19.stl` | 19 | `FDI_36.stl` | 左下第一磨牙 |
| `Tooth_20.stl` | 20 | `FDI_35.stl` | 左下第二前磨牙 |
| `Tooth_21.stl` | 21 | `FDI_34.stl` | 左下第一前磨牙 |
| `Tooth_22.stl` | 22 | `FDI_33.stl` | 左下犬牙 |
| `Tooth_23.stl` | 23 | `FDI_32.stl` | 左下侧切牙 |
| `Tooth_24.stl` | 24 | `FDI_31.stl` | 左下中切牙 |
| `Tooth_25.stl` | 25 | `FDI_41.stl` | 右下中切牙 |
| `Tooth_26.stl` | 26 | `FDI_42.stl` | 右下侧切牙 |
| `Tooth_27.stl` | 27 | `FDI_43.stl` | 右下犬牙 |
| `Tooth_28.stl` | 28 | `FDI_44.stl` | 右下第一前磨牙 |
| `Tooth_29.stl` | 29 | `FDI_45.stl` | 右下第二前磨牙 |
| `Tooth_30.stl` | 30 | `FDI_46.stl` | 右下第一磨牙 |
| `Tooth_31.stl` | 31 | `FDI_47.stl` | 右下第二磨牙 |
| `Tooth_32.stl` | 32 | `FDI_48.stl` | 右下第三磨牙 |

预备牙的对应关系：

| 原始文件 | 当前文件 | 说明 |
|---|---|---|
| `Tooth_3_Prep.stl` | `FDI_16_Prep.stl` | 右上第一磨牙预备牙 |
| `Tooth_19_Prep.stl` | `FDI_36_Prep.stl` | 左下第一磨牙预备牙 |

## 使用建议

程序读取独立牙位时，建议直接以文件名中的 FDI 编号作为牙位标识。例如：

```text
FDI_16.stl      -> 牙位 16 的完整解剖模型
FDI_16_Prep.stl -> 牙位 16 的预备牙模型
```

不要把 Universal 编号与 FDI 编号直接按数值等同。例如，Universal `3` 对应的是 FDI `16`，而不是 FDI `13` 或 `03`。

## 参考

- [Dental Merlin：Universal 1–32 library](https://academy.3pointx.com/addon)
- [ADA：Universal Tooth Designation System](https://www.ada.org/-/media/project/ada-organization/ada/ada-org/files/publications/cdt/universal_tooth_designation_system_valueset_2.pdf)
