# ComfyUI-See-through

一个将 [See-through](https://github.com/shitagaki-lab/see-through) 集成到 ComfyUI 的插件 — 从单张动漫插画自动分解出可操作的 2.5D 分层模型，带深度排序，可直接用于 Live2D 工作流。

[English](README.md)

论文：[arxiv:2602.03749](https://arxiv.org/abs/2602.03749)（已被 ACM SIGGRAPH 2026 条件接收）

## 功能特性

- **单图分层分解** — 输入一张动漫角色图片，输出多达 24 个语义透明图层（头发、面部、眼睛、服装、饰品等）
- **深度估计** — 通过微调的 Marigold 为每层自动生成深度图，确定正确的绘制顺序
- **智能拆分** — 眼睛、耳朵、手套自动拆分为左/右；头发通过深度聚类拆分为前/后
- **PSD 导出** — 直接在浏览器下载分层 PSD 文件（前端 ag-psd 实现，无需 Python 依赖）
- **深度 PSD** — 可单独导出深度 PSD，用于 3D/视差工作流
- **预览输出** — 合成重建预览作为标准 ComfyUI IMAGE 输出
- **HuggingFace 自动下载** — 首次使用时自动从 HuggingFace 下载模型

## 节点说明

| 节点 | 说明 |
|------|------|
| **SeeThrough Load LayerDiff Model** | 加载 LayerDiff SDXL 管线（图层生成） |
| **SeeThrough Load Depth Model** | 加载 Marigold 深度估计管线 |
| **SeeThrough Decompose** | 完整管线：LayerDiff + Marigold 深度 + 后处理 |
| **SeeThrough Save PSD** | 保存图层 PNG + 元数据；通过浏览器按钮下载 PSD |

## 安装

将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dmMaze/ComfyUI-See-through.git
```

安装依赖：

```bash
cd ComfyUI-See-through
pip install -r requirements.txt
```

重启 ComfyUI，**SeeThrough** 节点将出现在 `SeeThrough` 分类下。

### 依赖

仅需在 ComfyUI 基础之上额外安装 4 个 Python 包：

- `diffusers` — Hugging Face 扩散管线
- `accelerate` — 模型加载加速
- `opencv-python` — 图像处理
- `scikit-learn` — 基于深度的图层拆分（KMeans 聚类）

### 模型

首次使用时自动从 HuggingFace 下载：

| 模型 | HuggingFace 仓库 | 用途 |
|------|-------------------|------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | 基于 SDXL 的透明图层生成 |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | 动漫微调的单目深度估计 |

也可手动下载模型放到 `ComfyUI/models/SeeThrough/` 目录。

## 使用方法

### 基本工作流

1. 添加 **SeeThrough Load LayerDiff Model** 和 **SeeThrough Load Depth Model** 节点
2. 添加 **SeeThrough Decompose** 节点 — 连接两个模型和一个 **Load Image** 节点
3. 添加 **SeeThrough Save PSD** — 连接 `parts` 输出
4. 添加 **Preview Image** — 连接 `preview` 输出
5. 运行工作流
6. 点击 Save PSD 节点上的 **Download PSD** 按钮生成并下载 PSD 文件

### 示例工作流

`workflows/` 目录中提供了预设工作流：

| 工作流 | 分辨率 | 步数 | 左右拆分 | 说明 |
|--------|--------|------|----------|------|
| `seethrough-basic.json` | 1280 | 30 | 是 | 标准质量，推荐使用 |
| `seethrough-highres.json` | 2048 | 50 | 是 | 高质量 + 保存预览图 |
| `seethrough-fast.json` | 1024 | 15 | 否 | 快速预览，质量较低 |

将 `.json` 文件拖入 ComfyUI 即可加载工作流。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seed` | 42 | 随机种子，用于结果复现 |
| `resolution` | 1280 | 处理分辨率（图像会居中填充为正方形） |
| `num_inference_steps` | 30 | 扩散去噪步数（越多质量越好，速度越慢） |
| `tblr_split` | true | 是否将对称部位（眼睛、耳朵、手套）拆分为左/右 |

## 输出图层

分解产生的语义图层包括：

**身体部位：** 前发、后发、颈部、上衣、手套、下装、腿饰、鞋子、尾巴、翅膀、物件

**头部部位：** 头饰、面部、虹膜、眉毛、眼白、睫毛、眼镜、耳朵、耳饰、鼻子、嘴巴

每个图层都是带透明度的 RGBA 图像，定位在画布上的正确位置。

## 致谢

本插件封装了 [shitagaki-lab](https://github.com/shitagaki-lab) 的 [See-through](https://github.com/shitagaki-lab/see-through) 研究项目。

PSD 生成使用浏览器端的 [ag-psd](https://github.com/nicasiomg/ag-psd) 库。

## 许可证

MIT
