# ComfyUI-See-through

A ComfyUI plugin that wraps [See-through](https://github.com/shitagaki-lab/see-through) — an AI system that decomposes a single anime illustration into manipulatable 2.5D layer-decomposed models with depth ordering, ready for Live2D workflows.

[中文说明](README_CN.md)

Paper: [arxiv:2602.03749](https://arxiv.org/abs/2602.03749) (Conditionally accepted to ACM SIGGRAPH 2026)

## Features

- **Single-Image Layer Decomposition** — Input one anime character image, get up to 24 semantic transparent layers (hair, face, eyes, clothing, accessories, etc.)
- **Depth Estimation** — Automatic depth map generation for each layer via fine-tuned Marigold, establishing correct drawing order
- **Smart Splitting** — Eyes, ears, handwear split into left/right; hair split into front/back via depth clustering
- **PSD Export** — Download layered PSD files directly from the browser (frontend ag-psd, no Python dependency)
- **Depth PSD** — Separate depth PSD export for 3D/parallax workflows
- **Preview Output** — Blended reconstruction preview as a standard ComfyUI IMAGE output
- **HuggingFace Auto-Download** — Models download automatically from HuggingFace on first use

## Nodes

| Node | Description |
|------|-------------|
| **SeeThrough Load LayerDiff Model** | Load the LayerDiff SDXL pipeline (layer generation) |
| **SeeThrough Load Depth Model** | Load the Marigold depth estimation pipeline |
| **SeeThrough Decompose** | Full pipeline: LayerDiff + Marigold depth + post-processing |
| **SeeThrough Save PSD** | Save layers as PNGs + metadata; download PSD via browser button |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-See-through.git
```

Install dependencies:

```bash
cd ComfyUI-See-through
pip install -r requirements.txt
```

Restart ComfyUI. The **SeeThrough** nodes will appear under the `SeeThrough` category.

### Dependencies

Only 4 additional Python packages beyond ComfyUI's base:

- `diffusers` — Hugging Face diffusion pipeline
- `accelerate` — Model loading acceleration
- `opencv-python` — Image processing
- `scikit-learn` — KMeans clustering for depth-based layer splitting

### Models

Models are downloaded automatically from HuggingFace on first use:

| Model | HuggingFace Repo | Purpose |
|-------|-------------------|---------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | SDXL-based transparent layer generation |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | Fine-tuned monocular depth for anime |

Alternatively, download models manually and place them in `ComfyUI/models/SeeThrough/`.

## Usage

### Basic Workflow

1. Add **SeeThrough Load LayerDiff Model** and **SeeThrough Load Depth Model** nodes
2. Add a **SeeThrough Decompose** node — connect both models and a **Load Image** node
3. Add **SeeThrough Save PSD** — connect the `parts` output
4. Add **Preview Image** — connect the `preview` output
5. Run the workflow
6. Click **Download PSD** button on the Save PSD node to generate and download the PSD file

### Example Workflows

Pre-made workflows are available in the `workflows/` directory:

| Workflow | Resolution | Steps | L/R Split | Description |
|----------|-----------|-------|-----------|-------------|
| `seethrough-basic.json` | 1280 | 30 | Yes | Standard quality, recommended |

Drag any `.json` file into ComfyUI to load the workflow.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `resolution` | 1280 | Processing resolution (image is center-padded to square) |
| `num_inference_steps` | 30 | Diffusion denoising steps (more = better quality, slower) |
| `tblr_split` | true | Split symmetric parts (eyes, ears, handwear) into left/right |

## Output Layers

The decomposition produces semantic layers including:

**Body parts:** front hair, back hair, neck, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects

**Head parts:** headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth

Each layer is an RGBA image with transparency, positioned at its correct location in the canvas.

## Credits

This plugin wraps the [See-through](https://github.com/shitagaki-lab/see-through) research project by [shitagaki-lab](https://github.com/shitagaki-lab).

PSD generation uses [ag-psd](https://github.com/nicasiomg/ag-psd) in the browser.

## License

MIT
