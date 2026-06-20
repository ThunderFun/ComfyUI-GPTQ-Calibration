# ComfyUI-GPTQ-Calibration

ComfyUI custom node for collecting per-layer activation statistics (Hessians and optional amax) for external quantization tools like GPTQ, OBQ, and ConvRot.

**No quantization is performed.** This node produces `.pt` files containing calibration data that can be consumed offline by your quantization tool of choice.

> **⚠️ WARNING:** This code has not been thoroughly tested. Verify outputs before relying on it.

*Developed with AI assistance.*

## Installation

Clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ThunderFun/ComfyUI-GPTQ-Calibration.git
```

### Dependencies

`torch` and `numpy` are already shipped with ComfyUI and require no extra installation.

For GPU-accelerated Hadamard rotation (ConvRot), install [Triton](https://github.com/triton-lang/triton):

```bash
pip install triton
```

Triton is optional — without it, rotation falls back to a CPU matmul path.

## Nodes

### Calibration Data Collector

Collects per-layer Hessians (and optionally activation amax) from a loaded diffusion model. The model weights are never modified.

For the complete and authoritative list of inputs and their descriptions, hover over each input in the ComfyUI node UI. The table below is a summary:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | MODEL | — | Loaded diffusion model (FP16/BF16/FP32) |
| `conditioning` | CONDITIONING | — | Pre-encoded conditioning from CLIPTextEncode or similar |
| `num_steps` | INT | 4 | Denoising steps per sample |
| `num_samples` | INT | 16 | Independent samples to accumulate over |
| `seed` | INT | 0 | Seed for noise and timestep sampling |
| `hessian_block_size` | INT | 128 | Diagonal block size (ignored when `hessian_format='dlr'`) |
| `hessian_format` | COMBO | `dlr` | Hessian format: `dlr` (recommended), `block`, or `full` |
| `dlr_rank` | INT | 128 | Rank for DLR Hessian (only used when `hessian_format='dlr'`) |
| `collect_amax` | BOOLEAN | True | Also collect max(abs(x)) per layer |
| `output_path` | STRING | `output/calibration.pt` | Where to save the calibration file |
| `latent_height` | INT | 64 | Latent spatial height (128 for 1024px, 64 for 512px) |
| `latent_width` | INT | 64 | Latent spatial width (128 for 1024px, 64 for 512px) |
| `convrot` | BOOLEAN | False | Enable ConvRot Hadamard rotation |
| `rot_size` | INT | 256 | Hadamard group size (must be power of 2, up to 4096) |
| `permuquant` | BOOLEAN | False | Enable PermuQuant channel reordering |
| `piso` | BOOLEAN | False | Collect Hessian diagonal for PiSO data-aware scale optimization |
| `sigma_min` | FLOAT | 0.0 | Lower bound of the sigma range to sample |
| `sigma_max` | FLOAT | 1.0 | Upper bound of the sigma range to sample |
| `force_cpu_hook` | BOOLEAN | False | Force hook-side processing to CPU (GPU OOM workaround) |

**Output:** `calibration_path` (STRING) — path to the saved `.pt` file.

### Dual Model Calibration Data Collector

Calibrates two models used together via dual-model CFG. Mirrors `DualModelGuider`: positive conditioning runs through `model`, negative conditioning (often image-only) runs through `model_negative`, and CFG is applied between them at each step.

Outputs two `.pt` files — one per model — each in the same schema as the single-model node.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | MODEL | — | Positive (conditional) model |
| `model_negative` | MODEL | — | Negative (unconditional) model |
| `positive` | CONDITIONING | — | Positive conditioning |
| `negative` | CONDITIONING | *(optional)* | Negative conditioning (leave disconnected for image-only pass) |
| `cfg` | FLOAT | 4.0 | CFG value between the two models (must be > 1.0) |
| `num_steps` | INT | 4 | Denoising steps per sample |
| `num_samples` | INT | 16 | Independent samples to accumulate over |
| `seed` | INT | 0 | Seed for noise and timestep sampling |
| `hessian_block_size` | INT | 128 | Diagonal block size (ignored when `hessian_format='dlr'`) |
| `hessian_format` | COMBO | `dlr` | Hessian format: `dlr` (recommended), `block`, or `full` |
| `dlr_rank` | INT | 128 | Rank for DLR Hessian (only used when `hessian_format='dlr'`) |
| `collect_amax` | BOOLEAN | True | Also collect max(abs(x)) per layer |
| `output_path_positive` | STRING | `output/calibration_positive.pt` | Where to save the positive model's calibration file |
| `output_path_negative` | STRING | `output/calibration_negative.pt` | Where to save the negative model's calibration file |
| `latent_height` | INT | 64 | Latent spatial height |
| `latent_width` | INT | 64 | Latent spatial width |
| `convrot` | BOOLEAN | False | Enable ConvRot Hadamard rotation |
| `rot_size` | INT | 256 | Hadamard group size |
| `permuquant` | BOOLEAN | False | Enable PermuQuant channel reordering |
| `piso` | BOOLEAN | False | Collect Hessian diagonal for PiSO |
| `sigma_min` | FLOAT | 0.0 | Lower bound of the sigma range to sample |
| `sigma_max` | FLOAT | 1.0 | Upper bound of the sigma range to sample |
| `force_cpu_hook` | BOOLEAN | False | Force hook-side processing to CPU |

**Outputs:** `calibration_path_positive` (STRING), `calibration_path_negative` (STRING).

## Full Hessian Mode (`hessian_format='full'`)

> **WARNING:** Full Hessian mode has not been thoroughly tested and may not work properly. Use at your own risk.

Set `hessian_format` to `full` to store the complete `n × n` Hessian matrix for each layer. This is paper-accurate but:

- **Disk space**: O(n²) per layer — can produce multi-GB files for large models
- **Memory**: Hessians are memory-mapped to disk (`.gptq_hessian_tmp/` directory created next to the output file)
- **Cleanup**: The temporary mmap directory is automatically deleted after collection completes
- **Block size**: `hessian_block_size` is ignored when `hessian_format='full'`

Alternatively, setting `hessian_block_size=0` with `hessian_format='block'` also produces a full memory-mapped Hessian.

## DLR Hessian (Default)

DLR (Diagonal + Low-Rank) is the default Hessian format. It approximates the full `n × n` Hessian as `H ≈ diag(D) + UUᵀ`, where `D` is the exact per-channel diagonal and `U` is a rank-`r` factor capturing cross-channel correlations.

- **Storage**: `O(n + n·r)` — comparable to block-diagonal, but captures cross-block correlations that block-diagonal misses
- **Inverse**: computed in `O(nr²)` via the Woodbury identity (vs `O(n³)` for full Hessian)
- **Recommended rank**: 64–256 (`dlr_rank=128` is a good default)
- **Memory equivalence**: `dlr_rank=128` uses roughly the same memory as `hessian_block_size=128`

DLR is implemented via a truncated SVD streaming sketch that processes activations in batches, periodically compressing via SVD truncation. The exact diagonal is accumulated separately and preserved exactly in the output.

## ConvRot Rotation

ConvRot applies a Hadamard rotation to activations before computing Hessians. This makes the Hessians more block-diagonal, which can improve quantization quality.

- **When to use**: When you want better quality at the same block size, or the same quality at a smaller block size
- **Recommended settings**: `convrot=True`, `rot_size=256`, `hessian_block_size=128`
- **GPU acceleration**: Install `triton` for O(N log N) FHT rotation on GPU; without it, a CPU matmul fallback is used

## Sigma Clipping

By default, calibration samples across the full noise range (sigma 0 to 1). The `sigma_min` and `sigma_max` inputs restrict the sampling range. This is useful for MoE models where different experts handle different noise regimes (e.g., Wan 2.2 A14B with a boundary at sigma ~ 0.875).

The effective number of sampling steps may be smaller than `num_steps` when clipping is active — this is expected.

## Companion Tool

For the actual quantization step, see [int_crush_converter](https://github.com/ThunderFun/int_crush_converter).

## Technical Details

- **Hessians** are computed as x.T @ x (the Gram matrix of layer activations)
- **DLR** (default) streams activations through a truncated SVD sketch, preserving the exact diagonal separately. Output is `{D, U}` representing `H ≈ diag(D) + UUᵀ`
- **Block-diagonal** stores only diagonal blocks of the full Hessian — saves disk space with minimal quality loss
- **Full Hessian** (`hessian_format='full'`, or `hessian_block_size=0`) stores the complete `n × n` matrix, memory-mapped to disk
- **Rotation** (ConvRot) applies a normalized Hadamard transform to activations, making Hessians more diagonal
- **amax** tracks the running maximum of abs(x) per layer, useful for activation quantization

## License

MIT
