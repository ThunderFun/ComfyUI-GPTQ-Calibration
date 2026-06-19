# ComfyUI-GPTQ-Calibration

ComfyUI custom node for collecting per-layer activation statistics (Hessians and optional amax) for external quantization tools like GPTQ, OBQ, and ConvRot.

**No quantization is performed.** This node produces a single `.pt` file containing calibration data that can be consumed offline by your quantization tool of choice.

> **⚠️ WARNING:** This code has not been thoroughly tested. Verify outputs before relying on it.

*Developed with AI assistance.*

## Installation

Clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ThunderFun/ComfyUI-GPTQ-Calibration.git
```

No extra dependencies are required — torch and numpy are already shipped with ComfyUI.

## Node

**Calibration Data Collector** collects per-layer Hessians (and optionally activation amax) from a loaded diffusion model. The model weights are never modified.

### Inputs

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

### Output

`calibration_path` (STRING) — path to the saved `.pt` file.

## Warning: Full Hessian Mode (`hessian_block_size=0`)

When `hessian_block_size` is set to `0`, the node stores the full Hessian matrix for each layer instead of diagonal blocks. This is paper-accurate but:

- **Disk space**: O(n^2) per layer — can produce multi-GB files for large models
- **Memory**: Hessians are memory-mapped to disk (`.gptq_hessian_tmp/` directory created next to the output file)
- **Cleanup**: The temporary mmap directory is automatically deleted after collection completes

For most users, the default `hessian_block_size=128` is recommended.

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

## Sigma Clipping

By default, calibration samples across the full noise range (sigma 0 to 1). The `sigma_min` and `sigma_max` inputs restrict the sampling range. This is useful for MoE models where different experts handle different noise regimes (e.g., Wan 2.2 A14B with a boundary at sigma ≈ 0.875).

The effective number of sampling steps may be smaller than `num_steps` when clipping is active — this is expected.

## Companion Tool

For the actual quantization step, see [int_crush_converter](https://github.com/ThunderFun/int_crush_converter).

## Technical Details

- **Hessians** are computed as x.T @ x (the Gram matrix of layer activations)
- **DLR** (default) streams activations through a truncated SVD sketch, preserving the exact diagonal separately. Output is `{D, U}` representing `H ≈ diag(D) + UUᵀ`
- **Block-diagonal** stores only diagonal blocks of the full Hessian — saves disk space with minimal quality loss
- **Full Hessian** (`hessian_block_size=0`) stores the complete `n × n` matrix, memory-mapped to disk
- **Rotation** (ConvRot) applies a normalized Hadamard transform to activations, making Hessians more diagonal
- **amax** tracks the running maximum of abs(x) per layer, useful for activation quantization

## License

MIT
