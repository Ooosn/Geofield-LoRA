# Geofield-LoRA

This repository contains the code for Geofield-LoRA.

## Contents

- `train.py`: training entrypoint
- `eval.py`: evaluation entrypoint
- `cache_conditions.py`: precompute condition embeddings
- `model/`: Geofield-LoRA and baseline model implementations
- `data/`: dataset loading, template splits, and condition registry code
- `configs/release/`: release configs for `16x8`, `16x16`, and `32x16`
- `assets/`: example visualizations

## Demo

| `32x16` field fly-through | `16x16` field fly-through |
| --- | --- |
| ![Geofield-LoRA 32x16 field fly-through](assets/pca_gaussian_field_ray_chase_software_32x16_v1.gif) | ![Geofield-LoRA 16x16 field fly-through](assets/pca_gaussian_field_ray_chase_software_16x16_v1.gif) |

## Environment Setup

We recommend Python 3.11 and a CUDA-enabled PyTorch install.

```bash
cd open_source

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/bigscience-workshop/promptsource.git
```

Set the package root once:

```bash
export GSLORA_ROOT=$(pwd)
```

## Base Models

Main release settings:

- backbone: `Qwen/Qwen2.5-1.5B-Instruct`
- frozen condition encoder: `Qwen/Qwen3-Embedding-0.6B`
- adapted modules: `q_proj`, `v_proj`

## Condition Cache

Before training, build the condition embedding cache:

```bash
python cache_conditions.py --config configs/release/geofield_clean57_k16r8.yaml
```

This writes the cache under `runtime/data/`.

## Training

Single GPU:

```bash
python train.py --config configs/release/geofield_clean57_k16r8.yaml
```

Multi GPU:

```bash
torchrun --nproc_per_node=4 train.py --config configs/release/geofield_clean57_k16r8.yaml
```

Available configs:

- `configs/release/geofield_clean57_k16r8.yaml`
- `configs/release/geofield_clean57_k16r16.yaml`
- `configs/release/geofield_clean57_k32r16.yaml`

## Evaluation

```bash
python eval.py \
  --config configs/release/geofield_clean57_k16r8.yaml \
  --adapter_path runtime/checkpoints/<run_name>/best_heldout_adapter/adapter.pt
```

## Notes

- `runtime/` is not included here; caches, downloaded models, and checkpoints
  are generated locally.
- `promptsource` is installed from GitHub because the old PyPI package does not
  provide a compatible build for modern Python versions.
