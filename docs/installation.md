# Devkit installation

## Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3.10

## Install with conda (recommended)

```bash
git clone git@github.com:nureasoning/nureasoning-devkit.git
cd nureasoning-devkit
conda env create -f environment.yml
conda activate nureasoning
pip install -e .
```

The environment installs the pinned CUDA 12.8 PyTorch stack, `transformers`, `peft`, `huggingface_hub`, and all plotting/geometry dependencies. `pip install -e .` registers the `nureasoning` package so modules can be run as:

```bash
python -m nureasoning.dataset.download --help
```

## Install with pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Install with uv (Linux x86_64, NVIDIA GPU)

From the repository root:

```bash
uv sync --locked
uv run python -c "import torch, nureasoning; print(nureasoning.__version__, torch.__version__, torch.cuda.is_available())"
uv run python -m nureasoning.reasoning.modules.selftest_metrics
```

uv uses Python 3.10 and creates `.venv/` without changing system Python.
`pyproject.toml` reuses `setup.py` metadata and `requirements.txt`, constrains
direct dependencies to the reference `environment.yml` versions, and selects
the CUDA 12.8 PyTorch index explicitly. `uv.lock` pins resolved dependencies.
After changing dependencies, run `uv lock` before `uv sync --locked`.

Prefix the Python commands in these docs with `uv run`, for example
`uv run python -m nureasoning.planning.benchmark --help`. This installs software
only, not model weights or datasets. nuVLA inference still needs a trained
checkpoint. Keep vLLM in a separate environment as described below.

## Reasoning training / testing

Reasoning **training** (`python -m nureasoning.reasoning.train`) uses the `nureasoning` environment above.

Reasoning **evaluation** (`python -m nureasoning.reasoning.evaluate`) also stays in `nureasoning`. It does **not** load weights: it POSTs chat-completions to an OpenAI-compatible HTTP server (default `http://127.0.0.1:8000/v1`). No inference server is bundled in this package.

To score a **local** checkpoint, serve it yourself in a **second terminal**. The reference numbers use vLLM against a [merged LoRA checkpoint](reasoning.md#4-merge-the-adapter). **Do not** `pip install vllm` into the `nureasoning` environment: vLLM pins its own PyTorch build and will overwrite the CUDA 12.8 training stack above.

Create a dedicated env once:

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```

Terminal 2 — leave this running (it blocks and owns the GPUs):

```bash
conda activate vllm
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ./reasoning_workspace/merged_qwen3.5-4b-multiframe \
  --served-model-name nureasoning-4b-sft \
  --tensor-parallel-size 8 \
  --max-model-len 24576 \
  --dtype bfloat16 \
  --mm-processor-kwargs '{"max_pixels": 200704}' \
  --enable-prefix-caching
```

Terminal 1 — still `nureasoning`; run `python -m nureasoning.reasoning.evaluate` there. Set `--tensor-parallel-size` to the number of GPUs on that server. `--mm-processor-kwargs max_pixels` must match `max_pixels` in the training config. `--served-model-name` is the string evaluate uses as `--model`.

Train, merge, serve, and score commands are on the [reasoning page](reasoning.md).

## Verify

```bash
python -c "import torch, transformers, shapely, matplotlib; import nureasoning; print(nureasoning.__version__, torch.__version__)"
python -m nureasoning.reasoning.modules.selftest_metrics
```

Next: [download and set up the dataset](dataset_setup.md).
