# KernelEvo
<div align="center">
  <img src="logo/banner.svg" alt="Kernel Evo banner" />
</div>

**Evolutionary generation of efficient GPU kernels** using [GigaEvo](https://github.com/KhrulkovV/gigaevo-core-internal).  
Define a task, run evolution with an LLM backend, extract and compare optimized programs.

---

## Features

- **Custom tasks** — Define your own kernel tasks in KernelBench format and evolve them.
- **KernelBench integration** — Use existing [KernelBench](https://github.com/ScalingIntelligence/KernelBench) problems.
- **Remote or local execution** — Run validation locally or via a remote eval server.
---

## Requirements

- **Python** >= 3.12
- **LLM API** — OpenAI-compatible (e.g. [OpenRouter](https://openrouter.ai), or a local server like SGLang).  
- **Redis** — Used by GigaEvo for experiment state.

---

## Installation

### From source

```bash
git clone https://github.com/svtdanny/kernel_evo.git
cd kernel_evo
pip install -e . --ignore-requires-python
```

> **Note:** `--ignore-requires-python` relaxes the Python version check (KernelBench may declare 3.10 but works on 3.12).  
> For custom branches of `gigaevo` or `kernelbench`, edit the Git URLs in `pyproject.toml`.

### Docker

Pull and run (when a pre-built image is published):

```bash
docker pull svtdanny/kernel-evo:latest
docker run --rm kernel-evo:latest kernel-evo --help
```

To build the image yourself (e.g. for private dependencies or development), see **[build/README.md](build/README.md)**.

---

## Custom kernel task

To evolve your own kernel, create a task in **KernelBench format**. Example layout:

```
tasks/
└── armt_associate/
    └── task.py
```

See `tasks/armt_associate` in this repo for a reference. You can also use any existing task from [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

---

## Run evolution

Evolution can use a **local** or **remote** LLM (e.g. SGLang, OpenRouter). Examples below use OpenRouter and a remote eval server.

### 1. Start the eval server (optional, for remote validation)

In a separate terminal:

```bash
kernel-evo eval_server --port 15000
```

### 2. Evolve with a custom task

```bash
OPENAI_API_KEY="sk-or-v1-..." kernel-evo evolve \
  --problem-path tasks/armt_associate/task.py \
  --experiment-name custom_associate \
  --backend triton \
  --precision fp16 \
  --model-name openai/gpt-oss-120b \
  --llm-base-url https://openrouter.ai/api/v1 \
  --redis-db 1 \
  --max-generations 40 \
  --max-mutations-per-generation 1 \
  --validator-debug --validator-debug-dir outputs/validate_logs \
  --llm-log-dir outputs/traces --llm-log-port 14005 \
  --stdout-dir outputs/logs --disable-insights-lineage \
  --execution-mode remote_execution
```

### 3. Evolve with a KernelBench task

```bash
OPENAI_API_KEY="<KEY>" kernel-evo evolve \
  --level 1 \
  --problem-id 36 \
  --experiment-name kb_level1_36 \
  --dataset-src huggingface \
  --dataset-name ScalingIntelligence/KernelBench \
  --backend triton \
  --precision fp16 \
  --model-name <MODEL> \
  --llm-base-url https://openrouter.ai/api/v1 \
  --redis-db 2 \
  --max-generations 10 \
  --max-mutations-per-generation 2 \
  --validator-debug --validator-debug-dir <dir_for_validations_debug> \
  --llm-log-dir <dir_for_logs>
```

---

## Monitor progress

```bash
cd gigaevo/outputs/<DATE>/<EXPERIMENT_START>
tensorboard --logdir .
```

Use TensorBoard to find iterations with good performance before extracting programs.

---

## Extract a program

Export the program from a specific iteration (e.g. after inspecting TensorBoard):

```bash
kernel-evo extract \
  --redis-db 2 \
  --iteration 8 \
  --redis-prefix "kernel_evo" \
  --output-file prog8.py
```

---

## Compare two programs

### Custom task

```bash
kernel-evo compare \
  --program-a prog_a.py \
  --program-b prog_b.py \
  --problem-path tasks/armt_associate/task.py \
  --backend triton \
  --precision fp16 \
  --num-perf-trials 300
```

### KernelBench task

```bash
kernel-evo compare \
  --program-a prog_a.py \
  --program-b prog_b.py \
  --dataset-src huggingface \
  --dataset-name ScalingIntelligence/KernelBench \
  --level 1 \
  --problem-id 36 \
  --backend triton \
  --precision fp16
```

---

## CLI overview

| Command         | Description                          |
|----------------|--------------------------------------|
| `evolve`       | Run evolution (custom or KernelBench) |
| `eval_server`  | Start remote validation server       |
| `extract`      | Export program by iteration from Redis |
| `compare`      | Compare two programs (correctness + perf) |
