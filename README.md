# Kernel generation

## installation

```
https://github.com/svtdanny/kernel_generation.git
cd kernel_generation
pip install -e . --ignore-requires-python
```

For custom branches of gigaevo, kernelbench see `pyproject.toml`

## Custom kernel

To evolve your own kernel, you need to create task in KernelBench format.  
See example in `tasks/armt_associate`:
```
tasks/
└── armt_associate/
    ├── task.py
    └── tests/
        └── test_task.py
```

Or you can use existing task from KernelBench (see run scripts below)

## Run

Can be evolved with local or remote model (in example, sglang_server and openrouter were used)

### With custom kernel

Run in different terminal:    
```
python3 -m kernel_generation eval_server --port 15000
```

```
OPENAI_API_KEY="sk-or-v1-" python3 -m kernel_generation evolve \
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
  --stdout-dir outputs/logs --disable-insights-lineage --execution-mode remote_execution
```

<!-- ```
OPENAI_API_KEY=EMPTY python3 scripts/generate_and_eval_single_sample_gigaevo.py \
  --problem-path tasks/armt_associate/task.py \
  --backend triton \
  --precision fp16 \
  --max-tokens 64000 \
  --model-name Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
  --llm-base-url http://127.0.0.1:30000/v1 \
  --redis-db 2 \
  --max-generations 40 \
  --max-mutations-per-generation 2 \
  --validator-debug --validator-debug-dir <dir_for_validations_debug> --llm-log-dir <dir_for_logs>
``` -->

### With KernelBench task

```
OPENAI_API_KEY=<KEY> python3 -m kernel_generation evolve \
  --level 1 \
  --problem-id 36 \
  --dataset-src huggingface \
  --backend triton \
  --precision fp16 \
  --model-name <MODEL> \
  --llm-base-url https://openrouter.ai/api/v1 \
  --redis-db 2 \
  --max-generations 10 \
  --max-mutations-per-generation 2 \
  --validator-debug --validator-debug-dir <dir_for_validations_debug> --llm-log-dir <dir_for_logs>
```


## Monitor progress

```
cd gigaevo/outputs/<DATE>/<EXPERIMENT_START>  
tensorboard --logdir .
```

## Extract program with given id 
__Ps:__ use tensorboard to see iterations with good performance

```
python3 -m kernel_generation extract --redis-db 2 --iteration 8 --redis-prefix "kernel_generation" --output-file prog8.py
```

## Compare two programs

### Custom task

```
python3 -m kernel_generation compare \
  --program-a prog_a.py \
  --program-b prog_b.py \
  --problem-path tasks/armt_associate/task.py \
  --backend triton \
  --precision fp16 --num-perf-trials 300
```

### Kernel bench task

```
python3 -m kernel_generation compare \
  --program-a prog_a.py \
  --program-b prog_b.py \
  --dataset-src huggingface \
  --dataset-name ScalingIntelligence/KernelBench \
  --level 1 \
  --problem-id 36 \
  --backend triton \
  --precision fp16
```
