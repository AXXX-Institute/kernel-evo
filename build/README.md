# Build and deployment

## Docker

### Build the image

From the **repository root**:

```bash
docker build -f build/Dockerfile -t kernel-evo:latest .
```

**Private GitHub dependencies (gigaevo, kernelbench):**  
If those repos are private, pass a [GitHub PAT](https://github.com/settings/tokens) so pip can clone them. The token is passed as a BuildKit secret and is **not** stored in image layers:

```bash
export GITHUB_TOKEN=ghp_xxxx   # or use a file: --secret id=github_token,src=$HOME/.github_token
docker build -f build/Dockerfile --secret id=github_token,env=GITHUB_TOKEN -t kernel-evo:latest .
```

Omit `--secret` and `GITHUB_TOKEN` when the dependencies are public (e.g. after making repos open-source).

### Run with custom problems

Custom problems are directories that contain `task.py` (see repo `tasks/armt_associate` for the expected format). Mount your problem directory into the container and pass `--problem-path` to the `evolve` command.

**Example: one custom task directory**

```bash
# Your host directory with a custom problem, e.g.:
#   my_problems/
#   └── my_kernel/
#       └── task.py

docker run --rm -it \
  -v /path/on/host/my_problems:/problems \
  kernel-evo:latest \
  kernel-evo evolve \
    --problem-path /problems/my_kernel \
    --experiment-name my_run \
    --model-name openai/gpt-oss-120b
```

**Example: KernelBench problem (no custom path)**

```bash
docker run --rm -it \
  kernel-evo:latest \
  kernel-evo evolve \
    --level 1 \
    --problem-id 42 \
    --experiment-name kb_run \
    --model-name openai/gpt-oss-120b
```
