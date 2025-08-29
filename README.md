
# TianGong AI vLLM Serve

## Env Preparing

Setup `venv`:

```bash

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm --torch-backend=auto

uv pip install vllm --torch-backend=auto

uv pip freeze > requirements.txt
```

Auto lint:
```bash
uv pip install black
black .
```
