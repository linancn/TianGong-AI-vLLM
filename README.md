
# TianGong AI vLLM Serve

## Env Preparing

Setup `venv`:

```bash

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm --torch-backend=auto
uv pip install vllm --torch-backend=auto
uv pip install modelscope>=1.18.1

uv pip freeze > requirements.txt
```

Auto lint:
```bash
uv pip install black
black .
```
Serving:
```bash
npm install -g pm2

pm2 start "vllm serve openai-mirror/gpt-oss-20b \
  --max-model-len 4096 \
  --max-num-seqs 16" \
  --name vllm-gpt20b \
  --env VLLM_USE_MODELSCOPE=True
```
