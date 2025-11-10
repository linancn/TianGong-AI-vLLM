
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

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade vllm modelscope --torch-backend=auto
uv pip install --upgrade vllm modelscope --torch-backend=auto

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

pm2 start gpt-oss-120b.config.json

pm2 start qwen-2.5vl-72b.config.json
pm2 start qwen-3vl-30b.config.json
pm2 start qwen-3vl-32b.config.json

pm2 start qwen-3-embedding-0.6b.config.json


VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-20b

VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-120b --port 8001

VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ --max-model-len 16384 --port 8002

VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --max-model-len 16384 --port 8002

VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-32B-Instruct --max-model-len 16384 --port 8002

VLLM_USE_MODELSCOPE=True vllm serve ZhipuAI/GLM-4.6 \
  --tensor-parallel-size 2

VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size 3

VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90

CUDA_VISIBLE_DEVICES=0 VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-Embedding-0.6B --port 8004 \
  --gpu-memory-utilization 0.24
```

Test Cuda (optional):

```bash
watch -n 1 nvidia-smi
```
