# TianGong AI vLLM Serve

## 概览
- 项目依赖写入 `pyproject.toml`：核心运行依赖为 `vllm` 与 `modelscope>=1.31.0`，开发依赖包含 `black`。
- uv 默认使用清华镜像（PyPI 回退）安装与解析依赖，可按需覆盖。
- 提供多款模型的 vLLM 启动示例与 pm2 配置。

## 前置要求
- Python 3.12
- uv（极快的 Python 包管理器）
- GPU 环境（CUDA 驱动等）
- 可选：Node.js + pm2（用于进程守护）

## 使用 uv 完成环境准备
1) 安装 uv  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) 创建并激活虚拟环境：  
```bash
uv venv --python 3.12
source .venv/bin/activate
```

3) 安装项目依赖（使用清华镜像，PyPI 回退）：  
```bash
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --upgrdae
```

4) 开发工具（Black）：  
```bash
uv sync --group dev
uv run black .
```

5) 直接按需安装/重装核心包（与 `pyproject.toml` 对齐）：  
```bash
# 首选：使用清华镜像并让 vLLM 自动选择 torch 后端
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm --torch-backend=auto

# 官方源（或自带镜像配置）的 vLLM 自动后端
uv pip install vllm --torch-backend=auto

# 指定最低版本的 ModelScope
uv pip install modelscope
```

6) 升级：  
- 升级当前锁定版本到最新兼容版本：  
  ```bash
  uv sync --upgrade
  ```
- 单独升级核心包：  
  ```bash
  uv pip install --upgrade vllm modelscope
  ```

> 镜像配置位于 `pyproject.toml` 的 `tool.uv.pip`，默认清华镜像 + PyPI 回退；可用环境变量 `UV_INDEX_URL`/`UV_EXTRA_INDEX_URL` 或命令参数临时覆盖。

## 运行与服务
### 使用 pm2 管理
```bash
npm install -g pm2
pm2 start gpt-oss-120b.config.json
pm2 start qwen-2.5vl-72b.config.json
pm2 start qwen-3vl-30b.config.json
pm2 start qwen-3vl-32b.config.json
pm2 start qwen-3-embedding-0.6b.config.json
```

### 直接启动 vLLM
```bash
VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-20b
VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-120b --port 8001

VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ --max-model-len 16384 --port 8002
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --max-model-len 16384 --port 8002
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --max-model-len 32768 --port 8002
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-32B-Instruct --max-model-len 16384 --port 8002

VLLM_USE_MODELSCOPE=True vllm serve ZhipuAI/GLM-4.6 --tensor-parallel-size 2
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --tensor-parallel-size 3
VLLM_USE_MODELSCOPE=True vllm serve openai-mirror/gpt-oss-120b --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.90

CUDA_VISIBLE_DEVICES=0 VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-Embedding-0.6B --port 8004 --gpu-memory-utilization 0.24
```

## 监控 GPU（可选）
```bash
watch -n 1 nvidia-smi
```
