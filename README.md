# TianGong AI vLLM Serve

## 概览
- 使用 uv 管理 Python 依赖，默认启用清华镜像（PyPI 作为回退）。
- 提供多款模型的 vLLM 启动示例与 pm2 配置。
- 可选开发工具：Black 代码格式化。

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
uv sync 
```

4) 开发工具（Black）：  
```bash
uv sync --group dev
uv run black .
```

5) 升级依赖：  
- 升级当前锁定版本到最新兼容版本：  
  ```bash
  uv sync --upgrade
  ```  
- 单独升级核心包：  
  ```bash
  uv pip install --upgrade vllm modelscope
  ```

6) Blackwell（CUDA 13）推荐安装 vLLM nightly（Qwen3.5 官方配方）：  
```bash
source .venv/bin/activate
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

> 说明：`vllm/vllm-openai:cu130-nightly` 是 Docker 镜像标签，不是 `uv pip --extra-index-url` 可用的索引地址。  
> 如果走 Docker（Blackwell 推荐镜像）：`vllm/vllm-openai:cu130-nightly`。

> 镜像配置已写入 `pyproject.toml`：`tool.uv.pip.index-url` 指向清华，`tool.uv.index` 将清华设为默认索引，并保留 PyPI 作为备用源。

## 按命令切换/覆盖镜像源
- 临时指定清华（兼容 pip 语法，适用于 `uv pip`）：  
  ```bash
  uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
                  --extra-index-url https://pypi.org/simple \
                  vllm
  ```
- 临时指定官方 PyPI 或其他源（环境变量，对所有解析命令有效，如 `uv sync` / `uv lock` / `uv add`）：  
  ```bash
  UV_INDEX_URL=https://pypi.org/simple \
  UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  uv sync --upgrade vllm
  ```
- 为项目命令覆盖默认镜像（命令行或环境变量，优先级最高）：  
  ```bash
  UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
  UV_INDEX_URL=https://pypi.org/simpleu uv sync --upgrade
  ```

## 运行与服务
### 使用 pm2 管理
```bash
npm install -g pm2
pm2 start gpt-oss-120b.config.json
pm2 start qwen-2.5vl-72b.config.json
pm2 start qwen-3vl-30b.config.json
pm2 start qwen-3vl-32b.config.json
pm2 start qwen-3.5-122b-a10b-fp8.config.json
pm2 start qwen-3-embedding-0.6b.config.json

pm2 save
pm2 resurrect
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
