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
uv sync --upgrade
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
uv pip install modelscope
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
pm2 start qwen-3.5-397b-a17b-int4-size4.config.json

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

### 自定义 chat template（兼容 `developer` 角色）
仓库内置了 [chat-templates/qwen3_5_openai_developer.jinja](/home/david/projects/TianGong-AI-vLLM/chat-templates/qwen3_5_openai_developer.jinja)，用于把 OpenAI 风格的 `developer` 消息映射为 Qwen3.5 模板里的首条 `system` 指令。`qwen-3.5-122b-*` 和 `qwen-3.5-397b-*` 的 pm2 配置已经接入这个参数。

直接命令行启动时，也可以手动追加：
```bash
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --port 7730 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --quantization moe_wna16 \
  --chat-template ./chat-templates/qwen3_5_openai_developer.jinja
```

请求示例：
```bash
curl http://127.0.0.1:7730/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    "messages": [
      {"role": "developer", "content": "你是一个严谨的代码助手。"},
      {"role": "user", "content": "解释一下 prefix caching 的作用。"}
    ]
  }'
```

如果你的客户端同时还会发送 `system`，建议只保留一种高优先级指令角色，优先统一成 `developer`，避免模板层语义重复。

## 轻量 Codex 代理（`/v1/responses` -> `/v1/chat/completions`）
当下游是 Codex SDK、上游是 vLLM 时，可以先走仓库内置的轻量代理 [scripts/codex_vllm_proxy.py](/home/david/projects/TianGong-AI-vLLM/scripts/codex_vllm_proxy.py)。它只做当前这条链路最需要的兼容层：
- 将前置 `developer/system` 合并为一条 system 指令
- 过滤掉会触发 vLLM `/v1/responses` 严格校验的 `reasoning` input item
- 用内存保存 `previous_response_id` 对应的多轮消息
- 把 `/v1/responses` 请求转成上游 `/v1/chat/completions`

直接启动：
```bash
source .venv/bin/activate
python scripts/codex_vllm_proxy.py --upstream http://127.0.0.1:7730 --port 7740
```

用 pm2 启动：
```bash
pm2 start codex-vllm-proxy.config.json
pm2 save
```

环境变量：
```bash
CODEX_VLLM_PROXY_UPSTREAM=http://127.0.0.1:7730
CODEX_VLLM_PROXY_PORT=7740
CODEX_VLLM_PROXY_TIMEOUT=600
CODEX_VLLM_PROXY_MAX_HISTORY=200
```

请求示例：
```bash
curl http://127.0.0.1:7740/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    "stream": false,
    "input": [
      {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "你是一个严谨的代码助手。"}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "解释一下 prefix caching 的作用。"}]}
    ]
  }'
```

## codex 配置

在.codex/config.toml 中配置

```
model = "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
model_provider = "vllm"
model_reasoning_effort = "medium"
personality = "pragmatic"

[model_providers.vllm]
name = "vllm"
base_url = "http://192.168.1.143:7740/v1"
experimental_bearer_token = "local-demo"
wire_api = "responses"
```

> 注意：这个代理的 `previous_response_id` 历史是内存态，重启后不会保留。

## 监控 GPU（可选）
```bash
watch -n 1 nvidia-smi
```
