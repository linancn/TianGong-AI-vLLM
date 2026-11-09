# TianGong AI vLLM Serve

通过 Docker 下载、校验和运行多个模型，支持 vLLM 推理与 Laya 专用 SDK。同一个模型可在不同机器或同一机器上创建多个独立实例；GPU 数量、TP/DP/PP、上下文长度、显存预算、端口和模型目录按实例配置。

- `deploy/profiles/`：可复用模型配置，固定镜像、权重清单、默认参数和验收脚本。
- `deploy/examples/`：机器实例模板，包含单卡、三卡 embedding 与四卡生成模型示例。
- `deployments/<实例名>.json`：本机私有实例配置，不纳入 Git。
- `deploy/manage.sh`：统一管理入口；每个实例拥有独立容器、网络和缓存卷。

## 本机 Embedding

```bash
./deploy/manage.sh init embedding embedding
# 编辑 deployments/embedding.json：GPU、模型目录、端口、并行方式等
./deploy/manage.sh pull embedding
./deploy/manage.sh download embedding  # 复用缓存时可改用 verify
./deploy/manage.sh start embedding
./deploy/manage.sh check embedding
```

默认模型名 `Qwen/Qwen3-Embedding-8B`，地址 `http://127.0.0.1:7710/v1`，输出 4096 维向量。详细说明见 [Embedding 部署](docs/embedding.md)。可用 `init embedding-single <实例名>` 创建另一台机器的单卡配置，再按实际显存调整参数。

## Laya 多语言决策

Laya 按每卡一个完整模型副本运行，统一入口默认 `http://127.0.0.1:7720/v1/decisions`。`deploy/manage.sh <action> laya` 管理整个实例，单卡到三卡只需调整机器配置的 gpus 并重启。详见 [Laya 部署与调用](docs/laya.md)。

## 多模型与跨机器

```bash
./deploy/manage.sh profiles
./deploy/manage.sh list
./deploy/manage.sh init model chat-main
# 配置 deployments/chat-main.json 后，下载和启动该实例
./deploy/manage.sh pull chat-main
./deploy/manage.sh download chat-main
./deploy/manage.sh start chat-main
./deploy/manage.sh status all
./deploy/manage.sh stop embedding
```

`start` 保留已有容器；修改配置用 `restart`；已离线导入镜像用 `start-loaded` / `restart-loaded`。`all` 仅包含本机 `deployments/` 中明确配置的实例。启动多个实例前检查端口和共享 GPU 显存预算。Docker `unless-stopped` 管理退出恢复，主动停止的实例不会在重启时自行恢复。

已有四卡 Qwen3.8 部署仍可通过 `.env` 与 `manage.sh <action> model` 管理，保留原 Compose 身份；新部署优先使用实例文件。迁移为实例文件时须按[多模型管理](docs/models.md)保留原项目名称。`.env` 可保存实例通过 `api_key_env` 引用的密钥；默认不启用鉴权。

## GPUStack 模型托管

统一模型入口与跨主机副本由 `bash deploy/gpustack/manage.sh` 管理控制面，`worker.sh` 管理 GPU worker。首次安装执行 init/config/pull/start，已有部署沿用数据库与私有配置。模型和原生部署器互斥，详见 [GPUStack 控制面](docs/gpustack.md)。

[LCA-Nemotron-Embed-1B](docs/lca-nemotron.md) 使用固定 Sentence Transformers 参考镜像，每卡一个 GPUStack custom deployment；统一鉴权路由提供显式 query/document、2048 维向量，并通过完整 HTTP/reference 检索基准验收。

[H3 NVFP4 音视频服务](docs/h3.md) 固定完整模型组件、ModelScope 镜像与离线运行环境；已通过 GPUStack 单卡 480P/768P 带音频生成验收，使用参考图/视频输入。

[DeepSeek V4.1 Flash NVFP4](docs/deepseek-v41.md) 使用固定权重清单、SM120 兼容镜像和 GPUStack 单节点八卡部署，保留原生 1,048,576 token 上下文；提供文本、图片、工具调用与长输入验收脚本。

## 文档与开发

| 文档 | 内容 |
| --- | --- |
| [多模型与机器实例](docs/models.md) | 新模型注册、机器配置、参数覆盖、生命周期 |
| [Embedding 部署](docs/embedding.md) | PM2 迁移、兼容性验收、调用示例与回滚 |
| [生成模型部署](docs/deployment.md) | Qwen3.8 的资源要求与旧 `.env` 部署 |
| [跨机迁移](docs/host-migration.md) | 离线镜像和模型迁移 |
| [架构](docs/architecture.md) | 模型配置、机器实例与 Docker 边界 |
| [依赖](docs/dependencies.md) | 镜像和开发工具版本、升级方法 |
| [验证](docs/validation.md) | 本地回归、真实验收与证据 |
| [调优](docs/performance-tuning.md) / [运行时修复](docs/runtime-patches.md) | Qwen3.8 优化与镜像补丁 |
| [生成模型 API](docs/ai-integration.md) | Chat、图片和工具调用 |

```bash
export UV_PROJECT_ENVIRONMENT=.venv-dev
uv sync --locked --group dev
uv run --group dev black --check scripts tests deploy/vllm/patches deploy/laya deploy/lca-nemotron
uv run --group dev ruff check scripts tests deploy/vllm/patches deploy/laya deploy/lca-nemotron
uv run --group dev pytest
```

推理库只安装在镜像内；宿主运维脚本仅依赖 Python 标准库与 curl。原生迁移期间保留旧 `.venv` 供回退，不再用于常规启动。
