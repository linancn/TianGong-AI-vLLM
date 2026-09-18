# TianGong AI vLLM Serve

## 直接复制给 AI：启动、恢复、停止

在部署机器打开本仓库作为工作区，复制以下任一段。凭证从本地配置读取，无需粘贴到对话。

**首次初始化并启动：**

```text
请初始化并启动当前工作区的 TianGong AI vLLM Serve。先读 AGENTS.md、README.md 和 docs/deployment.md，检查本项目容器、端口、GPU、模型缓存和私有配置。复用已有配置；缺失时按 .env.example 初始化，默认不启用 API key；仅在配置明确要求鉴权时填写 key，不输出密钥。模型仅用 Docker 运行，按固定清单下载并校验，检查容器 CUDA。通过 deploy/manage.sh start model 启动并等候健康，执行 deploy/manage.sh check 完成真实推理验收。报告服务地址、模型名、验收结果及未解决问题，不操作其他项目服务。
```

**恢复或排障：**

```text
请恢复当前工作区的模型服务。先读 AGENTS.md 和 docs/deployment.md，检查本项目 Compose 状态、GPU、容器日志、端口和实际请求，定位原因。复用模型与私有配置，start 只补起缺失服务；配置变更才 restart。不要重启共享 Docker daemon、清空全局缓存或恢复其他项目 PM2 进程。修复后执行真实验收，不只检查 /health。报告原因、修复和仍存在的限制，不输出密钥。
```

**安全停止：**

```text
请停止当前工作区的模型设施。确认本项目容器归属，停止新增请求，检查 /metrics 中 running/waiting 请求并等待收敛。执行 deploy/manage.sh stop all，确认容器退出和对应 GPU 显存释放。保留模型、私有配置、编译缓存和验收结果，不清理其他服务。说明下次 start model 的恢复方式。
```

通过 Docker 提供 Qwen3.8-Flash-Next-NVFP4 的 OpenAI-compatible 文本、图片和工具调用 API。默认四卡 TP4 + EP4、GPU PLE、自动调优开启、MTP 3；性能对照见[调优指南](docs/performance-tuning.md)。模型来自指定的 [ModelScope 仓库](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)，修订与文件摘要固定在清单中。

## 启动与调用

前提：Linux、可用的 NVIDIA 驱动和 Container Toolkit、Docker Compose、Python 3.12+、curl；模板为四卡部署。准备约 133 GB 模型空间，另外预留镜像和缓存空间。详细兼容条件见[部署说明](docs/deployment.md)。

```bash
cp -n .env.example .env
# 编辑 .env，核对 GPU 和镜像；VLLM_API_KEY 默认留空
./deploy/manage.sh download
./deploy/manage.sh pull
./deploy/manage.sh start model
./deploy/manage.sh status model
./deploy/manage.sh check
```

默认地址 `http://127.0.0.1:7730/v1`，模型名 `nv-community/Qwen3.8-Flash-Next-NVFP4`。默认无需 API key 或 Authorization 请求头。可在 `.env` 中设置非空 `VLLM_API_KEY` 并重启模型启用 Bearer 鉴权。`check` 根据配置验证对应模式并保存真实响应，不输出凭证。外部客户端使用部署机器地址。

| 入口 | 用途 |
| --- | --- |
| `deploy/manage.sh start/restart/stop/status/logs model` | 模型生命周期 |
| `deploy/manage.sh check` | 鉴权、文本、流式、工具、图片、推理验收 |
| `deploy/manage.sh verify` | 检查全部模型文件 SHA256 |
| `examples/chat.http` | Chat、流式和工具调用示例 |

重复 start 保留已有容器；修改配置用 restart。Docker 自动恢复由 `unless-stopped` 管理；主动 stop 的服务不会在重启机器时自行恢复。

## 文档与开发

| 文档 | 内容 |
| --- | --- |
| [部署与恢复](docs/deployment.md) | 配置、下载、启动、停止、故障恢复 |
| [跨机迁移](docs/host-migration.md) | Git 更新、局域网复制模型/镜像、校验、切换与回滚 |
| [架构](docs/architecture.md) | 模型服务边界、目录和生命周期 |
| [AI 接入](docs/ai-integration.md) | API、鉴权、模型名、推理和多轮限制 |
| [调优](docs/performance-tuning.md) | 上下文、并发、GPU 和验证边界 |
| [依赖](docs/dependencies.md) | 容器版本、轻量 Python 锁文件和升级 |
| [运行时修复](docs/runtime-patches.md) | 自动调优缓存根因、容器补丁和回归 |
| [验证](docs/validation.md) | 本地回归、真实验收与证据 |

```bash
uv sync --locked --group dev
uv run --group dev black --check scripts tests deploy/vllm/patches
uv run --group dev ruff check scripts tests deploy/vllm/patches
uv run --group dev pytest
```

宿主开发环境只安装格式化、检查和测试工具；运维脚本仅依赖 Python 标准库与 curl。旧原生 vLLM 启动配置已退出当前部署，历史内容通过 Git 查看。
