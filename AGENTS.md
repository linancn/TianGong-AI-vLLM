# TianGong AI vLLM Serve 协作说明

本仓库部署 ModelScope `nv-community/Qwen3.8-Flash-Next-NVFP4`，推理引擎只在 Docker 内运行。默认 TP4 + EP4、GPU PLE、FlashInfer 自动调优、MTP 3 与 NCCL；API key 默认空且可配置。架构与资料组织参考 `unstructure-serve`，本项目只保留模型服务，不引入应用代理或文档解析队列。

## 文档与修改约定

- 版本发布直接提交并 push 到 `main`，不创建功能分支或 PR。

- 修改代码、配置或行为时，同步维护本文件及相关专题文档。README 保持用途和入口简明；部署步骤、接口合同、验证边界分别集中到专题说明。
- 根目录文档仅 README.md、AGENTS.md；专题在 docs，部署在 deploy，运维辅助在 scripts，调用示例在 examples。
- 文档使用相对文件路径；命令以仓库根目录为工作目录。不要写机器专属路径、凭证或逐轮操作流水账；历史部署由 Git 追溯。
- `.env`、模型、下载中间文件、日志和验收证据保持私有。公开模板为 `.env.example`；镜像摘要、模型修订和校验和可纳入 Git。
- 统一入口 `deploy/manage.sh`。start 不重建已有容器，配置变更使用 restart；跨机导入已核验镜像使用 start-loaded/restart-loaded（禁止隐式构建/拉取），迁移流程见 docs/host-migration.md；GPU 使用 Docker 原生 CDI 与官方 toolkit-base；保留 FlashInfer 自动调优，相关故障须排查修复；本地镜像修复由 Dockerfile 和上游文件 SHA256 固定，详见 docs/runtime-patches.md；Docker `unless-stopped` 负责恢复，不再由 PM2 启动 vLLM。
- 禁止在宿主 `.venv` 安装 vLLM/Torch/CUDA。`pyproject.toml` 和 `uv.lock` 仅管理开发工具。不得用旧 Qwen3.5 模板覆盖新模型自带模板。
- 模型固定清单在 `deploy/vllm/model-manifest.json`。下载逐文件 SHA256 校验；模型挂载只读，编译缓存独立卷。更新模型须重新生成并审查清单、镜像兼容性和真实验收。
- 停止、清理只针对本项目。先查在途请求和服务归属，不使用全局 PM2 删除、Docker prune 或共享缓存清空。常规 stop 保留模型、缓存和证据。

## 主要入口

| 文件 | 职责 |
| --- | --- |
| `deploy/manage.sh` | Compose 生命周期、模型下载/校验、真实验收 |
| `deploy/vllm/compose.yaml` | 四卡 TP + EP 模型（PLE 驻留 GPU）、健康检查和日志轮转 |
| `deploy/vllm/Dockerfile` / `deploy/vllm/patches/` | 固定上游镜像和分布式调优缓存修复 |
| `deploy/vllm/serve.sh` | 可选 MTP 参数组装，exec 启动 vLLM |
| `scripts/benchmark.py` / `check_autotune.py` | 真实 SSE 性能对照、容器内四 rank 调优回归 |
| `deploy/vllm/model-manifest.json` | ModelScope 固定修订与逐文件 SHA256 |
| `scripts/download_model.py` | 标准库 + curl 断点下载和完整校验 |
| `scripts/smoke_test.py` | 真实 HTTP 文本、角色、流式、工具、视觉和推理验收 |

## 合同与验证

- 原生模型 API 7730。模型名以 `.env` 的 `SERVED_MODEL_NAME` 为准。
- API key 默认未配置／为空，业务请求无需鉴权；非空 `VLLM_API_KEY` 启用 Bearer 鉴权，验收脚本按配置检查对应行为；健康检查不等于可推理。不要把 Compose 展开配置或 `.env` 内容写入日志、对话、Git。
- 发布前执行 `uv run --group dev black --check scripts tests deploy/vllm/patches`、`uv run --group dev ruff check scripts tests deploy/vllm/patches`、`uv run --group dev pytest`、`bash -n deploy/manage.sh deploy/vllm/serve.sh`、`deploy/manage.sh config`。
- 真实验收执行 `deploy/manage.sh check`。失败必须报告，不能仅凭 health 返回 200 宣称部署完成。
- PCIe IPC / FlashInfer 0.7.0rc3 候选实验已按用户要求暂停，方案记录在 docs/performance-tuning.md；用户明确恢复前不执行升级或开启该通信后端。
- 性能复验用 scripts/benchmark.py；MTP 与通信优化须核对实际 backend、草稿接受率和功能验收，当前镜像不提供 PCIe IPC 接口，保持对应开关为 0。
- 质量与性能测量只描述实际样本和配置；短请求、纯色图片测试不代表长上下文、复杂图表质量或并发容量。
