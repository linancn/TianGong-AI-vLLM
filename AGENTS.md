# TianGong AI vLLM Serve 协作说明

本仓库面向不同 GPU 机器按需部署独立的 vLLM 模型服务，推理引擎只在 Docker 内运行。已验收的四卡模板：ModelScope `nv-community/Qwen3.8-Flash-Next-NVFP4` 使用 262144 token（256K）上下文、TP4 + EP4、GPU PLE、FlashInfer 自动调优、MTP 3 与 NCCL；`nv-community/Nemotron-3-Embed-8B-BF16` 使用 DP4、每卡一个 BF16 副本。三卡 Embed DP3 已在三张 RTX PRO 6000 Blackwell 与 Unstructure Serve 的 MinerU DP3 上完成有限联合试跑；目标宿主使用私有 legacy NVIDIA GPU Compose overlay，公开 CDI 模板尚未在其 Snap Docker 环境原样验收。其他卡数与模型的 profile 方案见 docs/template-design.md，尚不能仅改 `.env` 就启用。各服务按需启动，API key 默认空且可配置。本项目只保留模型服务，不引入应用代理或文档解析队列。

## 文档与修改约定

- 版本发布直接提交并 push 到 `main`，不创建功能分支或 PR。

- 修改代码、配置或行为时，同步维护本文件及相关专题文档。README 保持用途和入口简明；部署步骤、接口合同、验证边界分别集中到专题说明。
- 根目录文档仅 README.md、AGENTS.md；专题在 docs，部署在 deploy，运维辅助在 scripts，调用示例在 examples。
- 文档使用相对文件路径；命令以仓库根目录为工作目录。不要写机器专属路径、凭证或逐轮操作流水账；历史部署由 Git 追溯。
- `.env`、模型、下载中间文件、日志和验收证据保持私有。公开模板为 `.env.example`；镜像摘要、模型修订和校验和可纳入 Git。
- 统一入口 `deploy/manage.sh`；`model`、`embed` 和 `embed3` 是独立 Compose 项目，均须显式指定，`all` 仅为 `model` 兼容别名。同一主机上 `embed` 与 `embed3` 互斥，避免重复装载相同模型。start 不重建已有容器，配置变更使用 restart；两种 Embed 从固定上游摘要拉取并标记可迁移的本地镜像，启动不隐式拉取。跨机导入已核验镜像使用 start-loaded/restart-loaded，迁移流程见 docs/host-migration.md；公开 Compose 使用 Docker 原生 CDI 与官方 toolkit-base。三卡目标宿主的 Snap Docker 已生成 CDI 规格，但当前 daemon 未扫描其目录；现场试跑使用私有 legacy NVIDIA GPU Compose overlay，公开模板尚未在该宿主原样验收。Qwen 保留 FlashInfer 自动调优，相关故障须排查修复；本地镜像修复由 Dockerfile 和上游文件 SHA256 固定，详见 docs/runtime-patches.md；Docker `unless-stopped` 负责恢复，不再由 PM2 启动 vLLM。
- 禁止在宿主 `.venv` 安装 vLLM/Torch/CUDA。`pyproject.toml` 和 `uv.lock` 仅管理开发工具。不得用旧 Qwen3.5 模板覆盖新模型自带模板。
- 模型固定清单在 `deploy/vllm/model-manifest.json` 与 `deploy/embed/model-manifest.json`；`embed3` 共用后者。默认下载到仓库内 Git 忽略的 `models/Qwen3.8-Flash-Next-NVFP4/` 与 `models/Nemotron-3-Embed-8B-BF16/`，三卡和四卡 Embed 共用后一个目录。下载逐文件 SHA256 校验；模型挂载只读，三个 Compose 项目的缓存卷独立。更新模型须重新生成并审查清单、镜像兼容性和真实验收。
- 各部署机在私有 `.env` 中核对 GPU、监听地址、端口及显存条件；启动四卡 Embed 必须填写 `EMBED_GPU_MEMORY_UTILIZATION`，启动三卡 Embed 必须填写 `EMBED3_GPU_MEMORY_UTILIZATION`。两个模板的单请求上限和每副本单次调度 token 上限均为 32768，最多 4 序列；一个满长请求可能占满该次调度的 token 额度，不能将四个上限相乘当作单批容量。Nemotron Embed 是无 KV cache 的 encoder-only pooling 模型；在当前固定的 vLLM 0.25 中，显存比例主要检查启动时空闲显存是否达到 GPU 总显存乘以该值，不是显存预留或进程上限。Qwen 的 `GPU_MEMORY_UTILIZATION` 则参与 KV cache 容量计算。运行 MinerU 等同卡服务时须逐卡测量长请求后的显存余量，再做共同负载验收；32 GiB 与 96 GiB 卡分别选择私有检查比例及并发预算，不沿用另一台机器的值。历史与现行样本及边界见 `docs/validation.md`。Qwen 与 Embed 不自动共驻留。
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
| `deploy/embed/compose.yaml` / `model-manifest.json` | 独立四卡 DP4 Embed 服务、固定 ModelScope 修订与摘要 |
| `deploy/embed3/compose.yaml` | 独立三卡 DP3 Embed 服务，共用 Embed 模型清单与固定镜像，独立缓存 |
| `scripts/download_model.py` | 标准库 + curl 断点下载和完整校验 |
| `scripts/smoke_test.py` | 真实 HTTP 文本、角色、流式、工具、视觉和推理验收 |
| `scripts/check_embed.py` | 真实 HTTP 向量维度、归一化和检索顺序验收 |
| `docs/template-design.md` | 多模型、GPU 拓扑和私有部署实例的后续改造方案 |

## 合同与验证

- Qwen API 默认端口 7730，四卡与三卡 Embed 均默认只监听本机 7731，且同一主机互斥；跨主机访问在私有 `.env` 中绑定各自局域网 IP。模型名分别以 `.env` 的 `SERVED_MODEL_NAME`、`EMBED_SERVED_MODEL_NAME` 与 `EMBED3_SERVED_MODEL_NAME` 为准。Embed 检索使用 `/v2/embed` 的 `input_type=query/document`，不同用途的输入不可混用。
- API key 默认未配置／为空，业务请求无需鉴权；非空 `VLLM_API_KEY`、`EMBED_API_KEY` 或 `EMBED3_API_KEY` 分别为对应服务启用 Bearer 鉴权，验收脚本按配置检查对应行为；健康检查不等于可推理。不要把 Compose 展开配置或 `.env` 内容写入日志、对话、Git。
- 发布前执行 `uv run --group dev black --check scripts tests deploy/vllm/patches`、`uv run --group dev ruff check scripts tests deploy/vllm/patches`、`uv run --group dev pytest`、`bash -n deploy/manage.sh deploy/vllm/serve.sh`、`deploy/manage.sh config model`；配置四卡或三卡 Embed 的机器还须分别执行 `deploy/manage.sh config embed` 或 `deploy/manage.sh config embed3`。
- 真实验收按本次实际启动的服务分别执行 `deploy/manage.sh check model`、`deploy/manage.sh check embed` 或 `deploy/manage.sh check embed3`；Embed 检查还核对 `/v1/models` 的实际长度及一条 `truncate=NONE` 的满长输入，运行前应先确认同卡负载与显存余量。失败必须报告，不能仅凭 health 返回 200 宣称部署完成。与 MinerU 共卡时还要在共同负载下核对显存峰值、错误与延迟；四卡 Ada 与三卡 Blackwell 已完成的样本及其边界见 docs/validation.md。
- PCIe IPC / FlashInfer 0.7.0rc3 候选实验已按用户要求暂停，方案记录在 docs/performance-tuning.md；用户明确恢复前不执行升级或开启该通信后端。
- 性能复验用 scripts/benchmark.py；MTP 与通信优化须核对实际 backend、草稿接受率和功能验收，当前镜像不提供 PCIe IPC 接口，保持对应开关为 0。
- 质量与性能测量只描述实际样本和配置；短请求、纯色图片测试不代表长上下文、复杂图表质量或并发容量。
