# TianGong AI vLLM Serve 协作说明

本项目通过 Docker 下载、校验和运行多个模型（vLLM 与专用 SDK）。模型配置与机器实例必须分离：`deploy/profiles/` 固定镜像、模型清单、默认参数和验收；私有 `deployments/<实例名>.json` 定义 GPU 列表、TP/DP/PP、上下文、显存预算、端口、路径及参数覆盖。同一模型可有多个实例，不把当前机器的三卡/四卡拓扑硬编码进通用管理逻辑。

## 修改与部署约定

- 版本发布直接提交并 push 到 main，不创建功能分支或 PR。
- 修改代码、配置或行为时同步维护本文件和专题文档。README 保持用途与入口简明；根目录文档仅 README.md、AGENTS.md，专题在 docs，部署在 deploy，脚本在 scripts，示例在 examples。
- 文档使用相对路径，不写主机专属路径、凭证或逐轮操作流水账。`.env`、`deployments/`、模型、日志和验收证据保持私有；可复用模板放 `deploy/examples/`，镜像摘要与模型 revision/SHA256 可纳入 Git。
- 统一入口 `deploy/manage.sh <action> <实例名>`；模型配置驱动管理，不为每个新增模型增加 shell 分支。start 不重建已有容器，配置变化使用 restart；start-loaded/restart-loaded 禁止构建或拉取。all 仅包含 deployments 中的显式实例。
- 每实例独立 Compose 项目、端口、网络和编译缓存；权重只读挂载，可共享已验证目录。启动前核对 GPU 归属和显存预算，不自动抢占其他实例或项目资源。
- GPU 注入按机器选择：cdi 使用 Docker 原生 CDI，nvidia 使用 NVIDIA runtime、设备预留及 UVM（兼容当前 Snap Docker）。不能只设置 --gpus 而忽略主机 runtime 模式。
- Docker unless-stopped 管理退出恢复，不再由 PM2 启动新推理实例。停止只针对指定实例，先检查在途请求；禁止全局 PM2 删除、Docker prune 或共享缓存清空。stop 保留磁盘资源。
- 禁止在宿主 .venv 安装 vLLM/Torch/CUDA。pyproject.toml/uv.lock 仅管理开发工具。原生迁移期间使用 `UV_PROJECT_ENVIRONMENT=.venv-dev`，保留旧 .venv 与私有 PM2 配置供回退；真实验收通过后才移除对应 PM2 项并保存列表。
- 权重固定 ModelScope revision、大小与 SHA256；更新时在新目录下载核验。容器推理依赖整体随固定镜像升级，不在启动时 pip update。不要把一种模型的模板或参数强加给另一种模型。

## 入口与兼容

| 入口 | 职责 |
| --- | --- |
| `deploy/manage.sh` / `scripts/manage_models.py` | 实例初始化、列举、Docker 生命周期、下载与验收 |
| `deploy/profiles/` / `deploy/examples/` | 可复用模型配置与机器实例模板 |
| `deploy/embedding/model-manifest.json` | Qwen3-Embedding-8B 的权重清单 |
| `deploy/vllm/` | Qwen3.8 权重清单、镜像补丁及旧 Compose 参考 |
| `scripts/download_model.py` | 断点下载、逐文件校验 |
| `scripts/check_embedding.py` / `scripts/smoke_test.py` | 模型专用真实 API 验收 |

新部署按 docs/models.md 使用实例文件。旧 .env 生成模型通过显式 model 兼容转换，保留 tiangong-vllm/model 与旧缓存，不属于 all；迁移为显式实例时保留 project_name/service_name。不能因另一个模型缺少权重或 GPU 而阻止当前实例的操作。

- 模型配置 runtime 默认为 vllm；replicated-http 为每张 GPU 创建一个独立 HTTP worker 与统一 Nginx 网关，不允许 TP/DP/PP 参数。扩缩容必须 restart，worker 各自只看到一张 GPU。新专用 SDK 通过此运行方式接入。

## GPUStack 控制面

- `deploy/gpustack/worker.sh`独立管理CDI worker，私有注册配置在`.secrets/gpustack-worker/worker.env`；只在已分配的托管GPU上验收，不自动接管旧容器。说明见docs/gpustack-workers.md；Snap/NVIDIA runtime主机需单独适配，不套用CDI模板。
- `deploy/gpustack/manage.sh` / `scripts/manage_gpustack.py` 独立管理正式控制面，版本和镜像摘要固定在 `deploy/gpustack/release.json`。私有配置位于 `deployments/gpustack/server.json`，不加入原生模型的 all 范围。控制面默认只监听回环，不挂载 Docker socket、不分配 GPU，显式禁用内嵌 worker。
- Ubuntu 控制机可选 `deploy/gpustack/install-docker-ubuntu.sh`，默认 --check 只读；--install 由原主机 sudo 执行，保留 daemon 配置并设置 ip-forward-no-drop，已有 Docker 时不自动改宿主网络。此操作独立于模型部署与 GPUStack 启动。
- 先接入现有模型 Provider 验证路由，再通过 UI 注册 worker 和独立正式 deployment。代理旧入口不等于接管生命周期；同一实例只能由原生部署器或 GPUStack 之一管理，不自动转换或接管生产容器。
- init 不覆盖已有配置；bootstrap 与 API key 留在 `.secrets/gpustack/` 且权限为 600。config 静默校验；start/restart 不拉取或构建；stop 保留数据。check 仅验证认证模型列表，不能代替推理、分流和故障恢复验收。操作与边界见 docs/gpustack.md，配置检查另执行 `bash deploy/gpustack/manage.sh config`。

- 正式 Compose 项目为 tiangong-gpustack，持久卷 tiangong-gpustack_data。变更名称须冷备并复制原卷，保留路由 ID、注册身份与 API key；核验后才删除旧资源。业务使用稳定主路由，节点路由仅用于诊断。
- 旧 Nemotron embed/embed3 显式命令由 manage.sh 转发到 manage-legacy.sh，保留互斥与 32768 token 配置；它们不属于通用 all。固定权重见 deploy/embed/model-manifest.json。GPUStack 托管时不得再启动同一原生实例。
- 下载器保留旧 --model 与新 --group 参数别名；兼容入口的 download/verify 同样使用固定清单，不恢复已退役的服务。
- 业务接入示例使用 GPUStack 统一网关与应用专用 Bearer key，不提供旧节点端口的兼容转发模板；原生部署模板只用于显式独立部署或回退。

## 已有模型约束

- DeepSeek-V4.1-Flash-1017-NVFP4：固定 ModelScope 清单与 vLLM 0.33.0 MXFP4 Engram 回移补丁，见 docs/deepseek-v41.md。GPUStack 单节点八卡单副本，优先保留 1,048,576 token 上下文；不能用健康检查或短输入证明满长容量。真实验收脚本为 scripts/check_deepseek_v41.py，覆盖文本、工具、图片、reasoning、SSE、长上下文召回和明确标注缓存条件的吞吐。SM120 不套用 SM100 MegaMoE 参数，不开启已暂停的 PCIe IPC 实验。

- H3 NVFP4：见 docs/h3.md；固定 FastH3 Omni NVFP4 QAD 和原版视频/音频 VAE 清单，ModelScope 镜像仍按原 HF 散列验证。已通过 SM120 单卡、GPUStack 鉴权路由的 5 秒 480P/768P 音视频验收；默认 36 层解码器、逐层卸载、运行并发 1。快速解码器、长视频与压力测试尚未验收。专用脚本与固定依赖镜像位于 scripts/download_hf_snapshot.py、scripts/check_h3.py、deploy/h3，GPUStack 模板位于 deploy/examples/h3-gpustack-*.json，不加入原生 all；不能把 Ref2VA 描述成无需参考的纯文生视频。

- LCA-Nemotron-Embed-1B：固定发布版 Sentence Transformers 参考路径，GPUStack 每卡一个独立 custom deployment，统一鉴权 generic-proxy 路由。显式 query/document、512/2048 tokens、2048 维；不接受 dimensions 或 encoding_format。构建、资源限制、固定依赖和完整 HTTP/reference benchmark 验收见 docs/lca-nemotron.md；实现 deploy/lca-nemotron，check_script 为 scripts/check_lca_nemotron.py。不得沿用通用 Nemotron 的 vLLM 镜像或 API 合同。
- Laya Multilingual：SDK 0.3.4，默认 API 7720 的 `/v1/decisions`；1024 tokens，最多 32 问题/请求，每卡一个完整模型。单卡验收通过再扩到三卡；不得把它配置为跨卡模型并行。SDK 的 tokenizer 修复只作用于临时副本，模型目录只读；禁止把 CPU 回退当作 GPU 部署成功。见 docs/laya.md。
- Qwen3-Embedding-8B：默认模型名 Qwen/Qwen3-Embedding-8B，默认 API 7710，4096 维输出；显式 pooling runner 和 Matryoshka 声明支持 dimensions。三卡 DP3 是一个机器模板，单卡也应能配置。不得混入生成模型的 MTP/工具解析参数。
- Qwen3.8-Flash-Next-NVFP4：默认 API 7730、262144 token 上下文、TP4 + EP4、GPU PLE、FlashInfer 自动调优、MTP 3 与 NCCL。保留自动调优，相关故障须修复；运行时补丁由 Dockerfile 和上游 SHA256 固定，见 docs/runtime-patches.md。
- PCIe IPC / FlashInfer 0.7.0rc3 实验已暂停，用户明确恢复前不升级或开启该通信后端。当前生成镜像不提供对应接口，开关保持 0。

## 验证与发布

- 默认 API key 空；非空 key 从实例 api_key_env 指向的私有变量读取，业务请求使用 Bearer。不得输出 Compose 展开配置、.env 或凭证。
- 本地检查：设置 `UV_PROJECT_ENVIRONMENT=.venv-dev` 后执行 `uv run --group dev black --check scripts tests deploy/vllm/patches deploy/laya deploy/lca-nemotron`、`uv run --group dev ruff check scripts tests deploy/vllm/patches deploy/laya deploy/lca-nemotron`、`uv run --group dev pytest`；执行 `bash -n deploy/manage.sh deploy/vllm/serve.sh deploy/lca-nemotron/build.sh` 与目标实例 `manage.sh config`。GPUStack 托管的 LCA 使用独立 backend/deployment 模板与其 check_script，不加入原生模型管理器的 all。
- 真实验收用 `manage.sh check <实例名>`。embedding 检查单条/批量、中英文、4096/1024 维、base64、并发、错误输入和检索排序；原生迁移另对比旧向量。Laya 检查中英文 typed decisions、概率范围、错误输入和请求分布；生成模型检查文本、角色、SSE、工具、图片和推理。失败必须报告，health 不等于验收。
- 新模型必须配置对应 check_script 才能声明通过验收；不要用替身测试冒充 GPU/模型测试。质量和性能结论只覆盖实际样本；短输入不代表最大上下文或持续并发能力。
