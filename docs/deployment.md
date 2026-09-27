# 部署与恢复

## 前置条件

命令均在仓库根目录执行。需要 Linux、NVIDIA GPU 驱动、NVIDIA Container Toolkit、Docker Engine 28.3+ / Compose 2.24.4+ 和 Python 3.12+、curl。宿主机不安装 vLLM/Torch；uv 仅用于开发工具。

Qwen 模板使用四张 Blackwell GPU、TP=4 + EP=4。NVFP4、Qwen4Exp 架构和 FP8 PLE 混合量化均要求镜像支持，不能任意替换为老版本。模型卡要求至少包含 vLLM 提交 `d4d703caf908786416585ceb1f369e2e0363358b`；固定镜像已包含混合量化 MTP 修复，模板默认使用 MTP 3，实测依据见[性能调优](performance-tuning.md)。来源：[Qwen 模型卡](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)。

Embed 四卡模板为 DP4/TP1；三卡模板为 DP3/TP1，每张卡装载一个 Nemotron-3-Embed-8B-BF16 副本。`deploy/embed/` 提供四卡 Compose 和固定模型清单，`deploy/embed3/` 提供三卡 Compose；两者共用固定 vLLM 0.25 镜像及模型清单。两个模板现在均将单请求序列与每副本单次调度 token 上限设为 32768，最多 4 序列。已在三张 RTX PRO 6000 Blackwell 与四张 RTX 5000 Ada 上分别完成无截断满长输入验收，但显存余量不同，须逐机验证长请求与 MinerU 等服务的共同峰值。每张选定 GPU 均须能容纳 BF16 权重、运行时与业务请求的峰值显存；不能仅根据权重文件大小判断可运行。模型原生长度依据：[NVIDIA 模型说明](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/README.md)；实测边界见[验证](validation.md)。

公开 Compose 模板使用 Docker 原生 CDI 设备映射。对于从 apt 安装的 Docker，安装 NVIDIA 官方 `nvidia-container-toolkit-base` 即可生成 CDI 规格，不需要配置 legacy runtime 或重启 Docker。按 [NVIDIA 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)配置官方软件源后执行：

```bash
sudo apt-get install -y nvidia-container-toolkit-base=1.20.0-1
nvidia-ctk cdi list
systemctl is-enabled nvidia-cdi-refresh.service nvidia-cdi-refresh.path
```

刷新服务为 oneshot，成功退出后显示 inactive 正常；`.path` 与开机启动负责自动刷新。驱动变化后设备列表不正确时执行 `sudo systemctl restart nvidia-cdi-refresh.service`，不要重启 Docker。

Snap Docker 自带 NVIDIA 工具链，也会生成 CDI 规格；三卡现场的规格文件位于 `/var/snap/docker/current/etc/cdi/nvidia.yaml`，但其 Docker daemon 当前只扫描 `/etc/cdi` 与 `/var/run/cdi`。因此现场实际以私有 legacy NVIDIA GPU Compose 覆盖启动。该覆盖保存在被忽略的 `output/instances/embed3.compose.yaml`；存在时统一入口为 `embed3` 附加覆盖。现场的真实请求与联合负载结果不能证明公开 CDI 模板在该 Snap Docker 上可原样启动。依据：[Canonical Docker Snap 的 NVIDIA 支持与 daemon 配置](https://github.com/canonical/docker-snap)、[Docker 原生 CDI 目录配置](https://docs.docker.com/reference/cli/dockerd/#configure-cdi-devices)、[NVIDIA CDI 规格说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)。

要在 Snap Docker 上试用公开 CDI Compose，可先核对 `docker info` 的 CDI spec directories 与 Snap 生成的规格文件。若 daemon 未扫描该目录，无需直接更换 Docker 安装方式：在整机维护窗口备份 `/var/snap/docker/current/config/daemon.json`，保留其现有设置，并将 `cdi-spec-dirs` 设为包含 `/etc/cdi`、`/var/run/cdi` 和 `/var/snap/docker/current/etc/cdi`。验证 JSON 后重启 Snap Docker daemon，再用 `docker info` 核对目录，并以临时 GPU 容器和本项目真实请求验收公开 Compose。当前三卡现场的 Docker 未启用 live restore，daemon 重启会中断同一引擎下的其他容器；应先与这些服务的负责人安排维护窗口，不在本项目常规部署时操作共享 daemon。完成原生 CDI 验收后，才能移除私有 legacy 覆盖。迁移到新宿主时仍须重新核对 GPU 运行时及该覆盖的适用性。

检查 `nvidia-smi`、`docker compose version`、`/dev/nvidia-uvm`，并确认所用端口与 GPU 的归属。Qwen 模型文件总量约 132.7 GB，Embed 约 15.9 GB；额外预留下载、容器镜像和编译缓存空间。默认下载目录均位于本仓库：`models/Qwen3.8-Flash-Next-NVFP4/` 与 `models/Nemotron-3-Embed-8B-BF16/`。目录由下载命令创建，权重文件被 Git 忽略；三卡和四卡 Embed 共用同一份 Nemotron 权重。

## 私有配置

```bash
cp -n .env.example .env
chmod 600 .env
```

默认 `VLLM_API_KEY` 留空，也可不配置，业务请求无需 Authorization。需要鉴权时，将非空 key 写入 `.env` 并执行 `./deploy/manage.sh restart model`，客户端随后携带 `Authorization: Bearer <key>`；清空该值并重启即可关闭鉴权。可用 `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'` 生成随机 key，不要提交或输出到对话。模型 API 默认在所有网卡监听；只本机使用可设 `VLLM_HOST=127.0.0.1`。

| 配置 | 模板值 / 含义 |
| --- | --- |
| `VLLM_IMAGE` | 本地修复镜像 `tiangong-vllm:qwen38-autotune-v1`；Dockerfile 固定上游 digest |
| `MODEL_DIR` | `./models/Qwen3.8-Flash-Next-NVFP4`，相对仓库根 |
| `SERVED_MODEL_NAME` | `nv-community/Qwen3.8-Flash-Next-NVFP4` |
| `VLLM_PORT` | 7730 |
| `VLLM_API_KEY` | 空／未配置：关闭鉴权；非空：启用 Bearer |
| `GPU_0..GPU_3` / `TENSOR_PARALLEL_SIZE` | GPU 0/1/2/3，TP4 + EP4 |
| `GPU_MEMORY_UTILIZATION` | Qwen 模板为 0.85；是每张卡的引擎预算，启用前按机器及同卡服务核对 |
| `MAX_MODEL_LEN` | 262144；请求输入与输出合计上限 |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | 16 / 8192 |
| `SPECULATIVE_CONFIG` | 模板为 `{"method":"mtp","num_speculative_tokens":3}`；空值关闭 MTP |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` | 默认 0；当前 FlashInfer 缺少 IPC 接口，设置 1 仍会回退 NCCL |

Embed 使用同一私有 `.env`，但三卡和四卡为独立 Compose 项目和容器；同一主机只启动其中一种。四卡模板的主要配置如下：

| 配置 | 模板值 / 含义 |
| --- | --- |
| `EMBED_IMAGE` | 本地标签 `tiangong-embed:vllm0.25.0`；`pull embed` 从 `deploy/manage.sh` 固定的上游摘要拉取并标记，便于跨机导出/导入 |
| `EMBED_MODEL_DIR` / `EMBED_SERVED_MODEL_NAME` | 只读模型目录 / API 中使用的 Embed 模型名 |
| `EMBED_HOST` / `EMBED_PORT` | 默认 `127.0.0.1` / 7731；局域网访问时在私有 `.env` 中绑定本机局域网 IP |
| `EMBED_API_KEY` | 空／未配置：关闭鉴权；非空：仅 Embed API 启用 Bearer |
| `EMBED_GPU_0..EMBED_GPU_3` | 四个副本使用的 GPU；与同卡其他服务逐卡核对 |
| `EMBED_GPU_MEMORY_UTILIZATION` | **必须按部署机填写**；本模型在 vLLM 0.25 中用于启动时空闲显存检查，模板留空，没有跨机器通用比例 |
| `EMBED_MAX_MODEL_LEN` / `EMBED_MAX_NUM_BATCHED_TOKENS` / `EMBED_MAX_NUM_SEQS` | 32768 / 32768 / 4；一个满长请求可占满单副本该次调度的 token 额度 |

三卡模板使用独立的 `EMBED3_*` 配置：`EMBED3_GPU_0..EMBED3_GPU_2` 指定三张卡，`EMBED3_HOST=127.0.0.1`、`EMBED3_PORT=7731` 为默认入口；它与四卡模板在同一主机互斥。`EMBED3_IMAGE` 默认与四卡模板相同，`EMBED3_MODEL_DIR` 默认使用同一只读权重目录。`EMBED3_SERVED_MODEL_NAME`、`EMBED3_API_KEY`、`EMBED3_MAX_MODEL_LEN`、`EMBED3_MAX_NUM_BATCHED_TOKENS` 和 `EMBED3_MAX_NUM_SEQS` 分别独立配置，后三者为 32768 / 32768 / 4。`EMBED3_GPU_MEMORY_UTILIZATION` **必须按部署机填写**，公开模板不提供跨机器通用比例。三卡缓存卷独立于四卡。

Qwen 与 Embed 的利用率参数在当前服务中的作用不同。Qwen 是生成模型，`GPU_MEMORY_UTILIZATION` 会参与可用 KV cache 容量的计算。Nemotron Embed 是非因果注意力的 encoder-only pooling 模型，运行时不建立 KV cache；在当前 vLLM 0.25 实现中，`EMBED_GPU_MEMORY_UTILIZATION` 与 `EMBED3_GPU_MEMORY_UTILIZATION` 主要将**GPU 总显存 × 配置比例**与启动瞬间的空闲显存比较，空闲不足即拒绝启动。它不预分配该比例的显存，也不是运行期间的进程上限或其他服务的隔离边界。依据：[NVIDIA 模型配置](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/config.json)中的 `is_causal=false`、`pooling=avg`，以及 [vLLM 0.25 的启动检查](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/v1/worker/utils.py)和[跳过 encoder-only KV cache 的实现](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/v1/worker/gpu/attn_utils.py)。

每台机器仍须根据 GPU 容量、启动时已有进程占用、输入长度和并发，在私有 `.env` 设定检查比例并预留请求峰值余量。与 MinerU 等服务共卡时，分别测量其权重、KV cache 与请求峰值，以及 Embed 权重、运行时和请求峰值；降低 Embed 的比例只会放宽启动检查，不能保证其运行期显存变小。四张 32 GiB Ada 卡在满长并发请求后的最低已测空闲仅 3505 MiB，不能据此增加批量 token 上限或宣称 MinerU 高峰共存安全；三张约 96 GiB Blackwell 卡余量更大，若需在同一副本的一次调度中容纳多条满长请求，可另行评估更大的 `EMBED3_MAX_NUM_BATCHED_TOKENS`，并重新做长请求与共同负载验收。Qwen 与 Embed 不自动共驻留，Qwen 的 0.85 模板值不适用于未经重新核算的共卡场景。

配置采用简单 `KEY=value`，避免插值、内联注释或多行值，以便 Compose 与标准库运维脚本一致读取。不要执行会将密钥展开到终端的 `docker compose config`；按选择的模板使用 `deploy/manage.sh config model`、`config embed` 或 `config embed3` 做静默检查。

## 模型准备与启动

```bash
./deploy/manage.sh download model
./deploy/manage.sh pull model
./deploy/manage.sh config model
./deploy/manage.sh start model
./deploy/manage.sh status model
./deploy/manage.sh logs model
```

下载从 ModelScope 固定提交取文件，curl 支持中断续传，SHA256 验证完成后原子替换单个文件。`verify` 检查完整清单；不要在模型正在加载时修改权重。推理只读本地权重并启用离线模式，不在启动时静默换模型版本。

`pull` 会拉取固定基础镜像并构建修复层；也可使用 `build` 复用本地基础镜像。首次加载、编译可能需较长时间。`logs` 为持续跟踪，Ctrl-C 只退出日志查看。健康后执行：

```bash
./deploy/manage.sh check model
```


从另一台机器复制模型和镜像时，按[跨机迁移](host-migration.md)校验，并使用 `start-loaded` 启动已导入镜像；需要重建容器时用 `restart-loaded`，避免隐式下载或构建。

## Embed 准备与启动

四卡模板先填写私有 `.env` 的 `EMBED_GPU_MEMORY_UTILIZATION`，核对四张 GPU、端口、启动空闲显存和已有服务。在不影响其他项目的前提下，按下列顺序启动独立 Embed 服务：

```bash
./deploy/manage.sh download embed
./deploy/manage.sh verify embed
./deploy/manage.sh pull embed
./deploy/manage.sh config embed
./deploy/manage.sh start embed
./deploy/manage.sh status embed
./deploy/manage.sh check embed
```

`download embed` 从 `deploy/embed/model-manifest.json` 指定的 ModelScope revision 下载并逐文件校验 SHA256，`verify embed` 可复查本地文件。`pull embed` 获取固定摘要的上游镜像，并赋予 `EMBED_IMAGE` 指定的本地标签。Embed 的 `start`/`restart` 均只使用本地镜像；跨机导入后可使用 `start-loaded embed`。Embed 默认只供本机访问；需要局域网访问时在私有 `.env` 设置该机的 `EMBED_HOST`，三卡则设置 `EMBED3_HOST`。完整请求合同见[AI 接入](ai-integration.md)。`check embed` 同时验证短检索和一条无截断满长请求，运行前应确认同卡负载与显存余量；健康检查通过不能代替它。

三卡模板填写私有 `EMBED3_GPU_MEMORY_UTILIZATION`、`EMBED3_GPU_0..EMBED3_GPU_2` 和端口配置后，执行相同操作，但组名为 `embed3`：

```bash
./deploy/manage.sh download embed3
./deploy/manage.sh verify embed3
./deploy/manage.sh pull embed3
./deploy/manage.sh config embed3
./deploy/manage.sh start embed3
./deploy/manage.sh status embed3
./deploy/manage.sh check embed3
```

`download/verify embed3` 仍使用 `deploy/embed/model-manifest.json`，`pull embed3` 使用与四卡相同的固定 vLLM 0.25 摘要，默认标记同一可迁移镜像。三卡项目为 `tiangong-embed3`，默认本机端口 7731，缓存卷独立。启动 `embed3` 前须停止同一主机上的 `embed`；管理脚本会拒绝两种 Embed 服务同时运行。三卡 Blackwell 现场已通过 `check embed3` 的真实向量与无截断满长输入验收；共同负载样本与限制见[验证](validation.md)。

Qwen 与两种 Embed 的 `start`、`restart`、`stop`、`status`、`logs`、`download`、`verify`、`config`、`check` 均分别指定 `model`、`embed` 或 `embed3`。不带组名和 `all` 是 `model` 的兼容入口，不会同时操作其他项目。修改所选 Embed 配置后使用对应的 `restart`；此操作会中断在途 Embed 请求。

## 维护、停止、恢复

1. 停止新增请求，查看目标服务 `/metrics` 的运行与等待请求，等待归零；同时确认服务及 GPU 属于哪个项目。
2. 使用 `./deploy/manage.sh stop model`、`stop embed` 或 `stop embed3` 停止目标服务，保留模型和缓存。
3. 恢复时对同一服务执行对应的 `start`，随后执行对应的真实验收。
4. 修改配置后对目标服务执行对应的 `restart`，重建容器以应用变更；此操作会中断该服务的在途请求。

`restart: unless-stopped` 配合已启用的 Docker 系统服务完成机器重启恢复，无需 PM2。主动 stop 的容器保持停止。不要使用全局 Docker 清理或重启共享 daemon。

旧原生部署迁移时先识别 PM2 名称及进程树，确认请求排空后删除对应记录、保存 PM2 列表，再删除明确归属的旧模型与环境。不要按“Qwen”字符串批量删除其他项目的 embedding 服务或共享缓存。

## 故障排查

- 镜像拉取或 PyPI TLS 超时：重试，使用已配置且可信的网络出口；不要修改共享 Docker 配置来掩盖问题。国内 Python 镜像可用于锁文件解析，仍须保留包版本和哈希。
- CUDA 初始化失败：检查容器 GPU 可见性和 UVM 设备；`nvidia-smi` 正常不保证 CUDA 分配成功。在目标镜像内执行一次 CUDA tensor 运算。
- 不支持模型架构／FP8 PLE：核对模型 revision 和镜像内实际 vLLM 版本，不用旧 Qwen 模板或更换模型冒充修复。
- NVFP4 `Intermediate size padding`：保持模板中的 `--enable-expert-parallel`。此模型专家宽度 640，纯 TP4 后每片 160，当前 FLASHINFER_CUTLASS 的 gated 权重 padding 不支持该形状；EP4 保持专家完整宽度。
- 自动调优缓存停滞：确认使用[修复镜像](runtime-patches.md)，保留自动调优并检查各 rank 缓存条目；不要用关闭调优作为默认解决方案。
- OOM：确认其他进程占用，降低上下文、并发或多模态容量；记录修改并重做验收。
- Embed 启动提示缺少 `EMBED_GPU_MEMORY_UTILIZATION` 或 `EMBED3_GPU_MEMORY_UTILIZATION`：在私有 `.env` 按本机启动空闲显存填写，不把一个机器的试验值复制成通用默认。各副本必须逐卡检查可用显存。
- 共卡时单独验收通过、同时运行却 OOM 或延迟剧增：核对每个服务的实际 GPU、MinerU 等生成模型的 KV cache 与请求峰值，以及 Embed 的计算显存。调整各自适用的参数并共同负载复验；降低 Embed 的显存比例不能替代降低请求峰值。
- HTTP 401：启用鉴权时，客户端须使用 `.env` 的 key；已清空配置但仍返回 401 时，确认执行了 `restart model` 以应用变更。
- `/health` 正常但首请求挂起：检查 worker、编译和模型加载日志，必须以真实请求返回验收。

升级前记录当前镜像摘要、模型清单和配置；保留仍需回退的镜像。回退只恢复已知兼容组合，不覆盖正在使用的模型文件。
