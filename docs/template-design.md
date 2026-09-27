# 多模型与多机器部署模板设计

本文是多模型实例架构方案。现有真实验收基线包括：Qwen3.8-Flash-Next-NVFP4 在四张 RTX PRO 6000 Blackwell Max-Q 96 GB 上运行 TP4 + EP4；Nemotron-3-Embed-8B-BF16 在四张 RTX 5000 Ada 32 GiB 上运行 DP4；Nemotron Embed 在三张 RTX PRO 6000 Blackwell Max-Q 95.6 GiB 上运行 DP3，并与独立 Unstructure Serve 的 MinerU DP3 完成有限联合试跑。三卡现场使用私有 legacy NVIDIA GPU Compose 覆盖，MinerU 为 4.0.5/vLLM 0.21，因此与四卡 Ada 的软件组合不同。其他 GPU 型号、显存容量、DP1/DP2 及新增模型都需要单独验证，不能因为配置可生成就标记为可用。

## 目标与边界

一台机器可按需求启动零个或多个独立 vLLM 实例；不同机器选择各自适用的模型、运行时与 GPU 拓扑。模型与镜像版本仍固定，GPU、端口、显存预算和密钥留在部署机的私有配置。每个实例单独管理容器、缓存、验收和生命周期，避免一项服务的操作影响另一项。

模板不自动判断某模型适合某 GPU。注册新组合须核对模型架构、精度、镜像中的 CUDA 与 vLLM、GPU 计算能力、实际显存，以及模型专属 API 和业务样本。现有 Qwen 的 NVFP4、GPU PLE、FlashInfer 自动调优、MTP 和修复镜像是该模型的配置，不能移植为所有 LLM 的默认参数。Embed 的 pooling、`/v2/embed` 与 DP 副本语义也不能当作生成模型参数。

## 三层配置

| 层 | 内容 | 作用 |
| --- | --- | --- |
| 模型／运行时 profile | 模型 ID、ModelScope 固定 revision 与逐文件 SHA256 清单；镜像 digest 或固定 Dockerfile/补丁；任务类型、精度、模型专属 vLLM 参数、API 合同和验收脚本 | 固定经过审查的一套模型与运行时，不让部署机任意拼接不兼容参数 |
| GPU 拓扑 profile | 所需 GPU 数量、TP/DP/EP 关系、已验证的 GPU 架构与容量、设备映射规则、经过验收的上下文及批处理起点 | 明确一套可部署的并行方式；卡数相同不代表模型兼容 |
| 私有部署 instance | 选择上述两个 profile、实例名、GPU 设备映射方式、主机监听地址与端口、模型目录、每卡启动所需空闲显存、KV 容量或自动 KV 目标、预留空间、API key | 表达某台机器实际运行什么；不得提交到 Git |

建议将公开 profile 放在 `deploy/profiles/`，保留 `deploy/vllm/model-manifest.json` 和 `deploy/embed/model-manifest.json` 作为当前固定清单。私有 instance 放在已忽略的 `output/instances/`，由公开示例文件说明字段；渲染出的 Compose 放在 `output/generated/`，文件权限限制为当前用户。密钥使用每实例的私有环境文件，`plan`、`status` 和错误信息均不得输出其值。实例名仅允许安全的短标识，不能用未经检查的输入拼接文件路径、Compose 项目名或 shell 命令。

当前固定入口与验收状态如下。`model`、`embed`、`embed3` 是已落地的静态模板；下文的通用 profile／instance 渲染器仍是后续设计，不应当作现有命令使用。

| 模型／运行时 profile | 拓扑 profile | 当前入口与已知边界 |
| --- | --- | --- |
| `qwen38-nvfp4-patched` | `blackwell-tp4-ep4` | `model`；四张 RTX PRO 6000 Blackwell Max-Q 96 GB 已验收，使用固定镜像、补丁、模型清单和专属推理验收 |
| `nemotron3-embed8b-bf16-vllm025` | `ada-dp4` | `embed`；四张 RTX 5000 Ada 32 GiB 已验收，使用固定 vLLM 0.25 镜像、模型清单和向量验收 |
| `nemotron3-embed8b-bf16-vllm025` | `blackwell-dp3` | `embed3`；三张 RTX PRO 6000 Blackwell Max-Q 已完成真实向量 API 和旧版 MinerU DP3 的短时联合试跑；现场使用私有 legacy GPU 覆盖，公开 CDI 配置仍待该宿主验证 |

针对现有机器，硬件清单可归纳成三类，避免按主机 IP 复制公共模板：

| 硬件类 | 可登记的模型拓扑 | 当前状态 |
| --- | --- | --- |
| `blackwell-6000x3` | Nemotron Embed DP3 + 独立的 MinerU DP3 | 已完成有限共卡试跑；现场 MinerU 为 4.0.5/vLLM 0.21，使用私有 GPU 覆盖；本仓库不在三卡机部署 Qwen |
| `blackwell-6000x4` | Qwen TP4 + EP4 | 已在该型号四卡机器验收；不同主机仍使用独立私有实例和容量检查 |
| `ada-5000x4` | Nemotron Embed DP4 + TP1 | 已在该型号四卡机器验收，并与独立 MinerU 服务完成有限联合负载试验 |

两台同为四卡 Blackwell 的主机共享 `blackwell-6000x4` 公共拓扑；各自当前运行的模型、端口、GPU 占用和请求峰值放在私有 instance。GPU 计算利用率为 0% 不表示已占用的显存可以分给新模型，99% 也不能单独说明服务是否达到业务性能目标。

三卡 Blackwell 的 MinerU 由 Unstructure Serve 原有 `model`/`app` 三卡入口管理，两个仓库各自负责模型与验证，不复制 MinerU 镜像到本仓库。该现场的 Snap Docker 已生成 CDI 规格，但当前 daemon 未扫描其目录；`embed3` 使用私有 `output/instances/embed3.compose.yaml` legacy GPU 覆盖，公开 CDI Compose 尚未在其上原样验收。Qwen 不列入该机器的模板：固定模型的 `linear_num_key_heads=16`、`linear_num_value_heads=48`，当前 vLLM 的 GDN 状态维度为 `128×16×2 + 128×48 = 10240`，TP3 会触发整除检查。依据：[固定模型配置](https://www.modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4/resolve/bb325e902fbb183287fcf009deba45d59abeb079/config.json)、[固定 vLLM 的 GDN 分片](https://github.com/vllm-project/vllm/blob/dee37d89115db4c94a820a79a78a7828e141c910/vllm/models/qwen4_exp/nvidia/model.py#L759-L779)。

未来若需在单卡或双卡机器运行 Embed，应增加单独的 DP1/DP2 拓扑 profile，并记录实际加载、真实请求、峰值显存与质量验收。若新 LLM 需要不同镜像、量化、chat template、tokenizer、解析器或 API 检查，应增加新的模型／运行时 profile，不能复用旧模型名伪装兼容。

## 实例与 Compose 的关系

每个新实例采用独立 Compose project、服务、缓存卷和容器日志；模型权重可在相同固定 revision 下只读共享，下载时仍使用锁和逐文件校验。实例配置可以引用同一批 GPU，但这表示主动共卡，必须通过显存计划与共同负载验收。新实例的项目名按 `tiangong-vllm-<实例名>` 生成；同一实例的 `start`、`restart`、`stop`、`status`、`logs`、`check` 始终作用于同一项目。主机端口不得与现有服务冲突。

公开 Compose 中 GPU CDI `devices` 条目数、vLLM 的 TP/DP 参数和实例 GPU 列表必须一致；特殊宿主的私有 GPU 覆盖也须达到相同的可见设备数量与顺序，并单独验收。管理器根据已登记的 profile 与私有 instance 渲染配置，不接受未经审查的任意 vLLM 命令行字符串。配置中的模型路径解析为部署机本地路径，容器内仍只读挂载；缓存按实例隔离，镜像或模型修订变化时检查缓存兼容性。`config` 只做静默语法验证，不能把展开后的 Compose 或密钥写入日志和验收证据。

拓扑校验按 TP × PP × DP 计算实际 GPU 数；EP 是专家在这些 rank 上的并行设置，不额外乘一次 GPU 数。当前 Qwen 是 TP4、EP4、DP1，共四张卡；四卡 Embed 是 TP1、DP4，三卡 Embed 是 TP1、DP3。新增拓扑仍须按模型与运行时约束验证 rank 映射。

现有 `model`、`embed`、`embed3` 与 `all` 继续保留为固定入口，其中 `all` 仍仅指当前 Qwen `model`，绝不表示“启动所有实例”。同机只能启动 `embed` 与 `embed3` 其中一种。迁移旧服务时保持既有 Compose project、服务名和缓存卷，先让新管理入口指向同一容器；不得以新项目名直接再启动一份同端口、同 GPU 的副本。只有明确执行排空、停止、切换、真实验收后，才将旧实例迁入新的命名规则。

## 显存配置与计划检查

私有 instance 面向部署者使用绝对 GiB，由 profile 决定字段含义。自动 KV 的生成模型使用 `engine_memory_target_gib`，供 vLLM 推算 KV 容量；固定 KV 的生成模型分别使用 `kv_cache_gib` 与 `startup_required_free_gib`；当前 Embed pooling 模型没有 KV，使用 `startup_required_free_gib`。所有实例另有 `reserve_gib` 用于计划检查。管理器将目标 GiB 或启动所需空闲 GiB 换算成 vLLM 比例：

```text
gpu_memory_utilization = profile 对应的 GiB 字段 / GPU 总显存 GiB
```

计算依据是 GPU **总显存**，不是启动前的空闲显存；应检查结果处于 `(0, 1]`，并记录换算后的值。一个 vLLM 实例若只接受统一比例，选中的 GPU 容量不一致时，同一比例会形成不同的绝对值。初版应拒绝这种组合，直到有明确的逐卡适配与验收方案。绝对 GiB 字段也不是进程显存硬上限；驱动、模型加载、计算工作区、CUDA Graph 和请求峰值仍需余量。`plan` 应结合当前占用、已声明的同卡服务实测峰值与保留空间逐卡报告结果，不能将两个实例的启动所需空闲显存直接相加当作稳态占用；一次空闲显存采样也不能替代峰值测量。

对确实支持 `--kv-cache-memory-bytes` 的生成模型运行时，`kv_cache_gib` 只控制 KV 缓存，不限制模型权重或整个进程。固定 KV 后 `--gpu-memory-utilization` 不再计算 KV 大小，但仍检查启动空闲显存；不能把固定 KV 当作通用“显存上限”，也不能未经核对就套在当前 vLLM 0.25 的 Embed pooling profile 上。模型／运行时 profile 应声明支持哪种显存参数及其优先级，拒绝互相矛盾的配置。即使使用绝对 GiB 用户接口，渲染出的 vLLM 仍可能使用比例参数；这样可让同一个模板在不同容量机器上由各自 instance 明确给出目标，而不复制单机百分比。

当前版本的依据见 [vLLM 0.25 启动空闲检查](https://docs.vllm.ai/en/v0.25.0/api/vllm/v1/worker/utils/)、[vLLM 0.28 固定 KV 分配](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/worker/gpu_worker/)及 [Nemotron Embed 模型配置](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/config.json)。升级运行时时要重新核对实现，不把此设计当成所有版本的永久行为。

`plan <实例名>` 为只读入口，至少核对：profile 与模型清单、镜像身份、GPU 设备映射方式与数量、GPU 架构与容量、TP×DP 拓扑、端口、模型文件完整性、同卡已知服务与预算余量；输出模型名、GPU、端口、预算和待解决问题，不显示 API key。计划通过只代表静态条件满足。`start` 前仍需检查服务归属和在途请求；上线后分别执行专属真实 API 验收，共卡时再测联合负载的峰值显存、错误和延迟。不同请求长度与并发容量不因 `plan` 通过而获得保证。

## 管理命令与迁移阶段

保持 `deploy/manage.sh` 为统一入口。目标命令形式为 `deploy/manage.sh <操作> <实例名>`，操作包含 `plan`、`download`、`verify`、`pull` 或 `build`、`config`、`start`、`restart`、`start-loaded`、`restart-loaded`、`stop`、`status`、`logs`、`check`。下载、镜像准备和验收均由实例选中的模型／运行时 profile 决定。`start` 不重建已有容器，配置变化使用 `restart`；`status` 或 `plan` 应报告渲染配置与运行容器的摘要是否一致，防止旧容器被误认为已经应用新参数。启动使用固定且已准备的本地镜像，不隐式换到新的上游版本；跨机导入仍使用 loaded 命令。各实例的验收只检查其真实 API，不能把 Qwen 的聊天测试作为任意 LLM 的通用测试。

建议按以下顺序实施，每一步独立评审和测试：

1. 建立 profile 清单与私有 instance 的字段规范，实现只读 `plan` 和配置校验；现有容器保持原启动方式。
2. 将现有 Qwen、四卡 Embed 与三卡 Embed 注册为已验收范围各异的模板，让旧命令别名解析到原项目、服务名、模型目录与缓存卷；对比生成参数与现有配置后再切换管理入口。
3. 增加实例独立的 Compose 渲染、下载／镜像准备及生命周期；测试重复启动、配置漂移、错误 GPU 数量、重复设备或端口、密钥脱敏和模型文件损坏等失败路径。
4. 需要其他卡数或模型时，先增加候选 profile，在对应机器完成镜像兼容、真实 API、容量与共同负载验收；通过后再在文档中标记该组合已验证。

整个迁移过程不自动启动 Qwen 与 Embed，不停止其他项目服务，不清理共享 Docker 资源，也不改变已暂停的 PCIe IPC 实验。当前部署步骤仍以[部署说明](deployment.md)为准；真实验收边界见[验证](validation.md)。
