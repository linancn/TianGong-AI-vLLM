# 性能与容量调优

## 当前默认

四卡 TP4 + EP4，GPU 引擎预算 0.85，上下文 262144（256K，输入与输出合计），并发序列上限 16，每轮批处理 token 预算 8192。每请求最多四图、禁用视频。使用原模型混合量化 metadata：主模型 NVFP4 专家层、FP8 PLE 和 FP8 MTP，不改权重。

- `--engram-config '{"cpu_offload":false}'`：PLE 随 TP 分片驻留 GPU，利用充足显存避免默认 CPU lookup。
- `--enable-flashinfer-autotune`：明确保留调优；分布式缓存命中与持久化修复见[运行时修复](runtime-patches.md)。
- `SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}`：基于下述测试选用三 token MTP；清空可关闭，改为 1 可使用单 token 草稿。
- 通信使用 NCCL。`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0` 保持默认。

模型原生 `max_position_embeddings=262144`，当前不启用 YaRN。256K 为单请求上限，`MAX_NUM_SEQS=16` 不保证 16 个满长请求可同时驻留；容量取决于启动时分配的 KV cache。以下性能数据仍对应 65536 上下文配置。

## Embed 容量与共卡

Embed 有四卡 DP4 和三卡 DP3 两种模板，每卡一个 BF16 模型副本，TP1；它们分别提供四个或三个请求吞吐路径，不等同于跨卡切分一个模型。固定清单中的四个 safetensors 权重文件合计约 14.8 GiB，运行时、模型加载、批处理和请求计算仍会另外占用显存。Nemotron Embed 以 encoder-only pooling 运行，不分配生成模型的 KV cache。`EMBED_GPU_MEMORY_UTILIZATION` 或 `EMBED3_GPU_MEMORY_UTILIZATION` 必须按所选模板在每台部署机的私有 `.env` 中填写；当前固定的 vLLM 0.25 用它对启动瞬间的空闲显存做总显存比例检查，不按此比例预留显存或限制运行期进程占用，详见[部署说明](deployment.md)。Qwen 的 `GPU_MEMORY_UTILIZATION=0.85` 仍参与其 KV cache 容量计算，两服务不会自动共驻留。

Embed 模型原生最多 32768 token，两种模板均以 `MAX_MODEL_LEN=4096`、`MAX_NUM_BATCHED_TOKENS=4096`、`MAX_NUM_SEQS=4` 为起点，对应 `EMBED_*` 或 `EMBED3_*` 前缀。增加文本长度或并发时，逐步测量每张选定卡的启动峰值、请求峰值、错误和延迟；只改变限制值并不能保证容量。精度保持 BF16，若改为 FP8 或其他权重，须按新模型和实际语料重新验证向量质量。

与 MinerU 等服务共卡时，先检查每个 GPU 上已有进程的权重、KV cache、计算峰值及剩余空间，再为 Embed 设置启动检查阈值并留运行期显存余量。Embed 请求越长、批量和并发越大，峰值计算显存可能越高；降低 `EMBED_GPU_MEMORY_UTILIZATION` 或 `EMBED3_GPU_MEMORY_UTILIZATION` 不会自动缩小这些分配。单项服务各自验收通过后，还须在共同负载下复验显存峰值、OOM、响应延迟和业务质量。具体机器的阈值与测量保存在私有配置及证据中，不写入共享模板。检索合同和验证边界见[AI 接入](ai-integration.md)与[验证](validation.md)。

2026-09-27 的四卡 Ada 试运行使用私有阈值 0.62：一个 8 页 PDF 解析与 8 次约 3401 token 的 Embed 请求并发，0.2 秒间隔的 80 个样本中，最低空闲 9094 MiB、最高已用 23129 MiB。`0.62` 只影响启动检查；运行中的空闲显存可以低于该比例对应的容量。此记录描述当次样本，持续高并发和生产最坏负载仍需另外验证；完整请求及各 rank 结果见[验证](validation.md)。

三卡 DP3 模板已在三张 RTX PRO 6000 Blackwell Max-Q（每卡约 95.6 GiB）上与 MinerU DP3 完成一次短时联合试跑。Nemotron BF16 三个 worker 每卡约占 16224 MiB；2 页 PDF advanced 解析与 8 次每次约 20749 字符的 Embed 请求全部通过，8 次向量请求各耗时 0.30～0.70 秒。按 0.25 秒间隔的 16 个显存样本，三张卡的最高已用分别为 30145、30005、31013 MiB，最低空闲分别为 67189、67337、66329 MiB。向量请求设置 `truncate=END`，4096 token 模板可能截断输入；这些数字既不是未截断长上下文性能，也不是持续高并发峰值。这次试跑使用 MinerU 4.0.5/vLLM 0.21，GPU 通过私有 legacy Compose 覆盖映射；现场现已升级至 MinerU 4.0.7/vLLM 0.28，升级后又完成一组 2 页 PDF 与 8 次文档向量请求的短时联合样本，但不能把旧结果套用到新软件组合或公开 CDI 启动路径。三卡启动检查比例按目标机已有进程占用填写，不沿用四卡私有阈值；完整边界见[验证](validation.md)。

## 2026-09-18 实测

硬件为 4 × RTX PRO 6000 Blackwell Max-Q 96 GB（SM120），Xeon w5-3425（12 核/24 线程）、约 250 GiB RAM，PCIe 单机无 NVLink。引擎为[固定基础镜像及修复层](dependencies.md)，GPU PLE、自动调优、TP4 + EP4 在各组保持相同。每组使用单独容器启动，先完成真实功能验收。

`python3 scripts/benchmark.py --label <名称>`：每个场景先预热，再运行三组；固定生成 256 token、temperature=0、thinking=false、SSE、ignore_eos。短输入实测 66 token；长输入为 384 条合成记录，实测 13526 token。每请求使用不同的起始 ID 避免整段 prefix cache 命中。以下是三组中位数，吞吐包含 prefill 和网络时间；四路指标为四请求合计吞吐。

| 配置 | 短输入单路 token/s | 长输入单路 token/s | 短输入四路合计 token/s | 草稿接受率 |
| --- | ---: | ---: | ---: | ---: |
| 不启用 MTP | 115.5 | 76.1 | 375.0 | — |
| MTP 1 | 168.0 | 96.6 | 536.9 | 79.7% |
| **MTP 3（默认）** | **229.8** | **113.0** | **610.6** | 60.0% |

MTP 3 相对基线提升约 **99% / 49% / 63%**。接受率按整个 benchmark（含预热）的服务端 accepted/draft token 增量计算；MTP 1、MTP 3 的每次草稿步平均输出长度（1 + accepted/drafts）分别约 1.80、2.80。三 token 草稿的接受百分比更低，但每次目标模型验证产出更多 token，实际吞吐更高。

| 配置 | 短输入单路 TTFT | 长输入单路 TTFT | 短输入四路 TTFT |
| --- | ---: | ---: | ---: |
| 不启用 MTP | 109 ms | 1238 ms | 226 ms |
| MTP 1 | 112 ms | 1266 ms | 241 ms |
| MTP 3 | 46 ms | 1264 ms | 176 ms |

长输入首 token 时间主要由 prefill 决定，MTP 的收益集中在随后生成阶段。上述样本不构成完整能力评分、64K 极限长度验证、16 路容量保证或生产 SLA。输出文本不保证逐 token 一致；独立功能验收检查算术、角色指令、流式、工具参数、图片、reasoning 分离，以及长输入中两个相距较远记录的精确字段。

私有证据位于 `output/benchmarks/`：基线 `20260918T085555210313Z.json`、MTP 1 `20260918T090423288152Z.json`、MTP 3 `20260918T090855232275Z.json`；对应 `<label>-container.json` 保存运行参数。新测量文件内也记录镜像 ID、实际命令和不含凭证的调优环境。不要只凭手写 label 判断实际配置。

## 其他优化的评估

**PCIe IPC all-reduce：暂未采用。** 四卡 P2P 读检查全部 OK，vLLM 也提供 `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` 开关，但固定镜像中的 FlashInfer 0.6.18.post1 不提供 `PcieIpcAllReduceWorkspace`。开启后日志明确回退到 PYNCCL，因此 `mtp3-pcie` 测量实际仍为 MTP 3 + NCCL，其波动不计作 IPC 收益。保留默认 0；未来升级基础镜像后，先确认实际后端启用，再单独测量。不要只看环境变量已设置就认定优化生效。

**MTP 多步融合：使用引擎支持的通用路径。** 当前 Qwen4 QSA state attention 不支持 fused multi-step draft decode，引擎会在草稿步间重建 attention metadata。上表的 MTP 3 收益是在该实际路径测得，没有强行替换 attention backend 或关闭正确性检查。

**量化与容量：维持当前配置。** 当前显存充足，未为节约显存另行降低 KV 精度，也未启用 CPU PLE offload。更长上下文、DP2/TP2 或更高并发需要独立业务样本验证；不要把 B200 的测量直接套到 RTX PRO 6000。

## 后续实验待办（已暂停）

用户决定先记录、不执行 PCIe IPC 升级实验；当前继续使用已验证的 MTP 3 + NCCL，自动调优保持开启。仅在用户明确恢复此项工作后再构建或部署候选版本。

已核实 [FlashInfer v0.7.0rc3 源码](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0rc3/flashinfer/comm/__init__.py)提供 `PcieIpcAllReduceWorkspace`，并存在 [PyPI 预发布包](https://pypi.org/project/flashinfer-python/0.7.0rc3/)。当前固定的 0.6.18.post1 没有该接口；这只是已验证镜像的版本边界，不代表硬件没有优化空间。

恢复后的顺序：

1. 在独立候选 Docker 镜像中固定预发布包及配套依赖，核对 CUDA/SM120、vLLM 与接口签名兼容性。
2. 重新审查自动调优补丁。现有补丁固定上游文件 SHA256，不能把旧补丁不加检查地应用到新库。
3. 验证四卡 P2P、all-reduce 数值、CUDA Graph、缓存与带缓存重启；确认日志显示真实 IPC 后端，而非回退 NCCL。
4. 与当前 MTP 3 + NCCL 做同口径 A/B，比较单路、四路及业务并发下的 TTFT、端到端吞吐和正确性。
5. 只有确有收益且验收通过，才更新默认镜像、配置和文档；不预先承诺提升百分比。

官方接口说明强调，[支持某个 shape 不等于更快](https://docs.flashinfer.ai/api/comm.html#pcie-ipc-allreduce)，仍须根据实际 PCIe 拓扑调优。当前不升级运行库、不修改通信开关。

## 调整与复验

1. 确认无在途请求，保存当前配置、镜像、模型清单和证据。
2. 每次只改变一个参数，通过 `deploy/manage.sh restart model` 应用，等待健康并执行 `deploy/manage.sh check`。
3. 运行相同 benchmark，比较 TTFT、端到端吞吐和 MTP 接受率；同时核对日志中的实际 backend 与 speculative 配置。
4. 保留性能和功能都通过的配置，恢复最终默认后再次验证带缓存重启。缓存问题应修复一致性，不以关闭自动调优或清空共享缓存代替。
