# 性能与容量调优

## 当前默认

四卡 TP4 + EP4，GPU 引擎预算 0.85，上下文 65536，并发序列上限 16，每轮批处理 token 预算 8192。每请求最多四图、禁用视频。使用原模型混合量化 metadata：主模型 NVFP4 专家层、FP8 PLE 和 FP8 MTP，不改权重。

- `--engram-config '{"cpu_offload":false}'`：PLE 随 TP 分片驻留 GPU，利用充足显存避免默认 CPU lookup。
- `--enable-flashinfer-autotune`：明确保留调优；分布式缓存命中与持久化修复见[运行时修复](runtime-patches.md)。
- `SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}`：基于下述测试选用三 token MTP；清空可关闭，改为 1 可使用单 token 草稿。
- 通信使用 NCCL。`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0` 保持默认。

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
