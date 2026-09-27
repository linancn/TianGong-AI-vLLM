# 验证

## 静态与本地回归

```bash
uv sync --locked --group dev
uv run --group dev black --check scripts tests deploy/vllm/patches
uv run --group dev ruff check scripts tests deploy/vllm/patches
uv run --group dev pytest
bash -n deploy/manage.sh deploy/vllm/serve.sh
./deploy/manage.sh config model
# 配置 Embed 的部署机还需填写私有启动空闲检查比例，再执行：
./deploy/manage.sh config embed
# 使用三卡 Embed 时填写 EMBED3_GPU_MEMORY_UTILIZATION，再执行：
./deploy/manage.sh config embed3
```

下载工具测试覆盖完整性判断、损坏文件拒收和下载失败不替换已有文件。单元测试不需要 GPU，不用替身结果冒充模型实测。

## 模型与真实 API

```bash
./deploy/manage.sh verify model
./deploy/manage.sh status model
./deploy/manage.sh check model
```

真实验收依次检查：

- `/health`、模型身份及鉴权配置：默认无 key 请求成功；配置非空 key 时正确 key 成功，未带 key 返回 401。
- 中文算术文本与正常 stop；system 指令处理。
- SSE 文本、正常结束与 `[DONE]`。
- 自动工具选择、工具名称、参数 JSON 和 `tool_calls` 结束原因。
- base64 图片输入，识别构造纯红图片。
- 开启 thinking 后的推理字段、正确答案及正文分离。
- 约 13.5K token 的 384 条合成记录中，准确提取 Record 0017 与 0371 的 output 字段。

每次运行保存独立 `output/validation/<UTC时间>.json`，包含耗时、真实响应及最终 passed/failed。任何断言失败返回非零，不能只看前几项通过。

## Embed 真实 API

```bash
./deploy/manage.sh verify embed
./deploy/manage.sh status embed
./deploy/manage.sh check embed
```

三卡入口将上述命令的组名换为 `embed3`，并先停止同一主机上的四卡 `embed`。`check embed` 与 `check embed3` 均经真实 HTTP 校验 `/health`、`/v1/models` 的模型身份与实际 `max_model_len`，并分别以 `input_type=query` 与 `input_type=document` 调用 `/v2/embed`。它们检查向量数量、每条 4096 维、数值有限、L2 范数接近 1，以及相关文档分数高于不相关文档；还会发送一条按当前固定模型提示词构造的 `truncate=NONE` 满长文档输入，检查实际计费 token 数等于配置上限。配置非空 `EMBED_API_KEY` 或 `EMBED3_API_KEY` 时，还检查不带 key 的模型查询返回 401。结果保存在私有 `output/validation/`，四卡与三卡的结果分别带 `embed` 或 `embed3` 标识；失败返回非零，不能以健康状态代替。满长检查会占用显存，运行前须确认同卡负载。

两篇短文档样本只证明接口和基本检索顺序；合成满长输入只证明长度边界及未截断，不能证明生产语料召回率、多语言质量或三/四副本并发容量。上述能力应以实际语料、输入长度和并发单独测试；调整模型、镜像、GPU 拓扑、启动检查阈值或批量参数后重新验收。

## 共卡验收

在 MinerU 或其他模型与 Embed 共用 GPU 的部署机上，先分别完成各服务的真实请求验收，再在代表性共同负载下记录每张卡的峰值显存、空闲余量、启动和请求时是否 OOM、错误率及延迟。还需观察 MinerU KV cache 使用与抢占，以及 Embed 的实际吞吐和长文本计算显存。Embed 是无 KV cache 的 encoder-only pooling 模型，其显存比例参数在当前 vLLM 0.25 中主要用于启动时空闲检查，不是运行期显存配额。只看单服务的 `/health`、静态检查比例或空闲显存，不足以说明共卡稳定。Qwen 与 Embed 不默认同时运行；若选择共卡，也要按相同方法单独评估。

2026-09-27 在三张 RTX PRO 6000 Blackwell Max-Q（每卡约 95.6 GiB）上试运行 `embed3`：停止旧 Qwen3-Embedding 后，固定 vLLM 0.25 镜像以 BF16、DP3/TP1 加载，容器 healthy，三个 worker 每卡约占 16224 MiB。`check embed3` 的真实 HTTP 模型身份、4096 维、L2 范数和检索顺序检查通过。该宿主的 Snap Docker 已生成 CDI 规格，但 daemon 未扫描其目录；现场使用私有 legacy NVIDIA GPU Compose 覆盖，这一结果尚不能证明公开 CDI 模板在该宿主原样启动。

上述联合试跑时，同机 MinerU DP3 为 4.0.5/vLLM 0.21，与四卡 Ada 上的 4.0.7/vLLM 0.28 不是相同的软件组合。一次短时联合样本中，2 页 PDF advanced 解析测试通过（单项约 3.94 秒），8 次每次约 20749 字符的 Embed 向量请求全部完成，单次耗时 0.30～0.70 秒；请求使用 `truncate=END` 且模板 `max_model_len=4096`，可能被截断，不能视作未截断的长上下文验证。联合过程约 4.69 秒，每 0.25 秒采样得到 16 个显存样本：三张卡的最高已用显存分别为 30145、30005、31013 MiB，最低空闲分别为 67189、67337、66329 MiB。另一次 MinerU DP3 的 p2 与 9 页 paper 解析测试通过，三个 rank 的成功计数均增长，耗时 30.34 秒。以上只覆盖单份 PDF 加 8 次向量请求的短时负载，不证明持续生产并发容量、最坏显存峰值或升级 MinerU 后的表现。

同日将该三卡 MinerU 更新至 4.0.7/vLLM 0.28.0，DP3 每 rank 固定 3 GiB KV。模型直连的 2 页与 9 页解析、同步 PDF、普通 PDF、两阶段 paper 和 DOCX 的真实应用 API 样本通过；重建宿主应用环境后，同步 2 页 PDF API 再次通过。Embed3 保持运行。旧版运行中三卡已用显存分别为 29565、29427、30431 MiB；升级并完成请求后的快照分别为 26051、26543、26157 MiB，分别低 3514、2884、4274 MiB。升级后每卡 Laya worker 约 2248 MiB、Embed worker 约 16664 MiB，MinerU worker 分别约 7118、7610、7224 MiB；旧版 MinerU worker 分别约 11098、10960、11966 MiB。刚启动后的较低显存快照不能代表预热后的占用；这些跨时间快照也不是业务峰值或显存变化的单一原因证明。

升级后的一次共同负载样本中，MinerU 解析 2 页 PDF 并返回 6 项，同时 8 次较长文档 Embed 请求全部通过，总历时 4.44 秒；重新执行的 `check embed3` 也通过检索顺序检查。按 0.25 秒间隔获得 19 个 GPU 样本，三卡峰值已用显存分别为 26051、26543、26155 MiB，最低空闲分别为 71283、70799、71187 MiB。结果保存在私有 `output/validation/` 中。该样本证明此次软件组合可完成这组并发请求；采样值不是最坏峰值，不能据此推断持续生产并发、复杂文档或更长输入下的容量。

2026-09-27 在四张 RTX 5000 Ada（每卡 32760 MiB）上试运行：MinerU 4 卡 DP4 使用每卡 3 GiB 固定 KV；Embed 使用固定 vLLM 0.25.0 镜像、BF16、DP4、4096 token 上限、4 序列，私有启动空闲检查比例为 0.62。

- 较短文本的联合样本：8 页 PDF advanced 解析返回 85 项、耗时 19.69 秒，四个 MinerU rank 均有成功计数增长；与其同时发出的 32 次 Embed 请求全部完成，四个 Embed rank 分别完成 8、8、9、7 次。每 0.25 秒采样的单卡最高已用显存为 22713 MiB，最低空闲为 9510 MiB。
- 较长文本的联合样本：8 页 PDF advanced 解析返回 85 项、耗时 19.28 秒；并发发出的 8 次约 3401 token Embed 请求全部完成。Embed 四个 rank 的成功计数增量依次为 3、2、1、2，MinerU 为 2、5、4、5。按 0.2 秒间隔取得 80 个显存样本，观测到单卡最低空闲 9094 MiB、最高已用 23129 MiB。原始证据保存在私有 `output/validation/20260927T084739876991Z-combined-long-official.json`。

此前独立发出的 8 次约 3401 token 文档向量请求也全部通过，四个 Embed rank 均参与。上述显存数字是采样窗口内的观测值，不是理论峰值；样本不覆盖持续高并发、复杂生产文档或其他显存容量的机器，也不能保证生产最坏负载下不会 OOM。

2026-09-28 将三卡和四卡 Embed 的私有配置都设为 `max_model_len=32768`、`max_num_batched_tokens=32768`、`max_num_seqs=4`，保持 BF16、DP3/DP4 与原有固定镜像。两台均用 `truncate=NONE` 的合成输入实际计费 32768 token 并返回 4096 维向量；三卡约 96 GiB Blackwell 的一次 32754 token 长请求期间，最低观测空闲约 67363 MiB。四卡 32 GiB Ada 与 MinerU 共卡，四条并发满长请求均返回 HTTP 200；又连续发送五条 30004 token 请求、随后并发发送四条 32768 token 请求，均成功。后一次约每 0.25 秒采样，单卡最低空闲为 3505 MiB；同机 Embedding 容器与 MinerU 容器均保持 healthy、重启次数为 0。长请求后各 Embed worker 的驻留显存不一致，连续请求使其他副本的显存占用上升；单卡剩余显存不能用四卡平均值代替。四条 API 请求并发不证明同一副本可在一次调度中容纳四条满长输入。上述 32K 样本没有叠加 MinerU 业务高峰，3505 MiB 不是联合峰值保证；此前 4096 token 配置下的联合负载余量不能沿用到 32K。

## 验证边界

Qwen 的纯色图片不代表 OCR、图表或复杂视觉质量；短文本不代表 256K 上下文质量或 16 路并发容量。模型升级、CUDA/驱动变化、GPU 拓扑变化后重跑真实验收。Qwen 启动健康检查最多有 30 分钟初始化宽限，Embed 为 10 分钟；仍在正常加载时不应不停重启。

## 256K 配置验收

当前上下文上限为 262144 token，保持 TP4 + EP4、GPU PLE、自动调优与 MTP 3。2026-09-21 使用既有镜像重建容器后，`/v1/models` 返回 `max_model_len=262144`，`deploy/manage.sh check` 全部通过。

补充边界样本包含 7000 条合成记录及填充文本：实际输入 262016 token，预留输出 128 token，成功提取首部、中部、尾部指定记录的三个字段，实际生成 18 token 并正常结束。相同输入将输出预算改为 129 token 后，总预算超过上限 1 token，服务返回 HTTP 400。请求、响应、耗时与断言结果保存于私有 `output/validation/` 的 `*-context-256k.json`。该样本验证长度边界及三处字段检索，不代表任意 256K 文本质量或满长并发性能。

## 已验证的部署基线

2026-09-18 在四张 RTX PRO 6000 Blackwell 96 GB（SM120）上完成真实验收：使用[依赖指南](dependencies.md)固定镜像、固定 ModelScope 清单、TP4 + EP4、GPU PLE、自动调优、MTP 3、65536 上下文和 16 序列配置。上述鉴权、文本、system、SSE、工具、图片、reasoning 及长输入检索检查全部通过，容器 healthy。实测重启范围为容器重建与缓存复用，不含整机重启。证据按时间保存在 `output/validation/`，部署摘要位于 `output/deployment.json`。

其中算术返回 `42`，工具返回结构化 `get_weather({"city":"Beijing"})`，图片返回 red，开启推理时 `12 × 13` 返回 156 且 reasoning 与 content 分离。以上是功能验收，不是长上下文或并发性能结论。

## 性能对照与调优回归

自动调优的分布式回归与缓存重启验证见[运行时修复](runtime-patches.md)。性能测量运行 `python3 scripts/benchmark.py --label <配置名称>`，先完成对应配置的真实功能验收，再比较基线、MTP 及通信后端。每个场景先预热，再重复三组测量；原始输出、请求 token 数、TTFT、墙钟吞吐及前后指标保存到 `output/benchmarks/`。运行中容器发生变化会拒绝该次结果。

测量使用固定 256 输出 token、temperature=0、关闭 thinking、SSE 和 ignore_eos。三场景是 66 token 短输入单路、13526 token 合成记录输入单路、短输入四路并发。每请求在 system 开头使用不同 ID，避免整段 prefix cache 命中；实际输入 token 数保存在证据中。指标为端到端吞吐（包含 prefill 和网络），不是纯 decode 峰值。固定长度输出仅用于性能对照，质量由独立的 smoke 验收检查；详细结果集中在[性能调优](performance-tuning.md)。
