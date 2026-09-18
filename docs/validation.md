# 验证

## 静态与本地回归

```bash
uv sync --locked --group dev
uv run --group dev black --check scripts tests deploy/vllm/patches
uv run --group dev ruff check scripts tests deploy/vllm/patches
uv run --group dev pytest
bash -n deploy/manage.sh deploy/vllm/serve.sh
./deploy/manage.sh config
```

下载工具测试覆盖完整性判断、损坏文件拒收和下载失败不替换已有文件。单元测试不需要 GPU，不用替身结果冒充模型实测。

## 模型与真实 API

```bash
./deploy/manage.sh verify
./deploy/manage.sh status model
./deploy/manage.sh check
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

## 验证边界

纯色图片不代表 OCR、图表或复杂视觉质量；短文本不代表 64K 上下文质量或 16 路并发容量。模型升级、CUDA/驱动变化、GPU 拓扑变化后重跑真实验收。启动健康检查最多有 30 分钟初始化宽限，不应因仍在编译而不停重启。

## 已验证的部署基线

2026-09-18 在四张 RTX PRO 6000 Blackwell 96 GB（SM120）上完成真实验收：使用[依赖指南](dependencies.md)固定镜像、固定 ModelScope 清单、TP4 + EP4、GPU PLE、自动调优、MTP 3、65536 上下文和 16 序列配置。上述鉴权、文本、system、SSE、工具、图片、reasoning 及长输入检索检查全部通过，容器 healthy。实测重启范围为容器重建与缓存复用，不含整机重启。证据按时间保存在 `output/validation/`，部署摘要位于 `output/deployment.json`。

其中算术返回 `42`，工具返回结构化 `get_weather({"city":"Beijing"})`，图片返回 red，开启推理时 `12 × 13` 返回 156 且 reasoning 与 content 分离。以上是功能验收，不是长上下文或并发性能结论。

## 性能对照与调优回归

自动调优的分布式回归与缓存重启验证见[运行时修复](runtime-patches.md)。性能测量运行 `python3 scripts/benchmark.py --label <配置名称>`，先完成对应配置的真实功能验收，再比较基线、MTP 及通信后端。每个场景先预热，再重复三组测量；原始输出、请求 token 数、TTFT、墙钟吞吐及前后指标保存到 `output/benchmarks/`。运行中容器发生变化会拒绝该次结果。

测量使用固定 256 输出 token、temperature=0、关闭 thinking、SSE 和 ignore_eos。三场景是 66 token 短输入单路、13526 token 合成记录输入单路、短输入四路并发。每请求在 system 开头使用不同 ID，避免整段 prefix cache 命中；实际输入 token 数保存在证据中。指标为端到端吞吐（包含 prefill 和网络），不是纯 decode 峰值。固定长度输出仅用于性能对照，质量由独立的 smoke 验收检查；详细结果集中在[性能调优](performance-tuning.md)。
