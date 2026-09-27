# 架构

Qwen 四卡、Embed 四卡与 Embed 三卡模板均有真实运行样本。三卡 Embed 与 MinerU 共卡的样本范围有限，并使用目标宿主的私有 GPU Compose 覆盖；公开 CDI 配置仍需按机器单独核验。面向其他模型、GPU 型号和卡数的三层 profile/实例方案见[模板架构方案](template-design.md)；当前管理脚本尚未实现任意拓扑渲染。

## 服务边界

```text
Chat / 图片 / 工具客户端 ──可选 Bearer──> Qwen Docker vLLM :7730 ──> 4 GPU TP + EP
查询 / 文档检索客户端 ──可选 Bearer──> Embed Docker vLLM :7731 ──> 4 GPU DP（每卡 TP1）
查询 / 文档检索客户端 ──可选 Bearer──> Embed3 Docker vLLM :7732 ──> 3 GPU DP（每卡 TP1）
                                             │
                                各自只读模型目录与缓存卷
```

Qwen 服务负责 tokenizer、模型原生 chat template、量化内核、KV cache、批处理和 OpenAI-compatible API。Embed 服务负责文本向量与检索提示词处理：四卡模板装载四个 BF16 副本，三卡模板装载三个。三卡模板已在三张 RTX PRO 6000 Blackwell 上与 Unstructure Serve 的 MinerU DP3 完成短时联合试跑，先后覆盖 MinerU 4.0.5/vLLM 0.21 与 4.0.7/vLLM 0.28；各次样本的范围见[验证](validation.md)。无需额外应用层、Redis、数据库或 PM2；Docker Compose 管理各个独立模型服务，Docker 负责退出恢复和日志轮转。

## 文件与数据

- `deploy/vllm/`：Qwen Compose、模型修订和完整性清单；`deploy/embed/`：四卡 Embed Compose 与模型清单；`deploy/embed3/`：三卡 Embed Compose，共用前者的模型清单和固定镜像；`deploy/manage.sh`：统一管理入口。
- `scripts/`：无第三方 Python 运行时依赖的下载、完整性校验和真实验收工具。
- `tests/`：下载完整性及失败处理测试。
- `docs/`：当前架构、部署和使用说明；`examples/`：可复制的 HTTP 请求。
- `.env`：私有部署覆盖及密钥；`.env.example`：公开配置骨架。
- `models/`：私有模型，推理容器只读挂载；独立 Compose volume 保存编译缓存。
- `output/validation/`：私有真实响应、耗时和失败原因；不纳入 Git。
- `output/instances/embed3.compose.yaml`：三卡目标宿主可选的私有 GPU Compose 覆盖；统一入口检测到该文件时附加，公开文件仍以 CDI 为默认。

## 生命周期

`deploy/manage.sh` 根据脚本位置定位仓库，不依赖调用方工作目录。`model`、`embed`、`embed3` 分别属于 Compose 项目 `tiangong-vllm`、`tiangong-embed`、`tiangong-embed3`；命令逐个选择服务。`embed` 与 `embed3` 在同一主机上互斥，启动一个之前须停止另一个。`all` 是 `model` 的兼容别名。start 不重建运行中容器；restart 重建以应用新环境和命令；stop 保留磁盘资源。

Qwen GPU ID 由四个 `GPU_0..GPU_3` 配置；张量并行参数须与实际模型拓扑匹配。四卡 Embed 由 `EMBED_GPU_0..EMBED_GPU_3` 选择 GPU、采用 DP4/TP1；三卡 Embed 由 `EMBED3_GPU_0..EMBED3_GPU_2` 选择 GPU、采用 DP3/TP1。变更 GPU 数量需同步调整 Compose 设备列表与并行参数。各容器内部端口为 8000，主机默认映射为 Qwen 7730、四卡 Embed 7731、三卡 Embed 7732；Embed 默认仅本机监听。三卡与四卡 Embed 默认共用只读模型权重和固定镜像，Compose 项目及缓存卷独立。公开 Compose 使用 CDI；目标三卡宿主的 Snap Docker 已生成 CDI 规格，但当前 daemon 未扫描其目录，因此现场使用私有 legacy NVIDIA GPU 覆盖。Embed 与 MinerU 可选用相同 GPU，但须按部署机私有配置核算显存并进行联合负载验收。

## API 合同

Qwen 客户端直接调用 `/v1/chat/completions`；对话历史由客户端完整传入。Embed 客户端使用 `/v2/embed`，分别指定 `query` 或 `document`。不提供额外 Responses 兼容层、检索数据库或内存会话存储。本项目承诺验收的接口见[AI 接入](ai-integration.md)，不因为引擎暴露其他路由就认为其已通过本项目验证。

不将旧模型专用模板强行移植到新模型。新模型的 system 指令、工具调用与 reasoning 必须通过真实请求验收。
