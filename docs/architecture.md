# 架构

## 服务边界

```text
Chat / 图片 / 工具客户端 ──可选 Bearer──> Docker vLLM :7730 ──> 4 GPU TP + EP
                                        │
                           只读模型目录 + 独立编译缓存卷
```

模型服务负责 tokenizer、模型原生 chat template、量化内核、KV cache、批处理和 OpenAI-compatible API。无需额外应用层、Redis、数据库或 PM2；Docker Compose 管理模型，Docker 负责退出恢复和日志轮转。

## 文件与数据

- `deploy/vllm/`：Compose、模型修订和完整性清单；`deploy/manage.sh`：统一管理入口。
- `scripts/`：无第三方 Python 运行时依赖的下载、完整性校验和真实验收工具。
- `tests/`：下载完整性及失败处理测试。
- `docs/`：当前架构、部署和使用说明；`examples/`：可复制的 HTTP 请求。
- `.env`：私有部署覆盖及密钥；`.env.example`：公开配置骨架。
- `models/`：私有模型，推理容器只读挂载；独立 Compose volume 保存编译缓存。
- `output/validation/`：私有真实响应、耗时和失败原因；不纳入 Git。

## 生命周期

`deploy/manage.sh` 根据脚本位置定位仓库，不依赖调用方工作目录。Compose 项目名固定 `tiangong-vllm`，只管理此项目的 model。start 不重建运行中容器；restart 重建以应用新环境和命令；stop 保留磁盘资源。

模型 GPU ID 由四个 `GPU_0..GPU_3` 配置；张量并行参数须与实际模型拓扑匹配。变更 GPU 数量需同步调整 Compose 设备列表。容器内端口固定 8000，主机默认映射 7730。

## API 合同

客户端直接调用模型 `/v1/chat/completions`；对话历史由客户端完整传入。不再提供额外 Responses 兼容层或内存会话存储。本项目承诺验收的接口见 AI 接入说明，不因为引擎暴露其他路由就认为其已通过本项目验证。

不将旧模型专用模板强行移植到新模型。新模型的 system 指令、工具调用与 reasoning 必须通过真实请求验收。
