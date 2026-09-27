# AI 接入

## 地址与鉴权

Qwen 默认 base URL 为 `http://127.0.0.1:7730/v1`，远程使用部署机器地址。模型名 `nv-community/Qwen3.8-Flash-Next-NVFP4`；以 `/v1/models` 与部署配置为准。默认无需 API key，省略 Authorization 请求头即可。仅当部署设置了非空 `VLLM_API_KEY` 时，业务请求才需要 `Authorization: Bearer <VLLM_API_KEY>`；key 从私有配置获取。

Qwen 主要接口：`GET /v1/models`、`POST /v1/chat/completions`。使用模型原生 system/user/assistant/tool 角色，developer 指令请由客户端改为开头的 system 消息。客户端自行传入完整历史；本项目不提供额外 Responses 代理。

## 文本、推理与流式

```json
{
  "model": "nv-community/Qwen3.8-Flash-Next-NVFP4",
  "messages": [{"role": "user", "content": "用一句话解释张量并行。"}],
  "max_tokens": 256,
  "temperature": 0.2,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

启用推理时将 `enable_thinking` 设为 true，并为推理及最终答案留足输出 token。reasoning parser 将推理与正文分离；客户端读取实际返回的 reasoning 字段，不把 `<think>` 标签当作答案。检查 `finish_reason`，`length` 表示输出被截断。

设置 `stream: true` 使用 SSE；读取 `data:` 事件直到 `[DONE]`。连接中断不代表模型已撤销请求，重试前评估是否可接受重复生成。

## 工具调用

传入 OpenAI 风格 `tools` 和 `tool_choice: auto`。检查 `message.tool_calls`，解析 `function.arguments` JSON；工具实际执行由客户端负责。执行后用相应 `tool_call_id` 的 tool 消息继续对话。服务不会执行模型生成的命令。

## 图片与限制

在 user content 数组中传 `text` 与 `image_url`；可使用 base64 data URL。模板每请求最多四张图片，视频配额为零。图片 token 计入总上下文。图片传输与业务输出可能包含私有内容，验收结果仅本地保存。

配置上限为输入加输出合计 262144 token；短请求测试不保证任意长上下文、复杂图表准确率或并发性能。HTTP 400 通常是请求或容量限制；401 是凭证问题；5xx 应结合容器日志排查，不把错误响应当成有效答案。

完整可编辑请求见 `examples/chat.http`。`/health` 仅确认服务存活，真实调用验收由 `deploy/manage.sh check model` 执行。

## Nemotron Embed 向量接口

Embed 是独立服务，三卡和四卡模板均默认仅在本机监听 `http://127.0.0.1:7731`；局域网调用须在私有配置中绑定所选主机的局域网 IP。模型名默认 `nv-community/Nemotron-3-Embed-8B-BF16`，以所选模板的 `EMBED_SERVED_MODEL_NAME` 或 `EMBED3_SERVED_MODEL_NAME` 和服务 `/v1/models` 为准。默认无需鉴权；仅当私有配置中对应的 `EMBED_API_KEY` 或 `EMBED3_API_KEY` 非空时，附加该服务的 Bearer key。它与 Qwen 的地址、key 和生命周期各自独立。

检索使用 `POST /v2/embed`，查询和待检索文档分别请求。`input_type` 为 `query` 或 `document`，服务按模型自带提示词处理原始文本；客户端无需再手动加 `query: ` 或 `passage: ` 前缀。示例请求：

```bash
curl --fail-with-body http://127.0.0.1:7731/v2/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"nv-community/Nemotron-3-Embed-8B-BF16","input_type":"query","texts":["什么是张量并行？"],"embedding_types":["float"],"truncate":"END"}'
```

建立索引时将 `input_type` 改为 `document`，`texts` 传文档片段。返回向量位于 `embeddings.float`，每条 4096 维且经 L2 归一化，可用点积比较相关度。两个模板的默认 `max_model_len` 均为 32768 token，包含服务按 `input_type` 添加的提示词 token；单副本每次调度的合计 token 上限也为 32768，因此一个满长输入可能占满该次调度的 token 额度。`MAX_NUM_SEQS=4` 不表示可在同一副本一次处理四条满长输入。需要确认完整输入未截断时使用 `truncate=NONE` 并检查响应的 `meta.billed_units.input_tokens`；业务文档仍须按实际长度分片，超限不能依赖静默截断。变更维度或截断方式会影响已建索引的一致性，须与查询端保持同一模型和处理方式。上述字段及提示词行为见 [NVIDIA 模型说明](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/README.md)和 [vLLM Embed 文档](https://docs.vllm.ai/en/v0.25.0/models/pooling_models/embed/)。

`deploy/manage.sh check embed` 或 `check embed3` 验证模型身份、4096 维、归一化、一个查询对两篇文档的检索顺序，以及与配置长度相符的无截断满长输入。合成满长输入只验证接口和长度边界，不能证明特定业务数据的召回率、长文本语义质量或三/四副本吞吐。

可编辑请求见 `examples/embed.http`。
