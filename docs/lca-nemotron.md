# LCA-Nemotron-Embed-1B

本服务由 GPUStack custom backend 托管，每个 deployment 只分配一张 GPU、一个模型进程。多卡容量通过多个独立 deployment 和一个 generic-proxy 路由提供。模型是 `Tiangong-LCA/LCA-Nemotron-Embed-1B`，不可与通用 Nemotron-3-Embed-8B 或已有向量索引互换。

## 固定发布与运行时

- 模型 revision：`6d77edaa3b92dd1b7e3dd8a152ea7170a8b581c5`。
- `model.safetensors`：4563688976 bytes，SHA256 `07fe9d06a60d89b269f62314a36c17be04448eed5f2a11001954e65badaa51cb`。
- 发布工具：`https://github.com/tiangong-lca/embedding-model`，commit `2d0c7d36a00b14dc1cc19d7ac6f94887f9d45d35`。其 `config/releases/lca-nemotron-embed-1b.json` 是全部 18 个文件的校验清单，包含发布版 `encode.py`。
- Linux x86_64 / Python 3.12；冻结的 `environments/nemotron-inference/uv.lock` 固定 Torch 2.11.0+cu129、Transformers 5.4.0、Sentence Transformers 5.3.0、Hugging Face Hub 1.8.0、safetensors 0.7.0。构建使用 uv 0.12.16。不要升级到其他模型现用的 Transformers/CUDA 版本。
- 固定 Python 基础镜像摘要见 [Dockerfile](../deploy/lca-nemotron/Dockerfile)。构建与 ML 安装只发生在镜像内，宿主开发环境不安装 Torch。保留最终 image ID、源码 commit、完整验收结果于私有 `output/`，部署后不覆盖已使用的镜像 tag。

先用固定工具的 `download_lca_release.py` 下载或 `--verify-only` 验证模型与 benchmark。已有文件校验失败时保留原副本，再补齐固定发布版本；不要跳过散列验证。

```bash
bash deploy/lca-nemotron/build.sh /path/to/embedding-model /path/to/linux-uv
```

## GPUStack 部署

当前后端模板使用 `st5.3.0-r2-custom` / 镜像 `tiangong-lca-nemotron:st5.3.0-r2`。GPUStack 2.3 的网关会将 JSON model 改写为选中的 deployment 名，并向后端发送 chunked 请求。r2 通过 `--served-model-name {{model_name}}` 接受这个确切内部别名，保留发布 ID 和固定响应 ID；同时支持有界 chunked JSON。累计 body 仍限制为 1000000 bytes，拒绝冲突的 Content-Length/Transfer-Encoding、无效分块、超长头和 trailers。依赖、权重与数值推理不变。r1 保留用于兼容旧网关的回退，不能直接用于上述请求合同。

客户端仍发送发布模型 ID。网关按已授权路由选择模型并改写 model，所以经网关不能用错误 model 字段证明后端拒绝陌生模型；此负例须直连受限后端验收。`check_lca_nemotron.py --gateway-rewrites-model` 显式验证网关始终返回本路由发布 ID，其他非法输入仍须拒绝；直连验收不传此选项。路由不能被客户端 model 字段用来选择其他服务。

1. 在目标 worker 中挂载专用父目录，例如 `/srv/models/lca-nemotron-embed-1b`，模型位于该目录下以完整 revision 命名的子目录。宿主和 worker 使用相同路径。仅添加此 mount，保留现有私有 worker overlay 的其他配置。
2. 导入 [backend 模板](../deploy/lca-nemotron/gpustack-backend.json)。导入 [deployment 模板](../deploy/examples/lca-nemotron-gpustack.json) 前替换节点、绝对模型路径和 GPU ID。`--allowed-clients` 必须只包含回环、该 worker IP 和获准的 GPUStack 控制面 IP。
3. 先部署一张卡验收。增加卡时复制 deployment，使用不同名称和单个 GPU ID，不设置 TP/DP，不在一个模型进程中共享三卡。
4. 新建 embedding 类别的 generic-proxy 路由，targets 指向这些新 deployment，weight 均为 1。使用 `/model/proxy/<route-id>/v1/embeddings`。新建仅 `inference` scope、仅允许该路由名的专用 key；保存到 `.secrets/gpustack/`，权限 600，不写文档或日志。

GPUStack 2.2.3 的 custom backend 默认以读写方式绑定模型父目录。本启动器在独立容器挂载命名空间中，对该 bind mount 执行 `remount,bind,ro`；若预期挂载不存在或重挂载失败则启动失败。该参数仅适用于当前 GPUStack privileged custom 容器；它不会改变宿主原目录的挂载标志。权重启动前逐文件校验；镜像使用离线模型加载，不执行 Hub remote code。

默认每卡 PyTorch allocator 上限为可见显存的 15%，批内编码 batch 8，等待队列 8，请求最多 32 段。这个上限约束 Torch allocator，不是整张卡或全部 CUDA 开销的隔离配额。上线前核对已有模型显存预算及负载，保留并验证共存余量。

## HTTP 合同

```json
{"model":"Tiangong-LCA/LCA-Nemotron-Embed-1B","input_type":"query","input":["生命周期评价与碳排放"]}
```

必须发送且仅发送 `model`、`input_type`、`input`。`input_type` 为 `query` 或 `document`；`input` 接受非空字符串或 1–32 个非空字符串，每段最多 100000 字符，原始 HTTP body 最多 1000000 bytes。拒绝 `dimensions`、`encoding_format` 等附加字段。客户端不要额外拼接 prompt。

返回顶层 `object=list`、固定 `model`、原 `input_type` 和按输入顺序编号的 `data[].embedding`，每段 2048 个有限浮点数，无 token usage。服务调用发布版 `LCAEmbedder`：query 前缀 `query: `、总长 512 tokens；document 前缀 `passage: `、总长 2048 tokens；attention-mask mean 包含 prompt，BF16 推理后转 float32，再 L2 normalize。一个模型进程串行处理，两种模式不会竞态修改截断长度。

GPUStack Bearer key 是客户端鉴权边界；后端只接受控制面、worker 和回环 IP 的连接，不直接对普通局域网客户端提供服务。此 IP 限制信任控制面及 worker 上的本地进程。`/health` 和 `/readyz` 只有权重校验、CUDA 加载及两种模式预热完成后才返回 200。`/metrics` 返回聚合请求数、队列和 Torch 显存峰值，不记录文本或向量。没有新增公网入口。

错误输入 400、超大 body 413、队列满 429、排队超过 30 秒 503、推理超过 120 秒 504。每容器最多 16 个 HTTP handler，套接字读取超时 10 秒。推理进程超时或退出会撤销 readiness、结束该容器，让 GPUStack `restart_on_error` 重建；队列中的请求会释放，不无限挂起。客户端对 429/503 退避重试，504 应限制重试次数，避免长请求循环。

## 验收与回退

本地运行格式、静态和 pytest 检查，另覆盖 `deploy/lca-nemotron`。HTTP、队列和进程测试只验证服务合同，不代表真实模型通过验收。

在同版本镜像中运行 [check_lca_nemotron.py](../scripts/check_lca_nemotron.py)，只读挂载固定模型、固定 benchmark、工具目录，另挂载可写的私有 output。通过 stdin 读取专用 key，禁止把 key 写入参数或打印。不要在宿主开发环境运行 ML 验收。验收可作为新模型容器内独立临时进程运行；提前预留参考模型临时显存。

```bash
python scripts/check_lca_nemotron.py \
  --base-url http://<controller>:17880/model/proxy/<route-id> \
  --model-dir /srv/models/lca-nemotron-embed-1b/6d77edaa3b92dd1b7e3dd8a152ea7170a8b581c5 \
  --data-dir /acceptance/benchmark --output-dir /acceptance/run-unique \
  --benchmark --key-stdin < /private/lca-inference-key
```

验收保存实际 HTTP 与同精度本地参考的向量、文本 hash、manifest、运行版本、Top10、逐 query 指标、延迟及产物 hash。全量重新编码 benchmark 的 15671 documents 和 100 queries，固定 revision `355ae546cbe5a3649c21934e84b368e360b89038`；不使用发布缓存代替服务验收。CPU float32 内积，同分按 document ID 排序。所有向量 cosine ≥ 0.999、max absolute error ≤ 0.02；`lca-fixed-five-reference-v1` 四项指标相对同精度参考绝对差 ≤ 1e-4。记录与公开基线不同之处，不混用常规 graded-qrels nDCG。

验收还应包括每个副本、鉴权拒绝、非法输入、长文截断、并发模式混合、路由分流，以及只重启一个新实例后的恢复。检查原有模型与业务容器的 ID/启动时间和真实请求，不能只以 readiness 宣布共存成功。延迟结论只覆盖记录的长度、batch、并发与样本数。

回退先停用新路由访问，再将本次新建的 LCA deployment 副本数设为 0、撤销其专用 key；核对新 GPU 进程退出。保留权重、镜像与验收证据，不停止原有 Nemotron/MinerU，不操作其他节点或旧路由，不清空共享缓存。业务索引重建和调用方切换属于业务项目，未包含在模型服务部署中。
