# DeepSeek V4.1 Flash 1017 NVFP4

本部署使用 ModelScope 的 `nv-community/DeepSeek-V4.1-Flash-1017-NVFP4`，由 GPUStack custom backend 管理单节点八卡实例。权重约 400 GiB；原生上下文上限为 1,048,576 tokens，包含输入与输出。已通过鉴权网关的 1,047,920-token 输入与三处信息召回；实际结果保存到私有 `output/deepseek-v41/`。

## 固定权重与运行环境

- `deploy/deepseek-v41/model-manifest.json` 固定 ModelScope revision、85 个文件的大小和 SHA256。使用现有 `scripts/download_model.py --manifest ... --directory ...` 下载和校验，权重只读挂载。
- 基础镜像为 vLLM 0.33.0，构建时必须提供不可变的 `BASE_IMAGE` 摘要；不在启动时安装依赖。容器包含 FlashInfer 0.7.2.post1，实际版本仍须通过目标镜像检查。
- `runtime-manifest.json` 保存基础镜像摘要与补丁来源；先加载该镜像，再运行 `bash deploy/deepseek-v41/build.sh`。构建入口只使用已存在的固定镜像，不隐式拉取。
- `engram-mxfp4.patch` 从上游 [vLLM #61530](https://github.com/vllm-project/vllm/pull/61530) 的 `c2e1c43648cfe7db4da42dac931561658c9f69ab` 回移。只修改 Engram 的 packed MXFP4 存储、反量化和格式验证；DP 共享内存分配适配到 0.33.0 的实现。
- 构建脚本先核对原始 `engram.py` 的 SHA256，再无 fuzz 应用补丁；源码变化时直接失败，不能静默套用到其他版本。目标 GPU 上另用上游用例比对 HBM/主机内存、不同 token 数和调度方式下的 MXFP4 查表结果。
- SM120 缓存页声明回移 [vLLM #65356](https://github.com/vllm-project/vllm/pull/65356) 的 `f5129016ec14ddbd7b5bdf6b42dbf697d0166caf`：sparse MLA 和 indexer 在 SM120 选择 64-token 页，保留其他架构行为；两个原始文件也须通过 SHA256 检查。实例显式设置 `--block-size=64`，避免默认 128 页在 DeepGEMM 预热时失败。
- 保持 PCIe IPC 实验关闭；八卡通信使用 NCCL。RTX PRO 6000 SM120 与 GB300 SM100 的内核能力不同，不能直接套用 GB300 的 MegaMoE 配置。

## GPUStack 配置

通用后端模板为 `deploy/examples/deepseek-v41-gpustack-backend.json`。实例的 worker/GPU 选择器、部署路径、注册身份和推理 key 只保存在私有配置。一个副本显式选择同一 worker 的八张 GPU；不能把副本数设为八。

参数保留完整上下文，使用 TP8 + EP、FP8 KV/indexer、64-token 页、GPU 常驻 Engram、8 条序列和每轮 4096 个 prefill tokens。SM120 使用 FP8 indexer；vLLM 0.33.0 的 MXFP4 indexer 限制为 SM10x，不能直接套用模型卡的 GB300 参数。DSpark 使用 5 个推测 tokens、probabilistic draft、block rejection，关闭 adaptive verification；此配置通过真实功能测试并提高了样本吞吐。长输入优先，不能只为提高短请求并发而缩小上下文。

后端直接提供 OpenAI Chat Completions，支持 `deepseek_v41` reasoning/tool parser。普通短答显式设置 `chat_template_kwargs.thinking=false`；推理模式设置 `thinking=true` 并给足输出 token 预算。不要将思考耗尽输出预算后的空 content 误报成模型加载失败。

先完成镜像构建、Engram GPU 对照和八卡 NCCL 检查，再创建模型 deployment。调优期设置 `restart_on_error=false`，避免配置失败后重复加载数百 GiB 权重；稳定验收后恢复退出重启策略。只修改当前模型，不更新共享后端或其他模型参数。

## 验收与调优

`scripts/check_deepseek_v41.py` 是必须显式运行的真实 API 验收，不属于看板探针。`--base-url` 使用不含 `/v1` 的服务根地址；推理 key 通过权限 600 的 `--key-file` 文件读取，不放到命令行。

- `--suite smoke`：中英文算术、system role、工具调用、thinking、图片与 SSE。
- `--suite context --input-tokens N`：先通过后端 `/tokenize` 计数，再执行三处审计码召回。输出同时记录实际 token 数、耗时和召回是否成功；请求完成与召回正确分开判断。
- `--suite benchmark --input-tokens N --output-tokens M --concurrency C --requests R`：记录 SSE 首字延迟、解码速度与总输出吞吐。该测试显式标记共享前缀，不能当作全部冷缓存的吞吐。

按短输入、32K、128K、较大输入和接近原生上限逐级验收；完整上下文测试预留输出空间。比较 DSpark 开关、批处理大小和并发时保持请求集与其他参数一致。记录错误、抢占、温度、降频和 GPU 利用率；没有测过的并发或图片数量不作容量承诺。

在八张 96 GiB RTX PRO 6000 SM120 上，8,110-token 合成输入、512-token 输出、共享前缀条件下，关闭 DSpark 的单请求解码约 98 tokens/s、四并发总吞吐约 299 tokens/s；开启后单请求解码约 173–197 tokens/s、四并发总吞吐约 361 tokens/s。单请求为四轮、四并发为八轮；总吞吐包含首字等待。结果只覆盖该样本和缓存条件，不能外推到所有任务或满长并发。功能验收覆盖中英文、system、tool、thinking、图片和 SSE，原始结果保留在私有 output 目录。

DSpark 的长输入测试：524,204 tokens 用时 126.43 秒；经 GPUStack 鉴权代理发送 1,047,920 tokens 用时 253.02 秒。两者均正确返回分布在输入 10%/50%/90% 处的三条审计码，实际 prompt token 计数与 `/tokenize` 一致。大输入预留了 256-token 输出预算，实际输出 20 tokens；这证明单请求接近原生上限可处理，不代表所有长文任务的质量或八请求满长并发。

## 恢复

先将本模型副本数缩到零，确认子容器退出，再调整参数或回退固定镜像。停止 worker 不保证其模型子容器停止。保留固定权重和编译缓存，不执行全局 prune；回退仅涉及当前 deployment。统一路由验收通过后才向调用方提供专用推理 key，动态实例端口仅用于诊断。

模型说明：[ModelScope 模型卡](https://www.modelscope.cn/models/nv-community/DeepSeek-V4.1-Flash-1017-NVFP4)。
