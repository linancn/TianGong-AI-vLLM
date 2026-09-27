# 依赖管理

## 隔离边界

Qwen 使用 `deploy/vllm/compose.yaml` 指定的本地构建 Docker 镜像；Embed 的 `deploy/manage.sh pull embed` 从固定摘要拉取上游镜像，再赋予 `deploy/embed/compose.yaml` 使用的本地标签。CUDA、Torch、vLLM、Transformers 及推理内核全部留在各自镜像内。宿主依赖 NVIDIA 驱动、支持 CDI 的 Docker 28.3+ / Compose 2.24.4+、Python 3.12+ 与 curl；apt Docker 使用 Container Toolkit base 1.20.0，Snap Docker 自带 NVIDIA 工具链且须核对 daemon 的 CDI 规格搜索目录，见[部署说明](deployment.md)。

`pyproject.toml` 无应用运行依赖；`uv.lock` 固定 Black、Ruff、pytest 等开发工具。标准库脚本可以直接 `python3` 执行。不要向宿主环境重新添加 Torch 或 vLLM，不修改系统 Python。

## 更新方法

1. Qwen 与 Embed 文件分别由 `deploy/vllm/model-manifest.json` 和 `deploy/embed/model-manifest.json` 固定 ModelScope revision、大小和 SHA256；更新时在新目录下载校验，不能原地覆盖运行中的权重。
2. Qwen 基础镜像使用验证过的 digest，在其上构建受 SHA256 检查约束的运行时修复层，见[运行时修复](runtime-patches.md)。Embed 使用独立固定 digest 的官方 vLLM 镜像，不继承 Qwen 专用补丁。切换镜像必须核对对应模型架构、精度、CUDA 与 GPU 兼容性，再完成真实 HTTP 验收。
3. 开发工具通过 `uv lock` 更新，并提交锁文件；`uv sync --locked --group dev` 安装。锁文件当前使用清华 PyPI 镜像解析。
4. 记录验证条件和已知边界，成功后更新示例配置及相关指南。保留需要回退的镜像和模型；不要全局 prune。

来源：[Qwen 模型卡](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)、[Nemotron Embed 模型卡](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/README.md)、[vLLM 仓库](https://github.com/vllm-project/vllm)。

## 已固定的引擎基线

基础镜像 `vllm/vllm-openai@sha256:dea7fa047caa114167efccdeb42321b12d2c11da1add8728aa2662cb8d6b8cd5`，vLLM `0.29.1rc1.dev347+gdee37d891`，构建提交 `dee37d89115db4c94a820a79a78a7828e141c910`，Torch `2.13.0+cu130`、CUDA 13.0、Transformers 5.17.0。它是开发构建，因此固定摘要而不在启动时追踪 nightly。

Embed 上游镜像摘要固定在 `deploy/manage.sh`；Compose 和 `.env.example` 使用可迁移的本地标签。NVIDIA 模型说明针对 BF16 `/v2/embed` 推荐 vLLM 0.25.0。更新此镜像时先核对镜像实际版本、GPU 架构与 DP4 加载，再重跑 Embed 的真实检索验收；Qwen 的运行时修复及实测性能结论不能套用到 Embed。

## 版本选择与升级判据

截至 2026-09-27，[vLLM 最新正式版为 0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)。本仓库固定版本是已测部署基线，不代表以后不升级，也不能仅凭版本号判断性能。当前 Qwen 镜像为固定提交的开发构建；0.30.0 发布说明包含 Qwen3.8 的 QSA、PLE、FP8 indexer 和 MTP 相关改进，值得作为独立候选镜像做同机对照，但尚无本项目 0.30.0 的性能或功能验收结论。当前补丁校验上游文件 SHA256，更换基础镜像须检查该修复是否已进入上游，并重新审查补丁，不能直接改 tag 启动。

Embed 0.25.0 是[NVIDIA 对 BF16 `/v2/embed` 的推荐版本](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16/blob/main/README.md#vllm-dependencies)，并非永久版本上限。若评估新版，保持 BF16 和相同模型修订，比较查询／文档向量合同、检索质量、吞吐、延迟、显存峰值、四副本分配和与 MinerU 的共卡负载。[MinerU 4.0.7 包元数据](https://pypi.org/pypi/mineru/4.0.7/json)的 `full` extra 声明 vLLM `<0.29.0`，不能将本仓库的 Qwen 候选镜像直接给 MinerU 使用；各服务维持独立运行时。

候选版本先固定镜像摘要，在隔离实例完成 `check`、长上下文、模型专属功能、MTP 接受率、FlashInfer 自动调优、缓存重启与同口径 benchmark，再比较显存及业务延迟；只有收益或必要修复得到证据支持，才替换默认摘要。当前 PCIe IPC／FlashInfer 0.7.0rc3 实验仍暂停，评估其他升级时保持该通信后端关闭。既有四卡 Qwen 样本与其边界见[性能调优](performance-tuning.md)，不能将其视为 0.30.0 的对照结果。
