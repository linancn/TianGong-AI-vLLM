# 依赖管理

## 隔离边界

模型推理使用 `deploy/vllm/compose.yaml` 指定的本地构建 Docker 镜像；CUDA、Torch、vLLM、Transformers 及推理内核全部留在镜像内。宿主仅依赖 NVIDIA 驱动、Container Toolkit base 1.20.0、支持 CDI 的 Docker 28.3+ / Compose 2.24.4+、Python 3.12+ 与 curl。

`pyproject.toml` 无应用运行依赖；`uv.lock` 固定 Black、Ruff、pytest 等开发工具。标准库脚本可以直接 `python3` 执行。不要向宿主环境重新添加 Torch 或 vLLM，不修改系统 Python。

## 更新方法

1. 模型文件由 `model-manifest.json` 固定 ModelScope revision、大小和 SHA256；更新时在新目录下载校验，不能原地覆盖运行中的权重。
2. 基础镜像使用验证过的 digest，在其上构建受 SHA256 检查约束的运行时修复层，见[运行时修复](runtime-patches.md)。切换镜像必须核对架构、混合量化、CUDA SM120 兼容性，再完成真实 HTTP 验收。
3. 开发工具通过 `uv lock` 更新，并提交锁文件；`uv sync --locked --group dev` 安装。锁文件当前使用清华 PyPI 镜像解析。
4. 记录验证条件和已知边界，成功后更新示例配置及相关指南。保留需要回退的镜像和模型；不要全局 prune。

来源：[模型卡](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)、[vLLM 仓库](https://github.com/vllm-project/vllm)。

## 已固定的引擎基线

基础镜像 `vllm/vllm-openai@sha256:dea7fa047caa114167efccdeb42321b12d2c11da1add8728aa2662cb8d6b8cd5`，vLLM `0.29.1rc1.dev347+gdee37d891`，构建提交 `dee37d89115db4c94a820a79a78a7828e141c910`，Torch `2.13.0+cu130`、CUDA 13.0、Transformers 5.17.0。它是开发构建，因此固定摘要而不在启动时追踪 nightly。
