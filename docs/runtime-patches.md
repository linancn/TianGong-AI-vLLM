# 固定镜像与运行时修复

## 构建边界

`deploy/vllm/Dockerfile` 从固定的 vLLM 官方镜像摘要构建 `tiangong-vllm:qwen38-autotune-v1`。只添加小型 Python 修复和启动脚本，不重新编译 vLLM/CUDA，不改模型权重。补丁脚本先校验两个上游源文件的 SHA256 和唯一替换位置，再编译检查结果；上游文件变化时构建失败，维护者必须重新审查。

`deploy/manage.sh build` 使用已有基础镜像构建；`pull` 拉取固定基础镜像并构建修复层；start/restart 自动构建。镜像标识和运行参数保存在私有部署/测量证据中。基础版本见[依赖指南](dependencies.md)。

## FlashInfer 调优缓存停滞

当前 FlashInfer `MoERunner.get_cache_key_extras()` 将 TP/EP rank 写入持久缓存键。上游 vLLM 的 `flashinfer_autotune()` 却只保存 rank 0 的缓存，再于下次启动将其广播给所有 rank。于是 rank 0 命中旧结果，其他 rank 没有自己的条目，进入 profiling。同步调优会对每个候选算法的耗时做 Gloo all-reduce；各 rank 调用次数不同导致等待。

实测原缓存只有 EP rank 0 的 42 项。原镜像再次启动可复现 rank 0 cache hit、其他 rank tuning 0% 停滞；这不是权重未下载、CUDA 不可见或显存不足。

修复位于 `deploy/vllm/patches/apply_runtime_fixes.py`：

1. 仅在 tuning 模式的每个 profile 中，对缓存命中标志取跨 rank 的 MIN；只有所有 rank 都命中才跳过 profiling。部分命中时所有 rank 重新测量，保留上游的同步耗时与算法选择。
2. 在共享缓存目录中，依次让每个 rank 保存。FlashInfer 自带的保存逻辑会合并已有文件条目，因此保存完整 EP 键，避免覆盖丢失。

自动调优继续开启。修复不删除缓存、不去掉 rank 键、不使用未调优的 heuristic 替代结果，且不在正常请求路径增加 collective。

该保存方式针对本项目单机、单容器、四卡共用同一缓存卷的部署。多主机独立文件系统不在此补丁的适用范围内。

## 回归方法

常规测试检查补丁上游指纹、替换位置与失败时不改文件。维护窗口可运行四进程 CUDA 小算子回归（必须有四张可用 GPU）：

```bash
./deploy/manage.sh build
docker run --rm --device nvidia.com/gpu=all -e OMP_NUM_THREADS=1 --shm-size 1g \
  -v "$PWD/scripts/check_autotune.py:/check.py:ro" \
  --entrypoint python3 tiangong-vllm:qwen38-autotune-v1 /check.py
```

回归构造所有 rank 仅加载 rank 0 持久键的情形，验证部分命中时所有 rank 测量、全部命中时没有重复 profiling，并核对算子输出。原镜像在此测试失败，修复镜像通过。

真实模型验收还须检查：已有不完整缓存可正常启动并补全、带完整缓存重启成功、`deploy/manage.sh check` 全部通过。旧缓存修复后已观察到 EP rank 0/1/2/3 各 42 项，共 168 项；这不是合成小算子结果。

## MTP 兼容

所固定镜像已包含 [vLLM #55513](https://github.com/vllm-project/vllm/pull/55513) 的混合量化 MTP 修复：block-FP8 MoE dispatch 与 MTP 量化层名重映射。MTP 是否启用仍由真实推理及性能对照决定，见[性能调优](performance-tuning.md)。
