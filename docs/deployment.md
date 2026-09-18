# 部署与恢复

## 前置条件

命令均在仓库根目录执行。需要 Linux、NVIDIA GPU 驱动、NVIDIA Container Toolkit、Docker Engine 28.3+ / Compose 2.24.4+ 和 Python 3.12+、curl。宿主机不安装 vLLM/Torch；uv 仅用于开发工具。

模板使用四张 Blackwell GPU、TP=4 + EP=4。NVFP4、Qwen4Exp 架构和 FP8 PLE 混合量化均要求镜像支持，不能任意替换为老版本。模型卡要求至少包含 vLLM 提交 `d4d703caf908786416585ceb1f369e2e0363358b`；固定镜像已包含混合量化 MTP 修复，模板默认使用 MTP 3，实测依据见[性能调优](performance-tuning.md)。来源：[模型卡](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)。

本部署使用 Docker 原生 CDI 设备映射。安装 NVIDIA 官方 `nvidia-container-toolkit-base` 即可，不需要配置 legacy runtime 或重启 Docker。按 [NVIDIA 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)配置官方软件源后执行：

```bash
sudo apt-get install -y nvidia-container-toolkit-base=1.20.0-1
nvidia-ctk cdi list
systemctl is-enabled nvidia-cdi-refresh.service nvidia-cdi-refresh.path
```

刷新服务为 oneshot，成功退出后显示 inactive 正常；`.path` 与开机启动负责自动刷新。驱动变化后设备列表不正确时执行 `sudo systemctl restart nvidia-cdi-refresh.service`，不要重启 Docker。

检查 `nvidia-smi`、`docker compose version`、`/dev/nvidia-uvm`，并确认 7730 端口和所选 GPU 的归属。模型文件总量约 132.7 GB；额外预留下载、容器镜像和编译缓存空间。

## 私有配置

```bash
cp -n .env.example .env
chmod 600 .env
```

默认 `VLLM_API_KEY` 留空，也可不配置，业务请求无需 Authorization。需要鉴权时，将非空 key 写入 `.env` 并执行 `./deploy/manage.sh restart model`，客户端随后携带 `Authorization: Bearer <key>`；清空该值并重启即可关闭鉴权。可用 `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'` 生成随机 key，不要提交或输出到对话。模型 API 默认在所有网卡监听；只本机使用可设 `VLLM_HOST=127.0.0.1`。

| 配置 | 模板值 / 含义 |
| --- | --- |
| `VLLM_IMAGE` | 本地修复镜像 `tiangong-vllm:qwen38-autotune-v1`；Dockerfile 固定上游 digest |
| `MODEL_DIR` | `./models/Qwen3.8-Flash-Next-NVFP4`，相对仓库根 |
| `SERVED_MODEL_NAME` | `nv-community/Qwen3.8-Flash-Next-NVFP4` |
| `VLLM_PORT` | 7730 |
| `VLLM_API_KEY` | 空／未配置：关闭鉴权；非空：启用 Bearer |
| `GPU_0..GPU_3` / `TENSOR_PARALLEL_SIZE` | GPU 0/1/2/3，TP4 + EP4 |
| `GPU_MEMORY_UTILIZATION` | 0.85；是每张卡的引擎预算 |
| `MAX_MODEL_LEN` | 65536；请求输入与输出合计上限 |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | 16 / 8192 |
| `SPECULATIVE_CONFIG` | 模板为 `{"method":"mtp","num_speculative_tokens":3}`；空值关闭 MTP |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` | 默认 0；当前 FlashInfer 缺少 IPC 接口，设置 1 仍会回退 NCCL |

配置采用简单 `KEY=value`，避免插值、内联注释或多行值，以便 Compose 与标准库运维脚本一致读取。不要执行会将密钥展开到终端的 `docker compose config`；使用 `deploy/manage.sh config` 做静默检查。

## 模型准备与启动

```bash
./deploy/manage.sh download
./deploy/manage.sh pull
./deploy/manage.sh config
./deploy/manage.sh start model
./deploy/manage.sh status model
./deploy/manage.sh logs model
```

下载从 ModelScope 固定提交取文件，curl 支持中断续传，SHA256 验证完成后原子替换单个文件。`verify` 检查完整清单；不要在模型正在加载时修改权重。推理只读本地权重并启用离线模式，不在启动时静默换模型版本。

`pull` 会拉取固定基础镜像并构建修复层；也可使用 `build` 复用本地基础镜像。首次加载、编译可能需较长时间。`logs` 为持续跟踪，Ctrl-C 只退出日志查看。健康后执行：

```bash
./deploy/manage.sh check
```


## 维护、停止、恢复

1. 停止新增请求，查看模型 `/metrics` 的 `vllm:num_requests_running`、`vllm:num_requests_waiting`，等待归零。
2. `./deploy/manage.sh stop model` 停止本项目服务，保留模型和编译缓存。
3. 恢复：`./deploy/manage.sh start model`，随后执行真实验收。
4. 修改配置后执行 `./deploy/manage.sh restart model`，重建容器以应用变更；此操作会中断在途请求。

`restart: unless-stopped` 配合已启用的 Docker 系统服务完成机器重启恢复，无需 PM2。主动 stop 的容器保持停止。不要使用全局 Docker 清理或重启共享 daemon。

旧原生部署迁移时先识别 PM2 名称及进程树，确认请求排空后删除对应记录、保存 PM2 列表，再删除明确归属的旧模型与环境。不要按“Qwen”字符串批量删除其他项目的 embedding 服务或共享缓存。

## 故障排查

- 镜像拉取或 PyPI TLS 超时：重试，使用已配置且可信的网络出口；不要修改共享 Docker 配置来掩盖问题。国内 Python 镜像可用于锁文件解析，仍须保留包版本和哈希。
- CUDA 初始化失败：检查容器 GPU 可见性和 UVM 设备；`nvidia-smi` 正常不保证 CUDA 分配成功。在目标镜像内执行一次 CUDA tensor 运算。
- 不支持模型架构／FP8 PLE：核对模型 revision 和镜像内实际 vLLM 版本，不用旧 Qwen 模板或更换模型冒充修复。
- NVFP4 `Intermediate size padding`：保持模板中的 `--enable-expert-parallel`。此模型专家宽度 640，纯 TP4 后每片 160，当前 FLASHINFER_CUTLASS 的 gated 权重 padding 不支持该形状；EP4 保持专家完整宽度。
- 自动调优缓存停滞：确认使用[修复镜像](runtime-patches.md)，保留自动调优并检查各 rank 缓存条目；不要用关闭调优作为默认解决方案。
- OOM：确认其他进程占用，降低上下文、并发或多模态容量；记录修改并重做验收。
- HTTP 401：启用鉴权时，客户端须使用 `.env` 的 key；已清空配置但仍返回 401 时，确认执行了 `restart model` 以应用变更。
- `/health` 正常但首请求挂起：检查 worker、编译和模型加载日志，必须以真实请求返回验收。

升级前记录当前镜像摘要、模型清单和配置；保留仍需回退的镜像。回退只恢复已知兼容组合，不覆盖正在使用的模型文件。
