# 复用模型与镜像迁移到其他机器

程序通过 GitHub `main` 更新；大模型和已验证的 Docker 镜像可通过局域网复制，不必在目标机重新下载。命令除特别标注外均以各自仓库根目录为工作目录。真实主机地址、传输记录和回滚材料保存在私有 `output/`，不写入公共配置。

## 先核对目标机

- 确认 SSH 登录身份与服务归属。跨机仅交换 SSH 公钥，私钥留在各自机器；不要复制整个 `.ssh`。
- 检查 GPU 型号、数量、显存、驱动、CPU 架构和空闲磁盘。当前四卡 TP4 + EP4 模板已在 SM120、四张 96 GB Blackwell 上验证；其他硬件应重新评估，不能只复制配置就宣称兼容。
- 模型约 132.7 GB，另外为镜像、传输归档和缓存预留空间。保持旧服务运行，直到新文件准备完成。
- 记录旧 Git 提交、旧服务名称与端口。已有 `.env`、未提交更改及旧依赖锁文件先保存到私有回滚目录；不要强制 reset 或覆盖凭证。
- 检查 Docker/CDI。缺少 toolkit-base 时按[部署说明](deployment.md)安装，不为迁移重启共享 Docker daemon。

## 更新程序与复制文件

在目标机确认工作区可快进后执行：

```bash
git status --short
git pull --ff-only origin main
mkdir -p models/Qwen3.8-Flash-Next-NVFP4 output/transfer output/rollback
```

源机器导出已验证镜像。基础镜像引用应与 Dockerfile 对应；下面以本机已经固定到该摘要的 nightly 标签为例，先核对 `docker image inspect`，不要先 pull 一个新的 nightly：

```bash
mkdir -p output/transfer
docker image save --output output/transfer/qwen38-docker.tar \
  tiangong-vllm:qwen38-autotune-v1 vllm/vllm-openai:nightly
sha256sum output/transfer/qwen38-docker.tar > output/transfer/images.sha256
```

在源机器设置 `TARGET_HOST` 为可免密登录的 SSH 目标，`TARGET_REPO` 为目标仓库绝对路径，然后复制。以下选项要求两端 rsync 支持 zstd（实测 rsync 3.2.7）；不支持时去掉三个压缩选项即可。

```bash
rsync -a --partial --append-verify --protect-args \
  --compress --compress-choice=zstd --compress-level=1 --info=progress2 \
  models/Qwen3.8-Flash-Next-NVFP4/ \
  "${TARGET_HOST}:${TARGET_REPO}/models/Qwen3.8-Flash-Next-NVFP4/"
rsync -a --partial --append-verify --protect-args --info=progress2 \
  output/transfer/qwen38-docker.tar output/transfer/images.sha256 \
  "${TARGET_HOST}:${TARGET_REPO}/output/transfer/"
```

断点恢复继续使用原命令；传输结束后仍须做完整 SHA256 校验。不要使用 `--delete` 清空目标目录，也不要复制源机 `.env`、`.venv`、PM2 全局状态或编译/调优缓存。

## 校验与导入

目标机从 `.env.example` 准备配置并核对 GPU、端口、模型目录；旧 `.env` 先备份，不能盲目覆盖。默认不启用 API key。随后执行：

```bash
sha256sum -c output/transfer/images.sha256
docker image load -i output/transfer/qwen38-docker.tar
./deploy/manage.sh verify
./deploy/manage.sh config
```

核对镜像架构、RootFS layer 列表与镜像 Config 是否与源机一致。Docker save/load 对多架构索引的处理可能使顶层 image ID 改变；本次迁移已遇到该现象，不能仅凭顶层 ID 不同判定层内容损坏。传输归档 SHA256 和解包后的层/config 都必须匹配。

在新镜像内做一次 CUDA tensor 运算，确认每张选定 GPU 可用。新机器建立独立编译和调优缓存；相同 GPU 型号也不直接沿用另一台机器的实测调优结果。

## 切换与验收

1. 停止新增提交，检查旧模型 running/waiting 请求，等待收敛。
2. 仅停止、移除本项目旧模型的 PM2 记录，保存 PM2 状态；不操作其他项目进程。原生依赖环境待该模型退出后再退休。
3. 使用已经导入且与当前代码对应的镜像启动，禁止隐式拉取和重建：

   ```bash
   ./deploy/manage.sh start-loaded model
   ./deploy/manage.sh status model
   ./deploy/manage.sh check
   ```

4. 检查模型身份、无 key 默认访问、文本、SSE、工具、图片、reasoning、长输入检索，再从另一台机器验证网络入口。
5. 配置变化或验证缓存重启时用 `restart-loaded model`；常规源码构建部署仍可用 `start/restart`。已导入镜像不包含后续代码补丁更新，更新修复层后必须重新 build 或导入新镜像。
6. 新服务验收通过后，按明确归属清理旧模型、环境与日志；保留所需回滚证据，不全局清空模型缓存或 Docker 资源。

客户端应使用新的 `SERVED_MODEL_NAME`。旧客户端若硬编码旧 Qwen 型号，需要更新该字段；不能将旧型号别名误当作仍在运行旧权重。

## 回滚边界

回滚材料仅供该项目恢复，不能用全局 `pm2 resurrect` 影响其他应用。切换失败时先停止新容器，确认端口/GPU 释放，再按保留的旧配置与环境恢复对应进程。验收前不删除唯一旧权重或唯一私有配置。
