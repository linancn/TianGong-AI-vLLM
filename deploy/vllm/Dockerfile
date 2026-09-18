FROM vllm/vllm-openai@sha256:dea7fa047caa114167efccdeb42321b12d2c11da1add8728aa2662cb8d6b8cd5
COPY deploy/vllm/patches/apply_runtime_fixes.py /opt/tiangong/apply_runtime_fixes.py
RUN python3 /opt/tiangong/apply_runtime_fixes.py
COPY deploy/vllm/serve.sh /opt/tiangong/serve.sh
LABEL ai.tiangong.runtime-fixes="flashinfer-distributed-cache-v1"
