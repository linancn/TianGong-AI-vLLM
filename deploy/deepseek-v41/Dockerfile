ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY engram-mxfp4.patch apply_engram_patch.py apply_sm120_patch.py /opt/tiangong/
RUN python3 /opt/tiangong/apply_engram_patch.py && python3 /opt/tiangong/apply_sm120_patch.py
ENV VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
ENTRYPOINT ["vllm", "serve"]
