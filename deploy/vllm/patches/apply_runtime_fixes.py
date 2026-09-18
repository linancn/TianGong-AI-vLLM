"""Apply narrowly scoped fixes to the pinned vLLM/FlashInfer image at build time."""

import hashlib
from pathlib import Path


def replace_checked(path: Path, expected_sha256: str, before: str, after: str) -> None:
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Upstream file changed; review patch before upgrading: {path}")
    if source.count(before) != 1:
        raise RuntimeError(f"Expected one patch anchor in {path}")
    result = source.replace(before, after)
    compile(result, str(path), "exec")
    path.write_text(result)


def main() -> None:
    packages = Path("/usr/local/lib/python3.12/dist-packages")
    replace_checked(
        packages / "flashinfer/autotuner/autotuner.py",
        "75bc65f5d1936867fd6c1fa5eca1629e7334b88110865abd14f2f52225c60ccc",
        """                    if not is_cache_hit:
                        # Active capture is safe""",
        """                    # TianGong: rank-dependent persisted keys can yield partial
                    # hits. Every rank must either profile or skip this bucket,
                    # otherwise per-tactic timing collectives deadlock.
                    if _tune_process_group is not None:
                        import torch.distributed as dist

                        backend = str(dist.get_backend(_tune_process_group)).lower()
                        hit = torch.tensor(
                            [int(is_cache_hit)], dtype=torch.int32,
                            device="cuda" if backend == "nccl" else "cpu",
                        )
                        dist.all_reduce(
                            hit, op=dist.ReduceOp.MIN, group=_tune_process_group
                        )
                        is_cache_hit = bool(hit.item())
                    if not is_cache_hit:
                        # Active capture is safe""",
    )
    replace_checked(
        packages / "vllm/model_executor/warmup/kernel_warmup.py",
        "1b40d25f1edeef59fe40b578aa9b0f63189bd5e2663af4982b01de800609351a",
        """    if world.world_size > 1:
        world.barrier()
    if is_leader:
        tuner.save_configs(str(cache_path))
""",
        """    if world.world_size > 1:
        world.barrier()
    # TianGong single-node shared cache: save every rank's distinct EP keys.
    # save_configs merges on-disk records; serialize writers to avoid lost keys.
    for cache_writer_rank in range(world.world_size):
        if world.rank_in_group == cache_writer_rank:
            tuner.save_configs(str(cache_path))
        if world.world_size > 1:
            world.barrier()
""",
    )


if __name__ == "__main__":
    main()
