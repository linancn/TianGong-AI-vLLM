"""Container-only distributed regression for partial FlashInfer cache hits."""

import datetime
import tempfile
from pathlib import Path


def worker(rank: int, world_size: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist
    from flashinfer.autotuner import (
        AutoTuner,
        TunableRunner,
        TuningConfig,
        autotune,
        set_autotune_process_group,
    )

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=15),
    )

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0, 1]

        def get_cache_key_extras(self, inputs):
            return (rank,)

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            return torch.add(inputs[0], 1, out=inputs[1])

    runner = Runner()
    tuner = AutoTuner.get()
    tuner.warmup = tuner.repeat = 1
    config = TuningConfig()
    inputs = [torch.zeros(16, device="cuda"), torch.empty(16, device="cuda")]
    key = tuner._get_cache_key("tiangong_cache_test", runner, ((16,), (16,)), config, (0,))
    # Mirror the old vLLM behavior: all ranks load rank 0's persisted key.
    tuner._file_configs[key.file_key] = ("Runner", 0)
    set_autotune_process_group(dist.group.WORLD)
    try:
        with autotune(True):
            _, tactic = tuner.choose_one("tiangong_cache_test", [runner], config, inputs)
        assert tactic in (0, 1)
        assert tuner.profiling_cache, "Every rank must profile when any rank misses"
        assert torch.equal(runner(inputs, tactic=tactic), torch.ones(16, device="cuda"))

        # All ranks now hit their own in-memory entries. No profiling is allowed.
        def unexpected_profile(*args, **kwargs):
            raise AssertionError("Warm cache unexpectedly reprofiled")

        tuner._profile_single_kernel = unexpected_profile
        with autotune(True):
            _, warm_tactic = tuner.choose_one("tiangong_cache_test", [runner], config, inputs)
        assert warm_tactic == tactic
        dist.barrier()
        if rank == 0:
            print("PASS: partial hits synchronize; all-hit warmup reuses tuned tactics", flush=True)
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(4, str(Path(directory) / "gloo")), nprocs=4, join=True)
