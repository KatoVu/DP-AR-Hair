import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from hairshifter.utils import strip_ddp_prefix, unwrap


def test_ddp_checkpoint_loads_into_plain_module():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        trained = DDP(nn.Linear(4, 2))
        fresh = nn.Linear(4, 2)

        # New checkpoints: saved unwrapped, load strictly as-is
        fresh.load_state_dict(unwrap(trained).state_dict())
        assert torch.equal(fresh.weight, trained.module.weight)

        # Old checkpoints: saved from the DDP wrapper with "module." keys
        old = trained.state_dict()
        assert all(k.startswith("module.") for k in old)
        fresh = nn.Linear(4, 2)
        fresh.load_state_dict(strip_ddp_prefix(old))
        assert torch.equal(fresh.weight, trained.module.weight)
        assert unwrap(fresh) is fresh
    finally:
        dist.destroy_process_group()
