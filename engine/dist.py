import os, torch, torch.distributed as dist

def init_distributed_mode(backend: str = "nccl"):
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("Distributed environment variables are not set. Use torchrun.")

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def is_main_process():
    return get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()
