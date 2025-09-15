import os, argparse, torch, torch.nn as nn
from utils.config import load_cfg
from utils.logger import get_logger
from utils.seed import set_seed
from engine.dist import init_distributed_mode, is_main_process
from modeling.registries import MODELS
from datasets.builder import build_data_bundle, build_data_bundle_test
from modeling.trainer import Trainer
from modeling import models
from remote_pdb import set_trace

def parse_args():
    p = argparse.ArgumentParser(description="DDHPose train (OpenGait-style)")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    p.add_argument("--log_to_file", action="store_true")
    p.add_argument("--iter", type=int, default=0)
    p.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    p.add_argument("--resume", type=str, default="")       
    p.add_argument("--checkpoint", type=str, default="", help="directory that contains checkpoints")
    return p.parse_args()

def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str, logger, strict_default=True):
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    obj = torch.load(ckpt_path, map_location="cpu")

    # unwrap common wrappers
    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state", "net", "module"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    state_dict = obj

    model.load_state_dict(state_dict, strict=strict_default)
    logger.info(f"Loaded checkpoint strictly: {ckpt_path}")
    return

def main():
    args = parse_args()
    init_distributed_mode("nccl")
    cfg = load_cfg(args.cfg)

    out_dir = os.path.join("outputs", cfg["DATASET"]["name"], cfg["MODEL"]["name"], cfg["ENGINE"]["save_name"])
    logger = get_logger(out_dir, to_file=args.log_to_file)
    if is_main_process(): logger.info(cfg)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    set_seed(cfg.get("RUNTIME", {}).get("seed", 42) + rank)
    
    # 训练数据
    if args.phase == "train":
        bundle = build_data_bundle(cfg, training=True)
    else:
        bundle_list = build_data_bundle(cfg, training=False)
        bundle = bundle_list[0]  # 随便取一个，主要是为了拿到 skeleton/joints 信息
        
    # 构建模型
    model_name = cfg["MODEL"]["name"]
    Model = getattr(models, model_name)      
    model = Model(**cfg["MODEL"]["backbone"], joints_left=bundle.joints_left, joints_right=bundle.joints_right)

    # 设备 & DDP
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
        find_unused_parameters=cfg.get("ENGINE", {}).get("find_unused_parameters", False))

    if args.resume:
        tag = args.resume if args.resume else args.evaluate
        ckpt_root = args.checkpoint or os.path.join(out_dir, "checkpoints")
        ckpt_path = os.path.join(ckpt_root, tag)
        if is_main_process():
            print("Loading checkpoint", ckpt_path)


    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Params: {n_params/1e6:.2f} M")
    
    trainer = Trainer(cfg, logger, out_dir)
    if args.phase == "train":
        if args.resume:
            trainer.train(model, bundle.train_loader, bundle.val_loader, bundle, ckpt_path)
        else:
            trainer.train(model, bundle.train_loader, bundle.val_loader, bundle)
    else:
        trainer.test(cfg, model, bundle_list, ckpt_path)

if __name__ == "__main__":
    main()
