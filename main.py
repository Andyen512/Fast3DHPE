import os, argparse, torch, torch.nn as nn
from utils.config import load_cfg
from utils.logger import get_logger
from utils.seed import set_seed
from modeling.dist import init_distributed_mode, is_main_process
from modeling.registries import MODELS
from datasets.builder import *
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
    p.add_argument("--evaluate", type=str, default="")       
    p.add_argument("--checkpoint", type=str, default="", help="directory that contains checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_mode("nccl")
    
    cfg = load_cfg(args.cfg)

    out_dir = os.path.join("outputs", cfg["DATASET"]["test_dataset"], cfg["MODEL"]["name"], cfg["ENGINE"]["save_name"])
    logger = get_logger(out_dir, to_file=args.log_to_file)
    if is_main_process(): logger.info(cfg)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    set_seed(cfg.get("RUNTIME", {}).get("seed", 1))
    
    # 训练数据
    if args.phase == "train":
        bundle = build_data_bundle(cfg, training=True)
    else:
        if cfg["DATASET"]["train_dataset"] == cfg["DATASET"]["test_dataset"]:
            bundle_list = build_data_bundle(cfg, training=False)
            bundle = bundle_list[0]  # 随便取一个，主要是为了拿到 skeleton/joints 信息
        else:
            bundle_list = build_data_bundle(cfg, training=False)
            bundle = bundle_list[0]  # 随便取一个，主要是为了拿到 skeleton/joints 信息

    skeleton = bundle.dataset.skeleton if bundle.dataset.skeleton is not None else None
    # 构建模型
    model_name = cfg["MODEL"]["name"]
    Model = getattr(models, model_name)    
    model = Model(**cfg["MODEL"]["backbone"], joints_left=bundle.joints_left, joints_right=bundle.joints_right, \
                  rootidx=cfg["DATASET"]["Root_idx"], dataset_skeleton=skeleton)
    # 设备 & DDP

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.get("ENGINE", {}).get("find_unused_parameters", False),
    )

    if args.resume or args.evaluate:
        tag = args.resume if args.resume else args.evaluate
        ckpt_root = args.checkpoint or os.path.join(out_dir, "checkpoints")
        ckpt_path = os.path.join(ckpt_root, tag)
        if is_main_process():
            print("Loading checkpoint", ckpt_path)


    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Params: {n_params/1e6:.2f} M")
    
    trainer = Trainer(cfg, logger, out_dir, device)
    if args.phase == "train":
        if args.resume:
            trainer.train(model, bundle.train_loader, bundle.val_loader, bundle, ckpt_path)
        else:
            trainer.train(model, bundle.train_loader, bundle.val_loader, bundle)

    else:
        trainer.test(cfg, model, bundle_list, ckpt_path)

if __name__ == "__main__":
    main()
