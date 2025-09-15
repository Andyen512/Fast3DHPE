import os, argparse, torch
from ddhpose.utils.config import load_cfg
from ddhpose.utils.logger import get_logger
from ddhpose.utils.seed import set_seed
from ddhpose.engine.dist import init_distributed_mode, is_main_process
from ddhpose.modeling.registries import MODELS
from ddhpose.datasets.builder import build_data_bundle
from ddhpose.engine.evaluator import evaluate
from ddhpose.utils.checkpoint import load_state_flexible

def parse_args():
    p = argparse.ArgumentParser(description="DDHPose test (OpenGait-style)")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--log_to_file", action="store_true")
    p.add_argument("--iter", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    init_distributed_mode("nccl")
    cfg = load_cfg(args.cfg)

    out_dir = os.path.join("outputs", cfg["DATASET"]["name"], cfg["MODEL"]["backbone"]["name"], cfg["ENGINE"]["save_name"])
    logger = get_logger(out_dir, to_file=args.log_to_file)
    if is_main_process(): logger.info(cfg)

    set_seed(cfg.get("RUNTIME", {}).get("seed", 42))

    bundle = build_data_bundle(cfg, training=False)
    model_name = cfg["MODEL"]["backbone"]["name"]
    Model = MODELS.get(model_name)
    model = Model(**cfg["MODEL"]["backbone"])

    ckpt = args.weights or cfg.get("ENGINE", {}).get("weights", "")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        load_state_flexible(model, state.get("state_dict", state))

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model.eval()

    results = evaluate(bundle.test_loader, cfg, model, bundle, return_predictions=True)
    if is_main_process():
        logger.info(results)

if __name__ == "__main__":
    main()
