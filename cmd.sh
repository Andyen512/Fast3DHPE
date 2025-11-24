# test
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 14622 --nproc_per_node=4 main.py  --cfg /home/lijunjie/caiqingyuan/ddhpose_opengait_integration_kit/ddhpose/configs/h36m_ddhpose.yaml  --phase test --checkpoint /home/lijunjie/caiqingyuan/ddhpose_opengait_integration_kit/ddhpose/outputs/h36m/DDHPose/h36m_mixste2/checkpoints --resume best_epoch.bin
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 14622 --nproc_per_node=8 main.py  --cfg ./configs/h36m_mixste2.yaml  --phase test --log_to_file --evaluate best_epoch.bin
# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 14622 --nproc_per_node=8 main.py  --cfg ./configs/h36m_ddhpose.yaml  --phase train --log_to_file