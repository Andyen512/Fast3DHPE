#!/bin/bash

# 设置内存使用阈值为100MiB
threshold=100

# 设置检查间隔时间（秒），例如5分钟
sleep_interval=600

while true; do
    # 检测GPU占用情况
    gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # 将GPU内存占用转换为数组
    gpu_usages=($gpu_usage)

    # 检测所有GPU是否空闲
    all_gpus_idle=true
    for mem in "${gpu_usages[@]}"; do
        if (( $(echo "$mem > $threshold" | bc -l) )); then
            all_gpus_idle=false
            break
        fi
    done

    # 如果所有GPU都空闲，执行训练命令
    if $all_gpus_idle; then
        echo "All GPUs are idle. Starting training..."
        # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 opengait/main.py --cfgs ./configs/dronegait2/gaitbase_multi_view_loss.yaml --phase train --log_to_file
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 14623 --nproc_per_node=8 main.py  --cfg ./configs/h36m_ddhpose.yaml  --phase train --log_to_file
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 14623 --nproc_per_node=8 main.py  --cfg ./configs/h36m_D3DP.yaml  --phase train --log_to_file
        # 假设只想运行一次训练任务，运行后退出循环
        break
        # 如果想在任务完成后继续监控，不要加 break，而是等待下一个间隔
    else
        echo "Not all GPUs are idle. Will check again in $sleep_interval seconds."
    fi
    # 等待指定的检查间隔时间
    sleep $sleep_interval
done
