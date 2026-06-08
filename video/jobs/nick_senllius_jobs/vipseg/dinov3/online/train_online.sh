#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=08:15:00
#SBATCH --output=jobs/nick_senllius_jobs/job_outputs/%j.out

module load 2025

__conda_setup="$('/sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /home/ncavagnero/miniconda3/envs/fvideomt

cd /home/ncavagnero/narges/frozen_videomt

export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((12000 + RANDOM % 20000))
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eno2np0"

export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/



srun --ntasks=2 --ntasks-per-node=1 bash -c "python train_net_video.py --num-gpus 4 --num-machines 2 \
  --machine-rank \$SLURM_PROCID \
  --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
  --config-file  configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
  MODEL.WEIGHTS /projects/prjs1742/Narges/frozen_videomt/training/vipseg/step2/vipseg_step4/model_final.pth \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.EVAL_PERIOD 40000 \
  SOLVER.MAX_ITER 40000 \
  OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/training/vipseg/step3/40k/vipseg_step1 "
  
DATAROOT='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panomasksRGB'
IMGSAVEROOT='/projects/prjs1742/Narges/frozen_videomt/training/vipseg/step3/40k/vipseg_step1/inference'
GT_JSONFILE='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json'

# ###VPQ
python utils/eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
# # ###STQ
python utils/eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE