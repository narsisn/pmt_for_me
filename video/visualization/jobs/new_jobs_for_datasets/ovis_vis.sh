module load Stages/2025
# module load Python/3.12.3
module load CUDA PyCUDA PyTorch torchvision

source /e/project1/seg4video/narges/libs/envs/videomt/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/e/home/jusers/norouzi2/jupiter/.cache/huggingface/hub/
python run_from_frames.py \
        --config-file configs/ovis/videomt/vit-large/videomt_online_ViTL.yaml \
        --weights  /e/scratch/seg4video/narges/frozen_videomt_checkpoints/frozen_videomt/ovis/dinov3/step3/ovis_online_10/model_final.pth \
        --frames-root  /e/scratch/seg4video/narges/video_datasets/ovis/valid \
        --output-root /e/scratch/seg4video/narges/visualization/PMT_just/OVIS/dinov3 \
        --val-list /e/project1/seg4video/narges/codes/2026_codes/fps_pipline_bppt/visualization/jobs/new_jobs_for_datasets/val_ovis.txt \
        --fps 2 \
        --windows_size -1

