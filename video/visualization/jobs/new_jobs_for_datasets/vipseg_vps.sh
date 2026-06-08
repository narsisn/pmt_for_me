export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/p/home/jusers/norouzi2/juwels/.cache/huggingface/hub/
python run_from_frames.py \
        --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
        --weights  /p/scratch/seg4video/narges/frozen_videomt/training/vipseg/dinov3/step3/vipseg_step1/model_final.pth \
        --frames-root  /p/scratch/seg4video/narges/video_datasets/VIPSeg/VIPSeg_720P/images \
        --output-root /p/scratch/seg4video/narges/visualization/GRU+PMT/vispseg/dinov3 \
        --val-list /p/scratch/seg4video/narges/video_datasets/VIPSeg/val.txt \
        --fps 2 \
        --windows_size -1