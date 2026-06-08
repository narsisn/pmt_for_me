export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/p/home/jusers/norouzi2/juwels/.cache/huggingface/hub/
python run_from_frames.py \
        --config-file configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
        --weights /p/scratch/seg4video/narges/frozen_videomt/training/yt-vis2021/dinov3/step3/gru/160k/yt-2021_online_dinov3_5/model_final.pth \
        --frames-root  /p/scratch/seg4video/narges/video_datasets/ytvis_2021/valid/JPEGImages \
        --output-root /p/scratch/seg4video/narges/visualization/GRU+PMT/YTVIS21/dinov3 \
        --fps 2 \
        --windows_size -1

