

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/pjancura/.cache/huggingface/hub/
python run_from_frames.py \
        --config-file configs/ovis/videomt/vit-large/videomt_online_ViTL_tbptt.yaml \
        --weights /projects/0/tesei0745/Narges/gru_videomt/ovis/dinov3/step3/160k/ovis_15_7_bppt_dataloader/model_0117499_56.9.pth \
        --frames-root  /projects/0/tesei0745/Narges/video-datasets/ovis/valid \
        --output-root /projects/0/tesei0745/Narges/vis_for_bppt/gru+bppt/OVIS/15-7_56.9-160k/dinov3 \
        --val-list /home/pjancura/Narges/codes/codes_2025/frozen_videomt/visualization/jobs/new_jobs_for_datasets/val_ovis.txt \
        --fps 2 \
        --windows_size -1

