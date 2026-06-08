python train_net_video.py --num-gpus 4 \
  --config-file configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
  --eval-only MODEL.WEIGHTS /mnt/adas7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/yt-2021/step3/60k/eomt_online_ytvis21_60k_postional_start_pretreained_1280_640_9/yt_2021_vit_large_63.1.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /mnt/adas7tb/narges/eval_output




# python benchmark.py \
#   --task fps \
#   --config-file  configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
#   --model-weights  /mnt/adas7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/yt-2021/step3/60k/eomt_online_ytvis21_60k_postional_start_pretreained_1280_640_9/yt_2021_vit_large_63.1.pth  \
#   --warmup-iters 100 

# export TIMM_FUSED_ATTN=0 
# python benchmark.py \
#   --task flops \
#   --config-file  configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml\
#   --model-weights  /mnt/adas7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/yt-2021/step3/60k/eomt_online_ytvis21_60k_postional_start_pretreained_1280_640_9/yt_2021_vit_large_63.1.pth  