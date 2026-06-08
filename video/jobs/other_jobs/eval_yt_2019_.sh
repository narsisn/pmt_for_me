python train_net_video.py --num-gpus 4 \
  --config-file configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \
  --eval-only MODEL.WEIGHTS /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /mnt/adas7tb/narges/eval_output




python video_demo.py \
  --config-file /home/narges/2025_codes/realse_videomt_code/videomt/configs/ovis/videomt/vit-large/videomt_online_ViTL.yaml \
  --input /mnt/second7tb/narges/datasets/youtube_long_videos/videos/rgb-images_backup/13588985_1920_1080_30fps \
  --output /mnt/adas7tb/narges/vis/test \
  --opts MODEL.WEIGHTS /mnt/second7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/ovis/step3/eomt_online_ovis_8_gpus_iter_160k_trackformer_wo_norm_wo_annealing_LLRD_0.6_one_lyaer_mlp_640_1/ovis_vit_large_52.5.pth
  


# python benchmark.py \
#   --task fps \
#   --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \
#   --model-weights  /mnt/adas7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/YT-2019/dinov2/vit-large/step3/60k/eomt_online_ytvis19_8_gpus_60k_trackformer_wo_x_norm_llrd_0.6_wo_masking_onelayer_q_postional_start_pretreained_1280_640_3/yt_2019_vit_large_68.6.pth  \
#   --warmup-iters 100 

# export TIMM_FUSED_ATTN=0 
# python benchmark.py \
#   --task flops \
#   --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml\
#   --model-weights  /mnt/adas7tb/narges/video_segmentation/videomt/cleaned_checkpoints/videomt_cleaned_checkpoints/YT-2019/dinov2/vit-large/step3/60k/eomt_online_ytvis19_8_gpus_60k_trackformer_wo_x_norm_llrd_0.6_wo_masking_onelayer_q_postional_start_pretreained_1280_640_3/yt_2019_vit_large_68.6.pth  