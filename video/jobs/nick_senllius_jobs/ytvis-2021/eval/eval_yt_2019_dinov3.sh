# python train_net_video.py --num-gpus 1 \
#   --config-file configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
#   --eval-only MODEL.WEIGHTS  /projects/prjs1742/Narges/frozen_videomt/training/yt_vis2021/dinov3/step3/yt-2021_online_dinov3_4/model_final.pth \
#   MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
#   OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/eval



python benchmark.py \
  --task fps \
  --config-file  configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt_vis2021/dinov3/step3/yt-2021_online_dinov3_4/model_final.pth  \
  --warmup-iters 100 \
  --model-type dinov3 


export TIMM_FUSED_ATTN=0 
python benchmark.py \
  --task flops \
  --config-file  configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml \
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt_vis2021/dinov3/step3/yt-2021_online_dinov3_4/model_final.pth  \
  --model-type dinov3 