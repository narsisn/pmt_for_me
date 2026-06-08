python train_net_video.py --num-gpus 1 \
  --config-file configs/ytvis19/videomt/vit-large/videomt_online_ViTL_dinov2.yaml \
  --eval-only MODEL.WEIGHTS  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/dinov2/step3/yt-2019_dinov2_online_3/model_final.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/eval



python benchmark.py \
  --task fps \
  --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL_dinov2.yaml \
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/dinov2/step3/yt-2019_dinov2_online_3/model_final.pth  \
  --warmup-iters 100 \
  --model-type dinov2 


export TIMM_FUSED_ATTN=0 
python benchmark.py \
  --task flops \
  --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL_dinov2.yaml\
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/dinov2/step3/yt-2019_dinov2_online_3/model_final.pth  \
  --model-type dinov2 