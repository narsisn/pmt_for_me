python train_net_video.py --num-gpus 2 \
  --config-file configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \
  --eval-only MODEL.WEIGHTS /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/eval



# python benchmark.py \
#   --task fps \
#   --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth  \
#   --warmup-iters 100 \
#   --model-type dinov3

# export TIMM_FUSED_ATTN=0 
# python benchmark.py \
#   --task flops \
#   --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml\
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth  \
#   --model-type dinov3