OUTPUT_DIR=/projects/prjs1742/Narges/frozen_videomt/training/vspw/dinov2/step3/640/vspw_step3_1/

python train_net_video.py --num-gpus 2 \
  --config-file configs/VSPW/videomt/vit-large/videomt_online_ViTL_dinov2.yaml \
  --eval-only MODEL.WEIGHTS  /projects/prjs1742/Narges/frozen_videomt/training/vspw/dinov2/step3/640/vspw_step3_1/model_final.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR ${OUTPUT_DIR}
  
DATAROOT='/projects/prjs1742/Narges/datasets/VSPW_480p'
IMGSAVEROOT="${OUTPUT_DIR}/inference"

if [ ! -d "$IMGSAVEROOT" ]; then
  echo "Missing prediction directory: $IMGSAVEROOT"
  exit 1
fi

python utils/eval_miou_vspw.py "$DATAROOT" "$IMGSAVEROOT"
python utils/eval_vc_vspw.py "$DATAROOT" "$IMGSAVEROOT"


# python benchmark.py \
#   --task fps \
#   --config-file  configs/VSPW/videomt/vit-large/videomt_online_ViTL_dinov2.yaml \
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vspw/dinov2/step3/640/vspw_step3_1/model_final.pth  \
#   --warmup-iters 200 \
#   --model-type dinov2 


# export TIMM_FUSED_ATTN=0 
# python benchmark.py \
#   --task flops \
#   --config-file  configs/VSPW/videomt/vit-large/videomt_online_ViTL_dinov2.yaml\
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vspw/dinov2/step3/640/vspw_step3_1/model_final.pth  \
#   --model-type dinov2 