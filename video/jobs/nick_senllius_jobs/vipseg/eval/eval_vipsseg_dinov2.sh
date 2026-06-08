python train_net_video.py --num-gpus 2 \
  --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL_dinov2.yaml \
  --eval-only MODEL.WEIGHTS  /projects/prjs1742/Narges/frozen_videomt/training/vipseg/dinov2/step3/vipseg_step3_1/model_0029999.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/training/vipseg/eval/dinov2_step3

DATAROOT='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panomasksRGB'
IMGSAVEROOT='/projects/prjs1742/Narges/frozen_videomt/training/vipseg/eval/dinov2_step3/inference'
GT_JSONFILE='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json'

###VPQ
python utils/eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
# # ###STQ
python utils/eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE

# python benchmark.py \
#   --task fps \
#   --config-file  configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL_dinov2.yaml \
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vipseg/dinov2/step3/vipseg_step3_1/model_final.pth  \
#   --warmup-iters 100 \
#   --model-type dinov2 


# export TIMM_FUSED_ATTN=0 
# python benchmark.py \
#   --task flops \
#   --config-file  configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL_dinov2.yaml\
#   --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vipseg/dinov2/step3/vipseg_step3_1/model_final.pth  \
#   --model-type dinov2 