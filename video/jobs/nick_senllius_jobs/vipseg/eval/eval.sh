

export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/



python train_net_video.py --num-gpus 2 \
  --config-file  configs/VIPSeg/videomt/vit-large/videomt_segmenter_ViTL.yaml \
  --eval-only MODEL.WEIGHTS /projects/prjs1742/Narges/frozen_videomt/training/vipseg/step2/vipseg_step4/model_0014999.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /projects/prjs1742/Narges/frozen_videomt/eval/vipseg/step2/vipseg_step5
  
DATAROOT='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panomasksRGB'
IMGSAVEROOT='/projects/prjs1742/Narges/frozen_videomt/eval/vipseg/step2/vipseg_step5/inference'
GT_JSONFILE='/projects/prjs1742/Narges/datasets/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json'

# ###VPQ
python utils/eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
# # ###STQ
python utils/eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE