export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/
OUTPUT_DIR=/projects/prjs1742/Narges/frozen_videomt/training/vspw/step2/vspw_step2

python train_net_video.py --num-gpus 2 \
  --config-file configs/VSPW/videomt/vit-large/videomt_segmenter_ViTL.yaml \
  MODEL.WEIGHTS /projects/prjs1742/Narges/pre-trained_models/dinov3/ps/coco_pan_vit_l_next_mf_640_videomt.ckpt \
  SOLVER.IMS_PER_BATCH 2 \
  TEST.EVAL_PERIOD 100 \
  SOLVER.MAX_ITER 40000 \
  OUTPUT_DIR ${OUTPUT_DIR}
  
DATAROOT='/projects/prjs1742/Narges/datasets/VSPW_480p'
IMGSAVEROOT="${OUTPUT_DIR}/inference"

if [ ! -d "$IMGSAVEROOT" ]; then
  echo "Missing prediction directory: $IMGSAVEROOT"
  exit 1
fi

python utils/eval_miou_vspw.py "$DATAROOT" "$IMGSAVEROOT"
python utils/eval_vc_vspw.py "$DATAROOT" "$IMGSAVEROOT"