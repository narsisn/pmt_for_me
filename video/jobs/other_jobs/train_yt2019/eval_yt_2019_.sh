
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/narges/.cache/huggingface/hub
python train_net_video.py  --num-gpus 2 \
  --config-file configs/ytvis19/videomt/vit-large/videomt_segmenter_ViTL.yaml \
  --resume MODEL.WEIGHTS /mnt/adas7tb/narges/frozen_videomt/pre-trained-models/frozen_eomt/is/coco_ins_vit_l_next_mf_640_videomt.ckpt \
  SOLVER.MAX_ITER 40000 \
  SOLVER.IMS_PER_BATCH 2 \
  TEST.EVAL_PERIOD 500 \
  OUTPUT_DIR /mnt/adas7tb/narges/frozen_videomt/train 



# python train_net_video.py  --num-gpus 2 \
#   --config-file configs/ytvis19/videomt/vit-large/videomt_segmenter_ViTL.yaml \
#   --eval-only MODEL.WEIGHTS  /mnt/second7tb/narges/frozen_videomt/training/yt-2019/step2/yt-2019_2/model_final.pth \
#   MODEL.BACKBONE.MASKED_ATTN_ENABLED False \
#   MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
#   OUTPUT_DIR /mnt/adas7tb/narges/frozen_videomt/eval
