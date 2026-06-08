export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/


# python benchmark.py \
#   --task fps \
#   --config-file  configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
#   --model-weights   /projects/prjs1742/Narges/frozen_videomt/training/vipseg/step3/20k/vipseg_step1/model_final.pth \
#   --model-type dinov3 \
#   --warmup-iters 100 

# [Batch 335] Proccesed Frames online: 5700, Time: 99856.73s, Average FPS online: 57.08
# [Batch 336] Proccesed Frames online: 5715, Time: 100116.33s, Average FPS online: 57.08
# [Batch 337] Proccesed Frames online: 5761, Time: 100912.40s, Average FPS online: 57.09
# [Batch 338] Proccesed Frames online: 5776, Time: 101172.94s, Average FPS online: 57.09
# [Batch 339] Proccesed Frames online: 5791, Time: 101432.79s, Average FPS online: 57.09
# [Batch 340] Proccesed Frames online: 5806, Time: 101694.15s, Average FPS online: 57.09
# [Batch 341] Proccesed Frames online: 5852, Time: 102489.49s, Average FPS online: 57.10
# [Batch 342] Proccesed Frames online: 5867, Time: 102749.49s, Average FPS online: 57.10

export TIMM_FUSED_ATTN=0 
python benchmark_vps.py \
  --task flops \
  --config-file  configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vipseg/step3/20k/vipseg_step1/model_final.pth \
  --model-type dinov3 