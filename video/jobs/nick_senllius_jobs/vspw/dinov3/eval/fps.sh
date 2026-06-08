export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/


python benchmark.py \
  --task fps \
  --config-file  configs/VSPW/videomt/vit-large/videomt_online_ViTL.yaml \
  --model-weights   /projects/prjs1742/Narges/frozen_videomt/training/vspw/step3/20k/vspw_step3_1/model_final.pth \
  --model-type dinov3 \
  --warmup-iters 200 

# [Batch 336] Proccesed Frames online: 8025, Time: 138189.10s, Average FPS online: 58.07
# [Batch 337] Proccesed Frames online: 8085, Time: 139222.97s, Average FPS online: 58.07
# [Batch 338] Proccesed Frames online: 8160, Time: 140516.91s, Average FPS online: 58.07
# [Batch 339] Proccesed Frames online: 8220, Time: 141552.16s, Average FPS online: 58.07
# [Batch 340] Proccesed Frames online: 8263, Time: 142295.36s, Average FPS online: 58.07
# [Batch 341] Proccesed Frames online: 8323, Time: 143330.24s, Average FPS online: 58.07
# [Batch 342] Proccesed Frames online: 8398, Time: 144624.66s, Average FPS online: 58.07

export TIMM_FUSED_ATTN=0 
python benchmark_vps.py \
  --task flops \
  --config-file  configs/VSPW/videomt/vit-large/videomt_online_ViTL.yaml\
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/vspw/step3/20k/vspw_step3_1/model_final.pth \
  --model-type dinov3 