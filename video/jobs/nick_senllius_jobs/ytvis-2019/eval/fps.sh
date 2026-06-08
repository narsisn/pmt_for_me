export WANDB_API_KEY=ce287f8c5da7131a9fb925d19e8f748f2742f57d
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=offline
export HF_HOME=/home/ncavagnero/.cache/huggingface/hub/



python benchmark.py \
  --task fps \
  --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth  \
  --model-type dinov3  \
  --warmup-iters 100 
  
# [Batch 297] Proccesed Frames online: 5509, Time: 51673.47s, Average FPS online: 106.61
# [Batch 298] Proccesed Frames online: 5539, Time: 51951.53s, Average FPS online: 106.62
# [Batch 299] Proccesed Frames online: 5575, Time: 52282.97s, Average FPS online: 106.63
# [Batch 300] Proccesed Frames online: 5609, Time: 52599.11s, Average FPS online: 106.64
# [Batch 301] Proccesed Frames online: 5624, Time: 52739.60s, Average FPS online: 106.64

export TIMM_FUSED_ATTN=0 
python benchmark.py \
  --task flops \
  --config-file  configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml\
  --model-weights  /projects/prjs1742/Narges/frozen_videomt/training/yt-vis/step3/yt-2019_onilne_correting_pe_3/model_final.pth \
  --model-type dinov3 
   