
python run_long_videos.py \
        --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
        --weights /projects/0/tesei0745/Narges/GRU_frozen_videomt/vipseg/dinov3/step3/vipseg_step1/model_final.pth \
        --video-root /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/TrafficCAM \
        --output-root  /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/TrafficCAM \
        --fps 5 \
        --windows_size -1

# python run_long_videos.py \
#         --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
#         --weights /projects/0/tesei0745/Narges/GRU_frozen_videomt/vipseg/dinov3/step3/vipseg_step1/model_final.pth \
#         --video-root /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/cityscapes \
#         --output-root  /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/cityscapes \
#         --fps 7 \
#         --windows_size -1

# python run_long_videos.py \
#         --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
#         --weights /projects/0/tesei0745/Narges/GRU_frozen_videomt/vipseg/dinov3/step3/vipseg_step1/model_final.pth \
#         --video-root /projects/0/tesei0745/Narges/vis_for_gijs/VidEoMT_VIS/Video_instance_segmentation/long_videos \
#         --output-root  /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/long_videos \
#         --fps 30 \
#         --windows_size -1

python run_long_videos.py \
        --config-file configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml \
        --weights /projects/0/tesei0745/Narges/GRU_frozen_videomt/vipseg/dinov3/step3/vipseg_step1/model_final.pth \
        --video-root /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/MAN_TruckScenes \
        --output-root  /projects/0/tesei0745/Narges/vis_for_gijs/new_vis_output/Video_panpti_segmentation/MAN_TruckScenes \
        --fps 7 \
        --windows_size -1
