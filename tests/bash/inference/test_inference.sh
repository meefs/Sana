#!/bin/bash
set -e

python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth

python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth

python tools/controlnet/inference_controlnet.py \
    --config=configs/sana_controlnet_config/Sana_600M_img1024_controlnet.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED/checkpoints/Sana_600M_1024px_ControlNet_HED.pth \
    --json_file=asset/controlnet/samples_controlnet.json

python scripts/inference_sana_sprint.py \
    --config=configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml \
    --model_path=hf://Lawrence-cj/Sana_Sprint_1600M_1024px/Sana_Sprint_1600M_1024px_36K.pth \
    --txt_file=asset/samples/samples_mini.txt

python inference_video_scripts/inference_sana_video.py \
    --config=configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
    --debug=true
