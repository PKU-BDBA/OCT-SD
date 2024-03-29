export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="OCT-SD/train_dataset.csv"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WANDB_MODE="offline"

accelerate launch --num_processes=8 --mixed_precision="fp16" OCT-SD Finetune/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=100 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts "normal" "mild retinal edema" "retinal pigment epithelium detachment" "Retinal pigment epithelial layer with strong reflection and bulge" "Cystic low-reflective areas of varying sizes can be seen between retinal layers" \
  --validation_epochs=1 \
  --output_dir="sd-OCT-model" \
  --report_to="wandb"\