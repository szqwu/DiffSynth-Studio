# checkpoint_path="./models/train/Wan2.2-TI2V-5B_lora128_SE_6to1/epoch-25.safetensors"
checkpoint_path="./models/train/Wan2.2-TI2V-5B_full_SE/step-3750.safetensors"
data_path="/data2/qiwu2/dl3dv_test_SE_train_val"
output_dir="wan_se_output_full_5to1"

CUDA_VISIBLE_DEVICES=7 python test_wan2.2_5to1.py \
    --checkpoint_path $checkpoint_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --use_dreamsim