# checkpoint_path="./models/train/Wan2.2-TI2V-5B_lora128_SE_6to1/epoch-25.safetensors"
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-5to1/epoch-19.safetensors"
data_path="/data2/qiwu2/dl3dv_test_plucker_separate_encoding"
output_dir="wan_14b_se_output_lora32_5to1"

CUDA_VISIBLE_DEVICES=6 python test_wan2.1_5to1.py \
    --checkpoint_path $checkpoint_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --use_dreamsim