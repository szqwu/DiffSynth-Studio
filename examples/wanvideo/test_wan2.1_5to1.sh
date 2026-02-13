# checkpoint_path="./models/train/Wan2.2-TI2V-5B_lora128_SE_6to1/epoch-25.safetensors"
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-5to1_3/epoch-59.safetensors"
# data_path="/data2/qiwu2/dl3dv_test_plucker"
# data_path="/data2/qiwu2/dl3dv_test_SE_train_val_960"
# data_path="/data2/qiwu2/dl3dv_test_bench_plucker_separate_encoding"
data_path="/data2/qiwu2/test_data"
output_dir="wan_14b_se_output_lora32_5to1"

CUDA_VISIBLE_DEVICES=6 python test_wan2.1_5to1.py \
    --checkpoint_path $checkpoint_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --use_dreamsim \
    # --height 480 \
    # --width 832