#!/bin/bash
# Waits for any running train_SE.py job to finish, then:
#   1) Tries 256x256
#   2) Falls back to 240x240 if 256x256 fails (e.g. OOM)

cd /data2/qiwu2/DiffSynth-Studio/examples/wanvideo

# --- Wait for current training to finish ---
if pgrep -f "train_SE.py" > /dev/null 2>&1; then
    echo "$(date): Waiting for current train_SE.py job to finish..."
    while pgrep -f "train_SE.py" > /dev/null 2>&1; do
        sleep 60
    done
    echo "$(date): Current job finished. Waiting 30s for GPU cleanup..."
    sleep 30
else
    echo "$(date): No running train_SE.py found. Starting immediately."
fi

# --- Try 256x256 ---
echo "============================================"
echo "$(date): Starting 256x256 training attempt"
echo "============================================"

bash model_training/lora/Wan2.1-SE-14B.sh

if [ $? -eq 0 ]; then
    echo "============================================"
    echo "$(date): Wan2.1-SE-14B completed successfully!"
    echo "============================================"
else
    echo "============================================"
    echo "$(date): Wan2.1-SE-14B FAILED (likely OOM)."
    echo "============================================"
    sleep 10
    # bash model_training/lora/Wan2.1-SE-CameraAdapter-14B-240x240.sh
fi
