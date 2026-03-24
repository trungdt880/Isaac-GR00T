#!/usr/bin/env bash

export WANDB_API_KEY=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

MODEL_DIR="/Data2/trungdt/tmp/full"
SEEN_FILE="$MODEL_DIR/evaluated_ckpts.txt"

DATASET_PATH="/root/.cache/huggingface/lerobot/uiuc-fruit-bag-oculus-full-30fps"
EMBODIMENT="NEW_EMBODIMENT"

python examples/Aloha16/eval_auto.py --model-dir "$MODEL_DIR" \
    --dataset-path "$DATASET_PATH" \
    --embodiment "$EMBODIMENT" \
    --wandb-project "finetune-gr00t-n1d6" \
    --wandb-run-id "owvnyuix" \
    --sleep 60

####################################
# MODEL_DIR="/Data2/trungdt/tmp/full"
# SEEN_FILE="$MODEL_DIR/evaluated_ckpts.txt"

# DATASET_PATH="/root/.cache/huggingface/lerobot/uiuc-fruit-bag-oculus-full-30fps"
# EMBODIMENT="NEW_EMBODIMENT"

# mkdir -p /tmp/open_loop_eval

# touch "$SEEN_FILE"

# while true; do
#     for ckpt in "$MODEL_DIR"/checkpoint-*; do
#         [ -d "$ckpt" ] || continue

#         # skip if already evaluated
#         if grep -Fxq "$ckpt" "$SEEN_FILE"; then
#             continue
#         fi

#         echo "Evaluating $ckpt"

#         name=$(basename "$ckpt")
#         save_path="/tmp/open_loop_eval"

#         python gr00t/eval/open_loop_eval.py \
#             --dataset-path "$DATASET_PATH" \
#             --embodiment-tag "$EMBODIMENT" \
#             --model-path "$ckpt" \
#             --traj-ids 0 89 \
#             --action-horizon 16 \
#             --steps 3000 \
#             --save_plot_path "$save_path" \
#             --modality-keys left_arm_joints left_gripper right_arm_joints right_gripper 2>& 1 | tee -a "$MODEL_DIR/eval.log"

#         echo "$ckpt" >> "$SEEN_FILE"
#     done

#     sleep 60
# done
