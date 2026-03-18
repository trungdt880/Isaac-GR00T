set -x -euo pipefail

export WANDB_API_KEY=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
python gr00t/eval/open_loop_eval.py \
    --dataset-path /root/.cache/huggingface/lerobot/uiuc-fruit-bag-oculus-last10eps \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /Data2/trungdt/tmp/aloha16_finetune/checkpoint-10000 \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 400 \
    --save_plot_path /tmp/open_loop_eval/traj_0_ckpt10k_fulldata_short.jpeg \
    --modality-keys left_arm_joints left_gripper right_arm_joints right_gripper
