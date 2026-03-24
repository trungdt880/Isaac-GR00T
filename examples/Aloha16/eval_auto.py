import os
import time
import subprocess
import re
import argparse
import wandb

MAE_PATTERN = re.compile(r"Average MAE across all trajs:\s*([0-9.]+)")
MSE_PATTERN = re.compile(r"Average MSE across all trajs:\s*([0-9.]+)")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--embodiment", default="NEW_EMBODIMENT", required=True)

    parser.add_argument("--save-path", default="/tmp/open_loop_eval")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-id", required=True)

    parser.add_argument("--sleep", type=int, default=60)

    return parser.parse_args()

def load_seen(seen_file):
    if not os.path.exists(seen_file):
        return set()
    with open(seen_file, "r") as f:
        return set(line.strip() for line in f)

def mark_seen(seen_file, ckpt):
    with open(seen_file, "a") as f:
        f.write(ckpt + "\n")

def extract_step(ckpt_path):
    return int(os.path.basename(ckpt_path).split("-")[-1])

def parse_mae(output):
    matches = MAE_PATTERN.findall(output)
    return float(matches[-1]) if matches else None

def parse_mse(output):
    matches = MSE_PATTERN.findall(output)
    return float(matches[-1]) if matches else None

def parse_output(output):
    mae = parse_mae(output)
    mse = parse_mse(output)
    return mae, mse

def evaluate_checkpoint(ckpt, args, log_file):
    print(f"Evaluating {ckpt}")
    step = extract_step(ckpt)

    cmd = [
        "python", "gr00t/eval/open_loop_eval.py",
        "--dataset-path", args.dataset_path,
        "--embodiment-tag", args.embodiment,
        "--model-path", ckpt,
        "--traj-ids", "0", "89",
        "--action-horizon", "16",
        "--steps", "3000",
        "--save_plot_path", args.save_path,
        "--modality-keys",
        "left_arm_joints", "left_gripper",
        "right_arm_joints", "right_gripper",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout

    with open(log_file, "a") as f:
        f.write(output)

    mae, mse = parse_output(output)
    print(f"MAE: {mae}, MSE: {mse}")

    wandb.log({"eval/mae": mae, "eval/mse": mse}, step=step)


def main():
    args = parse_args()

    seen_file = os.path.join(args.model_dir, "evaluated_ckpts.txt")
    log_file = os.path.join(args.model_dir, "eval.log")

    os.makedirs(args.save_path, exist_ok=True)
    open(seen_file, "a").close()

    wandb.init(
        project=args.wandb_project,
        group=args.wandb_run_id,  # ties eval + train together
        name=f"eval_{args.wandb_run_id}",
        job_type="eval",
    )

    while True:
        seen = load_seen(seen_file)

        ckpts = [
            os.path.join(args.model_dir, d)
            for d in os.listdir(args.model_dir)
            if d.startswith("checkpoint-")
        ]
        ckpts.sort(key=extract_step)

        for ckpt in ckpts:
            if ckpt in seen:
                continue

            evaluate_checkpoint(ckpt, args, log_file)
            mark_seen(seen_file, ckpt)

        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
