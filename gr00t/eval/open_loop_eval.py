from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro
from gr00t.data.types import (
    ModalityConfig,
)


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def smooth_actions(actions: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing independently to each action dimension."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(actions, sigma=sigma, axis=0)


def temporal_ensemble(
    all_chunks: list[tuple[int, np.ndarray]],
    total_steps: int,
    action_dim: int,
    exp_weight: float = 0.0,
) -> np.ndarray:
    """Blend overlapping action chunks using (optionally exponential) averaging.

    Args:
        all_chunks: list of (start_step, chunk_array) pairs.  chunk shape: (chunk_len, action_dim).
        total_steps: total number of timesteps to produce.
        action_dim: action dimensionality.
        exp_weight: if >0, newer chunks are weighted by exp(-exp_weight * age).
                    if 0, simple uniform average.
    """
    accum = np.zeros((total_steps, action_dim), dtype=np.float64)
    weights = np.zeros((total_steps, 1), dtype=np.float64)

    for start, chunk in all_chunks:
        chunk_len = min(len(chunk), total_steps - start)
        for j in range(chunk_len):
            w = np.exp(-exp_weight * j) if exp_weight > 0 else 1.0
            accum[start + j] += w * chunk[j, :action_dim]
            weights[start + j] += w

    weights = np.maximum(weights, 1e-8)
    return (accum / weights).astype(np.float32)


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    replan_steps: int | None = None,
    smooth_sigma: float = 0.0,
    save_plot_path=None,
):
    replan = replan_steps or action_horizon

    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    # TODO: tmp hardcode
    modality_configs["language"] = ModalityConfig(
        delta_indices=[0],
        modality_keys=[f'task'],
    )
    loader.modality_configs["language"] = ModalityConfig(
        delta_indices=[0],
        modality_keys=[f'task'],
    )

    use_ensemble = replan < action_horizon
    all_chunks: list[tuple[int, np.ndarray]] = []
    pred_action_across_time: list[np.ndarray] = []

    for step_count in range(0, actual_steps, replan):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)

        chunk_actions = []
        for j in range(action_horizon):
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            chunk_actions.append(concat_pred_action)
        chunk_array = np.stack(chunk_actions)

        if use_ensemble:
            all_chunks.append((step_count, chunk_array))
        else:
            end = min(step_count + replan, actual_steps)
            for j in range(end - step_count):
                pred_action_across_time.append(chunk_array[j] if j < len(chunk_array) else chunk_array[-1])

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]

    action_dim = gt_action_across_time.shape[1]
    if use_ensemble:
        pred_action_across_time_arr = temporal_ensemble(all_chunks, actual_steps, action_dim)
    else:
        pred_action_across_time_arr = np.array(pred_action_across_time)[:actual_steps]

    if smooth_sigma > 0:
        logging.info(f"Applying Gaussian smoothing (sigma={smooth_sigma})")
        pred_action_across_time_arr = smooth_actions(pred_action_across_time_arr, smooth_sigma)

    assert gt_action_across_time.shape == pred_action_across_time_arr.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time_arr.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time_arr) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time_arr))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time_arr.shape}")

    # Plot trajectory results
    resolved_plot_path = save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg"
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time_arr,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=resolved_plot_path,
    )

    # Save actions as .npz for downstream visualization (e.g. viser 3D viz)
    npz_path = str(Path(resolved_plot_path).with_suffix(".npz"))
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        pred_action=pred_action_across_time_arr,
        gt_action=gt_action_across_time,
        state_joints=state_joints_across_time,
        action_keys=np.array(action_keys),
        state_keys=np.array(state_keys),
    )
    logging.info(f"Saved actions to {npz_path}")

    txt_path = str(Path(resolved_plot_path).with_suffix(".txt"))
    with open(txt_path, "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    replan_steps: int | None = None
    """Receding-horizon: only use first K actions per chunk, then re-infer.
    If K < action_horizon, overlapping chunks are blended via temporal ensembling.
    Defaults to action_horizon (no replanning)."""

    smooth_sigma: float = 0.0
    """Gaussian smoothing sigma applied to the final predicted trajectory.
    0 = disabled. Try 1.0-3.0 to remove chunk-boundary jitter."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        if args.save_plot_path is not None:
            plot_path = Path(args.save_plot_path)
            if plot_path.is_file():
                logging.info(f"Plot path {plot_path} should be a folder, use parent folder instead")
                plot_path = plot_path.parent
            model_name = Path(args.model_path).name
            plot_path = plot_path / Path(args.model_path).parent.name /  f"traj_{traj_id}_{model_name}.jpeg"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        # policy.model.action_head.num_inference_timesteps = 16
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            replan_steps=args.replan_steps,
            smooth_sigma=args.smooth_sigma,
            save_plot_path=plot_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
