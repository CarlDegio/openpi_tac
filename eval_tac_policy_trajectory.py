"""Evaluate a TAC pi0.5 checkpoint on one local LeRobot episode and plot trajectories."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from typing import Any, Literal

import numpy as np
import pandas as pd
import tyro
from tqdm.auto import tqdm

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)

TacSubset = Literal["all", "clean", "smash"]
EvalSubset = Literal["clean", "smash"]
EVAL_COLUMNS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.state",
    "actions",
    "frame_index",
    "task_index",
)
TACTILE_EVAL_COLUMNS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_0",
    "observation.images.tactile_right_1",
)


@dataclasses.dataclass
class Args:
    tac_subset: TacSubset = "all"
    eval_subsets: tuple[EvalSubset, ...] = ("clean", "smash")
    tac: _config.TacMode = "false"
    repo_root: pathlib.Path = pathlib.Path("/home/ubuntu/tac_data")
    dataset_path: pathlib.Path | None = None
    episode_index: int = 0
    checkpoint_config: str | None = None
    exp_name: str | None = None
    checkpoint_step: int | None = None
    checkpoint_dir: pathlib.Path | None = None
    output_dir: pathlib.Path | None = None
    max_frames: int | None = None
    num_denoise_steps: int = 10
    prompt: str | None = None
    save_npy: bool = True


@dataclasses.dataclass(frozen=True)
class EvalTarget:
    subset: str
    dataset_path: pathlib.Path
    episode_index: int
    output_dir: pathlib.Path


@dataclasses.dataclass(frozen=True)
class ResolvedArgs:
    checkpoint_config: str
    checkpoint_dir: pathlib.Path
    output_dir: pathlib.Path
    targets: tuple[EvalTarget, ...]


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _discover_local_tac_repos(root: pathlib.Path, subset: TacSubset) -> list[pathlib.Path]:
    repos = [
        path
        for path in sorted(root.expanduser().iterdir())
        if (path / "meta" / "info.json").is_file() and (subset == "all" or f"_{subset}_" in path.name)
    ]
    if not repos:
        raise FileNotFoundError(f"No local TAC repos found under {root} for subset={subset!r}")
    return repos


def _default_checkpoint_config(subset: TacSubset) -> str:
    return f"pi05_tac_{subset}"


def _default_exp_name(subset: TacSubset) -> str:
    return f"tac_{subset}_pi05"


def _latest_checkpoint_step(checkpoint_root: pathlib.Path) -> int:
    steps = [int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint steps found under {checkpoint_root}")
    return max(steps)


def _infer_subset_from_repo_name(dataset_path: pathlib.Path, fallback: TacSubset) -> str:
    if "_clean_" in dataset_path.name:
        return "clean"
    if "_smash_" in dataset_path.name:
        return "smash"
    return fallback


def _resolve_args(args: Args) -> ResolvedArgs:
    checkpoint_config = args.checkpoint_config or _default_checkpoint_config(args.tac_subset)
    exp_name = args.exp_name or _default_exp_name(args.tac_subset)

    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        checkpoint_step = args.checkpoint_step
    else:
        checkpoint_root = pathlib.Path("checkpoints") / checkpoint_config / exp_name
        checkpoint_step = args.checkpoint_step or _latest_checkpoint_step(checkpoint_root)
        checkpoint_dir = (checkpoint_root / str(checkpoint_step)).resolve()

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else pathlib.Path(f"outputs/tac_eval_pi05_{args.tac_subset}_{checkpoint_step or 'custom'}_ep{args.episode_index}")
    )

    if args.dataset_path is not None:
        dataset_path = args.dataset_path.expanduser().resolve()
        targets = (
            EvalTarget(
                subset=_infer_subset_from_repo_name(dataset_path, args.tac_subset),
                dataset_path=dataset_path,
                episode_index=args.episode_index,
                output_dir=output_dir,
            ),
        )
    else:
        eval_subsets: tuple[str, ...] = args.eval_subsets if args.tac_subset == "all" else (args.tac_subset,)
        if args.tac_subset == "all" and not {"clean", "smash"}.issubset(eval_subsets):
            raise ValueError("--tac-subset all must evaluate at least one clean and one smash trajectory.")
        target_list = []
        for subset in eval_subsets:
            dataset_path = _discover_local_tac_repos(args.repo_root, subset)[0].resolve()
            target_list.append(
                EvalTarget(
                    subset=subset,
                    dataset_path=dataset_path,
                    episode_index=args.episode_index,
                    output_dir=output_dir / f"{subset}_{dataset_path.name}_ep{args.episode_index}"
                    if len(eval_subsets) > 1
                    else output_dir,
                )
            )
        targets = tuple(target_list)

    return ResolvedArgs(
        checkpoint_config=checkpoint_config,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        targets=targets,
    )


def _read_episode_frame(dataset_path: pathlib.Path, episode_index: int, tac_mode: _config.TacMode) -> pd.DataFrame:
    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    parquet_path = dataset_path / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")
    requested_columns = EVAL_COLUMNS + (TACTILE_EVAL_COLUMNS if tac_mode in ("true", "onlyload") else ())
    columns = [column for column in requested_columns if column in info["features"]]
    return pd.read_parquet(parquet_path, columns=columns)


def _episode_length(dataset_path: pathlib.Path, episode_index: int) -> int:
    for episode in _read_jsonl(dataset_path / "meta" / "episodes.jsonl"):
        if int(episode["episode_index"]) == episode_index:
            return int(episode["length"])
    raise ValueError(f"Episode {episode_index} was not found in {dataset_path / 'meta' / 'episodes.jsonl'}")


def _task_map(dataset_path: pathlib.Path) -> dict[int, str]:
    return {int(item["task_index"]): item["task"] for item in _read_jsonl(dataset_path / "meta" / "tasks.jsonl")}


def _make_observation(row: pd.Series, tasks: dict[int, str], prompt_override: str | None) -> dict[str, Any]:
    prompt = prompt_override
    if prompt is None:
        prompt = tasks[int(row["task_index"])]
    observation = {
        "observation.images.camera0": row["observation.images.camera0"],
        "observation.images.camera1": row["observation.images.camera1"],
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }
    for key in TACTILE_EVAL_COLUMNS:
        if key in row:
            observation[key] = row[key]
    return observation


def _load_policy(args: Args, resolved: ResolvedArgs):
    checkpoint_dir = resolved.checkpoint_dir
    train_config = dataclasses.replace(_config.get_config(resolved.checkpoint_config), tac=args.tac)
    action_horizon = train_config.model.action_horizon
    if train_config.model.action_dim != 32:
        raise ValueError(f"Expected pi05 TAC action_dim=32, got {train_config.model.action_dim}")

    logger.info("Loading policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_denoise_steps},
    )
    return policy, action_horizon


def _evaluate_target(
    args: Args, target: EvalTarget, policy, action_horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_path = target.dataset_path
    episode_len = _episode_length(dataset_path, target.episode_index)
    frame = _read_episode_frame(dataset_path, target.episode_index, args.tac)
    if len(frame) != episode_len:
        raise ValueError(f"Episode metadata length {episode_len} does not match parquet length {len(frame)}")

    tasks = _task_map(dataset_path)
    max_frames = episode_len if args.max_frames is None else min(args.max_frames, episode_len)
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    frame_index_chunks: list[np.ndarray] = []

    desc = f"{target.subset}/{dataset_path.name}/episode_{target.episode_index} chunks"
    for chunk_start in tqdm(range(0, max_frames, action_horizon), desc=desc):
        chunk_end = min(chunk_start + action_horizon, max_frames)
        obs = _make_observation(frame.iloc[chunk_start], tasks, args.prompt)
        result = policy.infer(obs)
        pred = np.asarray(result["actions"], dtype=np.float32)[: chunk_end - chunk_start]
        gt = np.stack(frame["actions"].iloc[chunk_start:chunk_end].to_numpy()).astype(np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match ground truth shape {gt.shape}")
        pred_chunks.append(pred)
        gt_chunks.append(gt)
        frame_index_chunks.append(frame["frame_index"].iloc[chunk_start:chunk_end].to_numpy(dtype=np.int64))

    return np.concatenate(pred_chunks), np.concatenate(gt_chunks), np.concatenate(frame_index_chunks)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    groups = {
        "left_xyz": (0, 3),
        "left_rot_cols": (3, 9),
        "left_gripper": (9, 10),
        "right_xyz": (10, 13),
        "right_rot_cols": (13, 19),
        "right_gripper": (19, 20),
    }
    metrics: dict[str, Any] = {
        "all_dims": {
            "mae": float(np.mean(np.abs(pred - gt))),
            "rmse": float(np.sqrt(np.mean(np.square(pred - gt)))),
        },
        "groups": {},
    }
    for name, (start, end) in groups.items():
        diff = pred[:, start:end] - gt[:, start:end]
        metrics["groups"][name] = {
            "dims_0_based": [start, end],
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        }
    return metrics


def _write_plots(pred: np.ndarray, gt: np.ndarray, frame_indices: np.ndarray, output_dir: pathlib.Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x = frame_indices
    names = [
        "left_dx",
        "left_dy",
        "left_dz",
        "left_r00",
        "left_r10",
        "left_r20",
        "left_r01",
        "left_r11",
        "left_r21",
        "left_gripper",
        "right_dx",
        "right_dy",
        "right_dz",
        "right_r00",
        "right_r10",
        "right_r20",
        "right_r01",
        "right_r11",
        "right_r21",
        "right_gripper",
    ]

    fig, axes = plt.subplots(5, 4, figsize=(20, 16), dpi=150, sharex=True)
    for dim, ax in enumerate(axes.ravel()):
        ax.plot(x, gt[:, dim], label="ground truth", linewidth=1.3)
        ax.plot(x, pred[:, dim], label="policy", linewidth=1.1, alpha=0.85)
        ax.set_title(f"{dim}: {names[dim]}")
        ax.grid(visible=True, linewidth=0.35, alpha=0.35)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.supxlabel("episode frame")
    fig.tight_layout()
    fig.savefig(output_dir / "actions_20d.png")
    plt.close(fig)

    gt_left = np.cumsum(gt[:, 0:3], axis=0)
    pred_left = np.cumsum(pred[:, 0:3], axis=0)
    gt_right = np.cumsum(gt[:, 10:13], axis=0)
    pred_right = np.cumsum(pred[:, 10:13], axis=0)

    fig = plt.figure(figsize=(16, 7), dpi=150)
    left_ax = fig.add_subplot(1, 2, 1, projection="3d")
    right_ax = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title, gt_xyz, pred_xyz in (
        (left_ax, "left cumulative delta xyz", gt_left, pred_left),
        (right_ax, "right cumulative delta xyz", gt_right, pred_right),
    ):
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="ground truth", linewidth=1.8)
        ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="policy", linewidth=1.5)
        ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="black", s=18, label="start")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_xyz_trajectory_3d.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150, sharex=True)
    for row, side, gt_xyz, pred_xyz in (
        (0, "left", gt_left, pred_left),
        (1, "right", gt_right, pred_right),
    ):
        for axis, label in enumerate("xyz"):
            ax = axes[row, axis]
            ax.plot(x, gt_xyz[:, axis], label="ground truth", linewidth=1.4)
            ax.plot(x, pred_xyz[:, axis], label="policy", linewidth=1.2, alpha=0.85)
            ax.set_title(f"{side} cumulative {label}")
            ax.grid(visible=True, linewidth=0.35, alpha=0.35)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.supxlabel("episode frame")
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_xyz_components.png")
    plt.close(fig)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    resolved = _resolve_args(args)
    resolved.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Resolved checkpoint_config=%s", resolved.checkpoint_config)
    logger.info("Resolved checkpoint_dir=%s", resolved.checkpoint_dir)
    for target in resolved.targets:
        logger.info(
            "Resolved eval target subset=%s dataset_path=%s episode_index=%d output_dir=%s",
            target.subset,
            target.dataset_path,
            target.episode_index,
            target.output_dir,
        )

    policy, action_horizon = _load_policy(args, resolved)
    summary = []
    for target in resolved.targets:
        target.output_dir.mkdir(parents=True, exist_ok=True)
        pred, gt, frame_indices = _evaluate_target(args, target, policy, action_horizon)
        _write_plots(pred, gt, frame_indices, target.output_dir)
        metrics = _compute_metrics(pred, gt)
        metrics.update(
            {
                "subset": target.subset,
                "dataset_path": str(target.dataset_path),
                "episode_index": target.episode_index,
                "checkpoint_config": resolved.checkpoint_config,
                "checkpoint_dir": str(resolved.checkpoint_dir),
                "tac": args.tac,
                "num_frames": int(pred.shape[0]),
                "action_dim": int(pred.shape[1]),
                "num_denoise_steps": args.num_denoise_steps,
            }
        )
        (target.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if args.save_npy:
            np.save(target.output_dir / "pred_actions.npy", pred)
            np.save(target.output_dir / "gt_actions.npy", gt)
            np.save(target.output_dir / "frame_indices.npy", frame_indices)
        summary.append(
            {
                "subset": target.subset,
                "dataset_path": str(target.dataset_path),
                "episode_index": target.episode_index,
                "output_dir": str(target.output_dir),
                "all_dims": metrics["all_dims"],
            }
        )
        logger.info("Wrote TAC policy evaluation outputs to %s", target.output_dir)

    (resolved.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote TAC policy evaluation summary to %s", resolved.output_dir / "summary.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
