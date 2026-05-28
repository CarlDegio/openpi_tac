"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json
import numpy as np
import pathlib
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.policies.tac_policy as tac_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


RED = "\033[31m"
RESET = "\033[0m"


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def discover_lerobot_repos(repo_root: str, tac_subset: _config.TacDatasetSubset = "all") -> tuple[str, ...]:
    root = pathlib.Path(repo_root).expanduser()
    if tac_subset == "all":
        return tuple(str(path) for path in sorted(root.iterdir()) if (path / "meta" / "info.json").is_file())
    return tuple(
        str(path)
        for path in sorted(root.iterdir())
        if (path / "meta" / "info.json").is_file() and f"_{tac_subset}_" in path.name
    )


def red_warning(message: str) -> None:
    print(f"{RED}[WARN] {message}{RESET}")


def validate_lerobot_repo(repo_id: str) -> bool:
    root = pathlib.Path(repo_id).expanduser()
    if not root.is_dir():
        # Remote HF repos are validated by the dataset loader.
        return True

    meta_dir = root / "meta"
    required_meta = [meta_dir / "info.json", meta_dir / "episodes.jsonl", meta_dir / "tasks.jsonl"]
    missing = [path for path in required_meta if not path.is_file()]
    if missing:
        red_warning(f"{root}: missing metadata files: {', '.join(str(path) for path in missing)}")
        return False

    ok = True
    try:
        info = json.loads((meta_dir / "info.json").read_text())
        episodes = [json.loads(line) for line in (meta_dir / "episodes.jsonl").read_text().splitlines() if line]
    except Exception as exc:
        red_warning(f"{root}: failed to read metadata: {exc}")
        return False

    if info.get("total_episodes") != len(episodes):
        ok = False
        red_warning(f"{root}: total_episodes={info.get('total_episodes')} but episodes.jsonl has {len(episodes)}")

    data_path = info.get("data_path")
    chunks_size = info.get("chunks_size")
    if data_path is None or chunks_size is None:
        red_warning(f"{root}: info.json missing data_path or chunks_size")
        return False

    total_rows = 0
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        expected_rows = int(episode["length"])
        parquet_path = root / data_path.format(
            episode_chunk=ep_idx // int(chunks_size),
            episode_index=ep_idx,
        )
        if not parquet_path.is_file():
            ok = False
            red_warning(f"{root}: missing parquet for episode {ep_idx}: {parquet_path}")
            continue

        try:
            parquet_file = pq.ParquetFile(parquet_path)
            rows = parquet_file.metadata.num_rows
            schema_names = set(parquet_file.schema_arrow.names)
        except Exception as exc:
            ok = False
            red_warning(f"{root}: unreadable parquet for episode {ep_idx}: {parquet_path} ({exc})")
            continue

        required_columns = {
            "observation.images.camera0",
            "observation.images.camera1",
            "observation.state",
            "actions",
            "timestamp",
            "episode_index",
            "task_index",
        }
        missing_columns = sorted(required_columns - schema_names)
        if missing_columns:
            ok = False
            red_warning(f"{root}: episode {ep_idx} missing columns: {missing_columns}")

        if rows != expected_rows:
            ok = False
            red_warning(f"{root}: episode {ep_idx} has {rows} parquet rows but metadata length is {expected_rows}")
        total_rows += rows

    if info.get("total_frames") != total_rows:
        ok = False
        red_warning(f"{root}: total_frames={info.get('total_frames')} but parquet rows sum to {total_rows}")

    if ok:
        print(f"[OK] {root}: {len(episodes)} episodes, {total_rows} frames")
    return ok


def validate_lerobot_repos(repo_ids: tuple[str, ...]) -> None:
    print(f"Validating parquet readability and metadata counts for {len(repo_ids)} dataset(s)...")
    failed = [repo_id for repo_id in repo_ids if not validate_lerobot_repo(repo_id)]
    if failed:
        red_warning(f"{len(failed)} dataset(s) failed validation; norm stats will still be computed unless interrupted.")


def _is_local_tac_data_config(data_config: _config.DataConfig) -> bool:
    repo_ids = tuple(data_config.repo_ids or ((data_config.repo_id,) if data_config.repo_id else ()))
    if not repo_ids or not all(pathlib.Path(repo_id).expanduser().is_dir() for repo_id in repo_ids):
        return False
    return any(transform.__class__.__name__ == "TacInputs" for transform in data_config.data_transforms.inputs)


def _iter_local_repo_episode_paths(repo_id: str):
    root = pathlib.Path(repo_id).expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        yield root / info["data_path"].format(
            episode_chunk=ep_idx // int(info["chunks_size"]),
            episode_index=ep_idx,
        )


def compute_local_tac_norm_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    max_frames: int | None = None,
) -> dict[str, normalize.NormStats]:
    """Fast norm stats path for local TAC datasets that avoids image decoding."""

    repo_ids = tuple(data_config.repo_ids or ((data_config.repo_id,) if data_config.repo_id else ()))
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}

    remaining_frames = max_frames
    total_frames = 0
    for repo_id in repo_ids:
        for parquet_path in _iter_local_repo_episode_paths(repo_id):
            table = pq.read_table(parquet_path, columns=["observation.state", "actions"])
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            raw_actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            if remaining_frames is not None:
                if remaining_frames <= 0:
                    break
                states = states[:remaining_frames]
                raw_actions = raw_actions[:remaining_frames]
                remaining_frames -= len(states)

            indices = np.minimum(
                np.arange(len(raw_actions))[:, None] + np.arange(action_horizon)[None, :],
                len(raw_actions) - 1,
            )
            actions = tac_policy.tac_actions_to_pi05(raw_actions[indices])
            stats["state"].update(states)
            stats["actions"].update(actions)
            total_frames += len(states)
        if remaining_frames is not None and remaining_frames <= 0:
            break

    if total_frames == 0:
        raise ValueError("No frames found while computing local TAC norm stats.")
    print(f"Computed local TAC stats from {len(repo_ids)} dataset(s), {total_frames} frame(s).")
    return {key: value.get_statistics() for key, value in stats.items()}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_root: str | None = None,
    tac_subset: _config.TacDatasetSubset = "all",
    validate_parquet: bool = True,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if repo_root is not None:
        repo_ids = discover_lerobot_repos(repo_root, tac_subset=tac_subset)
        if not repo_ids:
            raise ValueError(f"No LeRobot datasets found under repo_root={repo_root!r}, tac_subset={tac_subset!r}")
        data_config = dataclasses.replace(data_config, repo_id=repo_ids[0], repo_ids=repo_ids)
        if tac_subset != "all" and data_config.asset_id == "tac_all_pi05":
            data_config = dataclasses.replace(data_config, asset_id=f"tac_{tac_subset}_pi05")
        print(f"Discovered {len(repo_ids)} datasets under {repo_root} with tac_subset={tac_subset}:")
        for repo_id in repo_ids:
            print(f"  - {repo_id}")

    if validate_parquet and data_config.rlds_data_dir is None and data_config.repo_id is not None:
        validate_lerobot_repos(tuple(data_config.repo_ids or (data_config.repo_id,)))

    if _is_local_tac_data_config(data_config):
        norm_stats = compute_local_tac_norm_stats(data_config, config.model.action_horizon, max_frames)
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    configured_assets_dir = getattr(config.data, "assets", _config.AssetsConfig()).assets_dir
    if configured_assets_dir and "://" not in configured_assets_dir and data_config.asset_id:
        output_path = pathlib.Path(configured_assets_dir) / data_config.asset_id
    else:
        output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
