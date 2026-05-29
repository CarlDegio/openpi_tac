#!/usr/bin/env python3
# OpenPI TAC policy adapter for the ManiSkill-vitac websocket execution loop.
#
# Supported model names:
#   pi05_tac_clean -> config pi05_tac_clean, default experiment tac_clean_pi05
#   pi05_tac_smash -> config pi05_tac_smash, default experiment tac_smash_pi05
#
# Input obs dictionary passed to policy.infer, without tactile observations:
#   {
#     "observation.images.camera0": RGB left wrist image, uint8/float, shape (H, W, 3) or (..., H, W, 3)
#     "observation.images.camera1": RGB right wrist image, uint8/float, shape (H, W, 3) or (..., H, W, 3)
#     "observation.state": 20D robot state, float32, shape (20,) or (..., 20)
#     "prompt": language prompt string
#   }
#
# Output action dictionary sent to robot bridge:
#   {
#     "actions": float32 action chunk, shape (action_horizon, 20)
#   }
#
# TAC 20D action order:
#   left xyz: 0:3
#   left rotation first two matrix columns: 3:9
#   left gripper: 9
#   right xyz: 10:13
#   right rotation first two matrix columns: 13:19
#   right gripper: 19

from __future__ import annotations

import json
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import click
import cv2
import jax
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from client.interface_client import InterfaceClient  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    config_name: str
    default_exp_name: str


MODEL_REGISTRY = {
    "pi05_tac_clean": ModelSpec(config_name="pi05_tac_clean", default_exp_name="tac_clean_pi05"),
    "pi05_tac_smash": ModelSpec(config_name="pi05_tac_smash", default_exp_name="tac_smash_pi05"),
}


DEFAULT_OBS_KEYS = {
    "camera0": "observation.images.camera0",
    "camera1": "observation.images.camera1",
    "state": "observation.state",
}


class ObsSaver:
    """Asynchronously saves received obs_dict for interface verification."""

    def __init__(self, save_dir: str | Path, data_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / "eval_obs" / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_type = data_type
        self.save_queue = Queue(maxsize=100)
        self.save_thread: threading.Thread | None = None
        self.running = False
        self.step_count = 0

        with open(self.save_dir / "meta.json", "w") as f:
            json.dump({"data_type": data_type, "created_at": datetime.now().isoformat()}, f, indent=2)

        print(f"[ObsSaver] save directory: {self.save_dir}")

    def start(self) -> None:
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        print(f"[ObsSaver] stopped, total steps saved: {self.step_count}")

    def save_obs(self, obs: dict, step_idx: int | None = None, obs_seq: int | None = None) -> None:
        if not self.running:
            return
        self.step_count = step_idx if step_idx is not None else self.step_count + 1
        try:
            self.save_queue.put_nowait((self.step_count, obs_seq, obs))
        except Full:
            pass

    def _save_worker(self) -> None:
        while self.running or not self.save_queue.empty():
            try:
                step_idx, obs_seq, obs = self.save_queue.get(timeout=1.0)
            except Empty:
                continue
            self._save_single_obs(step_idx, obs, obs_seq)
            self.save_queue.task_done()

    def _to_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"shape": obj.shape, "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return obj

    def _save_single_obs(self, step_idx: int, obs: dict, obs_seq: int | None = None) -> None:
        step_dir = self.save_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(exist_ok=True)

        with open(step_dir / "obs_dict.pkl", "wb") as f:
            pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)

        summary = {
            "step_idx": step_idx,
            "obs_seq": obs_seq,
            "keys": sorted(obs.keys()) if isinstance(obs, dict) else None,
            "shapes": summarize_tree(obs),
        }
        with open(step_dir / "summary.json", "w") as f:
            json.dump(self._to_jsonable(summary), f, indent=2)

        if not isinstance(obs, dict):
            return

        for key, value in obs.items():
            if not isinstance(value, np.ndarray) or value.ndim < 3:
                continue
            if "camera" not in key and "rgb" not in key and "tactile" not in key:
                continue
            img = latest_image(value)
            if img is None:
                continue
            cv2.imwrite(str(step_dir / f"{key.replace('/', '_')}.jpg"), rgb_to_bgr_uint8(img))


def latest_image(value: Any) -> np.ndarray | None:
    image = np.asarray(value)
    while image.ndim > 3:
        image = image[-1]
    if image.ndim != 3:
        return None
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        return None
    return image


def latest_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    while vector.ndim > 1:
        vector = vector[-1]
    return vector


def rgb_to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        max_value = np.nanmax(image) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def summarize_tree(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): summarize_tree(v) for k, v in obj.items()}
    return type(obj).__name__


def get_required(raw_obs: dict, key: str, semantic_name: str) -> Any:
    if key not in raw_obs:
        available = ", ".join(sorted(str(k) for k in raw_obs.keys()))
        raise KeyError(f"Missing {semantic_name} key {key!r}. Available keys: {available}")
    return raw_obs[key]


def adapt_obs(
    raw_obs: dict,
    *,
    language_prompt: str,
    camera0_key: str,
    camera1_key: str,
    state_key: str,
) -> dict:
    if not isinstance(raw_obs, dict):
        raise TypeError(f"Expected robot obs to be a dict, got {type(raw_obs).__name__}")

    camera0 = latest_image(get_required(raw_obs, camera0_key, "camera0"))
    camera1 = latest_image(get_required(raw_obs, camera1_key, "camera1"))
    if camera0 is None:
        raise ValueError(f"Could not parse camera0 image from key {camera0_key!r}")
    if camera1 is None:
        raise ValueError(f"Could not parse camera1 image from key {camera1_key!r}")

    state = latest_vector(get_required(raw_obs, state_key, "state"))
    if state.shape[-1] != 20:
        raise ValueError(f"Expected 20D state at key {state_key!r}, got shape {state.shape}")

    prompt = raw_obs.get("prompt", language_prompt)
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    return {
        "observation.images.camera0": camera0,
        "observation.images.camera1": camera1,
        "observation.state": state,
        "prompt": str(prompt),
    }


def resolve_checkpoint(
    *,
    model_name: str,
    checkpoint_root: Path,
    exp_name: str | None,
    checkpoint_step: str | None,
    ckpt_dir: str | None,
) -> tuple[str, Path]:
    if model_name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model {model_name!r}. Choose one of: {choices}")

    spec = MODEL_REGISTRY[model_name]
    if ckpt_dir:
        checkpoint_dir = Path(ckpt_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (OPENPI_ROOT / checkpoint_dir).resolve()
    else:
        model_dir = checkpoint_root.expanduser()
        if not model_dir.is_absolute():
            model_dir = (OPENPI_ROOT / model_dir).resolve()
        model_dir = model_dir / model_name
        experiment_dir = model_dir / (exp_name or spec.default_exp_name)
        if checkpoint_step:
            checkpoint_dir = experiment_dir / str(checkpoint_step)
        else:
            checkpoint_dir = latest_checkpoint_step(experiment_dir)

    validate_checkpoint_dir(checkpoint_dir)
    return spec.config_name, checkpoint_dir


def latest_checkpoint_step(experiment_dir: Path) -> Path:
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    numeric_steps = [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not numeric_steps:
        raise FileNotFoundError(f"No numeric checkpoint step directories found in: {experiment_dir}")
    return max(numeric_steps, key=lambda path: int(path.name))


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not (checkpoint_dir / "params").exists() and not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Checkpoint directory must contain params/ or model.safetensors: {checkpoint_dir}"
        )
    if not (checkpoint_dir / "assets").exists():
        raise FileNotFoundError(f"Checkpoint directory must contain assets/ norm stats: {checkpoint_dir}")


def validate_actions(actions: Any, *, arm_num: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    expected_dim = 10 * arm_num
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk shape (horizon, {expected_dim}), got {actions.shape}")
    if actions.shape[-1] != expected_dim:
        raise ValueError(f"Expected action dim {expected_dim}, got {actions.shape[-1]} with shape {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy produced non-finite actions")
    return actions


@click.command()
@click.option(
    "--model",
    "model_name",
    type=click.Choice(sorted(MODEL_REGISTRY)),
    default="pi05_tac_clean",
    show_default=True,
    help="Our OpenPI TAC model preset to load.",
)
@click.option(
    "--checkpoint-root",
    default="checkpoints",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Root containing <model>/<experiment>/<step> checkpoint directories.",
)
@click.option("--exp-name", default=None, help="Experiment subdirectory under <checkpoint-root>/<model>.")
@click.option("--checkpoint-step", default=None, help="Checkpoint step directory name. Defaults to latest numeric step.")
@click.option("--ckpt-dir", default=None, help="Direct checkpoint directory override.")
@click.option("--data-type", "-dt", default="vision", show_default=True, help="Robot bridge data type.")
@click.option("--language-prompt", "-lp", default="", help="Language prompt injected into policy obs.")
@click.option("--save-obs/--no-save-obs", default=True, show_default=True, help="Save received observations.")
@click.option("--save-dir", default=SCRIPT_DIR, type=click.Path(path_type=Path), help="Directory for eval_obs.")
@click.option("--control-frequency", "-f", default=10.0, show_default=True, type=float, help="Control frequency in Hz.")
@click.option(
    "--controller-frequency",
    "-cf",
    default=80.0,
    show_default=True,
    type=float,
    help="Controller frequency in Hz.",
)
@click.option("--steps-per-inference", default=None, type=int, help="Defaults to model action_horizon.")
@click.option("--single-arm-mode", is_flag=True, help="Expect and send single-arm 10D actions.")
@click.option("--no-state-obs-mode", is_flag=True, help="Forwarded to robot bridge config.")
@click.option("--ip", default="127.0.0.1", show_default=True, help="Robot websocket host or URL.")
@click.option("--port", default="8000", show_default=True, help="Robot websocket port.")
@click.option("--add-port/--no-add-port", default=None, help="Whether to append --port to --ip.")
@click.option("--token", default=None, help="Robot bridge bearer token.")
@click.option("--max-obs-drain", default=1000, show_default=True, type=int, help="Max queued obs frames to drain.")
@click.option("--camera0-key", default=DEFAULT_OBS_KEYS["camera0"], show_default=True, help="Raw obs key for camera0.")
@click.option("--camera1-key", default=DEFAULT_OBS_KEYS["camera1"], show_default=True, help="Raw obs key for camera1.")
@click.option("--state-key", default=DEFAULT_OBS_KEYS["state"], show_default=True, help="Raw obs key for 20D state.")
@click.option("--dry-run", is_flag=True, help="Resolve config/checkpoint and exit before connecting.")
def main(
    model_name: str,
    checkpoint_root: Path,
    exp_name: str | None,
    checkpoint_step: str | None,
    ckpt_dir: str | None,
    data_type: str,
    language_prompt: str,
    save_obs: bool,
    save_dir: Path,
    control_frequency: float,
    controller_frequency: float,
    steps_per_inference: int | None,
    single_arm_mode: bool,
    no_state_obs_mode: bool,
    ip: str,
    port: str,
    add_port: bool | None,
    token: str | None,
    max_obs_drain: int,
    camera0_key: str,
    camera1_key: str,
    state_key: str,
    dry_run: bool,
) -> None:
    config_name, checkpoint_dir = resolve_checkpoint(
        model_name=model_name,
        checkpoint_root=checkpoint_root,
        exp_name=exp_name,
        checkpoint_step=checkpoint_step,
        ckpt_dir=ckpt_dir,
    )
    train_config = _config.get_config(config_name)
    action_horizon = int(train_config.model.action_horizon)
    steps_per_inference = action_horizon if steps_per_inference is None else int(steps_per_inference)

    print(f"[model] name={model_name}")
    print(f"[model] config={config_name}")
    print(f"[model] checkpoint={checkpoint_dir}")
    print(f"[model] action_horizon={action_horizon}")
    print(f"[bridge] steps_per_inference={steps_per_inference}")
    print(f"[obs] camera0_key={camera0_key}")
    print(f"[obs] camera1_key={camera1_key}")
    print(f"[obs] state_key={state_key}")

    if dry_run:
        return

    policy = policy_config.create_trained_policy(train_config, checkpoint_dir, default_prompt=language_prompt)
    client = InterfaceClient(ip, port, token=token, add_port=add_port)

    client.send_config(
        {
            "data_type": data_type,
            "language_prompt": language_prompt,
            "control_frequency": control_frequency,
            "controller_frequency": controller_frequency,
            "single_arm_mode": single_arm_mode,
            "no_state_obs_mode": no_state_obs_mode,
            "steps_per_inference": steps_per_inference,
            "action_horizon": action_horizon,
        }
    )

    arm_num = 1 if single_arm_mode else 2
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())

    obs_saver = None
    if save_obs:
        obs_saver = ObsSaver(save_dir=save_dir, data_type=data_type)
        obs_saver.start()

    with SharedMemoryManager():
        cv2.setNumThreads(2)

        print("Warming up policy inference")
        policy.reset()
        obs_seq, raw_obs, dropped_obs_count = client.recv_latest_obs(max_drain=max_obs_drain)
        print(f"[warmup] obs_seq={obs_seq} dropped_obs={dropped_obs_count}")
        print("[warmup] raw obs summary:", json.dumps(summarize_tree(raw_obs), indent=2))

        policy_obs = adapt_obs(
            raw_obs,
            language_prompt=language_prompt,
            camera0_key=camera0_key,
            camera1_key=camera1_key,
            state_key=state_key,
        )
        print("[warmup] policy obs summary:", json.dumps(summarize_tree(policy_obs), indent=2))

        result = policy.infer(policy_obs)
        actions = validate_actions(result["actions"], arm_num=arm_num)
        print(f"[warmup] action shape={actions.shape}, min={actions.min():.6f}, max={actions.max():.6f}")

        print("################################## Ready! ##################################")
        input("press enter to start...")
        client.send_state("start")

        try:
            policy.reset()
            last_status_log_time = time.monotonic()
            iter_idx = 0

            while True:
                iter_idx += 1
                obs_seq, raw_obs, dropped_obs_count = client.recv_latest_obs(max_drain=max_obs_drain)
                if obs_saver is not None:
                    obs_saver.save_obs(raw_obs, step_idx=iter_idx, obs_seq=obs_seq)

                policy_obs = adapt_obs(
                    raw_obs,
                    language_prompt=language_prompt,
                    camera0_key=camera0_key,
                    camera1_key=camera1_key,
                    state_key=state_key,
                )

                infer_start = time.monotonic()
                result = policy.infer(policy_obs)
                infer_elapsed = time.monotonic() - infer_start
                actions = validate_actions(result["actions"], arm_num=arm_num)
                client.send_action(actions, obs_seq=obs_seq)

                now = time.monotonic()
                if now - last_status_log_time >= 2.0:
                    print(
                        f"[main] iter={iter_idx} obs_seq={obs_seq} "
                        f"infer_time_ms={infer_elapsed * 1000.0:.1f} "
                        f"dropped_obs={dropped_obs_count}"
                    )
                    last_status_log_time = now

        except KeyboardInterrupt:
            print("Interrupted!")
            client.send_state("stop")
        finally:
            if obs_saver is not None:
                obs_saver.stop()
            client.close()


if __name__ == "__main__":
    main()
