import dataclasses
import io

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


def make_tac_example() -> dict:
    """Creates a random input example for the TAC policy."""
    return {
        "observation.images.camera0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.tactile_left_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.tactile_left_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.tactile_right_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.tactile_right_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(20).astype(np.float32),
        "prompt": "Pick up the blue test tube and use the test tube brush to clean it.",
    }


def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict) and image.get("bytes") is not None:
        image = Image.open(io.BytesIO(image["bytes"]))

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def tac_actions_to_pi05(actions: np.ndarray) -> np.ndarray:
    """Maps Mani ViTac 20D end-effector delta actions into PI05's 32D action space.

    Source 20D layout:
      left:  xyz(0:3), rotation first two matrix columns(3:9), gripper(9)
      right: xyz(10:13), rotation first two matrix columns(13:19), gripper(19)

    Target 32D layout keeps the original PI-style first 14 dimensions for
    joint-like values and grippers, then places the Cartesian deltas in extra
    dimensions.
    """

    actions = np.asarray(actions)
    if actions.shape[-1] != 20:
        raise ValueError(f"Expected TAC actions to have 20 dims, got shape {actions.shape}")

    pi_actions = np.zeros((*actions.shape[:-1], 32), dtype=actions.dtype)
    pi_actions[..., 0:6] = actions[..., 3:9]
    pi_actions[..., 6] = actions[..., 9]
    pi_actions[..., 7:13] = actions[..., 13:19]
    pi_actions[..., 13] = actions[..., 19]
    pi_actions[..., 14:17] = actions[..., 0:3]
    pi_actions[..., 17:20] = actions[..., 10:13]
    return pi_actions


def pi05_actions_to_tac(actions: np.ndarray) -> np.ndarray:
    """Inverse of `tac_actions_to_pi05` for policy outputs."""

    actions = np.asarray(actions)
    if actions.shape[-1] < 20:
        raise ValueError(f"Expected PI05 actions to have at least 20 dims, got shape {actions.shape}")

    tac_actions = np.zeros((*actions.shape[:-1], 20), dtype=actions.dtype)
    tac_actions[..., 0:3] = actions[..., 14:17]
    tac_actions[..., 3:9] = actions[..., 0:6]
    tac_actions[..., 9] = actions[..., 6]
    tac_actions[..., 10:13] = actions[..., 17:20]
    tac_actions[..., 13:19] = actions[..., 7:13]
    tac_actions[..., 19] = actions[..., 13]
    return tac_actions


@dataclasses.dataclass(frozen=True)
class TacInputs(transforms.DataTransformFn):
    """Inputs for Mani ViTac / TAC LeRobot datasets.

    Expected inputs:
    - observation.images.camera0: left wrist camera
    - observation.images.camera1: right wrist camera
    - observation.images.tactile_*: optional tactile cameras for tactile-conditioned models
    - observation.state: 20D proprioceptive state
    - actions: optional 20D end-effector delta action trajectory
    """

    model_type: _model.ModelType
    include_tactile: bool = False

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type for TAC inputs: {self.model_type}")

        left_wrist = _parse_image(data["observation.images.camera0"])
        right_wrist = _parse_image(data["observation.images.camera1"])
        base_image = np.zeros_like(left_wrist)

        inputs = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if self.include_tactile:
            tactile_map = {
                "tactile_left_0_rgb": "observation.images.tactile_left_0",
                "tactile_left_1_rgb": "observation.images.tactile_left_1",
                "tactile_right_0_rgb": "observation.images.tactile_right_0",
                "tactile_right_1_rgb": "observation.images.tactile_right_1",
            }
            for target_key, source_key in tactile_map.items():
                inputs["image"][target_key] = _parse_image(data[source_key])
                inputs["image_mask"][target_key] = np.True_

        if "actions" in data:
            inputs["actions"] = tac_actions_to_pi05(np.asarray(data["actions"], dtype=np.float32))

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class TacOutputs(transforms.DataTransformFn):
    """Maps PI05 32D action chunks back to TAC's 20D action layout."""

    def __call__(self, data: dict) -> dict:
        return {"actions": pi05_actions_to_tac(data["actions"])}
