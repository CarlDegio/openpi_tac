# TAC Pi0.5 Inference Adapter

This folder contains a local adaptation of the ManiSkill-vitac websocket inference loop for our OpenPI TAC checkpoints.

Example:

```bash
.venv/bin/python infer_1/infer.py \
  --model pi05_tac_clean \
  --checkpoint-root checkpoints \
  --ip <robot_ip_or_url> \
  --port <port> \
  --token <token> \
  --language-prompt "your task prompt"
```

Supported model presets:

- `pi05_tac_clean` resolves to config `pi05_tac_clean`, default experiment `tac_clean_pi05`.
- `pi05_tac_smash` resolves to config `pi05_tac_smash`, default experiment `tac_smash_pi05`.

Checkpoint lookup:

```text
<checkpoint-root>/<model>/<experiment>/<step>
```

If `--checkpoint-step` is omitted, the latest numeric step folder is used. `--ckpt-dir` can directly override the resolved checkpoint directory.

The OpenPI config is resolved by name from `src/openpi/training/config.py`; it is not expected to be a separate file inside the checkpoint directory.

Current observation mapping ignores tactile cameras and expects:

```text
observation.images.camera0
observation.images.camera1
observation.state
```

Override these raw robot obs keys with `--camera0-key`, `--camera1-key`, and `--state-key` if the bridge sends different names.
