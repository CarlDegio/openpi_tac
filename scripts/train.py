import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    wandb_id_path = ckpt_dir / "wandb_id.txt"
    if resuming and wandb_id_path.exists():
        run_id = wandb_id_path.read_text().strip()
        wandb.init(id=run_id, resume="allow", project=config.project_name)
    else:
        if resuming:
            logging.warning("Resuming checkpoint without wandb_id.txt; starting a new W&B run.")
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        wandb_id_path.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_tensorboard(config: _config.TrainConfig):
    enabled = os.environ.get("OPENPI_ENABLE_TENSORBOARD", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    log_dir = os.environ.get("OPENPI_TENSORBOARD_LOGDIR") or str(config.checkpoint_dir / "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on optional runtime package.
        logging.warning("TensorBoard logging disabled because SummaryWriter is unavailable: %s", exc)
        return None

    writer = SummaryWriter(log_dir=log_dir)
    logging.info("TensorBoard logging to %s", log_dir)
    return writer


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_checkpoint_steps(name: str, num_train_steps: int) -> set[int]:
    value = os.environ.get(name)
    if not value:
        return set()
    steps = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("%"):
            step = round(num_train_steps * float(part[:-1]) / 100.0)
        else:
            number = float(part)
            step = round(num_train_steps * number) if 0 < number <= 1 else round(number)
        steps.add(min(max(int(step), 1), num_train_steps))
    return steps


def _evenly_spaced_steps(start_step: int, end_step: int, count: int) -> set[int]:
    if count <= 0 or end_step <= start_step:
        return set()
    total_steps = end_step - start_step
    if count >= total_steps:
        return set(range(start_step, end_step))
    return set(int(step) for step in np.linspace(start_step, end_step - 1, num=count))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def action_mse_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    num_steps: int,
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    pred_actions = model.sample_actions(rng, observation, num_steps=num_steps)
    action_mse = jnp.mean(jnp.square(pred_actions - actions))
    return {"eval/action_mse": action_mse, "eval/action_rmse": jnp.sqrt(action_mse)}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    tensorboard_writer = init_tensorboard(config)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    camera_view_images = [
        np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1)
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    images_to_log = [wandb.Image(image) for image in camera_view_images]
    wandb.log({"camera_views": images_to_log}, step=0)
    if tensorboard_writer is not None:
        for i, image in enumerate(camera_view_images):
            tensorboard_writer.add_image(f"camera_views/sample_{i}", image, 0, dataformats="HWC")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    action_mse_evals = _env_int("OPENPI_ACTION_MSE_EVALS", 10)
    action_mse_num_steps = _env_int("OPENPI_ACTION_MSE_NUM_STEPS", 10)
    wait_after_checkpoint = _env_bool("OPENPI_WAIT_AFTER_CHECKPOINT", True)
    save_train_state = _env_bool("OPENPI_SAVE_TRAIN_STATE", True)
    checkpoint_steps = _env_checkpoint_steps("OPENPI_CHECKPOINT_STEPS", config.num_train_steps)
    if checkpoint_steps:
        logging.info("Checkpoint steps enabled: %s", sorted(checkpoint_steps))
        logging.info("Checkpoint save_train_state: %s", save_train_state)
    action_mse_steps = _evenly_spaced_steps(start_step, config.num_train_steps, action_mse_evals)
    if action_mse_steps:
        logging.info(
            "Action MSE eval enabled: %d evals, sample_actions num_steps=%d",
            len(action_mse_steps),
            action_mse_num_steps,
        )
    paction_mse_step = jax.jit(
        functools.partial(action_mse_step, config, num_steps=action_mse_num_steps),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if tensorboard_writer is not None:
                for key, value in reduced_info.items():
                    tensorboard_writer.add_scalar(key, float(value), step)
                tensorboard_writer.flush()
            infos = []
        if step in action_mse_steps:
            action_mse_rng = jax.random.fold_in(train_rng, step + 1_000_000)
            with sharding.set_mesh(mesh):
                action_mse_info = paction_mse_step(action_mse_rng, train_state, batch)
            action_mse_info = jax.device_get(action_mse_info)
            action_mse_str = ", ".join(f"{k}={v:.6f}" for k, v in action_mse_info.items())
            pbar.write(f"Step {step}: {action_mse_str}")
            wandb.log(action_mse_info, step=step)
            if tensorboard_writer is not None:
                for key, value in action_mse_info.items():
                    tensorboard_writer.add_scalar(key, float(value), step)
                tensorboard_writer.flush()
        batch = next(data_iter)

        completed_step = step + 1
        should_save_by_interval = (
            not checkpoint_steps
            and config.save_interval > 0
            and completed_step % config.save_interval == 0
            and completed_step > start_step
        )
        should_save_by_step = completed_step in checkpoint_steps and completed_step > start_step
        if should_save_by_interval or should_save_by_step or completed_step == config.num_train_steps:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                completed_step,
                save_train_state=save_train_state,
            )
            if wait_after_checkpoint:
                logging.info("Waiting for checkpoint save to finish before continuing training.")
                checkpoint_manager.wait_until_finished()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main(_config.cli())
