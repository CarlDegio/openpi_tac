#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="pi05_tac_all"
EXP_NAME="tac_all_pi05"
TAC_SUBSET="all"
AUTO_NORM_STATS="true"
ASYNC_CHECKPOINT="false"
WAIT_AFTER_CHECKPOINT="true"
CHECKPOINT_STEPS="50%,90%,100%"
SAVE_TRAIN_STATE="false"
MAX_CHECKPOINTS="3"
EMA_VALUE="true"
BATCH_SIZE="48"
NUM_WORKERS="2"
NUM_TRAIN_STEPS=""
LOG_INTERVAL=""
SAVE_INTERVAL="0"
GPUS="auto"
MAX_GPUS="0"
MIN_FREE_MEM_MB="30000"
MAX_GPU_UTIL="10"
FSDP_DEVICES="auto"
XLA_PREALLOCATE="false"
XLA_MEM_FRACTION="0.9"
WANDB_ENABLED="false"
WANDB_MODE_VALUE="local"
WANDB_HOST="127.0.0.1"
WANDB_PORT="8008"
WANDB_DIR_VALUE=""
OVERWRITE="false"
RESUME="false"
AUTO_RESUME="true"
OPENPI_DATA_HOME_VALUE="$ROOT_DIR/.openpi_cache"
HF_HOME_VALUE="/tmp/hf_home"
HF_DATASETS_CACHE_VALUE="/tmp/hf_datasets"
TENSORBOARD_ENABLED="true"
TENSORBOARD_HOST="127.0.0.1"
TENSORBOARD_PORT="8008"
TENSORBOARD_LOGDIR=""
ACTION_MSE_EVALS="10"
ACTION_MSE_NUM_STEPS="10"
TAC_MODE="false"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pi05_tac_train.sh [options] [-- extra train.py args]

Options:
  --config NAME              Training config name. Default: pi05_tac_all
  --exp-name NAME            Experiment name. Default: tac_all_pi05
  --tac-subset all|clean|smash
                             Use pi05_tac_all, pi05_tac_clean, or pi05_tac_smash. Default: all
  --auto-norm-stats          Compute missing TAC norm stats before training. Default: enabled
  --no-auto-norm-stats       Fail fast if TAC norm stats are missing
  --async-checkpoint         Let Orbax finalize checkpoints asynchronously. Default: disabled
  --no-async-checkpoint      Finish checkpoint saves before continuing training
  --ema BOOL|DECAY           Enable EMA, disable with false, or set decay value. Default: true
  --no-ema                   Disable EMA by passing --ema-decay None to train.py
  --wait-after-checkpoint    Wait for checkpoint save/finalize before continuing. Default: enabled
  --no-wait-after-checkpoint Do not add an explicit wait after checkpoint save
  --checkpoint-steps LIST    Comma-separated save steps or percentages. Default: 50%,90%,100%
  --save-train-state         Save full training state for resume. Default: disabled
  --no-save-train-state      Save only params/assets checkpoints
  --max-checkpoints N        Max checkpoints to keep. Default: 3
  --batch-size N             Global batch size. Must be divisible by GPU count. Default: 48
  --num-workers N            Data loader workers. Default: 2
  --num-train-steps N        Override training steps.
  --log-interval N           Override log interval.
  --save-interval N          Override save interval. 0 means save only at the final step. Default: 0
  --gpus LIST|auto           CUDA device ids, e.g. 1,2. Default: auto
  --max-gpus N               Max GPUs to use in auto mode. 0 means all idle GPUs. Default: 0
  --min-free-mem-mb N        Auto mode free-memory threshold. Default: 30000
  --max-gpu-util N           Auto mode utilization threshold. Default: 10
  --fsdp-devices N|auto      Override FSDP device count. Default: auto, equal to selected GPU count
  --xla-preallocate BOOL     XLA_PYTHON_CLIENT_PREALLOCATE. Default: true
  --xla-mem-fraction VALUE   XLA_PYTHON_CLIENT_MEM_FRACTION. Default: 0.9
  --wandb                    Enable wandb. Default: disabled
  --no-wandb                 Disable wandb.
  --wandb-mode MODE          local, offline, or online. Default: local
  --wandb-host HOST          Local wandb server host. Default: 127.0.0.1
  --wandb-port PORT          Local wandb server port. Default: 8008
  --wandb-dir PATH           Local wandb run directory. Default: checkpoints/<config>/<exp>/wandb
  --overwrite                Pass --overwrite to train.py
  --resume                   Pass --resume to train.py
  --no-auto-resume           Do not auto-resume when checkpoint dir exists
  --openpi-data-home PATH    OPENPI_DATA_HOME. Default: <repo>/.openpi_cache
  --hf-home PATH             HF_HOME. Default: /tmp/hf_home
  --hf-datasets-cache PATH   HF_DATASETS_CACHE. Default: /tmp/hf_datasets
  --tensorboard              Start TensorBoard. Default: enabled
  --no-tensorboard           Do not start TensorBoard or write TensorBoard logs
  --tb-host HOST             TensorBoard bind host. Default: 127.0.0.1
  --tb-port PORT             TensorBoard port. Default: 8008
  --tb-logdir PATH           TensorBoard logdir. Default: checkpoints/<config>/<exp>/tensorboard
  --action-mse-evals N       Number of evenly spaced action MSE evals. Default: 10
  --action-mse-num-steps N   Denoising steps for action MSE sampling. Default: 10
  --tac false|true|onlyload   false: baseline; onlyload: read tactile columns only; true: train tactile-conditioned model.
                             Default: false
  -h, --help                 Show this help.

Examples:
  scripts/run_pi05_tac_train.sh --gpus auto --max-gpus 2 --exp-name speed_test
  scripts/run_pi05_tac_train.sh --gpus 0,1,2,3 --batch-size 32 --tac-subset clean --num-train-steps 10000 --overwrite
  --resume or --overwrite --ema false (if your fsdp=1)
  * total samples = 4,730,344 / 2
  tmux attach -t pi
  --tac-subset clean or smash or all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_NAME="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --tac-subset) TAC_SUBSET="$2"; shift 2 ;;
    --auto-norm-stats) AUTO_NORM_STATS="true"; shift ;;
    --no-auto-norm-stats) AUTO_NORM_STATS="false"; shift ;;
    --async-checkpoint) ASYNC_CHECKPOINT="true"; shift ;;
    --no-async-checkpoint) ASYNC_CHECKPOINT="false"; shift ;;
    --ema) EMA_VALUE="$2"; shift 2 ;;
    --no-ema) EMA_VALUE="false"; shift ;;
    --wait-after-checkpoint) WAIT_AFTER_CHECKPOINT="true"; shift ;;
    --no-wait-after-checkpoint) WAIT_AFTER_CHECKPOINT="false"; shift ;;
    --checkpoint-steps) CHECKPOINT_STEPS="$2"; shift 2 ;;
    --save-train-state) SAVE_TRAIN_STATE="true"; shift ;;
    --no-save-train-state) SAVE_TRAIN_STATE="false"; shift ;;
    --max-checkpoints) MAX_CHECKPOINTS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --min-free-mem-mb) MIN_FREE_MEM_MB="$2"; shift 2 ;;
    --max-gpu-util) MAX_GPU_UTIL="$2"; shift 2 ;;
    --fsdp-devices) FSDP_DEVICES="$2"; shift 2 ;;
    --xla-preallocate) XLA_PREALLOCATE="$2"; shift 2 ;;
    --xla-mem-fraction) XLA_MEM_FRACTION="$2"; shift 2 ;;
    --wandb) WANDB_ENABLED="true"; shift ;;
    --no-wandb) WANDB_ENABLED="false"; shift ;;
    --wandb-mode) WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-host) WANDB_HOST="$2"; shift 2 ;;
    --wandb-port) WANDB_PORT="$2"; shift 2 ;;
    --wandb-dir) WANDB_DIR_VALUE="$2"; shift 2 ;;
    --overwrite) OVERWRITE="true"; shift ;;
    --resume) RESUME="true"; shift ;;
    --no-auto-resume) AUTO_RESUME="false"; shift ;;
    --openpi-data-home) OPENPI_DATA_HOME_VALUE="$2"; shift 2 ;;
    --hf-home) HF_HOME_VALUE="$2"; shift 2 ;;
    --hf-datasets-cache) HF_DATASETS_CACHE_VALUE="$2"; shift 2 ;;
    --tensorboard) TENSORBOARD_ENABLED="true"; shift ;;
    --no-tensorboard) TENSORBOARD_ENABLED="false"; shift ;;
    --tb-host) TENSORBOARD_HOST="$2"; shift 2 ;;
    --tb-port) TENSORBOARD_PORT="$2"; shift 2 ;;
    --tb-logdir) TENSORBOARD_LOGDIR="$2"; shift 2 ;;
    --action-mse-evals) ACTION_MSE_EVALS="$2"; shift 2 ;;
    --action-mse-num-steps) ACTION_MSE_NUM_STEPS="$2"; shift 2 ;;
    --tac) TAC_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$TAC_SUBSET" in
  all|clean|smash) ;;
  *) echo "--tac-subset must be one of: all, clean, smash" >&2; exit 2 ;;
esac
if [[ "$TAC_SUBSET" == "all" ]]; then
  case "$CONFIG_NAME" in
    pi05_tac_clean) TAC_SUBSET="clean" ;;
    pi05_tac_smash) TAC_SUBSET="smash" ;;
  esac
fi
if [[ "$CONFIG_NAME" == "pi05_tac_all" && "$TAC_SUBSET" != "all" ]]; then
  CONFIG_NAME="pi05_tac_$TAC_SUBSET"
fi
if [[ "$EXP_NAME" == "tac_all_pi05" && "$TAC_SUBSET" != "all" ]]; then
  EXP_NAME="tac_${TAC_SUBSET}_pi05"
fi

tac_norm_asset_id() {
  case "$1" in
    pi05_tac_all) echo "tac_all_pi05" ;;
    pi05_tac_clean) echo "tac_clean_pi05" ;;
    pi05_tac_smash) echo "tac_smash_pi05" ;;
    pi05_tac_blue_clean_01) echo "blue_clean_01_pi05" ;;
    *) return 1 ;;
  esac
}

select_auto_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; pass --gpus explicitly." >&2
    return 1
  fi

  local selected=()
  while IFS=',' read -r index used total util; do
    index="${index//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    local free=$((total - used))
    if (( free >= MIN_FREE_MEM_MB && util <= MAX_GPU_UTIL )); then
      selected+=("$index")
      if (( MAX_GPUS > 0 && ${#selected[@]} >= MAX_GPUS )); then
        break
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

  if (( ${#selected[@]} == 0 )); then
    echo "No idle GPUs found with free_mem >= ${MIN_FREE_MEM_MB}MB and util <= ${MAX_GPU_UTIL}%." >&2
    return 1
  fi

  local joined
  joined="$(IFS=,; echo "${selected[*]}")"
  echo "$joined"
}

if [[ "$GPUS" == "auto" ]]; then
  GPUS="$(select_auto_gpus)"
fi

GPU_COUNT="$(awk -F',' '{print NF}' <<<"$GPUS")"
if [[ "$FSDP_DEVICES" == "auto" ]]; then
  FSDP_DEVICES="$GPU_COUNT"
fi
if [[ -z "$TENSORBOARD_LOGDIR" ]]; then
  TENSORBOARD_LOGDIR="$ROOT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/tensorboard"
fi
if [[ -z "$WANDB_DIR_VALUE" ]]; then
  WANDB_DIR_VALUE="$ROOT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/wandb"
fi
CHECKPOINT_DIR="$ROOT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"

if [[ "$AUTO_RESUME" == "true" && "$OVERWRITE" != "true" && "$RESUME" != "true" && -d "$CHECKPOINT_DIR" ]]; then
  RESUME="true"
  echo "Checkpoint directory exists; enabling --resume: $CHECKPOINT_DIR"
fi

if (( BATCH_SIZE % GPU_COUNT != 0 )); then
  echo "Batch size $BATCH_SIZE must be divisible by selected GPU count $GPU_COUNT ($GPUS)." >&2
  exit 2
fi
if (( GPU_COUNT % FSDP_DEVICES != 0 )); then
  echo "Selected GPU count $GPU_COUNT must be divisible by FSDP devices $FSDP_DEVICES." >&2
  exit 2
fi

TRAIN_ARGS=(
  "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --batch-size "$BATCH_SIZE"
  --fsdp-devices "$FSDP_DEVICES"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$NUM_TRAIN_STEPS" ]]; then
  TRAIN_ARGS+=(--num-train-steps "$NUM_TRAIN_STEPS")
fi
if [[ -n "$LOG_INTERVAL" ]]; then
  TRAIN_ARGS+=(--log-interval "$LOG_INTERVAL")
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  TRAIN_ARGS+=(--save-interval "$SAVE_INTERVAL")
fi
case "${EMA_VALUE,,}" in
  true|yes|on|1) ;;
  false|no|off|0|none|null) TRAIN_ARGS+=(--ema-decay None) ;;
  *) TRAIN_ARGS+=(--ema-decay "$EMA_VALUE") ;;
esac
if [[ "$WANDB_ENABLED" == "true" ]]; then
  TRAIN_ARGS+=(--wandb-enabled)
else
  TRAIN_ARGS+=(--no-wandb-enabled)
fi
if [[ "$OVERWRITE" == "true" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
if [[ "$RESUME" == "true" ]]; then
  TRAIN_ARGS+=(--resume)
fi
case "${TAC_MODE,,}" in
  false|true|onlyload) TRAIN_ARGS+=(--tac "${TAC_MODE,,}") ;;
  *) echo "--tac must be one of: false, true, onlyload" >&2; exit 2 ;;
esac
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

echo "Working directory: $ROOT_DIR"
echo "CUDA_VISIBLE_DEVICES=$GPUS"
echo "Config: $CONFIG_NAME"
echo "Experiment: $EXP_NAME"
echo "TAC subset: $TAC_SUBSET"
echo "Auto norm stats: $AUTO_NORM_STATS"
echo "Async checkpoint: $ASYNC_CHECKPOINT"
echo "Wait after checkpoint: $WAIT_AFTER_CHECKPOINT"
echo "Checkpoint steps: $CHECKPOINT_STEPS"
echo "Save train state: $SAVE_TRAIN_STATE"
echo "Max checkpoints: $MAX_CHECKPOINTS"
echo "EMA: $EMA_VALUE"
echo "Batch size: $BATCH_SIZE"
echo "FSDP devices: $FSDP_DEVICES"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION"
echo "Num workers: $NUM_WORKERS"
echo "TAC mode: ${TAC_MODE,,}"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME_VALUE"
echo "HF_HOME=$HF_HOME_VALUE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE_VALUE"
if [[ "$WANDB_ENABLED" == "true" ]]; then
  echo "W&B mode: $WANDB_MODE_VALUE"
  echo "W&B dir: $WANDB_DIR_VALUE"
  if [[ "$WANDB_MODE_VALUE" == "local" ]]; then
    echo "W&B local server: http://$WANDB_HOST:$WANDB_PORT"
  fi
else
  echo "W&B: disabled"
fi
if [[ "$TENSORBOARD_ENABLED" == "true" ]]; then
  echo "TensorBoard: http://$TENSORBOARD_HOST:$TENSORBOARD_PORT"
  echo "TensorBoard logdir: $TENSORBOARD_LOGDIR"
else
  echo "TensorBoard: disabled"
fi
echo "Action MSE evals: $ACTION_MSE_EVALS"
echo "Action MSE sample_actions num_steps: $ACTION_MSE_NUM_STEPS"
echo
echo "Command:"
printf '  CUDA_VISIBLE_DEVICES=%q OPENPI_DATA_HOME=%q HF_HOME=%q HF_DATASETS_CACHE=%q XLA_PYTHON_CLIENT_PREALLOCATE=%q XLA_PYTHON_CLIENT_MEM_FRACTION=%q .venv/bin/python scripts/train.py' \
  "$GPUS" "$OPENPI_DATA_HOME_VALUE" "$HF_HOME_VALUE" "$HF_DATASETS_CACHE_VALUE" "$XLA_PREALLOCATE" "$XLA_MEM_FRACTION"
printf ' %q' "${TRAIN_ARGS[@]}"
echo
echo

TB_PID=""
WANDB_LOCAL_STARTED="false"

port_is_listening() {
  local port="$1"
  .venv/bin/python -c 'import socket, sys
try:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        result = sock.connect_ex(("127.0.0.1", int(sys.argv[1])))
    finally:
        sock.close()
except OSError:
    result = 1
sys.exit(0 if result == 0 else 1)' "$port"
}

start_wandb_server() {
  if [[ "$WANDB_ENABLED" != "true" || "$WANDB_MODE_VALUE" != "local" ]]; then
    return
  fi
  if port_is_listening "$WANDB_PORT"; then
    echo "W&B local server port $WANDB_PORT is already listening; reusing it."
    return
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is unavailable; falling back to WANDB_MODE=offline."
    WANDB_MODE_VALUE="offline"
    if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
      TENSORBOARD_ENABLED="true"
      TENSORBOARD_PORT="$WANDB_PORT"
      echo "Starting TensorBoard on port $TENSORBOARD_PORT for live local metrics instead."
    fi
    return
  fi
  if ! docker image inspect wandb/local >/dev/null 2>&1; then
    echo "Docker image wandb/local is unavailable; pulling it now."
    mkdir -p "$WANDB_DIR_VALUE"
    local wb_pull_log="$WANDB_DIR_VALUE/wandb_local_pull.log"
    if ! docker pull wandb/local >"$wb_pull_log" 2>&1; then
      echo "Failed to pull wandb/local; falling back to WANDB_MODE=offline."
      echo "Docker pull log: $wb_pull_log"
      WANDB_MODE_VALUE="offline"
      if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
        TENSORBOARD_ENABLED="true"
        TENSORBOARD_PORT="$WANDB_PORT"
        echo "Starting TensorBoard on port $TENSORBOARD_PORT for live local metrics instead."
      fi
      return
    fi
  fi

  local container_name="openpi-wandb-local"
  if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
    if ! docker start "$container_name" >/dev/null; then
      echo "Failed to start existing $container_name container; falling back to WANDB_MODE=offline."
      WANDB_MODE_VALUE="offline"
      if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
        TENSORBOARD_ENABLED="true"
        TENSORBOARD_PORT="$WANDB_PORT"
        echo "Starting TensorBoard on port $TENSORBOARD_PORT for live local metrics instead."
      fi
      return
    fi
  else
    if ! docker run -d \
      --name "$container_name" \
      -p "$WANDB_HOST:$WANDB_PORT:8080" \
      -v openpi-wandb-local:/vol \
      wandb/local >/dev/null; then
      echo "Failed to create $container_name container; falling back to WANDB_MODE=offline."
      WANDB_MODE_VALUE="offline"
      if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
        TENSORBOARD_ENABLED="true"
        TENSORBOARD_PORT="$WANDB_PORT"
        echo "Starting TensorBoard on port $TENSORBOARD_PORT for live local metrics instead."
      fi
      return
    fi
  fi

  sleep 2
  if ! port_is_listening "$WANDB_PORT"; then
    echo "W&B local server did not bind port $WANDB_PORT; falling back to WANDB_MODE=offline."
    WANDB_MODE_VALUE="offline"
    if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
      TENSORBOARD_ENABLED="true"
      TENSORBOARD_PORT="$WANDB_PORT"
      echo "Starting TensorBoard on port $TENSORBOARD_PORT for live local metrics instead."
    fi
    return
  fi
  WANDB_LOCAL_STARTED="true"
  echo "W&B local server started at http://$WANDB_HOST:$WANDB_PORT"
  echo
}

start_tensorboard() {
  if [[ "$TENSORBOARD_ENABLED" != "true" ]]; then
    return
  fi
  mkdir -p "$TENSORBOARD_LOGDIR"
  if ! .venv/bin/python -c "import tensorboard" >/dev/null 2>&1; then
    echo "TensorBoard package is not installed in .venv."
    echo "Install it with: uv pip install tensorboard"
    exit 1
  fi

  if port_is_listening "$TENSORBOARD_PORT"; then
    local container_name="openpi-wandb-local"
    if [[ "$WANDB_ENABLED" != "true" ]] && command -v docker >/dev/null 2>&1 \
      && docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
      echo "Stopping $container_name to free TensorBoard port $TENSORBOARD_PORT."
      docker stop "$container_name" >/dev/null
      sleep 1
    fi
  fi
  if port_is_listening "$TENSORBOARD_PORT"; then
    echo "TensorBoard port $TENSORBOARD_PORT is already in use. Stop the process using it or pass --tb-port." >&2
    exit 1
  fi

  local tb_log="$TENSORBOARD_LOGDIR/tensorboard_${TENSORBOARD_PORT}.log"
  .venv/bin/python -m tensorboard.main \
    --logdir "$TENSORBOARD_LOGDIR" \
    --host "$TENSORBOARD_HOST" \
    --port "$TENSORBOARD_PORT" \
    >"$tb_log" 2>&1 &
  TB_PID="$!"
  sleep 2
  if ! kill -0 "$TB_PID" >/dev/null 2>&1; then
    echo "TensorBoard failed to start. Last log lines:"
    tail -40 "$tb_log" || true
    TB_PID=""
    exit 1
  fi
  echo "TensorBoard started with PID $TB_PID at http://$TENSORBOARD_HOST:$TENSORBOARD_PORT"
  echo "TensorBoard process log: $tb_log"
  echo
}

cleanup() {
  if [[ -n "$TB_PID" ]]; then
    kill "$TB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="$GPUS"
export OPENPI_DATA_HOME="$OPENPI_DATA_HOME_VALUE"
export HF_HOME="$HF_HOME_VALUE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_VALUE"
export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PREALLOCATE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
export OPENPI_ACTION_MSE_EVALS="$ACTION_MSE_EVALS"
export OPENPI_ACTION_MSE_NUM_STEPS="$ACTION_MSE_NUM_STEPS"
export OPENPI_ASYNC_CHECKPOINT="$ASYNC_CHECKPOINT"
export OPENPI_WAIT_AFTER_CHECKPOINT="$WAIT_AFTER_CHECKPOINT"
export OPENPI_CHECKPOINT_STEPS="$CHECKPOINT_STEPS"
export OPENPI_SAVE_TRAIN_STATE="$SAVE_TRAIN_STATE"
export OPENPI_MAX_CHECKPOINTS="$MAX_CHECKPOINTS"

if asset_id="$(tac_norm_asset_id "$CONFIG_NAME")"; then
  norm_stats_path="/home/ubuntu/tac_data/openpi_assets/$asset_id/norm_stats.json"
  if [[ ! -f "$norm_stats_path" ]]; then
    if [[ "$AUTO_NORM_STATS" == "true" ]]; then
      echo "Missing TAC norm stats: $norm_stats_path"
      echo "Computing norm stats for $CONFIG_NAME before training."
      .venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
      echo
    else
      echo "Missing TAC norm stats: $norm_stats_path" >&2
      echo "Run: .venv/bin/python scripts/compute_norm_stats.py --config-name $CONFIG_NAME" >&2
      exit 1
    fi
  fi
fi

if [[ "$WANDB_ENABLED" == "true" ]]; then
  mkdir -p "$WANDB_DIR_VALUE"
  export WANDB_DIR="$WANDB_DIR_VALUE"
  if [[ "$WANDB_MODE_VALUE" == "local" ]]; then
    start_wandb_server
    if [[ "$WANDB_MODE_VALUE" == "local" ]]; then
      export WANDB_BASE_URL="http://$WANDB_HOST:$WANDB_PORT"
      if [[ -n "${WANDB_API_KEY:-}" ]]; then
        export WANDB_MODE="online"
      else
        export WANDB_MODE="offline"
        echo "W&B local server is running, but WANDB_API_KEY is not set."
        echo "Training will write local offline W&B runs. Open http://$WANDB_HOST:$WANDB_PORT, create/login an account, then rerun with WANDB_API_KEY set for live W&B logging."
        echo
      fi
    else
      export WANDB_MODE="$WANDB_MODE_VALUE"
    fi
  else
    export WANDB_MODE="$WANDB_MODE_VALUE"
  fi
else
  export WANDB_MODE="disabled"
fi
if [[ "$TENSORBOARD_ENABLED" == "true" ]]; then
  export OPENPI_ENABLE_TENSORBOARD=true
  export OPENPI_TENSORBOARD_LOGDIR="$TENSORBOARD_LOGDIR"
else
  export OPENPI_ENABLE_TENSORBOARD=false
fi

start_tensorboard
.venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}"
