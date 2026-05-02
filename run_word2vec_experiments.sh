#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

output_dir_name() {
  local alpha_only="$1"
  local min_freq="$2"
  local num_epoch="$3"
  local max_sentences="$4"
  local name="output_alpha_only_${alpha_only}_min_freq_${min_freq}_epoch_${num_epoch}"
  if [[ "${max_sentences}" != "0" ]]; then
    name="${name}_max_sentences_${max_sentences}"
  fi
  echo "${name}"
}

run_one() {
  local project_dir="$1"
  local model_name="$2"
  local experiment_name="$3"
  local alpha_only="$4"
  local min_freq="$5"
  local num_epoch="$6"
  local max_sentences="${7:-0}"

  local output_name
  output_name="$(output_dir_name "${alpha_only}" "${min_freq}" "${num_epoch}" "${max_sentences}")"
  local output_dir="${ROOT_DIR}/${project_dir}/outputs/${output_name}"
  mkdir -p "${output_dir}"
  local log_file="${output_dir}/run.log"

  echo "============================================================"
  echo "[RUN] ${model_name} / ${experiment_name}"
  echo "      ALPHA_ONLY=${alpha_only}"
  echo "      MIN_FREQ=${min_freq}"
  echo "      NUM_EPOCH=${num_epoch}"
  echo "      MAX_SENTENCES=${max_sentences}"
  echo "      log=${log_file}"
  echo "============================================================"

  (
    cd "${ROOT_DIR}/${project_dir}"
    ALPHA_ONLY="${alpha_only}" \
    MIN_FREQ="${min_freq}" \
    NUM_EPOCH="${num_epoch}" \
    MAX_SENTENCES="${max_sentences}" \
    uv run python train_pytorch.py
  ) 2>&1 | tee "${log_file}"
}

run_experiment_for_both_models() {
  local experiment_name="$1"
  local alpha_only="$2"
  local min_freq="$3"
  local num_epoch="$4"
  local max_sentences="${5:-0}"

  run_one "project1_cbow" "cbow" "${experiment_name}" "${alpha_only}" "${min_freq}" "${num_epoch}" "${max_sentences}"
  run_one "project2_skipgram" "skipgram" "${experiment_name}" "${alpha_only}" "${min_freq}" "${num_epoch}" "${max_sentences}"
}

echo "[word2vec] Starting 4 experiments x 2 models = 8 runs"
echo "[word2vec] Root: ${ROOT_DIR}"
echo "[word2vec] Logs: each run writes to its own outputs/output_*/run.log"

# 1. Original baseline: original tokenization, MIN_FREQ=1, 10 epochs.
run_experiment_for_both_models "baseline" 0 1 10 0

# 2. Improve vocabulary frequency threshold: MIN_FREQ=3, 10 epochs.
run_experiment_for_both_models "freq3" 0 3 10 0

# 3. Improve frequency threshold and train longer: MIN_FREQ=3, 20 epochs.
run_experiment_for_both_models "freq3_epoch20" 0 3 20 0

# 4. Improve frequency threshold, train longer, and filter non-alphabetic tokens.
run_experiment_for_both_models "freq3_epoch20_alpha" 1 3 20 0

echo "[word2vec] All experiments finished."
echo "[word2vec] Output folders:"
find "${ROOT_DIR}/project1_cbow/outputs" "${ROOT_DIR}/project2_skipgram/outputs" -maxdepth 1 -type d -name "output_*" | sort
