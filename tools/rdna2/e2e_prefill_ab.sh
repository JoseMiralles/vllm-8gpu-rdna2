#!/usr/bin/env bash
# End-to-end prefill A/B: the Triton path vs the all-HIP umbrella, TP4 on all
# four GPUs. This is the one comparison the per-kernel benches cannot make.
#
#   bash tools/rdna2/e2e_prefill_ab.sh            # both arms, in order
#   bash tools/rdna2/e2e_prefill_ab.sh triton     # one arm
#
# The gates are read at import time, so each arm needs its own server process.
# Prefill chunks are capped at --max-num-batched-tokens 2048, so T=2048 is the
# served shape; 8k/16k are the long-context chunked-prefill sizes.
#
# NOTE: the gates only cover the prefill ports (QSA metadata + sparse, GDN,
# hyper-connection, causal-conv1d). Decode-side HIP kernels are unconditional, so
# both arms have them and the difference isolates the prefill paths.
set -euo pipefail
repo=/home/wsantos/work/vllm-rdna
cd "$repo"
PORT=${PORT:-8000}
arms=${1:-both}

# One pattern per line, each bracketed so the pattern text cannot match this
# script's own command line and kill the caller's shell.
stop_server() {
  pkill -9 -f '[s]erve_v620_stable.sh|[s]erve_v620_v031.sh|[o]penai.api_server|[c]li.main serve|[V]LLM::' || true
  sleep 5
  fuser -k -9 /dev/kfd >/dev/null 2>&1 || true
  sleep 5
  pgrep -af '[o]penai.api_server|[c]li.main serve' && { echo "server still up"; return 1; }
  return 0
}

wait_ready() {
  for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "  ready after $((i * 10))s"
      return 0
    fi
    sleep 10
  done
  echo "  server did NOT become ready in 900s"
  return 1
}

run_arm() { # label  env...
  local label=$1
  shift
  echo "=== arm $label ==="
  stop_server
  env "$@" bash tools/rdna2/serve_v620_stable.sh \
    >".pi/e2e_$label.server.log" 2>&1 &
  if ! wait_ready; then
    tail -30 ".pi/e2e_$label.server.log"
    stop_server || true
    return 1
  fi
  bash tools/rdna2/bench_serve_gates_ab.sh "$label" || true
  stop_server
}

# Every prefill gate off: the Triton/FLA reference path.
TRITON_ENV=(
  VLLM_RDNA_ALL_HIP=0
  VLLM_RDNA_QSA_HIP=0
  VLLM_GDN_HIP_PREFILL=0
  VLLM_RDNA_HC_PREFILL_HIP=0
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=0
  VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
)
# The umbrella plus every individual gate: the all-HIP path.
HIP_ENV=(
  VLLM_RDNA_ALL_HIP=1
  VLLM_RDNA_QSA_HIP=1
  VLLM_GDN_HIP_PREFILL=1
  VLLM_RDNA_HC_PREFILL_HIP=1
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=1
  VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=1
)

case "$arms" in
  triton) run_arm triton "${TRITON_ENV[@]}" ;;
  hip) run_arm hip "${HIP_ENV[@]}" ;;
  both)
    run_arm triton "${TRITON_ENV[@]}"
    run_arm hip "${HIP_ENV[@]}"
    ;;
  *) echo "usage: $0 [both|triton|hip]"; exit 2 ;;
esac
echo "=== logs in $repo/.pi/serve_ab_{triton,hip}/ ==="
