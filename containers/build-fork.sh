#!/usr/bin/env bash
# containers/build-fork.sh — build this fork's runtime image under JoseMiralles' own
# registry namespace, FROM leapdragon's published base.
#
# The base image carries TheRock ROCm 7.14 + torch/triton built for gfx1030; it is public
# and is *not* rebuilt here (that takes 2-3 h — use build.sh --base). `build.sh` derives
# the base image name from REGISTRY_REPO, so building under your own namespace without
# this wrapper would look for a `josemiralles/...-base` image that does not exist. This
# wrapper pins the base explicitly so the fork builds out of the box.
#
#   ./containers/build-fork.sh                 # build ghcr.io/josemiralles/vllm-rdna2-qwen:latest
#   ./containers/build-fork.sh --push          # also push (docker login ghcr.io first)
#   ./containers/build-fork.sh --base          # rebuild the base too (FROM the pinned tarball, hours)
#   REGISTRY_REPO=... ./containers/build-fork.sh   # override the output namespace
#
# The tag/version and VLLM_VERSION_OVERRIDE are derived from the current git HEAD by
# containers/build.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REGISTRY_REPO="${REGISTRY_REPO:-ghcr.io/josemiralles/vllm-rdna2-qwen}"
export BASE_TAG="${BASE_TAG:-therock7.14.1-torch2.12-gfx1030}"
export BASE_IMAGE="${BASE_IMAGE:-ghcr.io/leapdragon/vllm-rdna2-qwen-base:${BASE_TAG}}"

echo "registry: $REGISTRY_REPO"
echo "base:     $BASE_IMAGE"
exec "$here/build.sh" "$@"
