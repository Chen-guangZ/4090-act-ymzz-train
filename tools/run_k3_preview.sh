#!/usr/bin/env bash
set -euo pipefail

TOOLS=$(cd "$(dirname "$0")" && pwd)
PY="${ISAACLAB_PYTHON:-$HOME/IsaacLab/.venv/bin/python}"
SCENE=${1:?Pass the K3 animation USD path}
export OMNI_KIT_ACCEPT_EULA=YES
FRAME="$(dirname "$SCENE")/preview_live.jpg"
"$PY" -u -B "$TOOLS/show_k3_animation.py" "$SCENE" --frame "$FRAME" &
viewer=$!
"$PY" -u -B "$TOOLS/serve_k3_view.py" "$FRAME" &
server=$!
cleanup() {
  kill "$viewer" "$server" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait -n "$viewer" "$server"
