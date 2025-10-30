#!/bin/bash
set -euo pipefail

XVFB_DISPLAY=${DISPLAY:-:99}

cleanup() {
  if [[ -n "${XVFB_PID:-}" ]]; then
    kill "${XVFB_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" >/dev/null 2>&1; then
  Xvfb "${XVFB_DISPLAY}" -screen 0 1280x720x24 -nolisten tcp &
  XVFB_PID=$!
  sleep 1
fi

export DISPLAY="${XVFB_DISPLAY}"

exec "$@"
