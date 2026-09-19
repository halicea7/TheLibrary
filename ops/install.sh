#!/usr/bin/env bash
# Install the API and worker as launchd user agents so they survive reboots.
#
# The worker runs max_jobs=1 on purpose: Ollama serialises requests against a single
# model, so a second concurrent job only makes both slower and can push a queued request
# past its client timeout.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv)"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS" "$DIR/ops/logs"

for name in api worker; do
  src="$DIR/ops/com.library-agent.$name.plist"
  dst="$AGENTS/com.library-agent.$name.plist"
  sed -e "s|__DIR__|$DIR|g" -e "s|__UV__|$UV|g" "$src" > "$dst"
  launchctl bootout "gui/$(id -u)/com.library-agent.$name" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$dst"
  echo "loaded com.library-agent.$name"
done

echo
echo "API:    http://127.0.0.1:8077"
echo "logs:   $DIR/ops/logs/"
echo "stop:   ./ops/uninstall.sh"
