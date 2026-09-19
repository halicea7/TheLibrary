#!/usr/bin/env bash
set -euo pipefail
for name in api worker; do
  launchctl bootout "gui/$(id -u)/com.library-agent.$name" 2>/dev/null \
    && echo "stopped com.library-agent.$name" || echo "com.library-agent.$name not loaded"
  rm -f "$HOME/Library/LaunchAgents/com.library-agent.$name.plist"
done
