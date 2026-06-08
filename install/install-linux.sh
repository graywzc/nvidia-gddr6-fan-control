#!/usr/bin/env bash
# Install fan_control.py and its systemd unit.
# Re-run safely; this script is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DEST=/usr/local/bin/nvidia-gddr6-fan-control
OBSERVER_DEST=/usr/local/bin/aipc_observer.py
UNIT_NAME=nvidia-gddr6-fan-control.service
UNIT_DEST=/etc/systemd/system/$UNIT_NAME

if [[ $EUID -ne 0 ]]; then
    echo "This script needs root (writes to /usr/local/bin and /etc/systemd)."
    echo "Re-run with: sudo $0"
    exit 1
fi

if ! command -v tailscale >/dev/null; then
    echo "WARN: 'tailscale' not found in PATH. The unit uses --listen-tailscale;"
    echo "      install Tailscale before enabling, or edit the unit to drop the flag."
fi

if [[ ! -x /usr/local/bin/gddr6 ]] && ! command -v gddr6 >/dev/null; then
    echo "WARN: 'gddr6' binary not found. Build/install it first:"
    echo "      https://github.com/olealgoritme/gddr6"
fi

echo "Installing $BIN_DEST"
install -m 755 "$REPO_ROOT/fan_control.py" "$BIN_DEST"

echo "Installing $OBSERVER_DEST"
install -m 644 "$REPO_ROOT/aipc_observer.py" "$OBSERVER_DEST"

echo "Installing $UNIT_DEST"
install -m 644 "$REPO_ROOT/systemd/$UNIT_NAME" "$UNIT_DEST"

# Ensure the state directory exists with sensible permissions.
install -d -m 755 /var/lib/nvidia-gddr6-fan-control

systemctl daemon-reload
systemctl enable "$UNIT_NAME"

# If the service is already running, restart to pick up the new binary.
if systemctl is-active --quiet "$UNIT_NAME"; then
    echo "Restarting running service"
    systemctl restart "$UNIT_NAME"
else
    echo "Starting service"
    systemctl start "$UNIT_NAME"
fi

echo
echo "Done. Useful commands:"
echo "  systemctl status $UNIT_NAME"
echo "  journalctl -u $UNIT_NAME -f"
echo "  systemctl restart $UNIT_NAME"
echo "  systemctl disable --now $UNIT_NAME    # to stop auto-start"
