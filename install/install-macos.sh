#!/usr/bin/env bash
# Build the SwiftPM app, install as /Applications/MenubarApp.app,
# and register a LaunchAgent so it starts at login.
# Re-run safely; this script is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT_PROJECT_DIR="$REPO_ROOT/macos/MenubarApp"
APP_BUNDLE_NAME="MenubarApp.app"
APP_INSTALL_DIR="/Applications"
APP_BUNDLE_PATH="$APP_INSTALL_DIR/$APP_BUNDLE_NAME"
BUNDLE_ID="com.graywzc.nvidia-gddr6-fan-menubar"
EXECUTABLE_NAME="MenubarApp"
PLIST_PATH="$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is for macOS." >&2
    exit 1
fi

if ! command -v swift >/dev/null; then
    echo "ERROR: 'swift' not found. Install Xcode Command Line Tools:"
    echo "       xcode-select --install"
    exit 1
fi

echo "Building release binary…"
cd "$SWIFT_PROJECT_DIR"
swift build -c release

BUILT_BIN="$(swift build -c release --show-bin-path)/$EXECUTABLE_NAME"
if [[ ! -x "$BUILT_BIN" ]]; then
    echo "ERROR: built binary not found at $BUILT_BIN" >&2
    exit 1
fi

echo "Assembling $APP_BUNDLE_PATH"
# Remove any prior install so we don't mix old files in.
rm -rf "$APP_BUNDLE_PATH"
mkdir -p "$APP_BUNDLE_PATH/Contents/MacOS"
mkdir -p "$APP_BUNDLE_PATH/Contents/Resources"

cp "$BUILT_BIN" "$APP_BUNDLE_PATH/Contents/MacOS/$EXECUTABLE_NAME"
chmod +x "$APP_BUNDLE_PATH/Contents/MacOS/$EXECUTABLE_NAME"

# Minimal Info.plist. LSUIElement=true hides the Dock icon at launch
# (NSApp.setActivationPolicy(.accessory) does the same at runtime, but
# setting it here avoids a brief Dock flash during cold start).
cat > "$APP_BUNDLE_PATH/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$EXECUTABLE_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>Nvidia GPU Fan Monitor</string>
    <key>CFBundleExecutable</key>
    <string>$EXECUTABLE_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
EOF

# Write the LaunchAgent that auto-starts the app on login.
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$BUNDLE_ID</string>
    <key>ProgramArguments</key>
    <array>
        <string>$APP_BUNDLE_PATH/Contents/MacOS/$EXECUTABLE_NAME</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
EOF

# Reload the LaunchAgent (bootout first if it's already loaded).
UID_NUM="$(id -u)"
launchctl bootout "gui/$UID_NUM/$BUNDLE_ID" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"

# Start it now too.
launchctl kickstart -k "gui/$UID_NUM/$BUNDLE_ID" >/dev/null 2>&1 || true

echo
echo "Done."
echo "  App:           $APP_BUNDLE_PATH"
echo "  LaunchAgent:   $PLIST_PATH"
echo
echo "Useful commands:"
echo "  open '$APP_BUNDLE_PATH'                   # launch manually"
echo "  pkill $EXECUTABLE_NAME                    # quit"
echo "  launchctl bootout gui/$UID_NUM/$BUNDLE_ID # stop autostart"
echo "  launchctl bootstrap gui/$UID_NUM '$PLIST_PATH' # re-enable autostart"
