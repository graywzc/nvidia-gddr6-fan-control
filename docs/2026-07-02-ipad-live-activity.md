# iPad Live Activity for GPU status

## Context

The user wants GPU status (VRAM temp, fan %, util) visible on the iPad "at the top of the screen, next to the clock/Wi-Fi/battery". iPadOS gives third-party apps no API to place items in the system status bar, so the agreed approach (confirmed via two AskUserQuestion rounds) is a **Live Activity**: it appears on the Lock Screen and, on iPadOS 26, surfaces in the top menu-bar area while multitasking — the closest sanctioned equivalent of a status-bar item.

All devices (iPad, PCs, GPU servers) share one Tailscale network, so the app can reach observers from anywhere; reachability is never the constraint — app suspension is.

Update mode (also confirmed): **app-driven only** — the Live Activity is started/updated by the app while it runs; once the app is suspended the activity shows its last values plus a visible staleness indicator. No APNs, no paid Apple Developer account, no observer changes. APNs push can be layered on later.

The existing iPad app lives at `ios/GPUFanControl/` (SwiftUI, iOS 17.0 target, hand-authored `project.pbxproj` with synthetic `A1…` object IDs). `StatusPoller` (`ios/GPUFanControl/GPUFanControl/StatusPoller.swift`) already polls each host's `/status` every second into `HostState` / `HostStatusPayload` (`HostStatus.swift`) — all the data the activity needs is already on-device.

All work happens in the session worktree `.claude/worktrees/claude/research-ipados-ios` (branch `claude/research-ipados-ios`, already created from latest origin/main).

## Changes

### 1. Shared ActivityKit attributes (new file, compiled into BOTH targets)

`ios/GPUFanControl/Shared/GPUStatusActivityAttributes.swift`:

- `struct GPUStatusActivityAttributes: ActivityAttributes` — fixed attributes: `hostID: String`, `hostName: String`.
- `ContentState: Codable, Hashable` — `vramTempC: Int?`, `fanPct: Int?`, `gpuUtilPct: Int?`, `powerW: Double?`, `updatedAt: Date`. (Derive hottest-GPU values the same way `HostSummaryRow` does, via `displayGPUs` + `compactMap(...).max()`.)
- Also compile `GPUFanControl/SharedViews.swift` into the widget target so the activity view reuses `colorFor(temp:)` and `MetricPill` for consistent temp coloring (it only imports SwiftUI, so it's safe to share).

### 2. New widget extension target `GPUFanControlWidgets`

New directory `ios/GPUFanControl/GPUFanControlWidgets/`:

- `GPUFanControlWidgetsBundle.swift` — `@main struct … : WidgetBundle` containing the live-activity widget.
- `GPUStatusLiveActivity.swift` — `Widget` with `ActivityConfiguration(for: GPUStatusActivityAttributes.self)`:
  - Lock-screen/banner view: host name, VRAM temp, fan %, util, power, and `Text(state.updatedAt, style: .relative)` ("updated Xs ago") so staleness is self-evident.
  - `dynamicIsland` closure is required by the API (used on iPhone; harmless on iPad): compact leading = temp, trailing = fan %, minimal = temp.
- `Info.plist` — `NSExtension` → `NSExtensionPointIdentifier = com.apple.widgetkit-extension`.

`project.pbxproj` edits (follow the existing synthetic-ID style, e.g. `A100…00B*`):

- New `PBXFileReference`s + group for the widget sources and the shared attributes file; add the shared file to BOTH app and extension `PBXSourcesBuildPhase`s.
- New `PBXNativeTarget` `GPUFanControlWidgets`, `productType = "com.apple.product-type.app-extension"`, product `GPUFanControlWidgets.appex`.
- App target: new `PBXCopyFilesBuildPhase` "Embed Foundation Extensions" (`dstSubfolderSpec = 13`) copying the `.appex`, plus `PBXTargetDependency`/`PBXContainerItemProxy` on the widget target.
- Widget build configs (Debug/Release): `PRODUCT_BUNDLE_IDENTIFIER = com.graywzc.gpufancontrol.widgets` (must be prefixed by the app's bundle ID), `INFOPLIST_FILE = GPUFanControlWidgets/Info.plist`, `SKIP_INSTALL = YES`, `IPHONEOS_DEPLOYMENT_TARGET = 17.0`, `TARGETED_DEVICE_FAMILY = "1,2"`, `CODE_SIGN_STYLE = Automatic`, `SWIFT_VERSION = 5.0`, `GENERATE_INFOPLIST_FILE = NO`, `LD_RUNPATH_SEARCH_PATHS` with `@executable_path/../../Frameworks`.
- Register target in `PBXProject.targets` and add its `XCConfigurationList`.

### 3. App-side Live Activity management

- `ios/GPUFanControl/GPUFanControl/Info.plist`: add `NSSupportsLiveActivities = true`.
- New `ios/GPUFanControl/GPUFanControl/LiveActivityManager.swift` (`@MainActor`, owned by `StatusPoller` or the app):
  - `setPinnedHost(_ host: Host?)` — persists pinned host ID in `UserDefaults`; starts `Activity.request` / ends the activity.
  - `update(host:payload:)` — builds `ContentState` from the payload; pushes via `activity.update(ActivityContent(state:…, staleDate: .now + 15))` so the system dims it when the app stops feeding it. Throttle: only update when values changed or ≥5 s elapsed.
  - On init, reclaim any existing `Activity<GPUStatusActivityAttributes>.activities` (app relaunch) and end orphans for unpinned hosts.
  - Live Activities expire after ~8 h; reclaim/restart on app foreground (`scenePhase` change in `GPUFanControlApp`).
- `StatusPoller.tick()` — after `states[id] = state`, if the host is pinned, call `liveActivityManager.update`.
- `HostListView.swift` — per-host context-menu/toolbar toggle "Pin Live Activity" (single pinned host at a time).

### 4. Tests

Add to `GPUFanControlTests`: `ContentState` JSON round-trip test (ActivityKit serializes it), and a throttle-decision test if the throttle logic is extracted as a pure function.

## Execution workflow (per standing user workflow)

1. Copy this plan into the repo as `docs/2026-07-02-ipad-live-activity.md` and commit it first.
2. Dispatch implementation to a local ccr agent (`/ccr-delegate`), review and iterate on its output; build fixes done locally.
3. Push branch `claude/research-ipados-ios`, check for existing PR (`gh pr list --head …`), create PR. **Do not merge — wait for user review** (standing rule).

## Verification

- `xcodebuild -project ios/GPUFanControl/GPUFanControl.xcodeproj -scheme GPUFanControl -destination 'platform=iOS Simulator,name=iPad Pro 11-inch (M5)' build` (Xcode 26.6 present).
- Run unit tests with `xcodebuild test` on the same destination.
- Launch in the iPad simulator (`/run` flow): add a host (aipc observer on Tailscale, port 8765, per memory), pin the Live Activity, verify it appears on the simulator Lock Screen with live values; lock the screen and confirm the "updated Xs ago" staleness readout; unpin ends the activity.
- Widget extension cannot be exercised beyond the simulator; final on-device check (iPadOS 26 top-bar presentation) is the user's sideload.

## Explicitly out of scope

- APNs/ActivityKit push updates (phase 2 if wanted later).
- Home/Lock-Screen widgets.
- Observer (`aipc_observer.py`) changes.
