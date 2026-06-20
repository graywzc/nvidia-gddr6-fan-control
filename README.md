# nvidia-gddr6-fan-control

VRAM-junction-temperature-driven fan control for NVIDIA GDDR6/GDDR6X GPUs on Linux, with SwiftUI macOS and iOS apps for live monitoring and remote curve editing.

Why this exists: NVIDIA's stock fan curve on Linux is driven by core temperature only. On RTX 3080/3090/A6000-class cards with GDDR6X, the memory junction temperature can sit ≥100°C while the core is happily under 70°C and the fans stay quiet. This project reads the actual VRAM junction temperature from the GPU's internal sensor, applies a user-defined fan curve, and exposes a small HTTP API so a Mac menubar app can monitor multiple GPUs and edit the curve live.

## Architecture

```
┌──────────────────┐         ┌──────────────────────────────────┐
│  macOS menubar   │ ◄────►  │  Linux host (RTX 30/40 series)   │
│  app (SwiftUI)   │  HTTP   │  ┌─────────────────────────────┐ │
│  ┌──────────┐    │  over   │  │ fan_control.py              │ │
│  │ 96° 91°  │    │  Tail-  │  │  ├─ spawns gddr6 (VRAM temp)│ │
│  └──────────┘    │  scale  │  │  ├─ applies curve via NVML  │ │
│  click → opens   │         │  │  └─ HTTP /status, /curve    │ │
│  curve editor    │         │  └─────────────────────────────┘ │
└──────────────────┘         └──────────────────────────────────┘
```

- **VRAM temp source:** the [`gddr6`](https://github.com/olealgoritme/gddr6) binary by olealgoritme reads the on-die VRAM junction-temperature register via `/dev/mem`. NVML still does not expose this on consumer cards as of driver 580.
- **Fan control:** `nvmlDeviceSetFanSpeed_v2` from NVML — no X server, no Xorg session, no Coolbits.
- **Power limiting:** NVML power-management APIs apply and report the board power cap when supported by the GPU/driver.
- **Observer dashboard:** the controller can serve an integrated llama.cpp/GPU request dashboard at `/observer`.
- **Transport:** plaintext HTTP, bound only to the host's Tailscale interface. Tailscale handles encryption and identity.
- **iOS client:** an iPhone/iPad app mirrors the macOS client features: multi-host monitoring, per-GPU telemetry, observer launch, fan-curve editing, and power-limit editing.

## Supported hardware

Anything supported by `gddr6` upstream and with `nvmlDeviceSetFanSpeed_v2` enabled in the driver. Verified on RTX 3090 (GA102) with driver 580.159.03. Likely works on:

- RTX 3070 / 3080 / 3080 Ti / 3090 / 3090 Ti
- RTX 4070 / 4080 / 4090 (incl. 4090 D)
- RTX A2000 / A4500 / A5000 / A6000, L4, L40S, A10

See [olealgoritme/gddr6 supported GPUs](https://github.com/olealgoritme/gddr6#supported-gpus) for the full list of cards that can have their VRAM temperature read.

## Dependencies

### Linux

- NVIDIA proprietary driver **≥ 525** (for NVML fan-control APIs). Driver 580+ recommended.
- Python **≥ 3.9** (uses `http.server`, ctypes, only the standard library — no `pip install` needed).
- `libpci-dev`, `cmake`, `build-essential` — only to build the `gddr6` binary.
- `gddr6` binary — see install steps below.
- `liquidctl` and/or `lm-sensors` (optional) — only for **case-fan** control; see [Case fans](#case-fans). GPU fan control works without them.
- Tailscale (optional but recommended) — provides network-layer auth so we don't need bearer tokens for the HTTP API.

### macOS

- macOS **≥ 13** (for SwiftUI `MenuBarExtra`).
- Xcode Command Line Tools (`xcode-select --install`) — provides the `swift` compiler. The full Xcode IDE is **not** required.
- Tailscale, on the same tailnet as the Linux hosts.

### iOS

- iOS/iPadOS **≥ 17**.
- Full Xcode, for opening and running `ios/GPUFanControl/GPUFanControl.xcodeproj`.
- Tailscale installed and connected to the same tailnet as the Linux hosts.

## Install

### Linux (run on each GPU host)

```bash
# 1. Build & install the gddr6 binary (the VRAM-temp reader)
sudo apt install -y libpci-dev cmake build-essential
cd ~/projects
git clone https://github.com/olealgoritme/gddr6.git
cd gddr6
./build_install.sh    # answer 'y' to install /usr/local/bin/gddr6

# 2. Install Tailscale (skip if already set up)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# 3. Install this project
cd ~/projects
git clone https://github.com/graywzc/nvidia-gddr6-fan-control.git
cd nvidia-gddr6-fan-control
sudo ./install/install-linux.sh
```

The installer copies `fan_control.py` to `/usr/local/bin/nvidia-gddr6-fan-control`, installs the systemd unit, creates `/var/lib/nvidia-gddr6-fan-control/` for the persisted curve, and enables + starts the service.

Verify:

```bash
sudo systemctl status nvidia-gddr6-fan-control
journalctl -u nvidia-gddr6-fan-control -f
```

The first log line should say `HTTP API listening on 100.x.x.x:8765` (the Tailscale IP).

#### Model-serving stack (for the observer's Install / Switch buttons)

The fan controller needs none of this. The observer dashboard's **Install** and
**Switch** buttons drive the [club-3090](https://github.com/graywzc/nvidia-gddr6-fan-control)
checkout's `scripts/setup.sh` (downloads model weights) and `scripts/switch.sh`
(`docker compose up` a variant), expected at `/home/<user>/projects/club-3090`.
Those need a model-serving stack the base install doesn't set up:

```bash
# 1. HuggingFace CLI (setup.sh downloads weights with it).
#    Ubuntu 24.04 ships an apt 'rich' with no pip RECORD, so install a
#    pip-managed rich first to avoid an uninstall error, then huggingface-hub.
sudo pip install --break-system-packages --ignore-installed rich
sudo pip install --break-system-packages 'huggingface-hub[hf_transfer]'

# 2. Container runtime + compose (switch.sh runs `docker compose`)
sudo apt install -y docker.io docker-compose-v2

# 3. GPU access inside containers (NVIDIA container toolkit)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Let the repo owner drive docker without sudo (the daemon runs these as
#    that user, not root)
sudo usermod -aG docker "$USER"
```

> **PATH gotcha:** the observer runs `setup.sh`/`switch.sh` via `runuser -u <owner>`
> with **no login shell**, so their PATH is the systemd daemon's
> (`…:/usr/local/bin:…:/usr/bin:…`) and excludes `~/.local/bin`. Install `hf`
> system-wide (as above, it lands in `/usr/local/bin`) — a `pipx` install in
> `~/.local/bin` works in your shell but the Install button won't find it.
> A `pipx` install therefore needs `sudo ln -s ~/.local/bin/hf /usr/local/bin/hf`.

Verify the stack the way the observer will use it:

```bash
# hf reachable under the daemon's environment
sudo env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  runuser -u "$USER" -- hf version
# both GPUs visible inside a container, run as the repo owner
sudo runuser -u "$USER" -- \
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Gated models (e.g. Qwen3.6) usually download without auth; if one needs a token,
put `HF_TOKEN=hf_...` in `/home/<user>/projects/club-3090/.env` (gitignored;
`setup.sh` sources it). Weights default to `<repo>/models-cache` (~20 GB); set
`MODEL_DIR=` in the same `.env` to relocate.

### macOS

```bash
# Make sure Xcode CLI tools and Tailscale are installed
xcode-select --install
# install Tailscale from https://tailscale.com/download/mac

cd ~/projects
git clone https://github.com/graywzc/nvidia-gddr6-fan-control.git
cd nvidia-gddr6-fan-control
./install/install-macos.sh
```

The installer runs `swift build -c release`, assembles `/Applications/MenubarApp.app`, and registers a per-user LaunchAgent so the app starts at every login.

After install, click the menubar item → **Add Host…** and enter each Linux host's Tailscale name (e.g. `aipc1`, `aipc`) with port `8765`. The temp appears within ~1 second.

### iOS

Open `ios/GPUFanControl/GPUFanControl.xcodeproj` in Xcode, select the
`GPUFanControl` scheme, and run it on an iPhone/iPad simulator or device. On a
real device, install and connect Tailscale first so hostnames such as `aipc1`
resolve the same way they do from the Mac.

## Usage

### Menubar

- Menubar label shows VRAM temps for each configured host, space-separated. Hottest temp colors the label (green < 85°C, yellow 85–94°C, red ≥ 95°C).
- Click the label for a popover with per-host detail (VRAM temp, current fan %, GPU model).
- Click a host row to open that host's observer dashboard in the browser.
- Slider icon next to each host opens the curve editor for that host.
- Bolt icon next to each host opens the power-limit editor for that host.

### iOS app

- Add the same Linux GPU hosts used by the macOS app, including optional bearer tokens.
- Monitor multiple hosts and per-GPU VRAM temperature, fan percentage, board power, power cap, and utilization.
- Open a host detail view for live telemetry and the rolling GPU-utilization sparkline.
- Open the observer dashboard in Safari.
- Edit fan curves per host/GPU, with the active temperature segment highlighted.
- Edit or restore board power limits per host/GPU when the GPU reports power-limit support.
- Swipe to delete hosts.

### Curve editor

- The current active segment (where the live VRAM temp falls) is highlighted.
- Edit waypoints; the chart preview updates live.
- **Apply** sorts the points by temperature, validates, and pushes the new curve to the host. The Linux controller switches to the new curve on its next iteration (≤ 1 s) and persists it to `/var/lib/nvidia-gddr6-fan-control/curve.json`.
- **Revert** reloads from the host.

### HTTP API

Bound to the Tailscale interface only. No auth required (Tailscale handles identity).

```
GET  http://<host>:8765/status
GET  http://<host>:8765/observer
GET  http://<host>:8765/observer/api/snapshot
PUT  http://<host>:8765/curve
       body: JSON list of [temp, fan_pct] pairs,
             temps strictly ascending, e.g.
             [[60,40],[80,55],[90,75],[95,90],[100,100]]
             Or {"gpu_index": 1, "curve": [[60,40],...]} for one GPU.
PUT  http://<host>:8765/power-limit
       body: {"power_limit_w": 250}
             Or {"gpu_index": 1, "power_limit_w": 250} for one GPU.
             Use null to restore the GPU default power limit.
PUT  http://<host>:8765/case-fans
       body: {"fan": "<fan id>", "duty_pct": 60}
             Sets a case fan to a manual duty (0–100%). Fan ids come from the
             "case_fans" array in GET /status. See "Case fans" below.
```

Example:

```bash
curl http://aipc1:8765/status
curl -X PUT -H 'Content-Type: application/json' \
     -d '[[60,40],[80,55],[90,75],[95,90],[100,100]]' \
     http://aipc1:8765/curve
curl -X PUT -H 'Content-Type: application/json' \
     -d '{"power_limit_w":250}' \
     http://aipc1:8765/power-limit
```

`GET /status` returns the controller's current view, e.g.:

```json
{"vram_temp_c": 88, "power_w": 215.4, "power_limit_w": 250.0,
 "power_limit_min_w": 100.0, "power_limit_max_w": 450.0,
 "power_limit_default_w": 350.0, "tdp_w": 350.0,
 "power_limit_supported": true,
 "gpus": [{"index": 0, "vram_temp_c": 88, "fan_pct": 62,
           "power_limit_w": 250.0, "curve": [[60,40],...]}],
 "curves": {"0": [[60,40],...]}, "power_limits": {"0": 250.0},
 "fan_pct": 62, "gpu_name": "...", "num_fans": 2, "curve": [[60,40],...],
 "case_fans": [{"id": "hwmon:nct6798:pwm2", "label": "Fractal front",
                "backend": "hwmon", "kind": "fan", "rpm": 820,
                "duty_pct": 50, "settable": true}],
 "updated_at": 1234.5,
 "wall_time": 1700000000.0, "dry_run": false}
```

`power_w` is the current board power draw in watts (NVML `nvmlDeviceGetPowerUsage`);
it is `null` on cards/drivers that don't expose it.
`power_limit_w` is the primary GPU's current NVML board power cap in watts.
`gpus` contains the per-GPU telemetry, curve, and power-limit view. PUT
`/power-limit` persists the cap to the controller state file and applies it on
startup.
`tdp_w` is NVML's default power-management limit, which comes from the card's
firmware/driver power target and is the value the macOS app labels as TDP.

`case_fans` is the current view of any case fans the controller found (empty if
none / disabled). Each entry looks like:

```json
{"id": "hwmon:nct6798:pwm2", "label": "nct6798 pwm2", "backend": "hwmon",
 "kind": "fan", "rpm": 820, "duty_pct": 50, "settable": true}
```

`duty_pct` is the last applied/observed duty (may be `null` for liquidctl fans,
which don't report per-fan duty). `kind` is `fan` or `pump`. `settable` is
`false` for pumps and for any channel an admin marks read-only in the config.

## Case fans

The GPU loop only drives the NVIDIA card's fans. Case fans (AIO radiator fans,
chassis fans) hang off separate controllers that differ per host, so they're
handled by two pluggable backends, both **optional** — if neither is present
the rest of the controller works unchanged:

- **liquidctl** — Corsair Commander Core / Commander Pro and the AIO pumps they
  host (iCUE H100i Capellix, MSI Coreliquid, …). The controller shells out to
  the `liquidctl` CLI (same subprocess pattern as `gddr6`).
- **hwmon** — motherboard fan headers via the Linux hwmon sysfs (e.g. an Asus
  ROG Strix B550-E's Nuvoton Super-I/O). Pure `/sys` reads/writes, no extra
  dependency.

This first cut **reads** fan speeds and applies a **manual duty %** — there are
no temperature-driven case-fan curves yet.

### Setup

```bash
# liquidctl (for Corsair/AIO fans) — pick one:
sudo apt install liquidctl            # distro package, or
pipx install liquidctl                # latest upstream

# hwmon (for motherboard headers): load the Super-I/O driver and let
# lm-sensors detect it. The module name depends on your board's chip.
sudo apt install lm-sensors
sudo sensors-detect                   # answer YES to load the right modules
sudo modprobe nct6775                 # e.g. for Nuvoton NCT67xx (B550-E)
sensors                               # confirm fanN / pwmN channels show up
```

The controller runs as root (it already needs root for NVML/`/dev/mem`), which
also covers liquidctl's USB access and writing to `pwmN_enable`/`pwmN`.

### How fans are discovered and addressed

With no config, the controller auto-discovers everything both backends expose
and assigns each a stable id:

- hwmon: `hwmon:<chip>:pwm<N>` (e.g. `hwmon:nct6798:pwm2`)
- liquidctl: `liquidctl:<device-slug>:fan<N>` and `…:pump`

Read them from `GET /status`, then set one:

```bash
curl http://aipc1:8765/status | jq '.case_fans'
curl -X PUT -H 'Content-Type: application/json' \
     -d '{"fan":"hwmon:nct6798:pwm2","duty_pct":80}' \
     http://aipc1:8765/case-fans
```

Manual duties are persisted to the state file and re-applied on restart. On a
clean shutdown (SIGINT/SIGTERM/SIGHUP), hwmon headers are returned to their
original BIOS/auto control mode; liquidctl fans keep running at their last
manual duty (these controllers have no firmware auto-curve to fall back to).

### Optional per-host config

Hardware differs per host, so an optional JSON file lets you friendly-label
fans, mark channels read-only (safety), and set the poll interval. Default path
`/etc/nvidia-gddr6-fan-control/case-fans.json` (override with
`--case-fan-config`); a missing file just means pure auto-discovery.

```json
{
  "poll_interval_s": 5,
  "fans": {
    "hwmon:nct6798:pwm2": {"label": "Fractal front", "settable": true},
    "hwmon:nct6798:pwm1": {"label": "CPU fan", "settable": false},
    "liquidctl:corsair-commander-core:fan1": {"label": "Top intake"}
  }
}
```

> **Safety:** by default every discovered fan header is settable (pumps are
> read-only). hwmon cannot tell a CPU-fan or pump header apart from a chassis
> fan — driving the wrong header to a low duty can overheat the CPU/AIO. Mark
> such channels `"settable": false` in the config once you've identified them.

## Configuration

Most options have sensible defaults; override via CLI flags or by editing the systemd unit.

| Flag | Default | What it does |
|---|---|---|
| `--listen-tailscale` | off | Resolve and bind only to the host's Tailscale IPv4 |
| `--listen-host` | `0.0.0.0` | HTTP bind address (ignored if `--listen-tailscale`) |
| `--listen-port` | `8765` | HTTP port |
| `--token-file` | none | File containing a bearer token; if unset, no auth |
| `--state-file` | `/var/lib/nvidia-gddr6-fan-control/curve.json` | Persisted settings; use `off` to disable persistence |
| `--gpus` | `all` | GPUs to control: `all` or a comma list like `0,1` |
| `--gpu` | none | Deprecated single-GPU mode; honours `--vram-source-index` |
| `--vram-source-index` | GPU index | With `--gpu`, which gddr6 VRAM temp index to drive it from |
| `--gddr6-bin` | `/usr/local/bin/gddr6` | Path to the gddr6 binary |
| `--power-limit-w` | none | Set board power limit in watts at startup for every controlled GPU |
| `--case-fans` / `--no-case-fans` | on | Enable or disable case-fan query/control (liquidctl + hwmon) |
| `--case-fan-config` | `/etc/nvidia-gddr6-fan-control/case-fans.json` | Per-host case-fan config (labels, settable allowlist, poll interval); missing file = auto-discovery |
| `--liquidctl-bin` | `liquidctl` | Path to the liquidctl binary for Corsair/AIO case fans |
| `--observer` / `--no-observer` | on | Enable or disable the integrated observer dashboard |
| `--observer-monitor-port` | `8020` | llama.cpp frontend port whose TCP connections are monitored |
| `--observer-container` | `beellama-qwen36-27b` | Docker container whose llama.cpp logs are tailed |
| `--dry-run` | off | Read temps and print decisions but never call NVML SetFanSpeed |

## Troubleshooting

- **`Connection refused` from the Mac:** the controller is binding to the wrong interface. Check `journalctl -u nvidia-gddr6-fan-control` — the "HTTP API listening on …" line should show your Tailscale IP. If it shows `127.0.0.1` or `0.0.0.0`, restart the service.
- **`NVML SetFanSpeed: Insufficient Permissions`:** the service isn't running as root. The systemd unit runs as root by default; verify with `systemctl show -p User nvidia-gddr6-fan-control` (should be empty / root).
- **VRAM temp shows `N/A`:** your GPU may not be in `gddr6`'s supported list; check with `sudo /usr/local/bin/gddr6`. If gddr6 sees it but our app doesn't, file an issue.
- **Fans stuck at a fixed % after a crash:** `nvmlDeviceSetFanSpeed_v2` puts the GPU into manual mode and only `SetDefaultFanSpeed_v2` (or a reboot) releases it. The controller restores auto on SIGINT/SIGTERM/SIGHUP and atexit, but not on SIGKILL or power loss. To recover manually:
  ```bash
  sudo python3 -c "
  import ctypes
  m = ctypes.CDLL('libnvidia-ml.so.1')
  m.nvmlInit_v2()
  h = ctypes.c_void_p(); m.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h))
  n = ctypes.c_uint(); m.nvmlDeviceGetNumFans(h, ctypes.byref(n))
  for i in range(n.value): m.nvmlDeviceSetDefaultFanSpeed_v2(h, i)
  m.nvmlShutdown()"
  ```
- **macOS menubar shows `—`:** host unreachable. Check `tailscale status` on both ends; `curl http://<host>:8765/status` from the Mac to isolate.

## Deployment

### Linux GPU hosts

Linux deployment can run automatically through GitHub Actions self-hosted
runners. Install one runner on each GPU host and give them these labels:

- `aipc`: `self-hosted`, `linux`, `aipc`
- `aipc1`: `self-hosted`, `linux`, `aipc1`

The runner user needs passwordless `sudo` for the installer because
`install/install-linux.sh` writes `/usr/local/bin`, installs the systemd unit,
and restarts `nvidia-gddr6-fan-control.service`.

After a merge to `main`, `.github/workflows/deploy-linux.yml` deploys to both
hosts in parallel. You can also run **Deploy Linux GPU Hosts** manually from the
GitHub Actions tab.

The macOS menubar app is still installed manually with:

```bash
./install/install-macos.sh
```

## Tests

Python unit tests (stdlib `unittest`, no GPU required):

```bash
python3 -m unittest discover -s tests
```

## File layout

```
fan_control.py                          # the Linux controller
aipc_observer.py                        # integrated llama.cpp/GPU observer dashboard
case_fans.py                            # case-fan backends (liquidctl + hwmon) + controller
tests/test_power.py                     # power-draw plumbing tests
tests/test_observer.py                  # observer request parser tests
tests/test_case_fans.py                 # case-fan backend/controller tests
systemd/nvidia-gddr6-fan-control.service
install/install-linux.sh
install/install-macos.sh
macos/MenubarApp/
    Package.swift
    Sources/MenubarApp/
        MenubarAppApp.swift             # @main, AppDelegate, MenuBarExtra scene
        MenubarContent.swift            # popover view, host rows, Add Host window
        CurveEditor.swift               # per-host curve editor window + chart
        StatusPoller.swift              # 1 Hz polling, PUT /curve client
        HostStatus.swift                # JSON models, Host config struct
```
