import SwiftUI

/// The view shown when the user clicks the menubar item.
struct MenubarContent: View {
    @EnvironmentObject var poller: StatusPoller
    let closePopover: () -> Void
    let openAddHost: () -> Void
    let openCurveEditor: (UUID) -> Void
    let openPowerLimitEditor: (UUID) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if poller.hosts.isEmpty {
                Text("No hosts configured")
                    .foregroundColor(.secondary)
                    .padding(.vertical, 4)
            } else {
                ForEach(poller.hosts) { host in
                    HostRow(
                        host: host,
                        state: poller.states[host.id],
                        openCurveEditor: openCurveEditor,
                        openPowerLimitEditor: openPowerLimitEditor
                    )
                        .padding(.vertical, 2)
                    Divider()
                }
            }

            Button {
                openAddHost()
            } label: {
                Label("Add Host…", systemImage: "plus.circle")
            }
            .buttonStyle(.borderless)

            Divider()

            HStack {
                Button("Close") {
                    closePopover()
                }
                .keyboardShortcut(.cancelAction)
                .buttonStyle(.borderless)

                Spacer()

                Button("Quit") {
                    NSApp.terminate(nil)
                }
                .keyboardShortcut("q")
                .buttonStyle(.borderless)
            }
        }
        .padding(10)
        .frame(width: 700)
        .onExitCommand {
            closePopover()
        }
    }
}

private struct HostRow: View {
    let host: Host
    let state: HostState?
    let openCurveEditor: (UUID) -> Void
    let openPowerLimitEditor: (UUID) -> Void
    @EnvironmentObject var poller: StatusPoller

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(host.name).font(.headline)
                    Spacer()
                    Button {
                        openPowerLimitEditor(host.id)
                    } label: {
                        Image(systemName: "bolt.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help("Edit power limit")
                    .disabled(state?.lastPayload?.powerLimitSupported == false)
                    Button {
                        openCurveEditor(host.id)
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help("Edit fan curve")
                    Button {
                        poller.removeHost(host)
                    } label: {
                        Image(systemName: "minus.circle")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help("Remove host")
                }
                if let p = state?.lastPayload, let t = p.vramTempC {
                    HStack(spacing: 12) {
                        Text("\(t)°")
                            .font(.system(size: 22, weight: .semibold, design: .rounded))
                            .foregroundColor(colorFor(temp: t))
                            .frame(width: 72, alignment: .leading)
                            .lineLimit(1)
                        if let fp = p.fanPct {
                            MetricText("fan \(fp)%", width: 76)
                        }
                        if let pw = p.powerW {
                            MetricText("\(Int(pw.rounded()))W", width: 58)
                        }
                        if let limit = p.powerLimitW {
                            MetricText("cap \(Int(limit.rounded()))W", width: 92)
                        }
                        if let util = p.gpuUtilPct {
                            MetricText("gpu \(util)%", width: 76)
                        }
                        if state?.isStale == true {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.yellow)
                                .help("Stale data")
                        }
                    }
                    if let gpu = p.gpuName {
                        Text(gpu).font(.caption).foregroundColor(.secondary)
                    }
                } else if let err = state?.lastError {
                    Text("Unreachable").foregroundColor(.red).font(.caption)
                    Text(err).font(.caption2).foregroundColor(.secondary)
                } else {
                    Text("Connecting…").foregroundColor(.secondary).font(.caption)
                }
            }

            if let history = state?.utilHistory, history.count >= 2 {
                Spacer(minLength: 0)
                VStack(alignment: .trailing, spacing: 2) {
                    UtilSparkline(history: history)
                        .frame(width: 210, height: 54)
                    Text("GPU %").font(.caption2).foregroundColor(.secondary)
                }
            }
        }
    }
}

private struct MetricText: View {
    let text: String
    let width: CGFloat

    init(_ text: String, width: CGFloat) {
        self.text = text
        self.width = width
    }

    var body: some View {
        Text(text)
            .foregroundColor(.secondary)
            .frame(width: width, alignment: .leading)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
    }
}

/// nvtop-style rolling line of recent GPU utilization. x is sample index,
/// y is a fixed 0…100 domain so line height reads as absolute utilization.
/// Styled to match CurveChart in CurveEditor.swift (Path, no chart dependency).
private struct UtilSparkline: View {
    let history: [Int]   // most-recent-last, 0..100

    private let pad: CGFloat = 3

    private func xFor(_ i: Int, width w: CGFloat) -> CGFloat {
        guard history.count > 1 else { return pad }
        let frac = CGFloat(i) / CGFloat(history.count - 1)
        return pad + frac * (w - 2 * pad)
    }

    private func yFor(_ v: Int, height h: CGFloat) -> CGFloat {
        let frac = CGFloat(v) / 100.0
        return (h - pad) - frac * (h - 2 * pad)
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            ZStack {
                // 0 / 50 / 100 gridlines.
                Path { p in
                    for f in stride(from: 0.0, through: 1.0, by: 0.5) {
                        let y = yFor(Int(f * 100), height: h)
                        p.move(to: CGPoint(x: pad, y: y))
                        p.addLine(to: CGPoint(x: w - pad, y: y))
                    }
                }
                .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)

                Path { p in
                    for (i, v) in history.enumerated() {
                        let pt = CGPoint(x: xFor(i, width: w), y: yFor(v, height: h))
                        if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                    }
                }
                .stroke(Color.accentColor, lineWidth: 1.5)
            }
        }
    }
}

/// Window for entering a new host. Lives as a top-level `Window` scene in
/// MenubarAppApp; opened with `openWindow(id: "addHost")`.
struct AddHostWindow: View {
    @EnvironmentObject var poller: StatusPoller
    @Environment(\.dismiss) private var dismiss
    var onClose: (() -> Void)? = nil
    @State private var name = ""
    @State private var hostname = ""
    @State private var port = "8765"
    @State private var token = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add Host").font(.headline)
            Form {
                TextField("Display name (e.g. aipc1)", text: $name)
                TextField("Hostname or IP (e.g. aipc1.tail-abc.ts.net)", text: $hostname)
                TextField("Port", text: $port)
                SecureField("Bearer token (optional)", text: $token)
            }
            HStack {
                Spacer()
                Button("Cancel") { close() }
                Button("Add") {
                    let portInt = Int(port) ?? 8765
                    poller.addHost(Host(
                        name: name.isEmpty ? hostname : name,
                        hostname: hostname,
                        port: portInt,
                        token: token
                    ))
                    close()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(hostname.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }
}

/// Color thresholds matching the menubar title color logic.
func colorFor(temp: Int) -> Color {
    switch temp {
    case ..<85:  return .green
    case 85..<95: return .yellow
    default:      return .red
    }
}
