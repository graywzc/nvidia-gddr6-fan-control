import SwiftUI

/// The view shown when the user clicks the menubar item.
struct MenubarContent: View {
    @EnvironmentObject var poller: StatusPoller
    @State private var showingAddHost = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if poller.hosts.isEmpty {
                Text("No hosts configured")
                    .foregroundColor(.secondary)
                    .padding(.vertical, 4)
            } else {
                ForEach(poller.hosts) { host in
                    HostRow(host: host, state: poller.states[host.id])
                        .padding(.vertical, 2)
                    Divider()
                }
            }

            Button {
                showingAddHost = true
            } label: {
                Label("Add Host…", systemImage: "plus.circle")
            }
            .buttonStyle(.borderless)

            Divider()

            Button("Quit") {
                NSApp.terminate(nil)
            }
            .keyboardShortcut("q")
            .buttonStyle(.borderless)
        }
        .padding(10)
        .frame(width: 280)
        .sheet(isPresented: $showingAddHost) {
            AddHostSheet { newHost in
                poller.addHost(newHost)
            }
        }
    }
}

private struct HostRow: View {
    let host: Host
    let state: HostState?
    @EnvironmentObject var poller: StatusPoller

    var body: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(host.name).font(.headline)
                    Spacer()
                    Button {
                        poller.removeHost(host)
                    } label: {
                        Image(systemName: "minus.circle")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.borderless)
                }
                if let p = state?.lastPayload, let t = p.vramTempC {
                    HStack(spacing: 8) {
                        Text("\(t)°")
                            .font(.system(size: 22, weight: .semibold, design: .rounded))
                            .foregroundColor(colorFor(temp: t))
                        if let fp = p.fanPct {
                            Text("fan \(fp)%")
                                .foregroundColor(.secondary)
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
        }
    }
}

/// Sheet for entering a new host.
private struct AddHostSheet: View {
    var onAdd: (Host) -> Void
    @Environment(\.dismiss) private var dismiss
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
                Button("Cancel") { dismiss() }
                Button("Add") {
                    let portInt = Int(port) ?? 8765
                    onAdd(Host(
                        name: name.isEmpty ? hostname : name,
                        hostname: hostname,
                        port: portInt,
                        token: token
                    ))
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(hostname.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 380)
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
