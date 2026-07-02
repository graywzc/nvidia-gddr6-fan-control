import SwiftUI
import UIKit

struct HostListView: View {
    @EnvironmentObject private var poller: StatusPoller
    @State private var showingAddHost = false

    var body: some View {
        NavigationStack {
            Group {
                if poller.hosts.isEmpty {
                    ContentUnavailableView(
                        "No Hosts",
                        systemImage: "desktopcomputer",
                        description: Text("Add a Linux GPU host to monitor VRAM temperature, fan speed, utilization, and power limits.")
                    )
                } else {
                    List {
                        ForEach(poller.hosts) { host in
                            NavigationLink(value: host) {
                                HostSummaryRow(host: host, state: poller.states[host.id])
                            }
                        }
                        .onDelete(perform: deleteHosts)
                    }
                }
            }
            .navigationTitle("GPU Fans")
            .navigationDestination(for: Host.self) { host in
                HostDetailView(hostID: host.id)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingAddHost = true
                    } label: {
                        Image(systemName: "plus")
                    }
                    .accessibilityLabel("Add Host")
                }
            }
            .sheet(isPresented: $showingAddHost) {
                AddHostView()
                    .environmentObject(poller)
            }
        }
    }

    private func deleteHosts(at offsets: IndexSet) {
        for index in offsets {
            poller.removeHost(poller.hosts[index])
        }
    }
}

private struct HostSummaryRow: View {
    let host: Host
    let state: HostState?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(host.name)
                        .font(.headline)
                    Text("\(host.hostname):\(host.port)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(temperatureSummary)
                    .font(.system(size: 22, weight: .semibold, design: .rounded))
                    .foregroundStyle(summaryColor)
            }

            if let payload = state?.lastPayload, !payload.displayGPUs.isEmpty {
                HStack(spacing: 10) {
                    Text("\(payload.displayGPUs.count) GPU\(payload.displayGPUs.count == 1 ? "" : "s")")
                    if let maxUtil = payload.displayGPUs.compactMap(\.gpuUtilPct).max() {
                        Text("GPU \(maxUtil)%")
                    }
                    if let maxFan = payload.displayGPUs.compactMap(\.fanPct).max() {
                        Text("fan \(maxFan)%")
                    }
                    if state?.isStale == true {
                        Label("Stale", systemImage: "exclamationmark.triangle.fill")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            } else if let error = state?.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            } else {
                Text("Connecting...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 5)
    }

    private var temperatures: [Int] {
        state?.lastPayload?.displayGPUs.compactMap(\.vramTempC) ?? []
    }

    private var temperatureSummary: String {
        temperatures.isEmpty ? "--°" : temperatures.map { "\($0)°" }.joined(separator: " / ")
    }

    private var summaryColor: Color {
        guard let maxTemp = temperatures.max() else { return .secondary }
        return colorFor(temp: maxTemp)
    }
}

struct HostDetailView: View {
    @EnvironmentObject private var poller: StatusPoller
    let hostID: UUID

    @State private var showingCurveEditor = false
    @State private var showingPowerEditor = false

    private var host: Host? { poller.hosts.first { $0.id == hostID } }
    private var state: HostState? { poller.states[hostID] }
    private var payload: HostStatusPayload? { state?.lastPayload }
    private var isPinned: Bool { poller.liveActivities.pinnedHostID == hostID }

    var body: some View {
        List {
            if let payload, !payload.displayGPUs.isEmpty {
                Section {
                    ForEach(payload.displayGPUs) { gpu in
                        GPUStatusBlock(gpu: gpu, isStale: state?.isStale == true)
                    }
                }

                if let history = state?.utilHistory, history.count >= 2 {
                    Section("GPU Utilization") {
                        UtilSparkline(history: history)
                            .frame(height: 88)
                        Text("Most recent 60 samples")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let error = state?.lastError {
                Section {
                    ContentUnavailableView("Host Unreachable", systemImage: "wifi.exclamationmark", description: Text(error))
                }
            } else {
                Section {
                    HStack {
                        ProgressView()
                        Text("Connecting...")
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section {
                Button {
                    showingCurveEditor = true
                } label: {
                    Label("Edit Fan Curve", systemImage: "slider.horizontal.3")
                }

                Button {
                    showingPowerEditor = true
                } label: {
                    Label("Edit Power Limit", systemImage: "bolt.fill")
                }
                .disabled(payload?.displayGPUs.allSatisfy { $0.powerLimitSupported == false } == true)

                Button {
                    openObserver()
                } label: {
                    Label("Open Observer Dashboard", systemImage: "safari")
                }
                .disabled(host?.observerURL == nil)

                Button {
                    toggleLiveActivityPin()
                } label: {
                    Label(
                        isPinned ? "Unpin Live Activity" : "Pin Live Activity",
                        systemImage: isPinned ? "pin.slash" : "pin"
                    )
                }
            }
        }
        .navigationTitle(host?.name ?? "Host")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showingCurveEditor) {
            NavigationStack {
                CurveEditor(hostID: hostID)
                    .environmentObject(poller)
            }
        }
        .sheet(isPresented: $showingPowerEditor) {
            NavigationStack {
                PowerLimitEditor(hostID: hostID)
                    .environmentObject(poller)
            }
        }
    }

    private func openObserver() {
        guard let url = host?.observerURL else { return }
        UIApplication.shared.open(url)
    }

    private func toggleLiveActivityPin() {
        if isPinned {
            poller.liveActivities.setPinnedHost(nil)
        } else if let host {
            poller.liveActivities.setPinnedHost(host)
        }
    }
}

struct AddHostView: View {
    @EnvironmentObject private var poller: StatusPoller
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var hostname = ""
    @State private var port = "8765"
    @State private var token = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Display name", text: $name)
                    TextField("Hostname or IP", text: $hostname)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    TextField("Port", text: $port)
                        .keyboardType(.numberPad)
                    SecureField("Bearer token", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
            }
            .navigationTitle("Add Host")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        addHost()
                    }
                    .disabled(!canAddHost)
                }
            }
        }
    }

    private var cleanedHostname: String {
        hostname.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var cleanedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var cleanedToken: String {
        token.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var portValue: Int? {
        Int(port.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private var canAddHost: Bool {
        guard !cleanedHostname.isEmpty, let portValue else { return false }
        return (1...65535).contains(portValue)
    }

    private func addHost() {
        guard let portValue, canAddHost else { return }
        poller.addHost(Host(
            name: cleanedName.isEmpty ? cleanedHostname : cleanedName,
            hostname: cleanedHostname,
            port: portValue,
            token: cleanedToken
        ))
        dismiss()
    }
}
