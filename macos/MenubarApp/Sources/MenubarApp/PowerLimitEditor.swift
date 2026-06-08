import SwiftUI

/// Per-host GPU board power-limit editor.
struct PowerLimitEditor: View {
    let hostID: UUID
    @EnvironmentObject var poller: StatusPoller

    @State private var localLimitW: Double = 0
    @State private var statusMessage: String? = nil
    @State private var statusIsError: Bool = false
    @State private var isApplying: Bool = false

    private var host: Host? { poller.hosts.first { $0.id == hostID } }
    private var payload: HostStatusPayload? { poller.states[hostID]?.lastPayload }
    private var minW: Double { payload?.powerLimitMinW ?? 0 }
    private var maxW: Double { payload?.powerLimitMaxW ?? max(minW, localLimitW) }
    private var tdpW: Double? { payload?.tdpW ?? payload?.powerLimitDefaultW }
    private var currentLimitW: Double? { payload?.powerLimitW }
    private var currentPowerW: Double? { payload?.powerW }
    private var canEdit: Bool {
        payload?.powerLimitSupported == true && minW > 0 && maxW > minW
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            if canEdit {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Limit")
                        Slider(value: $localLimitW, in: minW...maxW, step: 1)
                        TextField("watts", value: $localLimitW, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 72)
                        Text("W").foregroundColor(.secondary)
                    }

                    HStack {
                        Text("\(Int(minW.rounded()))W")
                        Spacer()
                        if let tdpW {
                            Text("TDP \(Int(tdpW.rounded()))W")
                        }
                        Spacer()
                        Text("\(Int(maxW.rounded()))W")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                }
            } else {
                Text("Power limit control unavailable")
                    .foregroundColor(.secondary)
            }

            HStack {
                Button("Reload") {
                    loadFromHost()
                }
                .disabled(isApplying)

                Spacer()

                Button("Default") {
                    apply(nil)
                }
                .disabled(isApplying || !canEdit)

                Button(isApplying ? "Applying…" : "Apply") {
                    apply(clampedLimit)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isApplying || !canEdit)
            }

            if let s = statusMessage {
                Text(s)
                    .font(.caption)
                    .foregroundColor(statusIsError ? .red : .secondary)
            }
        }
        .padding(16)
        .frame(width: 420)
        .onAppear(perform: loadFromHost)
        .onChange(of: currentLimitW) { _ in
            if !isApplying {
                loadFromHost()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(host?.name ?? "—").font(.headline)
                if let gpu = payload?.gpuName {
                    Text(gpu).font(.caption).foregroundColor(.secondary)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let currentPowerW {
                    Text("\(Int(currentPowerW.rounded()))W")
                        .font(.system(size: 18, weight: .semibold, design: .rounded))
                }
                if let currentLimitW {
                    Text("cap \(Int(currentLimitW.rounded()))W")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private var clampedLimit: Double {
        min(max(localLimitW.rounded(), minW), maxW)
    }

    private func loadFromHost() {
        if let currentLimitW {
            localLimitW = currentLimitW
        } else if let tdpW {
            localLimitW = tdpW
        }
        statusMessage = nil
        statusIsError = false
    }

    private func apply(_ watts: Double?) {
        guard let host = host else { return }
        isApplying = true
        statusMessage = nil
        Task {
            do {
                try await poller.putPowerLimit(host: host, watts: watts)
                statusMessage = "Applied at \(timeString())"
                statusIsError = false
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
                statusIsError = true
            }
            isApplying = false
        }
    }

    private func timeString() -> String {
        let df = DateFormatter()
        df.timeStyle = .medium
        return df.string(from: Date())
    }
}
