import SwiftUI

/// Per-host GPU board power-limit editor.
struct PowerLimitEditor: View {
    let hostID: UUID
    @EnvironmentObject var poller: StatusPoller

    @State private var localLimitW: Double = 0
    @State private var statusMessage: String? = nil
    @State private var statusIsError: Bool = false
    @State private var isApplying: Bool = false
    @State private var selectedGPUIndex: Int? = nil

    private var host: Host? { poller.hosts.first { $0.id == hostID } }
    private var payload: HostStatusPayload? { poller.states[hostID]?.lastPayload }
    private var availableGPUs: [GPUStatusPayload] { payload?.displayGPUs ?? [] }
    private var selectedGPU: GPUStatusPayload? {
        if let selectedGPUIndex,
           let gpu = availableGPUs.first(where: { $0.index == selectedGPUIndex }) {
            return gpu
        }
        return availableGPUs.first
    }
    private var minW: Double { selectedGPU?.powerLimitMinW ?? payload?.powerLimitMinW ?? 0 }
    private var maxW: Double { selectedGPU?.powerLimitMaxW ?? payload?.powerLimitMaxW ?? max(minW, localLimitW) }
    private var tdpW: Double? { selectedGPU?.powerLimitDefaultW ?? payload?.tdpW ?? payload?.powerLimitDefaultW }
    private var currentLimitW: Double? { selectedGPU?.powerLimitW ?? payload?.powerLimitW }
    private var currentPowerW: Double? { selectedGPU?.powerW ?? payload?.powerW }
    private var canEdit: Bool {
        (selectedGPU?.powerLimitSupported ?? payload?.powerLimitSupported) == true &&
        minW > 0 &&
        maxW > minW
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            gpuPicker

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
        .onAppear {
            syncSelectedGPU()
            loadFromHost()
        }
        .onChange(of: currentLimitW) { _ in
            if !isApplying {
                loadFromHost()
            }
        }
        .onChange(of: selectedGPUIndex) { _ in
            if !isApplying {
                loadFromHost()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(host?.name ?? "—").font(.headline)
                if let gpu = selectedGPU?.name ?? payload?.gpuName {
                    Text(gpu)
                        .font(.caption)
                        .foregroundColor(.secondary)
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

    private var gpuPicker: some View {
        Group {
            if availableGPUs.count > 1 {
                Picker("GPU", selection: selectedGPUBinding) {
                    ForEach(availableGPUs) { gpu in
                        Text("GPU \(gpu.index)").tag(gpu.index)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
    }

    private var selectedGPUBinding: Binding<Int> {
        Binding(
            get: { selectedGPU?.index ?? availableGPUs.first?.index ?? 0 },
            set: { selectedGPUIndex = $0 }
        )
    }

    private var clampedLimit: Double {
        min(max(localLimitW.rounded(), minW), maxW)
    }

    private func syncSelectedGPU() {
        guard selectedGPUIndex == nil else { return }
        selectedGPUIndex = availableGPUs.first?.index
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
        let gpuIndex = payload?.gpus?.isEmpty == false ? selectedGPU?.index : nil
        isApplying = true
        statusMessage = nil
        Task {
            do {
                try await poller.putPowerLimit(host: host, watts: watts, gpuIndex: gpuIndex)
                let target = selectedGPU.map { "GPU \($0.index)" } ?? "GPU"
                statusMessage = "Applied to \(target) at \(timeString())"
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
