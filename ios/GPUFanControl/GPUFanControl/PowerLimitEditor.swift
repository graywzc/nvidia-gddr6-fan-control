import SwiftUI

struct PowerLimitEditor: View {
    let hostID: UUID
    @EnvironmentObject private var poller: StatusPoller
    @Environment(\.dismiss) private var dismiss

    @State private var localLimitW: Double = 0
    @State private var statusMessage: String?
    @State private var statusIsError = false
    @State private var isApplying = false
    @State private var selectedGPUIndex: Int?

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
        Form {
            Section {
                header
                gpuPicker
            }

            Section("Limit") {
                if canEdit {
                    Slider(value: $localLimitW, in: minW...maxW, step: 1)
                    HStack {
                        Text("\(Int(minW.rounded())) W")
                        Spacer()
                        Text("\(Int(clampedLimit.rounded())) W")
                            .font(.title3.weight(.semibold))
                            .monospacedDigit()
                        Spacer()
                        Text("\(Int(maxW.rounded())) W")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    if let tdpW {
                        LabeledContent("Default", value: "\(Int(tdpW.rounded())) W")
                    }
                } else {
                    Text("Power limit control unavailable for the selected GPU.")
                        .foregroundStyle(.secondary)
                }
            }

            if let statusMessage {
                Section {
                    Text(statusMessage)
                        .foregroundStyle(statusIsError ? .red : .secondary)
                }
            }
        }
        .navigationTitle("Power Limit")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Done") {
                    dismiss()
                }
            }
            ToolbarItemGroup(placement: .bottomBar) {
                Button("Reload") {
                    loadFromHost()
                }
                .disabled(isApplying)
                Spacer()
                Button("Default") {
                    apply(nil)
                }
                .disabled(isApplying || !canEdit)
                Button(isApplying ? "Applying..." : "Apply") {
                    apply(clampedLimit)
                }
                .disabled(isApplying || !canEdit)
            }
        }
        .onAppear {
            syncSelectedGPU()
            loadFromHost()
        }
        .onChange(of: currentLimitW) {
            if !isApplying {
                loadFromHost()
            }
        }
        .onChange(of: selectedGPUIndex) {
            if !isApplying {
                loadFromHost()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text(host?.name ?? "Host")
                    .font(.headline)
                if let name = selectedGPU?.name ?? payload?.gpuName {
                    Text(name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let currentPowerW {
                    Text("\(Int(currentPowerW.rounded())) W")
                        .font(.title3.weight(.semibold))
                        .monospacedDigit()
                }
                if let currentLimitW {
                    Text("cap \(Int(currentLimitW.rounded())) W")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var gpuPicker: some View {
        if availableGPUs.count > 1 {
            Picker("GPU", selection: selectedGPUBinding) {
                ForEach(availableGPUs) { gpu in
                    Text("GPU \(gpu.index)").tag(gpu.index)
                }
            }
            .pickerStyle(.segmented)
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
        guard let host else { return }
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
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter.string(from: Date())
    }
}
