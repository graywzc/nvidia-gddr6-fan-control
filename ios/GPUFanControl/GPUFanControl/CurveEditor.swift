import SwiftUI

struct CurveEditor: View {
    let hostID: UUID
    @EnvironmentObject private var poller: StatusPoller
    @Environment(\.dismiss) private var dismiss

    @State private var localCurve: [[Int]] = []
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
    private var currentTemp: Int? { selectedGPU?.vramTempC ?? payload?.vramTempC }
    private var currentFanPct: Int? { selectedGPU?.fanPct ?? payload?.fanPct }
    private var currentCurve: [[Double]]? { selectedGPU?.curve ?? payload?.curve }

    var body: some View {
        Form {
            Section {
                header
                gpuPicker
                if localCurve.isEmpty {
                    Text("Waiting for a fan curve from the host.")
                        .foregroundStyle(.secondary)
                } else {
                    CurveChart(points: localCurve, currentTemp: currentTemp)
                        .frame(height: 180)
                        .listRowInsets(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 12))
                }
            }

            Section("Waypoints") {
                if localCurve.isEmpty {
                    Text("No editable curve is available yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(localCurve.indices, id: \.self) { index in
                        WaypointRow(
                            index: index,
                            temp: Binding(
                                get: { localCurve[index][0] },
                                set: { localCurve[index][0] = clamp($0, 0, 150) }
                            ),
                            fanPct: Binding(
                                get: { localCurve[index][1] },
                                set: { localCurve[index][1] = clamp($0, 0, 100) }
                            ),
                            canRemove: localCurve.count > 2,
                            isActiveSegment: isActiveIndex(index),
                            onRemove: { localCurve.remove(at: index) }
                        )
                    }
                }

                Button {
                    addPoint()
                } label: {
                    Label("Add Point", systemImage: "plus.circle")
                }
            }

            if let statusMessage {
                Section {
                    Text(statusMessage)
                        .foregroundStyle(statusIsError ? .red : .secondary)
                }
            }
        }
        .navigationTitle("Fan Curve")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Done") {
                    dismiss()
                }
            }
            ToolbarItemGroup(placement: .bottomBar) {
                Button("Revert") {
                    loadFromHost()
                }
                .disabled(isApplying)
                Spacer()
                Button(isApplying ? "Applying..." : "Apply") {
                    apply()
                }
                .disabled(isApplying || localCurve.count < 2)
            }
        }
        .onAppear {
            syncSelectedGPU()
            loadFromHost()
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
            if let temp = currentTemp {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(temp)°C")
                        .font(.system(size: 24, weight: .semibold, design: .rounded))
                        .foregroundStyle(colorFor(temp: temp))
                    if let fan = currentFanPct {
                        Text("fan \(fan)%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
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

    private func syncSelectedGPU() {
        guard selectedGPUIndex == nil else { return }
        selectedGPUIndex = availableGPUs.first?.index
    }

    private func loadFromHost() {
        guard let live = currentCurve else {
            localCurve = []
            statusMessage = nil
            statusIsError = false
            return
        }
        localCurve = live.map { [Int($0[0].rounded()), Int($0[1].rounded())] }
        statusMessage = nil
        statusIsError = false
    }

    private func addPoint() {
        let lastTemp = localCurve.last?[0] ?? 60
        let lastPct = localCurve.last?[1] ?? 50
        let newTemp = min(lastTemp + 5, 150)
        let newPct = min(lastPct + 10, 100)
        guard newTemp > lastTemp else {
            statusMessage = "Cannot extend beyond 150°C"
            statusIsError = true
            return
        }
        localCurve.append([newTemp, newPct])
    }

    private func apply() {
        let sorted = localCurve.sorted { $0[0] < $1[0] }
        for index in 1..<sorted.count where sorted[index][0] == sorted[index - 1][0] {
            statusMessage = "Duplicate temperature \(sorted[index][0])°C"
            statusIsError = true
            return
        }

        guard let host else { return }
        let gpuIndex = payload?.gpus?.isEmpty == false ? selectedGPU?.index : nil
        localCurve = sorted
        isApplying = true
        statusMessage = nil

        Task {
            do {
                try await poller.putCurve(host: host, curve: sorted, gpuIndex: gpuIndex)
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

    private func isActiveIndex(_ index: Int) -> Bool {
        guard let temp = currentTemp, index < localCurve.count else { return false }
        let lower = localCurve[index][0]
        let upper = index + 1 < localCurve.count ? localCurve[index + 1][0] : Int.max
        return temp >= lower && temp < upper
    }

    private func timeString() -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter.string(from: Date())
    }

    private func clamp(_ value: Int, _ low: Int, _ high: Int) -> Int {
        max(low, min(high, value))
    }
}

private struct WaypointRow: View {
    let index: Int
    @Binding var temp: Int
    @Binding var fanPct: Int
    let canRemove: Bool
    let isActiveSegment: Bool
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: isActiveSegment ? "play.fill" : "circle")
                    .foregroundStyle(isActiveSegment ? Color.accentColor : Color.secondary.opacity(0.5))
                    .frame(width: 16)
                Text("Point \(index + 1)")
                    .font(.headline)
                Spacer()
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "minus.circle")
                }
                .disabled(!canRemove)
            }

            Stepper(value: $fanPct, in: 0...100) {
                LabeledContent("Fan", value: "\(fanPct)%")
                    .monospacedDigit()
            }

            Stepper(value: $temp, in: 0...150) {
                LabeledContent("Temp", value: "\(temp)°C")
                    .monospacedDigit()
            }
        }
        .listRowBackground(isActiveSegment ? Color.accentColor.opacity(0.12) : nil)
    }
}

private struct CurveChart: View {
    let points: [[Int]]
    let currentTemp: Int?

    private let pad: CGFloat = 16
    private let tempMin: Double = 30
    private let tempMax: Double = 110

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                gridLayer(size: geometry.size)
                if points.count >= 2 {
                    curveLayer(size: geometry.size)
                    dotsLayer(size: geometry.size)
                }
                if let currentTemp {
                    currentTempLayer(temp: currentTemp, size: geometry.size)
                }
                axisLabels
            }
        }
    }

    private func xFor(_ temp: Double, width: CGFloat) -> CGFloat {
        let fraction = (temp - tempMin) / (tempMax - tempMin)
        return pad + CGFloat(fraction) * (width - 2 * pad)
    }

    private func yFor(_ fanPct: Double, height: CGFloat) -> CGFloat {
        let fraction = fanPct / 100
        return (height - pad) - CGFloat(fraction) * (height - 2 * pad)
    }

    private func gridLayer(size: CGSize) -> some View {
        Path { path in
            for value in stride(from: 0, through: 100, by: 25) {
                let y = yFor(Double(value), height: size.height)
                path.move(to: CGPoint(x: pad, y: y))
                path.addLine(to: CGPoint(x: size.width - pad, y: y))
            }
        }
        .stroke(Color.secondary.opacity(0.16), lineWidth: 0.5)
    }

    private func curveLayer(size: CGSize) -> some View {
        Path { path in
            for (index, point) in points.enumerated() {
                let chartPoint = CGPoint(
                    x: xFor(Double(point[0]), width: size.width),
                    y: yFor(Double(point[1]), height: size.height)
                )
                if index == 0 {
                    path.move(to: chartPoint)
                } else {
                    path.addLine(to: chartPoint)
                }
            }
        }
        .stroke(Color.accentColor, lineWidth: 2.5)
    }

    private func dotsLayer(size: CGSize) -> some View {
        ForEach(points.indices, id: \.self) { index in
            Circle()
                .fill(Color.accentColor)
                .frame(width: 8, height: 8)
                .position(
                    x: xFor(Double(points[index][0]), width: size.width),
                    y: yFor(Double(points[index][1]), height: size.height)
                )
        }
    }

    @ViewBuilder
    private func currentTempLayer(temp: Int, size: CGSize) -> some View {
        let value = Double(temp)
        if value >= tempMin && value <= tempMax {
            let x = xFor(value, width: size.width)
            Path { path in
                path.move(to: CGPoint(x: x, y: pad))
                path.addLine(to: CGPoint(x: x, y: size.height - pad))
            }
            .stroke(
                colorFor(temp: temp).opacity(0.8),
                style: StrokeStyle(lineWidth: 1.5, dash: [4, 4])
            )
        }
    }

    private var axisLabels: some View {
        VStack {
            Spacer()
            HStack {
                Text("\(Int(tempMin))°")
                Spacer()
                Text("\(Int(tempMax))°")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.horizontal, pad)
        }
    }
}
