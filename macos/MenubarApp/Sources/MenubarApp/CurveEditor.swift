import SwiftUI

/// Per-host fan curve editor. Loads the live curve from /status, lets the
/// user edit waypoints, and pushes the result via PUT /curve.
struct CurveEditor: View {
    let hostID: UUID
    @EnvironmentObject var poller: StatusPoller

    @State private var localCurve: [[Int]] = []
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
    private var currentTemp: Int? { selectedGPU?.vramTempC ?? payload?.vramTempC }
    private var currentFanPct: Int? { selectedGPU?.fanPct ?? payload?.fanPct }
    private var currentCurve: [[Double]]? { selectedGPU?.curve ?? payload?.curve }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            gpuPicker

            CurveChart(points: localCurve, currentTemp: currentTemp)
                .frame(height: 140)
                .padding(.horizontal, 2)

            Divider()

            // Waypoint editor
            ScrollView {
                VStack(spacing: 6) {
                    ForEach(localCurve.indices, id: \.self) { i in
                        WaypointRow(
                            index: i,
                            temp: Binding(
                                get: { localCurve[i][0] },
                                set: { localCurve[i][0] = clamp($0, 0, 150) }
                            ),
                            fanPct: Binding(
                                get: { localCurve[i][1] },
                                set: { localCurve[i][1] = clamp($0, 0, 100) }
                            ),
                            canRemove: localCurve.count > 2,
                            isActiveSegment: isActiveIndex(i),
                            onRemove: { localCurve.remove(at: i) }
                        )
                    }
                }
            }

            HStack {
                Button {
                    addPoint()
                } label: {
                    Label("Add point", systemImage: "plus.circle")
                }
                .buttonStyle(.borderless)

                Spacer()

                Button("Revert") {
                    loadFromHost()
                }
                .disabled(isApplying)

                Button(isApplying ? "Applying…" : "Apply") {
                    apply()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isApplying || localCurve.count < 2)
            }

            if let s = statusMessage {
                Text(s)
                    .font(.caption)
                    .foregroundColor(statusIsError ? .red : .secondary)
            }
        }
        .padding(16)
        .frame(minWidth: 460, minHeight: 460)
        .onAppear {
            syncSelectedGPU()
            loadFromHost()
        }
        .onChange(of: selectedGPUIndex) { _ in
            if !isApplying {
                loadFromHost()
            }
        }
    }

    // MARK: - Subviews

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(host?.name ?? "—").font(.headline)
                if let gpu = selectedGPU?.name ?? payload?.gpuName {
                    Text(gpu).font(.caption).foregroundColor(.secondary)
                }
            }
            Spacer()
            if let t = currentTemp {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(t)°C")
                        .font(.system(size: 18, weight: .semibold, design: .rounded))
                        .foregroundColor(colorFor(temp: t))
                    if let fp = currentFanPct {
                        Text("fan \(fp)%").font(.caption).foregroundColor(.secondary)
                    }
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

    // MARK: - Logic

    private func syncSelectedGPU() {
        guard selectedGPUIndex == nil else { return }
        selectedGPUIndex = availableGPUs.first?.index
    }

    private func loadFromHost() {
        guard let live = currentCurve else { return }
        localCurve = live.map { p in
            [Int(p[0].rounded()), Int(p[1].rounded())]
        }
        statusMessage = nil
        statusIsError = false
    }

    private func addPoint() {
        let lastTemp = localCurve.last?[0] ?? 60
        let lastPct = localCurve.last?[1] ?? 50
        let newTemp = min(lastTemp + 5, 150)
        let newPct = min(lastPct + 10, 100)
        if newTemp <= lastTemp {
            statusMessage = "Cannot extend beyond 150°C"
            statusIsError = true
            return
        }
        localCurve.append([newTemp, newPct])
    }

    private func apply() {
        if localCurve.count < 2 {
            statusMessage = "Curve must have at least 2 points"
            statusIsError = true
            return
        }
        // Sort by temp so the user can edit values freely and have rows
        // reorder on Apply. After sorting, the only way ascending order
        // can still fail is duplicate temps.
        let sorted = localCurve.sorted { $0[0] < $1[0] }
        for i in 1..<sorted.count where sorted[i][0] == sorted[i - 1][0] {
            statusMessage = "Duplicate temperature \(sorted[i][0])°C"
            statusIsError = true
            return
        }
        localCurve = sorted

        guard let host = host else { return }
        let gpuIndex = payload?.gpus?.isEmpty == false ? selectedGPU?.index : nil
        isApplying = true
        statusMessage = nil
        let curveCopy = sorted
        Task {
            do {
                try await poller.putCurve(host: host, curve: curveCopy, gpuIndex: gpuIndex)
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

    private func isActiveIndex(_ i: Int) -> Bool {
        // Highlight the segment whose lower bound is i: temps[i] <= current < temps[i+1].
        guard let t = currentTemp, i < localCurve.count else { return false }
        let lower = localCurve[i][0]
        let upper = i + 1 < localCurve.count ? localCurve[i + 1][0] : Int.max
        return t >= lower && t < upper
    }

    private func timeString() -> String {
        let df = DateFormatter()
        df.timeStyle = .medium
        return df.string(from: Date())
    }

    private func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(hi, v)) }
}

// MARK: - Waypoint row

private struct WaypointRow: View {
    let index: Int
    @Binding var temp: Int
    @Binding var fanPct: Int
    let canRemove: Bool
    let isActiveSegment: Bool
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: isActiveSegment ? "play.fill" : "circle")
                .foregroundColor(isActiveSegment ? .accentColor : .secondary.opacity(0.4))
                .frame(width: 14)

            Text("Point \(index + 1)").frame(width: 64, alignment: .leading)

            HStack(spacing: 2) {
                TextField("temp", value: $temp, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 56)
                Text("°C").foregroundColor(.secondary)
            }

            Text("→").foregroundColor(.secondary)

            HStack(spacing: 2) {
                TextField("fan", value: $fanPct, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 56)
                Text("%").foregroundColor(.secondary)
            }

            Spacer()

            Button(action: onRemove) {
                Image(systemName: "minus.circle")
                    .foregroundColor(canRemove ? .secondary : .secondary.opacity(0.3))
            }
            .buttonStyle(.borderless)
            .disabled(!canRemove)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(
            isActiveSegment
                ? Color.accentColor.opacity(0.12)
                : Color.clear
        )
        .cornerRadius(6)
    }
}

// MARK: - Chart

/// Simple line chart drawn with Path. (Swift Charts would also work; this
/// keeps the dependency surface minimal.)
private struct CurveChart: View {
    let points: [[Int]]
    let currentTemp: Int?

    // Domain: temps 30..110 (broad enough for typical curves), fan 0..100.
    private let pad: CGFloat = 12
    private let tempMin: Double = 30
    private let tempMax: Double = 110

    private func xFor(_ t: Double, width w: CGFloat) -> CGFloat {
        let frac = (t - tempMin) / (tempMax - tempMin)
        return pad + CGFloat(frac) * (w - 2 * pad)
    }

    private func yFor(_ p: Double, height h: CGFloat) -> CGFloat {
        // Flip: 0% at bottom, 100% at top.
        let frac = p / 100.0
        return (h - pad) - CGFloat(frac) * (h - 2 * pad)
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                gridLayer(size: geo.size)
                if points.count >= 2 {
                    curveLayer(size: geo.size)
                    dotsLayer(size: geo.size)
                }
                if let t = currentTemp {
                    currentTempLayer(temp: t, size: geo.size)
                }
                axisLabels
            }
        }
    }

    private func gridLayer(size: CGSize) -> some View {
        let w = size.width, h = size.height
        return Path { p in
            for f in stride(from: 0.0, through: 1.0, by: 0.25) {
                let y = yFor(f * 100, height: h)
                p.move(to: CGPoint(x: pad, y: y))
                p.addLine(to: CGPoint(x: w - pad, y: y))
            }
        }
        .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)
    }

    private func curveLayer(size: CGSize) -> some View {
        let w = size.width, h = size.height
        return Path { p in
            for (i, pt) in points.enumerated() {
                let x = xFor(Double(pt[0]), width: w)
                let y = yFor(Double(pt[1]), height: h)
                if i == 0 {
                    p.move(to: CGPoint(x: x, y: y))
                } else {
                    p.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }
        .stroke(Color.accentColor, lineWidth: 2)
    }

    private func dotsLayer(size: CGSize) -> some View {
        let w = size.width, h = size.height
        return ForEach(points.indices, id: \.self) { i in
            Circle()
                .fill(Color.accentColor)
                .frame(width: 7, height: 7)
                .position(
                    x: xFor(Double(points[i][0]), width: w),
                    y: yFor(Double(points[i][1]), height: h)
                )
        }
    }

    @ViewBuilder
    private func currentTempLayer(temp: Int, size: CGSize) -> some View {
        let td = Double(temp)
        if td >= tempMin && td <= tempMax {
            let w = size.width, h = size.height
            let x = xFor(td, width: w)
            Path { p in
                p.move(to: CGPoint(x: x, y: pad))
                p.addLine(to: CGPoint(x: x, y: h - pad))
            }
            .stroke(
                colorFor(temp: temp).opacity(0.8),
                style: StrokeStyle(lineWidth: 1.5, dash: [3, 3])
            )
        }
    }

    private var axisLabels: some View {
        VStack {
            Spacer()
            HStack {
                Text("\(Int(tempMin))°").font(.caption2).foregroundColor(.secondary)
                Spacer()
                Text("\(Int(tempMax))°").font(.caption2).foregroundColor(.secondary)
            }
            .padding(.horizontal, pad)
        }
    }
}
