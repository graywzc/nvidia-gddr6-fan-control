import SwiftUI

func colorFor(temp: Int) -> Color {
    switch temp {
    case ..<85: return .green
    case 85..<95: return .yellow
    default: return .red
    }
}

struct UtilSparkline: View {
    let history: [Int]
    private let pad: CGFloat = 4

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            ZStack {
                Path { path in
                    for value in stride(from: 0, through: 100, by: 50) {
                        let y = yFor(value, height: height)
                        path.move(to: CGPoint(x: pad, y: y))
                        path.addLine(to: CGPoint(x: width - pad, y: y))
                    }
                }
                .stroke(Color.secondary.opacity(0.18), lineWidth: 0.5)

                Path { path in
                    for (index, value) in history.enumerated() {
                        let point = CGPoint(
                            x: xFor(index, width: width),
                            y: yFor(value, height: height)
                        )
                        if index == 0 {
                            path.move(to: point)
                        } else {
                            path.addLine(to: point)
                        }
                    }
                }
                .stroke(Color.accentColor, lineWidth: 2)
            }
        }
    }

    private func xFor(_ index: Int, width: CGFloat) -> CGFloat {
        guard history.count > 1 else { return pad }
        let fraction = CGFloat(index) / CGFloat(history.count - 1)
        return pad + fraction * (width - 2 * pad)
    }

    private func yFor(_ value: Int, height: CGFloat) -> CGFloat {
        let fraction = CGFloat(value) / 100
        return (height - pad) - fraction * (height - 2 * pad)
    }
}

struct MetricPill: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8))
    }
}

struct GPUStatusBlock: View {
    let gpu: GPUStatusPayload
    let isStale: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(tempText)
                    .font(.system(size: 34, weight: .semibold, design: .rounded))
                    .foregroundStyle(tempColor)
                    .lineLimit(1)
                VStack(alignment: .leading, spacing: 2) {
                    Text("GPU \(gpu.index)")
                        .font(.headline)
                    Text(gpu.name ?? "NVIDIA GPU")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                if isStale {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                        .accessibilityLabel("Stale data")
                }
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), spacing: 8)], spacing: 8) {
                if let fan = gpu.fanPct {
                    MetricPill(title: "Fan", value: "\(fan)%")
                }
                if let power = gpu.powerW {
                    MetricPill(title: "Power", value: "\(Int(power.rounded())) W")
                }
                if let limit = gpu.powerLimitW {
                    MetricPill(title: "Cap", value: "\(Int(limit.rounded())) W")
                }
                if let util = gpu.gpuUtilPct {
                    MetricPill(title: "GPU", value: "\(util)%")
                }
            }
        }
        .padding(.vertical, 8)
    }

    private var tempText: String {
        guard let temp = gpu.vramTempC else { return "--°" }
        return "\(temp)°"
    }

    private var tempColor: Color {
        guard let temp = gpu.vramTempC else { return .secondary }
        return colorFor(temp: temp)
    }
}
