import ActivityKit
import SwiftUI
import WidgetKit

struct GPUStatusLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: GPUStatusActivityAttributes.self) { context in
            LockScreenContentView(hostName: context.attributes.hostName, state: context.state)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Text(tempText(context.state.vramTempC))
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(tempColor(context.state.vramTempC))
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text(fanText(context.state.fanPct))
                        .font(.title2.weight(.semibold))
                }
                DynamicIslandExpandedRegion(.bottom) {
                    HStack {
                        Text(context.attributes.hostName)
                        Spacer()
                        Text(context.state.updatedAt, style: .relative)
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                }
            } compactLeading: {
                Text(tempText(context.state.vramTempC))
                    .foregroundStyle(tempColor(context.state.vramTempC))
            } compactTrailing: {
                Text(fanText(context.state.fanPct))
            } minimal: {
                Text(tempText(context.state.vramTempC))
                    .foregroundStyle(tempColor(context.state.vramTempC))
            }
        }
    }
}

private struct LockScreenContentView: View {
    let hostName: String
    let state: GPUStatusActivityAttributes.ContentState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(hostName)
                .font(.headline)

            Text(state.updatedAt, style: .relative)
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("VRAM Temp")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(tempText(state.vramTempC))
                        .font(.system(size: 28, weight: .semibold, design: .rounded))
                        .foregroundStyle(tempColor(state.vramTempC))
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text("Fan")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(fanText(state.fanPct))
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text("GPU")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(utilText(state.gpuUtilPct))
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text("Power")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(powerText(state.powerW))
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                }
            }
        }
        .padding()
    }
}

private func tempText(_ temp: Int?) -> String {
    guard let temp else { return "--" }
    return "\(temp)°"
}

private func tempColor(_ temp: Int?) -> Color {
    guard let temp else { return .secondary }
    return colorFor(temp: temp)
}

private func fanText(_ fan: Int?) -> String {
    guard let fan else { return "--" }
    return "\(fan)%"
}

private func utilText(_ util: Int?) -> String {
    guard let util else { return "--" }
    return "\(util)%"
}

private func powerText(_ power: Double?) -> String {
    guard let power else { return "--" }
    return "\(Int(power.rounded())) W"
}
