import XCTest
@testable import GPUFanControl

final class GPUFanControlTests: XCTestCase {
    func testHostBuildsStatusAndObserverURLs() {
        let host = Host(name: "aipc1", hostname: "aipc1", port: 8765, token: "")

        XCTAssertEqual(host.statusURL?.absoluteString, "http://aipc1:8765/status")
        XCTAssertEqual(host.observerURL?.absoluteString, "http://aipc1:8765/observer")
    }

    func testLegacyStatusPayloadMirrorsPrimaryGPU() {
        let payload = HostStatusPayload(
            vramTempC: 88,
            powerW: 215.4,
            powerLimitW: 250,
            powerLimitMinW: 100,
            powerLimitMaxW: 450,
            powerLimitDefaultW: 350,
            tdpW: 350,
            powerLimitSupported: true,
            gpuUtilPct: 62,
            fanPct: 74,
            gpuName: "RTX 3090",
            numFans: 2,
            gpus: nil,
            curve: [[60, 40], [80, 55]],
            updatedAt: 123,
            wallTime: 456,
            dryRun: false
        )

        XCTAssertEqual(payload.displayGPUs.count, 1)
        XCTAssertEqual(payload.displayGPUs[0].index, 0)
        XCTAssertEqual(payload.displayGPUs[0].name, "RTX 3090")
        XCTAssertEqual(payload.displayGPUs[0].vramTempC, 88)
        XCTAssertEqual(payload.displayGPUs[0].fanPct, 74)
        XCTAssertEqual(payload.displayGPUs[0].powerLimitSupported, true)
    }

    func testMultiGPUStatusPayloadSortsDisplayOrder() {
        let payload = HostStatusPayload(
            vramTempC: nil,
            powerW: nil,
            powerLimitW: nil,
            powerLimitMinW: nil,
            powerLimitMaxW: nil,
            powerLimitDefaultW: nil,
            tdpW: nil,
            powerLimitSupported: nil,
            gpuUtilPct: nil,
            fanPct: nil,
            gpuName: nil,
            numFans: nil,
            gpus: [
                makeGPU(index: 2, temp: 91),
                makeGPU(index: 0, temp: 84),
                makeGPU(index: 1, temp: 87),
            ],
            curve: nil,
            updatedAt: 123,
            wallTime: 456,
            dryRun: false
        )

        XCTAssertEqual(payload.displayGPUs.map(\.index), [0, 1, 2])
        XCTAssertEqual(payload.displayGPUs.map(\.vramTempC), [84, 87, 91])
    }

    private func makeGPU(index: Int, temp: Int) -> GPUStatusPayload {
        GPUStatusPayload(
            index: index,
            name: "GPU \(index)",
            vramTempC: temp,
            fanPct: nil,
            numFans: nil,
            powerW: nil,
            gpuUtilPct: nil,
            curve: nil,
            powerLimitW: nil,
            powerLimitMinW: nil,
            powerLimitMaxW: nil,
            powerLimitDefaultW: nil,
            powerLimitSupported: nil
        )
    }
}
