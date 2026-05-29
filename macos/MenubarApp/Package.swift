// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MenubarApp",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "MenubarApp",
            // SwiftUI @main in a SwiftPM executable needs this flag so the
            // compiler treats the @main type as a library entry point rather
            // than expecting a top-level main.swift.
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        ),
    ]
)
