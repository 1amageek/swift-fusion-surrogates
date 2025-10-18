import Testing
@testable import FusionSurrogates

@Suite("Example Tests")
struct ExampleTests {
    @Test("Basic example test")
    func example() async throws {
        // Verify package is importable
        let inputNames = QLKNN.inputParameterNames
        #expect(inputNames.count == 10)
    }
}
