import Testing
@testable import FusionSurrogates

/// Basic API tests without MLX or Python dependencies
@Suite("Basic API Tests")
struct BasicAPITests {

    @Test("QLKNN input parameter names")
    func inputParameterNames() {
        let names = QLKNN.inputParameterNames
        #expect(names.count == 10)
        #expect(names.contains("Ati"))        // R/L_Ti
        #expect(names.contains("Ate"))        // R/L_Te
        #expect(names.contains("Ane"))        // R/L_ne
        #expect(names.contains("Ani"))        // R/L_ni
        #expect(names.contains("q"))          // Safety factor
        #expect(names.contains("smag"))       // Magnetic shear
        #expect(names.contains("x"))          // r/R
        #expect(names.contains("Ti_Te"))      // Temperature ratio
        #expect(names.contains("LogNuStar"))  // Collisionality
        #expect(names.contains("normni"))     // ni/ne
    }

    @Test("QLKNN output parameter names")
    func outputParameterNames() {
        let names = QLKNN.outputParameterNames
        #expect(names.count == 8)
        #expect(names.contains("efiITG"))     // Ion thermal flux (ITG)
        #expect(names.contains("efeITG"))     // Electron thermal flux (ITG)
        #expect(names.contains("efeTEM"))     // Electron thermal flux (TEM)
        #expect(names.contains("efeETG"))     // Electron thermal flux (ETG)
        #expect(names.contains("efiTEM"))     // Ion thermal flux (TEM)
        #expect(names.contains("pfeITG"))     // Particle flux (ITG)
        #expect(names.contains("pfeTEM"))     // Particle flux (TEM)
        #expect(names.contains("gamma_max"))  // Growth rate
    }

    @Test("FusionSurrogatesError descriptions")
    func errorDescriptions() {
        let error1 = FusionSurrogatesError.predictionFailed("test reason")
        #expect(error1.localizedDescription.contains("test reason"))

        let error2 = FusionSurrogatesError.invalidInput("invalid parameter")
        #expect(error2.localizedDescription.contains("invalid parameter"))
    }
}
