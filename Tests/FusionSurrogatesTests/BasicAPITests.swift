import Testing
@testable import FusionSurrogates

/// Basic API tests without MLX or Python dependencies
@Suite("Basic API Tests")
struct BasicAPITests {

    @Test("QLKNN input parameter names")
    func inputParameterNames() {
        let names = QLKNN.inputParameterNames
        #expect(names.count == 10)
        #expect(names.contains("R_L_Te"))
        #expect(names.contains("R_L_Ti"))
        #expect(names.contains("R_L_ne"))
        #expect(names.contains("R_L_ni"))
        #expect(names.contains("q"))
        #expect(names.contains("s_hat"))
        #expect(names.contains("r_R"))
        #expect(names.contains("Ti_Te"))
        #expect(names.contains("log_nu_star"))
        #expect(names.contains("ni_ne"))
    }

    @Test("QLKNN output parameter names")
    func outputParameterNames() {
        let names = QLKNN.outputParameterNames
        #expect(names.count >= 4)
        #expect(names.contains("chi_ion_itg"))
        #expect(names.contains("chi_electron_tem"))
        #expect(names.contains("chi_electron_etg"))
        #expect(names.contains("particle_flux"))
    }

    @Test("FusionSurrogatesError descriptions")
    func errorDescriptions() {
        let error1 = FusionSurrogatesError.pythonImportFailed("test_module")
        #expect(error1.localizedDescription.contains("test_module"))

        let error2 = FusionSurrogatesError.unsupportedModelVersion("9_99")
        #expect(error2.localizedDescription.contains("9_99"))

        let error3 = FusionSurrogatesError.predictionFailed("test reason")
        #expect(error3.localizedDescription.contains("test reason"))
    }
}
