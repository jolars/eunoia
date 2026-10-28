//! WASM optimizer names and compatibility mappings.

// wasm-bindgen generates conversions that reference every exported variant.
#![allow(deprecated)]

use eunoia::Optimizer;
use wasm_bindgen::prelude::*;

/// Optimizer options for WASM
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmOptimizer {
    /// @deprecated Use `CmaEsBoundedLm` instead.
    #[deprecated(note = "Use `CmaEsBoundedLm` instead.")]
    CmaEsTrf = 0,
    CmaEsLm = 1,
    CmaEs = 2,
    LevenbergMarquardt = 3,
    /// @deprecated Use `BoundedLevenbergMarquardt` instead.
    #[deprecated(note = "Use `BoundedLevenbergMarquardt` instead.")]
    Trf = 4,
    Lbfgs = 5,
    NelderMead = 6,
    /// CMA-ES escape with bounded Levenberg-Marquardt retries and polishing.
    CmaEsBoundedLm = 7,
    /// Levenberg-Marquardt with Coleman-Li box scaling.
    BoundedLevenbergMarquardt = 8,
}

// Legacy numeric values remain valid for existing JavaScript callers.
impl From<WasmOptimizer> for Optimizer {
    fn from(opt: WasmOptimizer) -> Self {
        match opt {
            WasmOptimizer::CmaEsBoundedLm | WasmOptimizer::CmaEsTrf => Optimizer::CmaEsBoundedLm,
            WasmOptimizer::CmaEsLm => Optimizer::CmaEsLm,
            WasmOptimizer::CmaEs => Optimizer::CmaEs,
            WasmOptimizer::LevenbergMarquardt => Optimizer::LevenbergMarquardt,
            WasmOptimizer::BoundedLevenbergMarquardt | WasmOptimizer::Trf => {
                Optimizer::BoundedLevenbergMarquardt
            }
            WasmOptimizer::Lbfgs => Optimizer::Lbfgs,
            WasmOptimizer::NelderMead => Optimizer::NelderMead,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_aliases_preserve_numeric_values_and_mappings() {
        for (variant, number, expected) in [
            (WasmOptimizer::CmaEsTrf, 0, Optimizer::CmaEsBoundedLm),
            (WasmOptimizer::CmaEsLm, 1, Optimizer::CmaEsLm),
            (WasmOptimizer::CmaEs, 2, Optimizer::CmaEs),
            (
                WasmOptimizer::LevenbergMarquardt,
                3,
                Optimizer::LevenbergMarquardt,
            ),
            (WasmOptimizer::Trf, 4, Optimizer::BoundedLevenbergMarquardt),
            (WasmOptimizer::Lbfgs, 5, Optimizer::Lbfgs),
            (WasmOptimizer::NelderMead, 6, Optimizer::NelderMead),
            (WasmOptimizer::CmaEsBoundedLm, 7, Optimizer::CmaEsBoundedLm),
            (
                WasmOptimizer::BoundedLevenbergMarquardt,
                8,
                Optimizer::BoundedLevenbergMarquardt,
            ),
        ] {
            assert_eq!(variant as u32, number);
            assert_eq!(Optimizer::from(variant), expected);
        }
    }
}
