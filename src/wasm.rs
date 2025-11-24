use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use console_error_panic_hook;

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

/// A circle representation for WASM with label
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmCircle {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    label: String,
}

#[wasm_bindgen]
impl WasmCircle {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, radius: f64, label: String) -> Self {
        Self {
            x,
            y,
            radius,
            label,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }
}

/// A diagram specification entry (set combination and size)
#[wasm_bindgen]
#[derive(Clone)]
pub struct DiagramSpec {
    input: String,
    size: f64,
}

#[wasm_bindgen]
impl DiagramSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(input: String, size: f64) -> Self {
        Self { input, size }
    }

    #[wasm_bindgen(getter)]
    pub fn input(&self) -> String {
        self.input.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> f64 {
        self.size
    }
}

/// Generate a simple test layout for debugging
#[wasm_bindgen]
pub fn generate_test_layout() -> Vec<WasmCircle> {
    vec![
        WasmCircle::new(100.0, 100.0, 50.0, "A".to_string()),
        WasmCircle::new(150.0, 100.0, 40.0, "B".to_string()),
        WasmCircle::new(125.0, 150.0, 30.0, "C".to_string()),
    ]
}

/// Compute initial layout (placeholder for now)
#[wasm_bindgen]
pub fn compute_layout(n_sets: usize) -> Vec<WasmCircle> {
    // Placeholder: return test circles
    (0..n_sets)
        .map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_sets as f64);
            let x = 200.0 + 100.0 * angle.cos();
            let y = 200.0 + 100.0 * angle.sin();
            let label = format!("{}", (b'A' + i as u8) as char);
            WasmCircle::new(x, y, 50.0, label)
        })
        .collect()
}

/// Generate layout from diagram specification
#[wasm_bindgen]
pub fn generate_from_spec(specs: Vec<DiagramSpec>) -> Result<Vec<WasmCircle>, JsValue> {
    use crate::diagram::{DiagramSpecBuilder, InputType};
    use crate::fitter::Fitter;

    // Build diagram spec using DiagramSpecBuilder
    let mut builder = DiagramSpecBuilder::new();

    for spec in &specs {
        let input = spec.input.trim();
        let size = spec.size;

        if input.is_empty() || size <= 0.0 {
            continue;
        }

        // Parse the input - it can be "A", "B", or "A&B"
        let sets: Vec<&str> = input.split('&').map(|s| s.trim()).collect();

        if sets.len() == 1 {
            // Single set
            builder = builder.set(sets[0], size);
        } else if sets.len() > 1 {
            // Intersection
            builder = builder.intersection(&sets, size);
        }
    }

    // Build the specification
    let diagram_spec = builder
        .input_type(InputType::Disjoint)
        .build()
        .map_err(|e| JsValue::from_str(&format!("Failed to build spec: {}", e)))?;

    // Fit the diagram using circles
    let mut fitter = Fitter::new(&diagram_spec);
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("Failed to fit diagram: {}", e)))?;

    // Convert circles to WasmCircle with labels
    let wasm_circles: Vec<WasmCircle> = diagram_spec
        .set_names()
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            layout.shape_for_set(name).map(|shape| {
                WasmCircle::new(
                    shape.center().x(),
                    shape.center().y(),
                    shape.radius(),
                    name.clone(),
                )
            })
        })
        .collect();

    Ok(wasm_circles)
}
