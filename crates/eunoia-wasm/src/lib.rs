//! WASM bindings for Eunoia diagram generation
//!
//! This crate provides WebAssembly bindings for the eunoia library,
//! enabling Euler and Venn diagram generation in web browsers.

use eunoia::geometry::diagram;
use eunoia::geometry::shapes::Circle;
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

/// Result of diagram generation including debug information
#[wasm_bindgen]
pub struct DiagramResult {
    circles: Vec<WasmCircle>,
    loss: f64,
    target_areas_json: String,
    fitted_areas_json: String,
}

#[wasm_bindgen]
impl DiagramResult {
    #[wasm_bindgen(getter)]
    pub fn circles(&self) -> Vec<WasmCircle> {
        self.circles.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    #[wasm_bindgen(getter)]
    pub fn target_areas_json(&self) -> String {
        self.target_areas_json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn fitted_areas_json(&self) -> String {
        self.fitted_areas_json.clone()
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
pub fn generate_from_spec(
    specs: Vec<DiagramSpec>,
    input_type: String,
) -> Result<Vec<WasmCircle>, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    // Parse input type
    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => {
            return Err(JsValue::from_str(
                "Invalid input type. Must be 'exclusive' or 'inclusive'",
            ))
        }
    };

    // Build diagram spec using DiagramSpecBuilder
    let mut builder = DiagramSpecBuilder::new();

    for spec in &specs {
        let input = spec.input.trim();
        let size = spec.size;

        if input.is_empty() || size < 0.0 {
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
        .input_type(input_type)
        .build()
        .map_err(|e| JsValue::from_str(&format!("Failed to build spec: {}", e)))?;

    // Fit the diagram using circles
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("Failed to fit diagram: {}", e)))?;

    // Convert circles to WasmCircle with labels
    let wasm_circles: Vec<WasmCircle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|shape: &Circle| {
                WasmCircle::new(
                    shape.center().x(),
                    shape.center().y(),
                    shape.radius(),
                    name.to_string(),
                )
            })
        })
        .collect();

    Ok(wasm_circles)
}

/// Generate layout from diagram specification with debug information
#[wasm_bindgen]
pub fn generate_from_spec_with_debug(
    specs: Vec<DiagramSpec>,
    input_type: String,
) -> Result<DiagramResult, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};
    use std::collections::HashMap;

    // Parse input type
    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => {
            return Err(JsValue::from_str(
                "Invalid input type. Must be 'exclusive' or 'inclusive'",
            ))
        }
    };

    // Build diagram spec using DiagramSpecBuilder
    let mut builder = DiagramSpecBuilder::new();

    for spec in &specs {
        let input = spec.input.trim();
        let size = spec.size;

        if input.is_empty() || size < 0.0 {
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
        .input_type(input_type)
        .build()
        .map_err(|e| JsValue::from_str(&format!("Failed to build spec: {}", e)))?;

    // Fit the diagram using circles
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("Failed to fit diagram: {}", e)))?;

    // Convert circles to WasmCircle with labels
    let wasm_circles: Vec<WasmCircle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|shape: &Circle| {
                WasmCircle::new(
                    shape.center().x(),
                    shape.center().y(),
                    shape.radius(),
                    name.to_string(),
                )
            })
        })
        .collect();

    // Collect target exclusive areas
    let mut target_areas_map: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target_areas_map.insert(combo.to_string(), area);
    }

    // Compute fitted exclusive areas using the same function as the optimizer
    use eunoia::geometry::diagram::compute_exclusive_areas_from_layout;
    let circles: Vec<Circle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| layout.shape_for_set(name).cloned())
        .collect();
    let fitted_exclusive = compute_exclusive_areas_from_layout(&circles, diagram_spec.set_names());

    let mut fitted_areas_map: HashMap<String, f64> = HashMap::new();
    for (combo, area) in fitted_exclusive {
        fitted_areas_map.insert(combo.to_string(), area);
    }

    // Convert to JSON strings
    let target_areas_json = serde_json::to_string(&target_areas_map)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize target areas: {}", e)))?;
    let fitted_areas_json = serde_json::to_string(&fitted_areas_map)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize fitted areas: {}", e)))?;

    Ok(DiagramResult {
        circles: wasm_circles,
        loss: layout.loss(),
        target_areas_json,
        fitted_areas_json,
    })
}

/// Get debug information as JSON string
#[wasm_bindgen]
pub fn get_debug_info(specs: Vec<DiagramSpec>, input_type: String) -> Result<String, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};
    use std::collections::HashMap;

    web_sys::console::log_1(&"[Rust] get_debug_info called".into());

    let input_type_enum = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

    web_sys::console::log_1(&"[Rust] Building spec...".into());
    let mut builder = DiagramSpecBuilder::new();
    for spec in &specs {
        let input = spec.input.trim();
        let size = spec.size;
        if input.is_empty() || size < 0.0 {
            continue;
        }
        let sets: Vec<&str> = input.split('&').map(|s| s.trim()).collect();
        if sets.len() == 1 {
            builder = builder.set(sets[0], size);
        } else if sets.len() > 1 {
            builder = builder.intersection(&sets, size);
        }
    }

    web_sys::console::log_1(&"[Rust] Building diagram spec...".into());
    let diagram_spec = builder
        .input_type(input_type_enum)
        .build()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    web_sys::console::log_1(&"[Rust] Fitting...".into());
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    web_sys::console::log_1(&"[Rust] Computing target areas...".into());
    let mut target: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target.insert(combo.to_string(), area);
    }

    web_sys::console::log_1(&"[Rust] Computing fitted areas...".into());
    use eunoia::geometry::diagram::compute_exclusive_areas_from_layout;
    let circles: Vec<Circle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| layout.shape_for_set(name).cloned())
        .collect();
    let fitted_exclusive = compute_exclusive_areas_from_layout(&circles, diagram_spec.set_names());
    let mut fitted: HashMap<String, f64> = HashMap::new();
    for (combo, area) in fitted_exclusive {
        fitted.insert(combo.to_string(), area);
    }

    web_sys::console::log_1(&"[Rust] Creating JSON...".into());
    let response = serde_json::json!({
        "loss": layout.loss(),
        "target_areas": target,
        "fitted_areas": fitted
    });

    web_sys::console::log_1(&"[Rust] Serializing...".into());
    serde_json::to_string(&response).map_err(|e| JsValue::from_str(&format!("{}", e)))
}
/// Get debug information as JSON string - takes raw inputs instead of DiagramSpec objects
#[wasm_bindgen]
pub fn get_debug_info_simple(
    inputs: Vec<String>,
    sizes: Vec<f64>,
    input_type: String,
) -> Result<String, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};
    use std::collections::HashMap;

    let input_type_enum = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

    let mut builder = DiagramSpecBuilder::new();
    for (input, size) in inputs.iter().zip(sizes.iter()) {
        if input.trim().is_empty() || *size < 0.0 {
            continue;
        }
        let sets: Vec<&str> = input.split('&').map(|s| s.trim()).collect();
        if sets.len() == 1 {
            builder = builder.set(sets[0], *size);
        } else if sets.len() > 1 {
            builder = builder.intersection(&sets, *size);
        }
    }

    let diagram_spec = builder
        .input_type(input_type_enum)
        .build()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut target: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target.insert(combo.to_string(), area);
    }

    use eunoia::geometry::diagram::compute_exclusive_areas_from_layout;
    let circles: Vec<Circle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| layout.shape_for_set(name).cloned())
        .collect();
    let fitted_exclusive = compute_exclusive_areas_from_layout(&circles, diagram_spec.set_names());
    let mut fitted: HashMap<String, f64> = HashMap::new();
    for (combo, area) in fitted_exclusive {
        fitted.insert(combo.to_string(), area);
    }

    let response = serde_json::json!({
        "loss": layout.loss(),
        "target_areas": target,
        "fitted_areas": fitted
    });

    serde_json::to_string(&response).map_err(|e| JsValue::from_str(&format!("{}", e)))
}

/// Generate layout from diagram specification (initial layout only, no optimization)
#[wasm_bindgen]
pub fn generate_from_spec_initial(
    specs: Vec<DiagramSpec>,
    input_type: String,
) -> Result<Vec<WasmCircle>, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    // Parse input type
    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => {
            return Err(JsValue::from_str(
                "Invalid input type. Must be 'exclusive' or 'inclusive'",
            ))
        }
    };

    // Build diagram spec
    let mut builder = DiagramSpecBuilder::new();

    for spec in &specs {
        let input = spec.input.trim();
        let size = spec.size;

        if input.is_empty() || size < 0.0 {
            continue;
        }

        let sets: Vec<&str> = input.split('&').map(|s| s.trim()).collect();

        if sets.len() == 1 {
            builder = builder.set(sets[0], size);
        } else if sets.len() > 1 {
            builder = builder.intersection(&sets, size);
        }
    }

    let diagram_spec = builder
        .input_type(input_type)
        .build()
        .map_err(|e| JsValue::from_str(&format!("Failed to build spec: {}", e)))?;

    // Fit using initial layout only
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit_initial_only()
        .map_err(|e| JsValue::from_str(&format!("Failed to fit diagram: {}", e)))?;

    // Convert to WasmCircles
    let wasm_circles: Vec<WasmCircle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|shape: &Circle| {
                WasmCircle::new(
                    shape.center().x(),
                    shape.center().y(),
                    shape.radius(),
                    name.to_string(),
                )
            })
        })
        .collect();

    Ok(wasm_circles)
}

/// Get debug info for initial layout only
#[wasm_bindgen]
pub fn get_debug_info_initial(
    inputs: Vec<String>,
    sizes: Vec<f64>,
    input_type: String,
) -> Result<String, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};
    use std::collections::HashMap;

    let input_type_enum = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

    let mut builder = DiagramSpecBuilder::new();
    for (input, size) in inputs.iter().zip(sizes.iter()) {
        if input.trim().is_empty() || *size < 0.0 {
            continue;
        }
        let sets: Vec<&str> = input.split('&').map(|s| s.trim()).collect();
        if sets.len() == 1 {
            builder = builder.set(sets[0], *size);
        } else if sets.len() > 1 {
            builder = builder.intersection(&sets, *size);
        }
    }

    let diagram_spec = builder
        .input_type(input_type_enum)
        .build()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
    let fitter = Fitter::<Circle>::new(&diagram_spec);
    let layout = fitter
        .fit_initial_only()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut target: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target.insert(combo.to_string(), area);
    }

    use diagram::compute_exclusive_areas_from_layout;
    let circles: Vec<Circle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| layout.shape_for_set(name).cloned())
        .collect();
    let fitted_exclusive = compute_exclusive_areas_from_layout(&circles, diagram_spec.set_names());
    let mut fitted: HashMap<String, f64> = HashMap::new();
    for (combo, area) in fitted_exclusive {
        fitted.insert(combo.to_string(), area);
    }

    let response = serde_json::json!({
        "loss": layout.loss(),
        "target_areas": target,
        "fitted_areas": fitted
    });

    serde_json::to_string(&response).map_err(|e| JsValue::from_str(&format!("{}", e)))
}
