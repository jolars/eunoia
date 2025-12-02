//! WASM bindings for Eunoia diagram generation
//!
//! This crate provides WebAssembly bindings for the eunoia library,
//! enabling Euler and Venn diagram generation in web browsers.

use eunoia::geometry::shapes::{Circle, Ellipse};
use eunoia::geometry::traits::Polygonize;
use eunoia::Optimizer;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use console_error_panic_hook;

/// Optimizer options for WASM
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmOptimizer {
    NelderMead,
    Lbfgs,
    ConjugateGradient,
    TrustRegion,
}

impl From<WasmOptimizer> for Optimizer {
    fn from(opt: WasmOptimizer) -> Self {
        match opt {
            WasmOptimizer::NelderMead => Optimizer::NelderMead,
            WasmOptimizer::Lbfgs => Optimizer::Lbfgs,
            WasmOptimizer::ConjugateGradient => Optimizer::ConjugateGradient,
            WasmOptimizer::TrustRegion => Optimizer::TrustRegion,
        }
    }
}

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

/// An ellipse representation for WASM with label
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmEllipse {
    pub x: f64,
    pub y: f64,
    pub semi_major: f64,
    pub semi_minor: f64,
    pub rotation: f64,
    label: String,
}

#[wasm_bindgen]
impl WasmEllipse {
    #[wasm_bindgen(constructor)]
    pub fn new(
        x: f64,
        y: f64,
        semi_major: f64,
        semi_minor: f64,
        rotation: f64,
        label: String,
    ) -> Self {
        Self {
            x,
            y,
            semi_major,
            semi_minor,
            rotation,
            label,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }
}

/// A point in 2D space
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmPoint {
    pub x: f64,
    pub y: f64,
}

#[wasm_bindgen]
impl WasmPoint {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A polygon representation for visualization
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmPolygon {
    vertices: Vec<WasmPoint>,
    label: String,
}

#[wasm_bindgen]
impl WasmPolygon {
    #[wasm_bindgen(getter)]
    pub fn vertices(&self) -> Vec<WasmPoint> {
        self.vertices.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn area(&self) -> f64 {
        // Calculate using shoelace formula
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }

        (area / 2.0).abs()
    }

    /// Calculate the centroid (geometric center) of the polygon.
    #[wasm_bindgen]
    pub fn centroid(&self) -> WasmPoint {
        if self.vertices.is_empty() {
            return WasmPoint { x: 0.0, y: 0.0 };
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut area = 0.0;
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let cross =
                self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y;
            area += cross;
            cx += (self.vertices[i].x + self.vertices[j].x) * cross;
            cy += (self.vertices[i].y + self.vertices[j].y) * cross;
        }

        area *= 0.5;
        if area.abs() < 1e-10 {
            cx = self.vertices.iter().map(|p| p.x).sum::<f64>() / n as f64;
            cy = self.vertices.iter().map(|p| p.y).sum::<f64>() / n as f64;
        } else {
            cx /= 6.0 * area;
            cy /= 6.0 * area;
        }

        WasmPoint { x: cx, y: cy }
    }

    /// Find the pole of inaccessibility (visual center) of the polygon.
    ///
    /// This is the most distant internal point from the polygon outline,
    /// which is better for label placement than the centroid for complex shapes.
    ///
    /// # Arguments
    ///
    /// * `precision` - Tolerance (smaller = more accurate but slower). Default: 1.0
    #[wasm_bindgen]
    pub fn pole_of_inaccessibility(&self, precision: f64) -> WasmPoint {
        use eunoia::geometry::primitives::Point;
        use eunoia::geometry::shapes::Polygon;

        // Convert to eunoia Polygon
        let points: Vec<Point> = self.vertices.iter().map(|p| Point::new(p.x, p.y)).collect();

        let polygon = Polygon::new(points);
        let pole = polygon.pole_of_inaccessibility(precision);

        WasmPoint {
            x: pole.x(),
            y: pole.y(),
        }
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

/// Result with ellipses and debug info
#[wasm_bindgen]
pub struct EllipseResult {
    ellipses: Vec<WasmEllipse>,
    loss: f64,
    target_areas_json: String,
    fitted_areas_json: String,
}

#[wasm_bindgen]
impl EllipseResult {
    #[wasm_bindgen(getter)]
    pub fn ellipses(&self) -> Vec<WasmEllipse> {
        self.ellipses.clone()
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

/// Result with polygons and debug info
#[wasm_bindgen]
pub struct PolygonResult {
    polygons: Vec<WasmPolygon>,
    circles: Vec<WasmCircle>,
    ellipses: Vec<WasmEllipse>,
    loss: f64,
    target_areas_json: String,
    fitted_areas_json: String,
}

#[wasm_bindgen]
impl PolygonResult {
    #[wasm_bindgen(getter)]
    pub fn polygons(&self) -> Vec<WasmPolygon> {
        self.polygons.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn circles(&self) -> Vec<WasmCircle> {
        self.circles.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ellipses(&self) -> Vec<WasmEllipse> {
        self.ellipses.clone()
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

/// A single region with its polygons
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmRegion {
    combination: String,
    polygons: Vec<WasmPolygon>,
}

#[wasm_bindgen]
impl WasmRegion {
    #[wasm_bindgen(getter)]
    pub fn combination(&self) -> String {
        self.combination.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn polygons(&self) -> Vec<WasmPolygon> {
        self.polygons.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn total_area(&self) -> f64 {
        self.polygons.iter().map(|p| p.area()).sum()
    }
}

/// Collection of region polygons for visualization
#[wasm_bindgen]
pub struct WasmRegionPolygons {
    regions: Vec<WasmRegion>,
}

#[wasm_bindgen]
impl WasmRegionPolygons {
    #[wasm_bindgen(getter)]
    pub fn regions(&self) -> Vec<WasmRegion> {
        self.regions.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.regions.len()
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
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<DiagramResult, JsValue> {
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
    let mut fitter = Fitter::<Circle>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
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

    // Extract areas from the same layout (no refitting!)
    let target_areas: std::collections::HashMap<String, f64> = diagram_spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let fitted_areas: std::collections::HashMap<String, f64> = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    Ok(DiagramResult {
        circles: wasm_circles,
        loss: layout.loss(),
        target_areas_json: serde_json::to_string(&target_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
        fitted_areas_json: serde_json::to_string(&fitted_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
    })
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
    shape_type: String,
    seed: Option<u64>,
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

    // Compute fitted areas based on shape type
    let fitted: HashMap<String, f64> = if shape_type == "ellipse" {
        let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        let layout = fitter
            .fit()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        layout
            .fitted()
            .iter()
            .map(|(combo, &area)| (combo.to_string(), area))
            .collect()
    } else {
        let mut fitter = Fitter::<Circle>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        let layout = fitter
            .fit()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        layout
            .fitted()
            .iter()
            .map(|(combo, &area)| (combo.to_string(), area))
            .collect()
    };

    let loss = if shape_type == "ellipse" {
        let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        fitter.fit().map(|l| l.loss()).unwrap_or(0.0)
    } else {
        let mut fitter = Fitter::<Circle>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        fitter.fit().map(|l| l.loss()).unwrap_or(0.0)
    };

    let mut target: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target.insert(combo.to_string(), area);
    }

    let response = serde_json::json!({
        "loss": loss,
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
    seed: Option<u64>,
) -> Result<DiagramResult, JsValue> {
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
    let mut fitter = Fitter::<Circle>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
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

    // Extract areas from the same layout (no refitting!)
    let target_areas: std::collections::HashMap<String, f64> = diagram_spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let fitted_areas: std::collections::HashMap<String, f64> = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    Ok(DiagramResult {
        circles: wasm_circles,
        loss: layout.loss(),
        target_areas_json: serde_json::to_string(&target_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
        fitted_areas_json: serde_json::to_string(&fitted_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
    })
}

/// Get debug info for initial layout only
#[wasm_bindgen]
pub fn get_debug_info_initial(
    inputs: Vec<String>,
    sizes: Vec<f64>,
    input_type: String,
    shape_type: String,
    seed: Option<u64>,
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

    // Compute fitted areas based on shape type
    let (loss, fitted) = if shape_type == "ellipse" {
        let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        let layout = fitter
            .fit_initial_only()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        let fitted_map: HashMap<String, f64> = layout
            .fitted()
            .iter()
            .map(|(combo, &area)| (combo.to_string(), area))
            .collect();

        (layout.loss(), fitted_map)
    } else {
        let mut fitter = Fitter::<Circle>::new(&diagram_spec);
        if let Some(s) = seed {
            fitter = fitter.seed(s);
        }
        let layout = fitter
            .fit_initial_only()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        let fitted_map: HashMap<String, f64> = layout
            .fitted()
            .iter()
            .map(|(combo, &area)| (combo.to_string(), area))
            .collect();

        (layout.loss(), fitted_map)
    };

    let mut target: HashMap<String, f64> = HashMap::new();
    for (combo, &area) in diagram_spec.exclusive_areas() {
        target.insert(combo.to_string(), area);
    }

    let response = serde_json::json!({
        "loss": loss,
        "target_areas": target,
        "fitted_areas": fitted
    });

    serde_json::to_string(&response).map_err(|e| JsValue::from_str(&format!("{}", e)))
}

/// Generate ellipse layout from diagram specification
#[wasm_bindgen]
pub fn generate_ellipses_from_spec(
    specs: Vec<DiagramSpec>,
    input_type: String,
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<EllipseResult, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

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
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let wasm_ellipses: Vec<WasmEllipse> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|shape: &Ellipse| {
                WasmEllipse::new(
                    shape.center().x(),
                    shape.center().y(),
                    shape.semi_major(),
                    shape.semi_minor(),
                    shape.rotation(),
                    name.to_string(),
                )
            })
        })
        .collect();

    // Extract areas from the same layout (no refitting!)
    let target_areas: std::collections::HashMap<String, f64> = diagram_spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let fitted_areas: std::collections::HashMap<String, f64> = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    Ok(EllipseResult {
        ellipses: wasm_ellipses,
        loss: layout.loss(),
        target_areas_json: serde_json::to_string(&target_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
        fitted_areas_json: serde_json::to_string(&fitted_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
    })
}

/// Generate circle layout and convert to polygons for rendering
#[wasm_bindgen]
pub fn generate_circles_as_polygons(
    specs: Vec<DiagramSpec>,
    input_type: String,
    n_vertices: usize,
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<PolygonResult, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

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
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut fitter = Fitter::<Circle>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let wasm_circles: Vec<WasmCircle> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|circle: &Circle| {
                WasmCircle::new(
                    circle.center().x(),
                    circle.center().y(),
                    circle.radius(),
                    name.to_string(),
                )
            })
        })
        .collect();

    let wasm_polygons: Vec<WasmPolygon> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|circle: &Circle| {
                let polygon = circle.polygonize(n_vertices);
                let vertices: Vec<WasmPoint> = polygon
                    .vertices()
                    .iter()
                    .map(|p| WasmPoint::new(p.x(), p.y()))
                    .collect();
                WasmPolygon {
                    vertices,
                    label: name.to_string(),
                }
            })
        })
        .collect();

    // Extract areas from the same layout (no refitting!)
    let target_areas: std::collections::HashMap<String, f64> = diagram_spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let fitted_areas: std::collections::HashMap<String, f64> = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    Ok(PolygonResult {
        polygons: wasm_polygons,
        circles: wasm_circles,
        ellipses: vec![],
        loss: layout.loss(),
        target_areas_json: serde_json::to_string(&target_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
        fitted_areas_json: serde_json::to_string(&fitted_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
    })
}

/// Generate ellipse layout and convert to polygons for rendering
#[wasm_bindgen]
pub fn generate_ellipses_as_polygons(
    specs: Vec<DiagramSpec>,
    input_type: String,
    n_vertices: usize,
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<PolygonResult, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

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
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let wasm_ellipses: Vec<WasmEllipse> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|ellipse: &Ellipse| {
                WasmEllipse::new(
                    ellipse.center().x(),
                    ellipse.center().y(),
                    ellipse.semi_major(),
                    ellipse.semi_minor(),
                    ellipse.rotation(),
                    name.to_string(),
                )
            })
        })
        .collect();

    let wasm_polygons: Vec<WasmPolygon> = diagram_spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|ellipse: &Ellipse| {
                let polygon = ellipse.polygonize(n_vertices);
                let vertices: Vec<WasmPoint> = polygon
                    .vertices()
                    .iter()
                    .map(|p| WasmPoint::new(p.x(), p.y()))
                    .collect();
                WasmPolygon {
                    vertices,
                    label: name.to_string(),
                }
            })
        })
        .collect();

    // Extract areas from the same layout (no refitting!)
    let target_areas: std::collections::HashMap<String, f64> = diagram_spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let fitted_areas: std::collections::HashMap<String, f64> = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    Ok(PolygonResult {
        polygons: wasm_polygons,
        circles: vec![],
        ellipses: wasm_ellipses,
        loss: layout.loss(),
        target_areas_json: serde_json::to_string(&target_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
        fitted_areas_json: serde_json::to_string(&fitted_areas)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?,
    })
}

/// Generate region polygons from circles (for filled diagram visualization)
#[wasm_bindgen]
pub fn generate_region_polygons_circles(
    specs: Vec<DiagramSpec>,
    input_type: String,
    n_vertices: usize,
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<WasmRegionPolygons, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

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
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut fitter = Fitter::<Circle>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    // Get region polygons using the plotting feature
    let region_polygons = layout.region_polygons(&diagram_spec, n_vertices);

    // Convert to WASM types
    let mut wasm_regions = Vec::new();
    for (combination, polygons) in region_polygons.iter() {
        let wasm_polygons: Vec<WasmPolygon> = polygons
            .iter()
            .map(|poly| {
                let vertices: Vec<WasmPoint> = poly
                    .vertices()
                    .iter()
                    .map(|p| WasmPoint::new(p.x(), p.y()))
                    .collect();
                WasmPolygon {
                    vertices,
                    label: combination.to_string(),
                }
            })
            .collect();

        wasm_regions.push(WasmRegion {
            combination: combination.to_string(),
            polygons: wasm_polygons,
        });
    }

    Ok(WasmRegionPolygons {
        regions: wasm_regions,
    })
}

/// Generate region polygons from ellipses (for filled diagram visualization)
#[wasm_bindgen]
pub fn generate_region_polygons_ellipses(
    specs: Vec<DiagramSpec>,
    input_type: String,
    n_vertices: usize,
    seed: Option<u64>,
    optimizer: Option<WasmOptimizer>,
) -> Result<WasmRegionPolygons, JsValue> {
    use eunoia::fitter::Fitter;
    use eunoia::spec::{DiagramSpecBuilder, InputType};

    let input_type = match input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        _ => return Err(JsValue::from_str("Invalid input type")),
    };

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
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    let mut fitter = Fitter::<Ellipse>::new(&diagram_spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    if let Some(opt) = optimizer {
        fitter = fitter.optimizer(opt.into());
    }
    let layout = fitter
        .fit()
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    // Get region polygons using the plotting feature
    let region_polygons = layout.region_polygons(&diagram_spec, n_vertices);

    // Convert to WASM types
    let mut wasm_regions = Vec::new();
    for (combination, polygons) in region_polygons.iter() {
        let wasm_polygons: Vec<WasmPolygon> = polygons
            .iter()
            .map(|poly| {
                let vertices: Vec<WasmPoint> = poly
                    .vertices()
                    .iter()
                    .map(|p| WasmPoint::new(p.x(), p.y()))
                    .collect();
                WasmPolygon {
                    vertices,
                    label: combination.to_string(),
                }
            })
            .collect();

        wasm_regions.push(WasmRegion {
            combination: combination.to_string(),
            polygons: wasm_polygons,
        });
    }

    Ok(WasmRegionPolygons {
        regions: wasm_regions,
    })
}
