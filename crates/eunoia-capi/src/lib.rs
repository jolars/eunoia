//! C ABI bindings for eunoia, backing the Julia package (`julia/Eunoia`).
//!
//! # Design: JSON in, JSON out
//!
//! A diagram spec is an irregular, string-keyed, variable-length payload (set
//! names, intersections, optional complement) and the output is a
//! variable-length list of one of four shape types. Marshalling that as flat C
//! structs would mean a bespoke layout plus manual length bookkeeping on both
//! sides. Instead the boundary is deliberately tiny: callers pass a JSON string
//! and receive a JSON string. The same choice the WASM layer already makes
//! internally (`*_json` getters).
//!
//! Exported symbols (all `extern "C"`):
//!
//! - `eunoia_euler(*const c_char) -> *mut c_char` — fit a diagram.
//! - `eunoia_venn(*const c_char) -> *mut c_char` — canonical Venn layout.
//! - `eunoia_version() -> *mut c_char` — crate version string.
//! - `eunoia_free(*mut c_char)` — free any string returned by the above.
//!
//! # Ownership
//!
//! Every returned `*mut c_char` is heap-allocated by Rust and **must** be
//! handed back to `eunoia_free` exactly once. The caller never frees it itself
//! and never holds it past that call.
//!
//! # Panics never cross the boundary
//!
//! Each entry point wraps its body in [`catch_unwind`]; a panic in the core is
//! converted into a JSON `{"ok": false, "error": ...}` response rather than
//! unwinding into the Julia runtime (which would be undefined behaviour).
//!
//! # Response envelope
//!
//! Success: `{"ok": true, "shape": ..., "shapes": [...], "metrics": {...}}`.
//! Failure: `{"ok": false, "error": "<message>"}`. Callers branch on `ok`.

// Mirrors the WASM crate: the shape constructors panic on bad input, so the
// `disallowed-methods` clippy.toml forbids them here. `try_new` is the FFI-safe
// path. This makes a stray `::new` a hard error rather than a latent abort.
#![deny(clippy::disallowed_methods)]

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::Serialize;
use serde::de::DeserializeOwned;

use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse, Rectangle, Square};
use eunoia::geometry::traits::{DiagramShape, Polygonize};
use eunoia::plotting::PlotOptions;
use eunoia::spec::{DiagramSpec, DiagramSpecBuilder, InputType};
use eunoia::{Fitter, Layout, VennDiagram};

use std::collections::{BTreeMap, HashMap};

// ============================================================================
// JSON contract — input
// ============================================================================

/// One `(combination, size)` pair. `combination` is a single set name (`"A"`)
/// or an intersection joined by `&` (`"A&B"`), matching the core spec syntax.
#[derive(serde::Deserialize)]
struct SetSpec {
    combination: String,
    size: f64,
}

#[derive(serde::Deserialize)]
struct EulerInput {
    sets: Vec<SetSpec>,
    #[serde(default)]
    shape: Option<String>,
    #[serde(default = "default_input_type")]
    input_type: String,
    #[serde(default)]
    complement: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(serde::Deserialize)]
struct VennInput {
    /// Set names, in order. Their count selects the canonical arrangement.
    names: Vec<String>,
    #[serde(default)]
    shape: Option<String>,
}

fn default_input_type() -> String {
    "exclusive".to_string()
}

// ============================================================================
// JSON contract — output
// ============================================================================

#[derive(Serialize)]
struct PointOut {
    x: f64,
    y: f64,
}

impl From<Point> for PointOut {
    fn from(p: Point) -> Self {
        PointOut { x: p.x(), y: p.y() }
    }
}

/// Tagged union of the four fittable shapes. `label_anchor` is the per-set
/// pole-of-inaccessibility label position (`PlotData::set_anchors`), falling
/// back to the shape centroid when the set had no dedicated anchor.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ShapeOut {
    Circle {
        label: String,
        x: f64,
        y: f64,
        radius: f64,
        label_anchor: PointOut,
    },
    Ellipse {
        label: String,
        x: f64,
        y: f64,
        semi_major: f64,
        semi_minor: f64,
        rotation: f64,
        label_anchor: PointOut,
    },
    Square {
        label: String,
        x: f64,
        y: f64,
        side: f64,
        label_anchor: PointOut,
    },
    Rectangle {
        label: String,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        label_anchor: PointOut,
    },
}

/// Container frame emitted only when the spec carried a `complement`.
#[derive(Serialize)]
struct ContainerOut {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[derive(Serialize)]
struct Metrics {
    loss: f64,
    stress: f64,
    diag_error: f64,
    iterations: usize,
    target_areas: BTreeMap<String, f64>,
    fitted_areas: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct LayoutOut {
    shape: String,
    shapes: Vec<ShapeOut>,
    metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    container: Option<ContainerOut>,
}

/// Success envelope: `ok: true` flattened over the layout fields.
#[derive(Serialize)]
struct OkResponse {
    ok: bool,
    #[serde(flatten)]
    layout: LayoutOut,
}

// ============================================================================
// Shape → ShapeOut
// ============================================================================

/// Per-shape conversion to the tagged output variant. Implementing this is the
/// only shape-specific code; everything downstream is generic over `S`.
trait ToShapeOut {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut;
}

impl ToShapeOut for Circle {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut {
        let c = self.center();
        ShapeOut::Circle {
            label,
            x: c.x(),
            y: c.y(),
            radius: self.radius(),
            label_anchor: anchor,
        }
    }
}

impl ToShapeOut for Ellipse {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut {
        let c = self.center();
        ShapeOut::Ellipse {
            label,
            x: c.x(),
            y: c.y(),
            semi_major: self.semi_major(),
            semi_minor: self.semi_minor(),
            rotation: self.rotation(),
            label_anchor: anchor,
        }
    }
}

impl ToShapeOut for Square {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut {
        let c = self.center();
        ShapeOut::Square {
            label,
            x: c.x(),
            y: c.y(),
            side: self.side(),
            label_anchor: anchor,
        }
    }
}

impl ToShapeOut for Rectangle {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut {
        let c = self.center();
        ShapeOut::Rectangle {
            label,
            x: c.x(),
            y: c.y(),
            width: self.width(),
            height: self.height(),
            label_anchor: anchor,
        }
    }
}

// ============================================================================
// Core extraction (generic over shape)
// ============================================================================

/// Pull the fitted shapes, label anchors, metrics, and optional container out
/// of a `Layout` into the serializable `LayoutOut`. Shared by `euler` and
/// `venn`.
fn extract<S>(layout: &Layout<S>, spec: &DiagramSpec, shape: &str) -> LayoutOut
where
    S: DiagramShape + Polygonize + ToShapeOut + Copy + 'static,
{
    // Per-set label anchors (pole of inaccessibility of `shape \ ⋃ others`).
    let plot = layout.plot_data(spec, PlotOptions::default());
    let anchors: HashMap<String, Point> = plot.set_anchors.into_iter().collect();

    let shapes = spec
        .set_names()
        .iter()
        .filter_map(|name| {
            layout.shape_for_set(name).map(|s| {
                let anchor = anchors.get(name).copied().unwrap_or_else(|| s.centroid());
                s.to_shape_out(name.clone(), anchor.into())
            })
        })
        .collect();

    let target_areas = spec
        .exclusive_areas()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();
    let fitted_areas = layout
        .fitted()
        .iter()
        .map(|(combo, &area)| (combo.to_string(), area))
        .collect();

    let container = layout.container().map(|r| {
        let c = r.center();
        ContainerOut {
            x: c.x(),
            y: c.y(),
            width: r.width(),
            height: r.height(),
        }
    });

    LayoutOut {
        shape: shape.to_string(),
        shapes,
        metrics: Metrics {
            loss: layout.loss(),
            stress: layout.stress(),
            diag_error: layout.diag_error(),
            iterations: layout.iterations(),
            target_areas,
            fitted_areas,
        },
        container,
    }
}

// ============================================================================
// euler / venn implementations
// ============================================================================

fn build_spec(input: &EulerInput) -> Result<DiagramSpec, String> {
    let input_type = match input.input_type.as_str() {
        "exclusive" => InputType::Exclusive,
        "inclusive" => InputType::Inclusive,
        other => {
            return Err(format!(
                "invalid input_type '{other}' (want exclusive|inclusive)"
            ));
        }
    };

    let mut builder = DiagramSpecBuilder::new();
    for s in &input.sets {
        let combination = s.combination.trim();
        if combination.is_empty() || s.size < 0.0 {
            continue;
        }
        let sets: Vec<&str> = combination.split('&').map(str::trim).collect();
        builder = match sets.len() {
            0 => builder,
            1 => builder.set(sets[0], s.size),
            _ => builder.intersection(&sets, s.size),
        };
    }
    if let Some(c) = input.complement {
        builder = builder.complement(c);
    }

    builder
        .input_type(input_type)
        .build()
        .map_err(|e| format!("failed to build spec: {e}"))
}

fn fit<S>(spec: &DiagramSpec, seed: Option<u64>) -> Result<Layout<S>, String>
where
    S: DiagramShape + Copy + 'static,
{
    let mut fitter = Fitter::<S>::new(spec);
    if let Some(s) = seed {
        fitter = fitter.seed(s);
    }
    fitter
        .fit()
        .map_err(|e| format!("failed to fit diagram: {e}"))
}

fn euler_impl(input: EulerInput) -> Result<LayoutOut, String> {
    let spec = build_spec(&input)?;
    let shape = input.shape.as_deref().unwrap_or("circle");
    match shape {
        "circle" => Ok(extract(&fit::<Circle>(&spec, input.seed)?, &spec, "circle")),
        "ellipse" => Ok(extract(
            &fit::<Ellipse>(&spec, input.seed)?,
            &spec,
            "ellipse",
        )),
        "square" => Ok(extract(&fit::<Square>(&spec, input.seed)?, &spec, "square")),
        "rectangle" => Ok(extract(
            &fit::<Rectangle>(&spec, input.seed)?,
            &spec,
            "rectangle",
        )),
        other => Err(format!(
            "invalid shape '{other}' (want circle|ellipse|square|rectangle)"
        )),
    }
}

fn venn_impl(input: VennInput) -> Result<LayoutOut, String> {
    let n = input.names.len();
    let refs: Vec<&str> = input.names.iter().map(String::as_str).collect();
    let shape = input.shape.as_deref().unwrap_or("circle");

    // Each arm builds the canonical Venn layout for `n` sets, then runs the
    // same generic extraction. `into_layout_and_spec` hands back the derived
    // spec so label anchors and metrics compute exactly as for `euler`.
    macro_rules! venn_arm {
        ($shape:ty, $name:literal) => {{
            let (layout, spec) = VennDiagram::<$shape>::new(n)
                .map_err(|e| format!("no {}-set Venn for {}: {e}", n, $name))?
                .with_names(&refs)
                .into_layout_and_spec();
            Ok(extract(&layout, &spec, $name))
        }};
    }

    match shape {
        "circle" => venn_arm!(Circle, "circle"),
        "ellipse" => venn_arm!(Ellipse, "ellipse"),
        "square" => venn_arm!(Square, "square"),
        "rectangle" => venn_arm!(Rectangle, "rectangle"),
        other => Err(format!(
            "invalid shape '{other}' (want circle|ellipse|square|rectangle)"
        )),
    }
}

// ============================================================================
// FFI boundary
// ============================================================================

/// Render a `Result<LayoutOut, String>` to the JSON envelope string.
fn to_json(result: Result<LayoutOut, String>) -> String {
    match result {
        Ok(layout) => serde_json::to_string(&OkResponse { ok: true, layout })
            .unwrap_or_else(|e| error_json(&format!("serialization failed: {e}"))),
        Err(error) => error_json(&error),
    }
}

fn error_json(message: &str) -> String {
    // Hand-built so it can't itself fail to serialize.
    let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
    format!("{{\"ok\":false,\"error\":\"{escaped}\"}}")
}

/// Shared entry-point body: read the C string, parse JSON, run `f`, and return
/// a freshly allocated JSON C string. Panics are caught and reported as errors.
fn run<I, F>(input: *const c_char, f: F) -> *mut c_char
where
    I: DeserializeOwned,
    F: FnOnce(I) -> Result<LayoutOut, String>,
{
    let json = catch_unwind(AssertUnwindSafe(|| {
        let parsed = parse_input::<I>(input)?;
        f(parsed)
    }))
    .map(to_json)
    .unwrap_or_else(|_| error_json("panic in eunoia core"));

    // `CString::new` only fails on interior NUL, which JSON never contains.
    CString::new(json)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

fn parse_input<I: DeserializeOwned>(input: *const c_char) -> Result<I, String> {
    if input.is_null() {
        return Err("null input pointer".to_string());
    }
    // SAFETY: caller guarantees `input` is a valid NUL-terminated C string for
    // the duration of this call (documented in the header / Julia wrapper).
    let s = unsafe { CStr::from_ptr(input) }
        .to_str()
        .map_err(|e| format!("input is not valid UTF-8: {e}"))?;
    serde_json::from_str(s).map_err(|e| format!("invalid JSON: {e}"))
}

/// Fit an Euler diagram. `input` is a JSON `EulerInput`; returns a JSON
/// envelope. Free the result with [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_euler(input: *const c_char) -> *mut c_char {
    run(input, euler_impl)
}

/// Build a canonical Venn diagram. `input` is a JSON `VennInput`; returns a
/// JSON envelope. Free the result with [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_venn(input: *const c_char) -> *mut c_char {
    run(input, venn_impl)
}

/// Return the crate version as a NUL-terminated C string. Free with
/// [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_version() -> *mut c_char {
    CString::new(env!("CARGO_PKG_VERSION"))
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Free a string previously returned by any `eunoia_*` function. Passing null
/// is a no-op; passing any other pointer, or freeing twice, is undefined.
///
/// # Safety
///
/// `ptr` must be a pointer returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eunoia_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: by contract `ptr` came from `CString::into_raw` above.
        drop(unsafe { CString::from_raw(ptr) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(f: unsafe extern "C" fn(*const c_char) -> *mut c_char, input: &str) -> String {
        let c_in = CString::new(input).unwrap();
        let out_ptr = unsafe { f(c_in.as_ptr()) };
        assert!(!out_ptr.is_null());
        let out = unsafe { CStr::from_ptr(out_ptr) }
            .to_str()
            .unwrap()
            .to_string();
        unsafe { eunoia_free(out_ptr) };
        out
    }

    #[test]
    fn euler_two_set_circle_fits() {
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":5},{"combination":"B","size":3},
                {"combination":"A&B","size":1}],"seed":1}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shape"], "circle");
        assert_eq!(v["shapes"].as_array().unwrap().len(), 2);
        assert_eq!(v["shapes"][0]["type"], "circle");
        assert!(v["metrics"]["loss"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn venn_three_set_ellipse() {
        let out = call(eunoia_venn, r#"{"names":["A","B","C"],"shape":"ellipse"}"#);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shapes"].as_array().unwrap().len(), 3);
        assert_eq!(v["shapes"][0]["type"], "ellipse");
    }

    #[test]
    fn bad_shape_is_reported_not_panicked() {
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":1}],"shape":"hexagon"}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["error"].as_str().unwrap().contains("hexagon"));
    }

    #[test]
    fn malformed_json_is_reported() {
        let out = call(eunoia_euler, "{not json");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
    }

    #[test]
    fn version_is_nonempty() {
        let ptr = eunoia_version();
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
        unsafe { eunoia_free(ptr) };
        assert!(!s.is_empty());
    }
}
