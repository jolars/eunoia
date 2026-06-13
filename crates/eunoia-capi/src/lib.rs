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
//! Success: `{"ok": true, "shape": ..., "shapes": [...], "metrics": {...},
//! "plot_data": {...}}`. The `plot_data` bundle (region pieces, region/set
//! anchors, region areas, shape outlines) mirrors the PyO3 binding so the Julia
//! side can render diagrams. Failure: `{"ok": false, "error": "<message>"}`.
//! Callers branch on `ok`.

// Mirrors the WASM crate: the shape constructors panic on bad input, so the
// `disallowed-methods` clippy.toml forbids them here. `try_new` is the FFI-safe
// path. This makes a stray `::new` a hard error rather than a latent abort.
#![deny(clippy::disallowed_methods)]

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::Serialize;
use serde::de::DeserializeOwned;

use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse, Polygon, Rectangle, Square};
use eunoia::geometry::traits::{DiagramShape, Polygonize};
use eunoia::loss::LossType;
use eunoia::plotting::{PlotData, PlotOptions};
use eunoia::spec::{DiagramSpec, DiagramSpecBuilder, InputType};
use eunoia::{Fitter, InitialSampler, Layout, MdsSolver, Optimizer, VennDiagram};

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

    // --- Phase 4(a) fitting knobs ---
    //
    // All optional; omitting a field leaves the corresponding `Fitter` default
    // untouched. Enum-valued knobs (`loss`/`optimizer`/`mds_solver`/
    // `initial_sampler`) are snake_case strings validated and mapped to core
    // enums by the `parse_*` helpers below — the core enums carry no serde
    // derives, so the capi is the string↔enum contract.
    /// `LossType` variant (snake_case, e.g. `"sum_absolute"`).
    #[serde(default)]
    loss: Option<String>,
    /// Smoothing `eps` for the six `smooth_*` losses; ignored otherwise.
    #[serde(default)]
    loss_eps: Option<f64>,
    #[serde(default)]
    n_restarts: Option<usize>,
    /// `Optimizer` variant (snake_case, e.g. `"cmaes_trf"`).
    #[serde(default)]
    optimizer: Option<String>,
    /// `MdsSolver` variant for the initial layout (snake_case).
    #[serde(default)]
    mds_solver: Option<String>,
    /// `InitialSampler` variant (snake_case).
    #[serde(default)]
    initial_sampler: Option<String>,
    #[serde(default)]
    cmaes_fallback_threshold: Option<f64>,
    #[serde(default)]
    max_iterations: Option<usize>,
    #[serde(default)]
    tolerance: Option<f64>,
    #[serde(default)]
    xtol: Option<f64>,
    #[serde(default)]
    ftol: Option<f64>,
    #[serde(default)]
    gtol: Option<f64>,
    #[serde(default)]
    jobs: Option<usize>,
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
    /// Per-region error keyed by combination string (always exclusive form).
    region_error: BTreeMap<String, f64>,
    target_areas: BTreeMap<String, f64>,
    fitted_areas: BTreeMap<String, f64>,
}

/// One connected component of a region: a CCW outer ring and any CW hole rings.
/// Vertices are `[x, y]` pairs. Mirrors [`eunoia::plotting::RegionPiece`].
#[derive(Serialize)]
struct RegionPieceOut {
    outer: Vec<[f64; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
}

/// Renderable geometry for a fitted layout, mirroring the PyO3 binding's
/// `plot_data` bundle. All coordinates are `[x, y]` pairs. Region keys are
/// canonical combination strings; set keys are set names.
#[derive(Serialize)]
struct PlotDataOut {
    region_pieces: BTreeMap<String, Vec<RegionPieceOut>>,
    region_anchors: BTreeMap<String, [f64; 2]>,
    region_areas: BTreeMap<String, f64>,
    set_anchors: BTreeMap<String, [f64; 2]>,
    shape_outlines: BTreeMap<String, Vec<[f64; 2]>>,
}

#[derive(Serialize)]
struct LayoutOut {
    shape: String,
    shapes: Vec<ShapeOut>,
    metrics: Metrics,
    plot_data: PlotDataOut,
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

fn poly_to_vec(poly: &Polygon) -> Vec<[f64; 2]> {
    poly.vertices().iter().map(|v| [v.x(), v.y()]).collect()
}

/// Serialize the renderable geometry of a `PlotData` into `PlotDataOut`. Sorted
/// containers (`iter_sorted`, `BTreeMap`) give deterministic JSON.
fn build_plot_data(plot: &PlotData) -> PlotDataOut {
    let region_pieces = plot
        .regions
        .iter_sorted()
        .map(|(combo, pieces)| {
            let out = pieces
                .iter()
                .map(|p| RegionPieceOut {
                    outer: poly_to_vec(&p.outer),
                    holes: p.holes.iter().map(poly_to_vec).collect(),
                })
                .collect();
            (combo.to_string(), out)
        })
        .collect();

    PlotDataOut {
        region_pieces,
        region_anchors: plot
            .region_anchors
            .iter()
            .map(|(k, p)| (k.clone(), [p.x(), p.y()]))
            .collect(),
        region_areas: plot
            .region_areas
            .iter()
            .map(|(k, &a)| (k.clone(), a))
            .collect(),
        set_anchors: plot
            .set_anchors
            .iter()
            .map(|(k, p)| (k.clone(), [p.x(), p.y()]))
            .collect(),
        shape_outlines: plot
            .shape_outlines
            .iter()
            .map(|(k, poly)| (k.clone(), poly_to_vec(poly)))
            .collect(),
    }
}

/// Pull the fitted shapes, label anchors, metrics, plot data, and optional
/// container out of a `Layout` into the serializable `LayoutOut`. Shared by
/// `euler` and `venn`.
fn extract<S>(layout: &Layout<S>, spec: &DiagramSpec, shape: &str) -> LayoutOut
where
    S: DiagramShape + Polygonize + ToShapeOut + Copy + 'static,
{
    // Per-set label anchors (pole of inaccessibility of `shape \ ⋃ others`).
    let plot = layout.plot_data(spec, PlotOptions::default());
    let plot_data = build_plot_data(&plot);
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
    let region_error = layout
        .region_error()
        .into_iter()
        .map(|(combo, error)| (combo.to_string(), error))
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
            region_error,
            target_areas,
            fitted_areas,
        },
        plot_data,
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

/// Default smoothing `eps` for the `smooth_*` losses when the caller omits
/// `loss_eps`. Matches the core's "~1% of typical residual magnitude" guidance
/// and the value its own tests/benches use (`smooth_*(1e-3)`).
const DEFAULT_LOSS_EPS: f64 = 1e-3;

/// Map a snake_case loss name to a [`LossType`]. The six `smooth_*` variants use
/// `eps` (falling back to [`DEFAULT_LOSS_EPS`]); the rest ignore it.
fn parse_loss(name: &str, eps: Option<f64>) -> Result<LossType, String> {
    let e = eps.unwrap_or(DEFAULT_LOSS_EPS);
    let loss = match name {
        "sum_squared" => LossType::SumSquared,
        "sum_absolute" => LossType::SumAbsolute,
        "sum_absolute_region_error" => LossType::SumAbsoluteRegionError,
        "sum_squared_region_error" => LossType::SumSquaredRegionError,
        "max_absolute" => LossType::MaxAbsolute,
        "max_squared" => LossType::MaxSquared,
        "root_mean_squared" => LossType::RootMeanSquared,
        "stress" => LossType::Stress,
        "diag_error" => LossType::DiagError,
        "log_sum_absolute" => LossType::LogSumAbsolute,
        "smooth_sum_absolute" => LossType::smooth_sum_absolute(e),
        "smooth_sum_absolute_region_error" => LossType::smooth_sum_absolute_region_error(e),
        "smooth_max_absolute" => LossType::smooth_max_absolute(e),
        "smooth_max_squared" => LossType::smooth_max_squared(e),
        "smooth_diag_error" => LossType::smooth_diag_error(e),
        "smooth_log_sum_absolute" => LossType::smooth_log_sum_absolute(e),
        other => {
            return Err(format!(
                "invalid loss '{other}' (want sum_squared|sum_absolute|\
                 sum_absolute_region_error|sum_squared_region_error|max_absolute|\
                 max_squared|root_mean_squared|stress|diag_error|log_sum_absolute|\
                 smooth_sum_absolute|smooth_sum_absolute_region_error|\
                 smooth_max_absolute|smooth_max_squared|smooth_diag_error|\
                 smooth_log_sum_absolute)"
            ));
        }
    };
    Ok(loss)
}

/// Map a snake_case optimizer name to an [`Optimizer`].
fn parse_optimizer(name: &str) -> Result<Optimizer, String> {
    match name {
        "levenberg_marquardt" => Ok(Optimizer::LevenbergMarquardt),
        "lbfgs" => Ok(Optimizer::Lbfgs),
        "nelder_mead" => Ok(Optimizer::NelderMead),
        "trf" => Ok(Optimizer::Trf),
        "cmaes_lm" => Ok(Optimizer::CmaEsLm),
        "cmaes_trf" => Ok(Optimizer::CmaEsTrf),
        other => Err(format!(
            "invalid optimizer '{other}' (want levenberg_marquardt|lbfgs|\
             nelder_mead|trf|cmaes_lm|cmaes_trf)"
        )),
    }
}

/// Map a snake_case MDS-solver name to an [`MdsSolver`].
fn parse_mds_solver(name: &str) -> Result<MdsSolver, String> {
    match name {
        "lbfgs" => Ok(MdsSolver::Lbfgs),
        "levenberg_marquardt" => Ok(MdsSolver::LevenbergMarquardt),
        other => Err(format!(
            "invalid mds_solver '{other}' (want lbfgs|levenberg_marquardt)"
        )),
    }
}

/// Map a snake_case initial-sampler name to an [`InitialSampler`].
fn parse_initial_sampler(name: &str) -> Result<InitialSampler, String> {
    match name {
        "uniform" => Ok(InitialSampler::Uniform),
        "latin_hypercube" => Ok(InitialSampler::LatinHypercube),
        other => Err(format!(
            "invalid initial_sampler '{other}' (want uniform|latin_hypercube)"
        )),
    }
}

/// Resolved, validated fitting knobs (capi-side; not a core type). Built once in
/// `euler_impl` and applied to a freshly constructed `Fitter` in [`fit`].
struct FitConfig {
    seed: Option<u64>,
    loss: Option<LossType>,
    n_restarts: Option<usize>,
    optimizer: Option<Optimizer>,
    mds_solver: Option<MdsSolver>,
    initial_sampler: Option<InitialSampler>,
    cmaes_fallback_threshold: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    xtol: Option<f64>,
    ftol: Option<f64>,
    gtol: Option<f64>,
    jobs: Option<usize>,
}

impl FitConfig {
    /// Parse and validate every enum string up front so a bad value surfaces
    /// regardless of the requested shape and each string is parsed exactly once.
    fn from_input(input: &EulerInput) -> Result<Self, String> {
        Ok(Self {
            seed: input.seed,
            loss: input
                .loss
                .as_deref()
                .map(|s| parse_loss(s, input.loss_eps))
                .transpose()?,
            n_restarts: input.n_restarts,
            optimizer: input
                .optimizer
                .as_deref()
                .map(parse_optimizer)
                .transpose()?,
            mds_solver: input
                .mds_solver
                .as_deref()
                .map(parse_mds_solver)
                .transpose()?,
            initial_sampler: input
                .initial_sampler
                .as_deref()
                .map(parse_initial_sampler)
                .transpose()?,
            cmaes_fallback_threshold: input.cmaes_fallback_threshold,
            max_iterations: input.max_iterations,
            tolerance: input.tolerance,
            xtol: input.xtol,
            ftol: input.ftol,
            gtol: input.gtol,
            jobs: input.jobs,
        })
    }
}

fn fit<S>(spec: &DiagramSpec, cfg: &FitConfig) -> Result<Layout<S>, String>
where
    S: DiagramShape + Copy + 'static,
{
    let mut fitter = Fitter::<S>::new(spec);
    if let Some(s) = cfg.seed {
        fitter = fitter.seed(s);
    }
    if let Some(l) = cfg.loss {
        fitter = fitter.loss_type(l);
    }
    if let Some(n) = cfg.n_restarts {
        fitter = fitter.n_restarts(n);
    }
    if let Some(o) = cfg.optimizer {
        fitter = fitter.optimizer(o);
    }
    if let Some(m) = cfg.mds_solver {
        fitter = fitter.initial_solver(m);
    }
    if let Some(s) = cfg.initial_sampler {
        fitter = fitter.initial_sampler(s);
    }
    if let Some(t) = cfg.cmaes_fallback_threshold {
        fitter = fitter.cmaes_fallback_threshold(t);
    }
    if let Some(i) = cfg.max_iterations {
        fitter = fitter.max_iterations(i);
    }
    if let Some(t) = cfg.tolerance {
        fitter = fitter.tolerance(t);
    }
    if let Some(x) = cfg.xtol {
        fitter = fitter.xtol(x);
    }
    if let Some(f) = cfg.ftol {
        fitter = fitter.ftol(f);
    }
    if let Some(g) = cfg.gtol {
        fitter = fitter.gtol(g);
    }
    if let Some(j) = cfg.jobs {
        fitter = fitter.jobs(j);
    }
    fitter
        .fit()
        .map_err(|e| format!("failed to fit diagram: {e}"))
}

fn euler_impl(input: EulerInput) -> Result<LayoutOut, String> {
    let spec = build_spec(&input)?;
    let cfg = FitConfig::from_input(&input)?;
    let shape = input.shape.as_deref().unwrap_or("circle");
    match shape {
        "circle" => Ok(extract(&fit::<Circle>(&spec, &cfg)?, &spec, "circle")),
        "ellipse" => Ok(extract(&fit::<Ellipse>(&spec, &cfg)?, &spec, "ellipse")),
        "square" => Ok(extract(&fit::<Square>(&spec, &cfg)?, &spec, "square")),
        "rectangle" => Ok(extract(&fit::<Rectangle>(&spec, &cfg)?, &spec, "rectangle")),
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

        // Per-region error is keyed by combination string.
        assert!(v["metrics"]["region_error"]["A&B"].is_number());

        // plot_data carries renderable geometry, all `[x, y]` pairs.
        let plot = &v["plot_data"];
        assert!(plot["region_pieces"]["A&B"].is_array());
        let outer = &plot["region_pieces"]["A&B"][0]["outer"];
        assert!(outer.is_array() && !outer.as_array().unwrap().is_empty());
        assert_eq!(outer[0].as_array().unwrap().len(), 2);
        assert!(plot["region_anchors"]["A&B"].is_array());
        assert!(plot["region_areas"]["A&B"].is_number());
        assert_eq!(plot["set_anchors"]["A"].as_array().unwrap().len(), 2);
        assert!(plot["shape_outlines"]["A"].is_array());
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

    // ------------------------------------------------------------------------
    // Phase 4(a) fitting knobs
    // ------------------------------------------------------------------------

    /// A two-set spec with a fixed seed reused across the knob tests.
    fn two_set(extra: &str) -> String {
        format!(
            r#"{{"sets":[{{"combination":"A","size":5}},{{"combination":"B","size":3}},
                {{"combination":"A&B","size":1}}],"seed":1{extra}}}"#
        )
    }

    #[test]
    fn euler_loss_type_is_honored() {
        // A plain non-default loss and a smooth loss with explicit eps both fit.
        for extra in [
            r#","loss":"sum_absolute""#,
            r#","loss":"smooth_max_absolute","loss_eps":0.01"#,
        ] {
            let out = call(eunoia_euler, &two_set(extra));
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["ok"], true, "input was {extra}");
            assert!(v["metrics"]["loss"].as_f64().unwrap() >= 0.0);
        }
    }

    #[test]
    fn euler_numeric_knobs_accepted() {
        let out = call(
            eunoia_euler,
            &two_set(
                r#","n_restarts":3,"max_iterations":50,"tolerance":1e-4,
                   "xtol":1e-7,"ftol":1e-7,"gtol":1e-7,
                   "cmaes_fallback_threshold":1e-2,"jobs":1"#,
            ),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shapes"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn euler_solver_and_sampler_knobs_accepted() {
        let out = call(
            eunoia_euler,
            &two_set(
                r#","optimizer":"levenberg_marquardt","mds_solver":"lbfgs",
                   "initial_sampler":"latin_hypercube""#,
            ),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn euler_bad_enum_values_are_reported() {
        for (extra, bad) in [
            (r#","loss":"frobnicate""#, "frobnicate"),
            (r#","optimizer":"genetic""#, "genetic"),
            (r#","mds_solver":"gauss""#, "gauss"),
            (r#","initial_sampler":"sobol""#, "sobol"),
        ] {
            let out = call(eunoia_euler, &two_set(extra));
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["ok"], false, "input was {extra}");
            assert!(v["error"].as_str().unwrap().contains(bad));
        }
    }

    #[test]
    fn euler_enum_error_surfaces_regardless_of_shape() {
        // A bad loss together with a non-circle shape must still error, proving
        // the knobs are validated before the shape match.
        let out = call(
            eunoia_euler,
            &two_set(r#","shape":"ellipse","loss":"frobnicate""#),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["error"].as_str().unwrap().contains("frobnicate"));
    }
}
