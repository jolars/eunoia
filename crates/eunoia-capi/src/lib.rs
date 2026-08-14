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
//! - `eunoia_place_labels(*const c_char) -> *mut c_char` — resolve
//!   collision-aware label positions (and leader-line geometry) for region
//!   polygons, given caller-measured label box sizes.
//! - `eunoia_place_set_labels(*const c_char) -> *mut c_char` — resolve
//!   exterior set-label positions that hug each set's own shape outline
//!   (no leader line), given caller-measured label box sizes.
//! - `eunoia_place_glyphs(*const c_char) -> *mut c_char` — pack equally-sized
//!   circular glyphs (one per data unit, eulerGlyphs-style) inside region
//!   polygons, given per-region counts.
//! - `eunoia_place_glyph_boxes(*const c_char) -> *mut c_char` — pack per-item
//!   text boxes (member names) inside region polygons, given caller-measured
//!   `w × h` sizes, at one auto-fitted diagram-wide scale.
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
//! anchors, set-anchor regions, region areas, shape outlines) mirrors the PyO3
//! binding so the Julia side can render diagrams. `eunoia_place_labels` instead
//! returns `{"ok": true,
//! "placements": {...}}`, `eunoia_place_glyphs` returns `{"ok": true,
//! "radius": ..., "positions": {...}, "unplaced": {...}}`, and
//! `eunoia_place_glyph_boxes` returns `{"ok": true, "scale": ...,
//! "boxes": {...}, "unplaced": {...}}`. Failure:
//! `{"ok": false, "error": "<message>"}`. Callers branch on `ok`.

// Mirrors the WASM crate: the shape constructors panic on bad input, so the
// `disallowed-methods` clippy.toml forbids them here. `try_new` is the FFI-safe
// path. This makes a stray `::new` a hard error rather than a latent abort.
#![deny(clippy::disallowed_methods)]

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::Serialize;
use serde::de::DeserializeOwned;

use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse, Polygon, Rectangle, RotatedRectangle, Square};
use eunoia::geometry::traits::{DiagramShape, Polygonize};
use eunoia::loss::LossType;
use eunoia::plotting::{
    ElbowOptions, ExteriorPolicy, GlyphArrangement, GlyphBoxOptions, GlyphOptions, LeaderStrategy,
    PlacementKind, PlacementStrategy, PlotData, PlotOptions, RegionPiece, RegionPolygons,
    SetLabelStrategy, TetherSource, classify_into_pieces, place_glyph_boxes, place_glyphs,
    place_labels, place_set_labels,
};
use eunoia::spec::{Combination, DiagramSpec, DiagramSpecBuilder, InputType};
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
    /// Raise the set-count ceiling (`DiagramSpecBuilder::max_sets`, default
    /// [`eunoia::constants::MAX_SETS`]); clamped core-side to
    /// `MAX_SETS_HARD_CAP`. Omitting it leaves the core default in place.
    #[serde(default)]
    max_sets: Option<usize>,

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

    // --- Phase 4(b) plot knobs ---
    //
    // All optional; omitting a field leaves the corresponding `PlotOptions`
    // default untouched. These are numeric (no enum tokens), so unlike the
    // 4(a) knobs they need no `parse_*` validation — they are forwarded
    // straight to the `PlotOptions` builder before `layout.plot_data`.
    /// Vertices per polygonized shape/region (`PlotOptions::n_vertices`, 200).
    #[serde(default)]
    n_vertices: Option<usize>,
    /// Polylabel anchor precision (`PlotOptions::label_precision`, 0.01).
    #[serde(default)]
    label_precision: Option<f64>,
    /// Sliver-rejection fraction (`PlotOptions::sliver_threshold`, 1e-3).
    #[serde(default)]
    sliver_threshold: Option<f64>,
}

#[derive(serde::Deserialize)]
struct VennInput {
    /// Set names, in order. Their count selects the canonical arrangement.
    names: Vec<String>,
    #[serde(default)]
    shape: Option<String>,
    /// Optional complement ("universe") size; draws a bounding container frame
    /// around the diagram (`VennDiagram::complement`). Omitting it leaves the
    /// diagram unframed.
    #[serde(default)]
    complement: Option<f64>,
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
    #[serde(rename = "rotated_rectangle")]
    RotatedRectangle {
        label: String,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        rotation: f64,
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
    /// For each set whose label anchored to a region, the canonical combination
    /// string of that region (a key into `region_anchors`). Lets renderers pair
    /// a set label with a region quantity by key instead of comparing anchor
    /// points by float equality. Omits sets that fell back to the whole-shape
    /// POI. Mirrors [`eunoia::plotting::PlotData::set_anchor_regions`].
    set_anchor_regions: BTreeMap<String, String>,
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

/// Success envelope: `ok: true` flattened over the payload fields (a
/// [`LayoutOut`] for `eunoia_euler`/`eunoia_venn`, a [`PlaceLabelsOut`] for
/// `eunoia_place_labels`).
#[derive(Serialize)]
struct OkResponse<T> {
    ok: bool,
    #[serde(flatten)]
    payload: T,
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

impl ToShapeOut for RotatedRectangle {
    fn to_shape_out(&self, label: String, anchor: PointOut) -> ShapeOut {
        let c = self.center();
        ShapeOut::RotatedRectangle {
            label,
            x: c.x(),
            y: c.y(),
            width: self.width(),
            height: self.height(),
            rotation: self.rotation(),
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
        set_anchor_regions: plot
            .set_anchor_regions
            .iter()
            .map(|(k, combo)| (k.clone(), combo.clone()))
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
fn extract<S>(
    layout: &Layout<S>,
    spec: &DiagramSpec,
    shape: &str,
    plot_opts: PlotOptions,
) -> LayoutOut
where
    S: DiagramShape + Polygonize + ToShapeOut + Copy + 'static,
{
    // Per-set label anchors (pole of inaccessibility of `shape \ ⋃ others`).
    let plot = layout.plot_data(spec, plot_opts);
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
    if let Some(m) = input.max_sets {
        builder = builder.max_sets(m);
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
        "mads" => Ok(Optimizer::Mads),
        "trf" => Ok(Optimizer::Trf),
        "cmaes" => Ok(Optimizer::CmaEs),
        "cmaes_lm" => Ok(Optimizer::CmaEsLm),
        "cmaes_trf" => Ok(Optimizer::CmaEsTrf),
        other => Err(format!(
            "invalid optimizer '{other}' (want levenberg_marquardt|lbfgs|\
             nelder_mead|mads|trf|cmaes|cmaes_lm|cmaes_trf)"
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

/// Build the `PlotOptions` for `layout.plot_data`, applying only the knobs the
/// caller set (mirroring `fit`'s `if let Some(..)` shape) so omitted fields keep
/// the `PlotOptions::default()` values.
fn plot_options_from_input(input: &EulerInput) -> PlotOptions {
    let mut opts = PlotOptions::default();
    if let Some(n) = input.n_vertices {
        opts = opts.n_vertices(n);
    }
    if let Some(p) = input.label_precision {
        opts = opts.label_precision(p);
    }
    if let Some(t) = input.sliver_threshold {
        opts = opts.sliver_threshold(t);
    }
    opts
}

fn euler_impl(input: EulerInput) -> Result<LayoutOut, String> {
    let spec = build_spec(&input)?;
    let cfg = FitConfig::from_input(&input)?;
    let plot_opts = plot_options_from_input(&input);
    let shape = input.shape.as_deref().unwrap_or("circle");
    match shape {
        "circle" => Ok(extract(
            &fit::<Circle>(&spec, &cfg)?,
            &spec,
            "circle",
            plot_opts,
        )),
        "ellipse" => Ok(extract(
            &fit::<Ellipse>(&spec, &cfg)?,
            &spec,
            "ellipse",
            plot_opts,
        )),
        "square" => Ok(extract(
            &fit::<Square>(&spec, &cfg)?,
            &spec,
            "square",
            plot_opts,
        )),
        "rectangle" => Ok(extract(
            &fit::<Rectangle>(&spec, &cfg)?,
            &spec,
            "rectangle",
            plot_opts,
        )),
        "rotated_rectangle" => Ok(extract(
            &fit::<RotatedRectangle>(&spec, &cfg)?,
            &spec,
            "rotated_rectangle",
            plot_opts,
        )),
        other => Err(format!(
            "invalid shape '{other}' (want circle|ellipse|square|rectangle|rotated_rectangle)"
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
            let mut venn = VennDiagram::<$shape>::new(n)
                .map_err(|e| format!("no {}-set Venn for {}: {e}", n, $name))?
                .with_names(&refs);
            if let Some(c) = input.complement {
                venn = venn
                    .complement(c)
                    .map_err(|e| format!("invalid complement: {e}"))?;
            }
            let (layout, spec) = venn.into_layout_and_spec();
            // Venn plot-tuning knobs are out of scope (slice (a)/(b) are
            // euler-only); the shared `extract` just gets the defaults.
            Ok(extract(&layout, &spec, $name, PlotOptions::default()))
        }};
    }

    match shape {
        "circle" => venn_arm!(Circle, "circle"),
        "ellipse" => venn_arm!(Ellipse, "ellipse"),
        "square" => venn_arm!(Square, "square"),
        "rectangle" => venn_arm!(Rectangle, "rectangle"),
        "rotated_rectangle" => venn_arm!(RotatedRectangle, "rotated_rectangle"),
        other => Err(format!(
            "invalid shape '{other}' (want circle|ellipse|square|rectangle|rotated_rectangle)"
        )),
    }
}

// ============================================================================
// Label placement — JSON contract + impl
// ============================================================================
//
// A separate entry point (`eunoia_place_labels`) rather than a `plot_data`
// field, because collision-aware placement needs the rendered label box sizes
// (font metrics) the core can't know at fit time. Mirrors the WASM
// `place_region_labels` surface, adapted to capi conventions: snake_case enum
// tokens validated up front (like the fitting knobs in `parse_*`/`FitConfig`).

/// One connected component of a region as supplied by the caller: an outer ring
/// plus any hole rings, each a list of `[x, y]` pairs. Re-paired into a
/// [`RegionPiece`] via [`classify_into_pieces`] (which is `#[non_exhaustive]`
/// and must not be hand-built).
#[derive(serde::Deserialize)]
struct PieceIn {
    outer: Vec<[f64; 2]>,
    #[serde(default)]
    holes: Vec<Vec<[f64; 2]>>,
}

/// An axis-aligned rectangle in the same coordinate space as the regions:
/// center plus full extents. Used for the fitted complement container and
/// for glyph keep-out boxes.
#[derive(serde::Deserialize)]
struct RectIn {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Leader strategy: the edge type plus the placement algorithm for it. All
/// fields optional. `r#type` is `"straight"` (default) or `"elbow"`;
/// `placement` selects the straight-edge exterior solver (`"raycast"` default,
/// `"force_directed"`, or `"matched"`) and is ignored for `"elbow"`. `margin`
/// applies to both edge types; `iterations` only to `force_directed`; `min_gap`
/// only to `elbow`.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct LeaderIn {
    r#type: Option<String>,
    placement: Option<String>,
    margin: Option<f64>,
    iterations: Option<usize>,
    min_gap: Option<f64>,
}

/// Placement strategy. All fields optional; omitted fields keep the core
/// defaults (straight leaders, raycast placement, POI tether).
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct StrategyIn {
    leader: Option<LeaderIn>,
    precision: Option<f64>,
    tether: Option<String>,
    leader_gap: Option<f64>,
}

/// Input to [`eunoia_place_labels`]. `regions` and `sizes` are keyed by
/// canonical combination strings; only regions present in both get a placement.
#[derive(serde::Deserialize)]
struct PlaceLabelsInput {
    regions: BTreeMap<String, Vec<PieceIn>>,
    sizes: BTreeMap<String, [f64; 2]>,
    #[serde(default)]
    container: Option<RectIn>,
    #[serde(default)]
    strategy: Option<StrategyIn>,
}

/// One resolved placement. `kind` is `interior` | `exterior_raycast` |
/// `exterior_force_directed` | `exterior_elbow` | `exterior_matched` (|
/// `unknown` for any future variant). `tether`/`leader_end`/`leader_waypoints`
/// are only set for exterior placements that need a leader line.
#[derive(Serialize)]
struct PlacementOut {
    anchor: [f64; 2],
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tether: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    leader_end: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    leader_waypoints: Vec<[f64; 2]>,
}

/// Success payload for [`eunoia_place_labels`]: `combo -> PlacementOut`.
#[derive(Serialize)]
struct PlaceLabelsOut {
    placements: BTreeMap<String, PlacementOut>,
}

/// Map a snake_case tether name to a [`TetherSource`].
fn parse_tether(name: &str) -> Result<TetherSource, String> {
    match name {
        "poi" => Ok(TetherSource::Poi),
        "boundary" => Ok(TetherSource::Boundary),
        other => Err(format!("invalid tether '{other}' (want poi|boundary)")),
    }
}

/// Resolve an optional [`StrategyIn`] into a core [`PlacementStrategy`],
/// validating every enum token up front so a bad value errors regardless of the
/// rest of the input (mirroring `FitConfig::from_input`).
fn strategy_from_input(strategy: Option<StrategyIn>) -> Result<PlacementStrategy, String> {
    let strategy = strategy.unwrap_or_default();
    let leader_in = strategy.leader.unwrap_or_default();
    let leader = match leader_in.r#type.as_deref() {
        None | Some("straight") => {
            let exterior = match leader_in.placement.as_deref() {
                None | Some("raycast") => ExteriorPolicy::Raycast {
                    margin: leader_in.margin,
                },
                Some("force_directed") => ExteriorPolicy::ForceDirected {
                    margin: leader_in.margin,
                    iterations: leader_in.iterations,
                },
                Some("matched") => ExteriorPolicy::Matched {
                    margin: leader_in.margin,
                },
                Some(other) => {
                    return Err(format!(
                        "invalid placement '{other}' (want raycast|force_directed|matched)"
                    ));
                }
            };
            LeaderStrategy::Straight(exterior)
        }
        Some("elbow") => LeaderStrategy::Elbow(
            ElbowOptions::default()
                .margin(leader_in.margin)
                .min_gap(leader_in.min_gap),
        ),
        Some(other) => {
            return Err(format!(
                "invalid leader type '{other}' (want straight|elbow)"
            ));
        }
    };
    let tether = match strategy.tether.as_deref() {
        None => TetherSource::Poi,
        Some(name) => parse_tether(name)?,
    };
    Ok(PlacementStrategy::default()
        .leader(leader)
        .precision(strategy.precision.unwrap_or(0.01))
        .tether(tether)
        .leader_gap(strategy.leader_gap.unwrap_or(0.0)))
}

fn place_labels_impl(input: PlaceLabelsInput) -> Result<PlaceLabelsOut, String> {
    let strategy = strategy_from_input(input.strategy)?;

    let container = input
        .container
        .map(|c| Rectangle::try_new(Point::new(c.x, c.y), c.width, c.height))
        .transpose()
        .map_err(|e| format!("invalid container: {e}"))?;

    let sizes: HashMap<String, (f64, f64)> = input
        .sizes
        .into_iter()
        .map(|(k, [w, h])| (k, (w, h)))
        .collect();

    // Rebuild each region's pieces through the public classifier: `RegionPiece`
    // is `#[non_exhaustive]`, so the outer/holes pairing must be re-derived from
    // a flat ring list rather than hand-built. Unparseable combo keys are
    // skipped (the caller simply sees no placement for them).
    let to_polygon = |pts: Vec<[f64; 2]>| -> Polygon {
        Polygon::new(pts.into_iter().map(|p| Point::new(p[0], p[1])).collect())
    };
    let mut region_map: HashMap<Combination, Vec<RegionPiece>> =
        HashMap::with_capacity(input.regions.len());
    for (key, pieces_in) in input.regions {
        // `Combination`'s `FromStr` is `Infallible`, so the match is exhaustive
        // with just the `Ok` arm (the `Err` variant is uninhabited).
        let Ok(combo) = key.parse::<Combination>();
        let pieces: Vec<RegionPiece> = pieces_in
            .into_iter()
            .flat_map(|p| {
                let mut rings = vec![to_polygon(p.outer)];
                rings.extend(p.holes.into_iter().map(to_polygon));
                classify_into_pieces(rings)
            })
            .collect();
        region_map.insert(combo, pieces);
    }
    let regions = RegionPolygons::from_map(region_map);

    let placements = place_labels(&regions, &sizes, container.as_ref(), &strategy);

    let placements = placements
        .into_iter()
        .map(|(key, p)| {
            let kind = match p.kind {
                PlacementKind::Interior => "interior",
                PlacementKind::ExteriorRaycast => "exterior_raycast",
                PlacementKind::ExteriorForceDirected => "exterior_force_directed",
                PlacementKind::ExteriorElbow => "exterior_elbow",
                PlacementKind::ExteriorMatched => "exterior_matched",
                // `PlacementKind` is `#[non_exhaustive]`; surface unknown future
                // kinds rather than failing to compile when one is added.
                _ => "unknown",
            };
            (
                key,
                PlacementOut {
                    anchor: [p.anchor.x(), p.anchor.y()],
                    kind,
                    tether: p.tether.map(|t| [t.x(), t.y()]),
                    leader_end: p.leader_end.map(|t| [t.x(), t.y()]),
                    leader_waypoints: p.leader_waypoints.iter().map(|t| [t.x(), t.y()]).collect(),
                },
            )
        })
        .collect();

    Ok(PlaceLabelsOut { placements })
}

// ============================================================================
// Set-label placement — JSON contract + impl
// ============================================================================
//
// The sibling of `eunoia_place_labels`, mirroring the WASM `place_set_labels`
// surface. Keyed by set rather than by region, and it hugs each set's own
// outline instead of exiting to the diagram border, so it needs the shape
// outlines (`plot_data.shape_outlines`) rather than the region pieces.

/// Set-label strategy. All fields optional; omitted fields keep the core
/// defaults (half-line-height margin, 180 candidate angles, no extra
/// obstacles, `precision = 0.01`).
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct SetLabelStrategyIn {
    margin: Option<f64>,
    angular_steps: Option<usize>,
    obstacles: Option<Vec<RectIn>>,
    precision: Option<f64>,
}

/// Input to [`eunoia_place_set_labels`]. `outlines` and `sizes` are keyed by
/// set name; only sets present in both get a placement. `outlines` values are
/// closed rings of `[x, y]` pairs — the `shape_outlines` of a `plot_data`
/// payload.
#[derive(serde::Deserialize)]
struct PlaceSetLabelsInput {
    outlines: BTreeMap<String, Vec<[f64; 2]>>,
    sizes: BTreeMap<String, [f64; 2]>,
    #[serde(default)]
    container: Option<RectIn>,
    #[serde(default)]
    strategy: Option<SetLabelStrategyIn>,
}

/// One resolved set-label placement. `kind` is always `exterior_set`; there is
/// no leader geometry because the label is adjacent to the shape it names.
#[derive(Serialize)]
struct SetPlacementOut {
    anchor: [f64; 2],
    kind: &'static str,
}

/// Success payload for [`eunoia_place_set_labels`]: `set -> SetPlacementOut`.
#[derive(Serialize)]
struct PlaceSetLabelsOut {
    placements: BTreeMap<String, SetPlacementOut>,
}

fn place_set_labels_impl(input: PlaceSetLabelsInput) -> Result<PlaceSetLabelsOut, String> {
    let strategy_in = input.strategy.unwrap_or_default();
    let mut strategy = SetLabelStrategy::default()
        .margin(strategy_in.margin)
        .precision(strategy_in.precision.unwrap_or(0.01));
    if let Some(steps) = strategy_in.angular_steps {
        strategy = strategy.angular_steps(steps);
    }
    if let Some(obstacles) = strategy_in.obstacles {
        let boxes: Vec<Rectangle> = obstacles
            .into_iter()
            .map(|r| Rectangle::try_new(Point::new(r.x, r.y), r.width, r.height))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("invalid obstacle: {e}"))?;
        strategy = strategy.obstacles(boxes);
    }

    let container = input
        .container
        .map(|c| Rectangle::try_new(Point::new(c.x, c.y), c.width, c.height))
        .transpose()
        .map_err(|e| format!("invalid container: {e}"))?;

    let sizes: HashMap<String, (f64, f64)> = input
        .sizes
        .into_iter()
        .map(|(k, [w, h])| (k, (w, h)))
        .collect();

    let outlines: HashMap<String, Polygon> = input
        .outlines
        .into_iter()
        .map(|(name, ring)| {
            (
                name,
                Polygon::new(ring.into_iter().map(|p| Point::new(p[0], p[1])).collect()),
            )
        })
        .collect();

    let placements = place_set_labels(&outlines, &sizes, container.as_ref(), &strategy)
        .into_iter()
        .map(|(key, p)| {
            (
                key,
                SetPlacementOut {
                    anchor: [p.anchor.x(), p.anchor.y()],
                    kind: "exterior_set",
                },
            )
        })
        .collect();

    Ok(PlaceSetLabelsOut { placements })
}

/// Glyph options for [`eunoia_place_glyphs`]. All fields optional; omitted
/// fields keep the core defaults (uniform arrangement, auto radius,
/// `gap = 0.25`, `seed = 0`, `precision = 0.01`, `max_attempts = 300`, no
/// obstacles). `arrangement` is `"uniform"` (default) or `"random"`.
/// `obstacles` are diagram-wide keep-out boxes (usually the caller's measured
/// label boxes); degenerate ones are ignored rather than rejected.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct GlyphOptionsIn {
    arrangement: Option<String>,
    radius: Option<f64>,
    gap: Option<f64>,
    seed: Option<u64>,
    precision: Option<f64>,
    max_attempts: Option<u32>,
    obstacles: Option<Vec<RectIn>>,
}

/// Input to [`eunoia_place_glyphs`]. `regions` and `counts` are keyed by
/// canonical combination strings (`""` for the complement region); only
/// regions present in both, with a positive count, get glyphs.
#[derive(serde::Deserialize)]
struct PlaceGlyphsInput {
    regions: BTreeMap<String, Vec<PieceIn>>,
    counts: BTreeMap<String, u64>,
    #[serde(default)]
    options: Option<GlyphOptionsIn>,
}

/// Success payload for [`eunoia_place_glyphs`]: the diagram-wide glyph
/// radius, glyph center points per combination, and any per-combination
/// shortfall (only populated when a fixed radius overflows a region).
#[derive(Serialize)]
struct PlaceGlyphsOut {
    radius: f64,
    positions: BTreeMap<String, Vec<[f64; 2]>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    unplaced: BTreeMap<String, u64>,
}

/// Resolve an optional [`GlyphOptionsIn`] into core [`GlyphOptions`],
/// validating the arrangement token up front (mirroring
/// [`strategy_from_input`]).
fn glyph_options_from_input(options: Option<GlyphOptionsIn>) -> Result<GlyphOptions, String> {
    let options = options.unwrap_or_default();
    let arrangement = match options.arrangement.as_deref() {
        None | Some("uniform") => GlyphArrangement::Uniform,
        Some("random") => GlyphArrangement::Random,
        Some(other) => {
            return Err(format!(
                "invalid arrangement '{other}' (want uniform|random)"
            ));
        }
    };
    let mut out = GlyphOptions::default()
        .arrangement(arrangement)
        .radius(options.radius);
    if let Some(gap) = options.gap {
        out = out.gap(gap);
    }
    if let Some(seed) = options.seed {
        out = out.seed(seed);
    }
    if let Some(precision) = options.precision {
        out = out.precision(precision);
    }
    if let Some(max_attempts) = options.max_attempts {
        out = out.max_attempts(max_attempts);
    }
    if let Some(obstacles) = options.obstacles {
        // `Rectangle::new`, not `try_new`: an empty label legitimately
        // measures 0 x 0, and the core drops degenerate boxes silently
        // rather than failing the whole call over one.
        out = out.obstacles(
            obstacles
                .into_iter()
                .map(|r| Rectangle::new(Point::new(r.x, r.y), r.width, r.height)),
        );
    }
    Ok(out)
}

fn place_glyphs_impl(input: PlaceGlyphsInput) -> Result<PlaceGlyphsOut, String> {
    let options = glyph_options_from_input(input.options)?;

    let counts: HashMap<String, usize> = input
        .counts
        .into_iter()
        .map(|(k, n)| match usize::try_from(n) {
            Ok(n) => Ok((k, n)),
            Err(_) => Err(format!("count for '{k}' does not fit in usize")),
        })
        .collect::<Result<_, _>>()?;

    // Rebuild each region's pieces through the public classifier, exactly as
    // in `place_labels_impl` — `RegionPiece` is `#[non_exhaustive]`.
    let to_polygon = |pts: Vec<[f64; 2]>| -> Polygon {
        Polygon::new(pts.into_iter().map(|p| Point::new(p[0], p[1])).collect())
    };
    let mut region_map: HashMap<Combination, Vec<RegionPiece>> =
        HashMap::with_capacity(input.regions.len());
    for (key, pieces_in) in input.regions {
        let Ok(combo) = key.parse::<Combination>();
        let pieces: Vec<RegionPiece> = pieces_in
            .into_iter()
            .flat_map(|p| {
                let mut rings = vec![to_polygon(p.outer)];
                rings.extend(p.holes.into_iter().map(to_polygon));
                classify_into_pieces(rings)
            })
            .collect();
        region_map.insert(combo, pieces);
    }
    let regions = RegionPolygons::from_map(region_map);

    let result = place_glyphs(&regions, &counts, &options);

    Ok(PlaceGlyphsOut {
        radius: result.radius,
        positions: result
            .positions
            .into_iter()
            .map(|(k, pts)| (k, pts.into_iter().map(|p| [p.x(), p.y()]).collect()))
            .collect(),
        unplaced: result
            .unplaced
            .into_iter()
            .map(|(k, n)| (k, n as u64))
            .collect(),
    })
}

/// Glyph-box options for [`eunoia_place_glyph_boxes`]. All fields optional;
/// omitted fields keep the core defaults (uniform row/shelf packing, auto
/// scale, `min_scale = 0.35`, `gap = 0.25`, `seed = 0`, `precision = 0.01`,
/// `max_attempts = 300`, no obstacles). `arrangement` is `"uniform"`
/// (default) or `"random"`. `gap` is a fraction of the row height, not of a
/// radius; `obstacles` behave exactly as in [`GlyphOptionsIn`].
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct GlyphBoxOptionsIn {
    arrangement: Option<String>,
    scale: Option<f64>,
    min_scale: Option<f64>,
    gap: Option<f64>,
    seed: Option<u64>,
    precision: Option<f64>,
    max_attempts: Option<u32>,
    obstacles: Option<Vec<RectIn>>,
}

/// Input to [`eunoia_place_glyph_boxes`]. `regions` and `sizes` are keyed by
/// canonical combination strings (`""` for the complement region); `sizes`
/// holds one `[width, height]` per item, **in the order the caller wants them
/// placed**, since the packer returns a prefix.
#[derive(serde::Deserialize)]
struct PlaceGlyphBoxesInput {
    regions: BTreeMap<String, Vec<PieceIn>>,
    sizes: BTreeMap<String, Vec<[f64; 2]>>,
    #[serde(default)]
    options: Option<GlyphBoxOptionsIn>,
}

/// Success payload for [`eunoia_place_glyph_boxes`]: the diagram-wide box
/// scale, one `[cx, cy, w, h]` per placed item, and any per-combination
/// shortfall.
#[derive(Serialize)]
struct PlaceGlyphBoxesOut {
    scale: f64,
    boxes: BTreeMap<String, Vec<[f64; 4]>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    unplaced: BTreeMap<String, u64>,
}

/// Resolve an optional [`GlyphBoxOptionsIn`] into core [`GlyphBoxOptions`],
/// validating the arrangement token up front (mirroring
/// [`glyph_options_from_input`]).
fn glyph_box_options_from_input(
    options: Option<GlyphBoxOptionsIn>,
) -> Result<GlyphBoxOptions, String> {
    let options = options.unwrap_or_default();
    let arrangement = match options.arrangement.as_deref() {
        None | Some("uniform") => GlyphArrangement::Uniform,
        Some("random") => GlyphArrangement::Random,
        Some(other) => {
            return Err(format!(
                "invalid arrangement '{other}' (want uniform|random)"
            ));
        }
    };
    let mut out = GlyphBoxOptions::default()
        .arrangement(arrangement)
        .scale(options.scale);
    if let Some(min_scale) = options.min_scale {
        out = out.min_scale(min_scale);
    }
    if let Some(gap) = options.gap {
        out = out.gap(gap);
    }
    if let Some(seed) = options.seed {
        out = out.seed(seed);
    }
    if let Some(precision) = options.precision {
        out = out.precision(precision);
    }
    if let Some(max_attempts) = options.max_attempts {
        out = out.max_attempts(max_attempts);
    }
    if let Some(obstacles) = options.obstacles {
        // `Rectangle::new`, not `try_new`: see `glyph_options_from_input`.
        out = out.obstacles(
            obstacles
                .into_iter()
                .map(|r| Rectangle::new(Point::new(r.x, r.y), r.width, r.height)),
        );
    }
    Ok(out)
}

fn place_glyph_boxes_impl(input: PlaceGlyphBoxesInput) -> Result<PlaceGlyphBoxesOut, String> {
    let options = glyph_box_options_from_input(input.options)?;

    let sizes: HashMap<String, Vec<(f64, f64)>> = input
        .sizes
        .into_iter()
        .map(|(k, items)| (k, items.into_iter().map(|wh| (wh[0], wh[1])).collect()))
        .collect();

    // Rebuild each region's pieces through the public classifier, exactly as
    // in `place_glyphs_impl` — `RegionPiece` is `#[non_exhaustive]`.
    let to_polygon = |pts: Vec<[f64; 2]>| -> Polygon {
        Polygon::new(pts.into_iter().map(|p| Point::new(p[0], p[1])).collect())
    };
    let mut region_map: HashMap<Combination, Vec<RegionPiece>> =
        HashMap::with_capacity(input.regions.len());
    for (key, pieces_in) in input.regions {
        let Ok(combo) = key.parse::<Combination>();
        let pieces: Vec<RegionPiece> = pieces_in
            .into_iter()
            .flat_map(|p| {
                let mut rings = vec![to_polygon(p.outer)];
                rings.extend(p.holes.into_iter().map(to_polygon));
                classify_into_pieces(rings)
            })
            .collect();
        region_map.insert(combo, pieces);
    }
    let regions = RegionPolygons::from_map(region_map);

    let result = place_glyph_boxes(&regions, &sizes, &options);

    Ok(PlaceGlyphBoxesOut {
        scale: result.scale,
        boxes: result
            .boxes
            .into_iter()
            .map(|(k, rects)| {
                let quads = rects
                    .into_iter()
                    .map(|r| [r.center().x(), r.center().y(), r.width(), r.height()])
                    .collect();
                (k, quads)
            })
            .collect(),
        unplaced: result
            .unplaced
            .into_iter()
            .map(|(k, n)| (k, n as u64))
            .collect(),
    })
}

// ============================================================================
// FFI boundary
// ============================================================================

/// Render a `Result<T, String>` to the JSON envelope string. On `Ok`, the
/// payload fields are flattened next to `ok: true`; on `Err`, the message
/// becomes `{"ok": false, "error": ...}`.
fn to_json<T: Serialize>(result: Result<T, String>) -> String {
    match result {
        Ok(payload) => serde_json::to_string(&OkResponse { ok: true, payload })
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
fn run<I, O, F>(input: *const c_char, f: F) -> *mut c_char
where
    I: DeserializeOwned,
    O: Serialize,
    F: FnOnce(I) -> Result<O, String>,
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

/// Resolve collision-aware label positions for a set of region polygons.
/// `input` is a JSON `PlaceLabelsInput` (region pieces + caller-measured label
/// sizes + an optional placement strategy); the success envelope carries a
/// `placements` map. Free the result with [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_place_labels(input: *const c_char) -> *mut c_char {
    run(input, place_labels_impl)
}

/// Resolve exterior set-label positions — one per set, hugging that set's own
/// shape, with no leader line. `input` is a JSON `PlaceSetLabelsInput` (shape
/// outlines + caller-measured label sizes + an optional strategy); the success
/// envelope carries a `placements` map keyed by set name. Free the result with
/// [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_place_set_labels(input: *const c_char) -> *mut c_char {
    run(input, place_set_labels_impl)
}

/// Pack equally-sized circular glyphs inside a set of region polygons.
/// `input` is a JSON `PlaceGlyphsInput` (region pieces + per-region counts +
/// optional glyph options); the success envelope carries the diagram-wide
/// `radius`, a `positions` map, and an `unplaced` map when a fixed radius
/// overflows. Free the result with [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_place_glyphs(input: *const c_char) -> *mut c_char {
    run(input, place_glyphs_impl)
}

/// Pack per-item text boxes (member names) inside a set of region polygons.
/// `input` is a JSON `PlaceGlyphBoxesInput` (region pieces + per-region
/// caller-measured `[width, height]` lists + optional glyph-box options); the
/// success envelope carries the diagram-wide `scale`, a `boxes` map of
/// `[cx, cy, w, h]` quads, and an `unplaced` map when a region overflows.
/// `boxes[key]` is always a prefix of `sizes[key]`. Free the result with
/// [`eunoia_free`].
#[unsafe(no_mangle)]
pub extern "C" fn eunoia_place_glyph_boxes(input: *const c_char) -> *mut c_char {
    run(input, place_glyph_boxes_impl)
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
    fn plot_data_exposes_set_anchor_regions() {
        // Two overlapping sets: both have an exclusive lobe, so each set label
        // anchors to its own exclusive region (`A` -> "A", `B` -> "B").
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":5},{"combination":"B","size":3},
                {"combination":"A&B","size":1}],"seed":1}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let plot = &v["plot_data"];

        let regions = plot["set_anchor_regions"].as_object().unwrap();
        assert_eq!(regions["A"], "A");
        assert_eq!(regions["B"], "B");

        // Documented invariant: set_anchors[s] == region_anchors[regions[s]],
        // so a binding can pair set labels with region quantities by key alone,
        // never by comparing anchor points with floating-point tolerance.
        for (set, combo) in regions {
            let combo = combo.as_str().unwrap();
            assert_eq!(
                plot["set_anchors"][set], plot["region_anchors"][combo],
                "set {set} should share an anchor with region {combo}"
            );
        }
    }

    #[test]
    fn euler_max_sets_is_forwarded() {
        // Lowering the cap below the set count makes the spec build fail,
        // proving `max_sets` reaches `DiagramSpecBuilder::max_sets`.
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":5},{"combination":"B","size":3},
                {"combination":"A&B","size":1}],"seed":1,"max_sets":1}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);

        // The same spec at the default cap fits cleanly.
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":5},{"combination":"B","size":3},
                {"combination":"A&B","size":1}],"seed":1,"max_sets":32}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shapes"].as_array().unwrap().len(), 2);
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
    fn euler_rotated_rectangle_fits_and_carries_rotation() {
        let out = call(
            eunoia_euler,
            r#"{"sets":[{"combination":"A","size":4},{"combination":"B","size":4},
                {"combination":"A&B","size":2}],"shape":"rotated_rectangle","seed":7}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shape"], "rotated_rectangle");
        assert_eq!(v["shapes"].as_array().unwrap().len(), 2);
        assert_eq!(v["shapes"][0]["type"], "rotated_rectangle");
        // The rotation field must be present in the tagged output.
        assert!(v["shapes"][0]["rotation"].is_number());
        assert!(v["metrics"]["loss"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn venn_three_set_rotated_rectangle() {
        let out = call(
            eunoia_venn,
            r#"{"names":["A","B","C"],"shape":"rotated_rectangle"}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shapes"].as_array().unwrap().len(), 3);
        assert_eq!(v["shapes"][0]["type"], "rotated_rectangle");
    }

    #[test]
    fn venn_complement_emits_container() {
        // A complement size frames the Venn diagram with a bounding container,
        // which `extract` surfaces as the top-level `container` field.
        let out = call(eunoia_venn, r#"{"names":["A","B"],"complement":4}"#);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let container = &v["container"];
        assert!(
            container.is_object(),
            "expected a container, got {container}"
        );
        assert!(container["width"].as_f64().unwrap() > 0.0);
        assert!(container["height"].as_f64().unwrap() > 0.0);

        // Without a complement there is no container frame.
        let out = call(eunoia_venn, r#"{"names":["A","B"]}"#);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v.get("container").is_none());
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
    fn euler_mads_and_cmaes_optimizers_accepted() {
        // The MADS (v1.6) and plain CMA-ES optimizer tokens both map to core
        // variants and fit cleanly.
        for opt in ["mads", "cmaes"] {
            let out = call(eunoia_euler, &two_set(&format!(r#","optimizer":"{opt}""#)));
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["ok"], true, "optimizer {opt}");
            assert_eq!(v["shapes"].as_array().unwrap().len(), 2);
        }
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

    // --- Phase 4(b) plot knobs ---

    fn outline_len(v: &serde_json::Value, set: &str) -> usize {
        v["plot_data"]["shape_outlines"][set]
            .as_array()
            .unwrap_or_else(|| panic!("missing shape_outlines[{set}]"))
            .len()
    }

    #[test]
    fn euler_plot_knobs_are_honored() {
        // `n_vertices` controls how densely each set's outline is polygonized,
        // so it is directly observable in `plot_data.shape_outlines`. A low
        // value yields a far coarser outline than the 200-vertex default.
        let coarse = call(eunoia_euler, &two_set(r#","n_vertices":40"#));
        let coarse: serde_json::Value = serde_json::from_str(&coarse).unwrap();
        assert_eq!(coarse["ok"], true);
        let coarse_len = outline_len(&coarse, "A");
        assert!(
            (30..=60).contains(&coarse_len),
            "n_vertices=40 outline had {coarse_len} points"
        );

        let default = call(eunoia_euler, &two_set(""));
        let default: serde_json::Value = serde_json::from_str(&default).unwrap();
        let default_len = outline_len(&default, "A");
        assert!(
            (150..=250).contains(&default_len),
            "default outline had {default_len} points"
        );
        assert!(coarse_len < default_len);
    }

    #[test]
    fn euler_plot_knobs_accepted() {
        // `label_precision` + `sliver_threshold` have no easily-asserted scalar
        // effect; check they are accepted and leave `plot_data` well-formed.
        let out = call(
            eunoia_euler,
            &two_set(r#","label_precision":0.05,"sliver_threshold":1e-2"#),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["plot_data"]["region_pieces"].is_object());
        assert!(outline_len(&v, "A") > 0);
    }

    // ------------------------------------------------------------------------
    // Phase 4(c) label placement
    // ------------------------------------------------------------------------

    /// Region pieces from a default two-set fit — exactly the `{combo:
    /// [{outer, holes}]}` shape `eunoia_place_labels` expects for `regions`.
    fn region_pieces() -> serde_json::Value {
        let out = call(eunoia_euler, &two_set(""));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        v["plot_data"]["region_pieces"].clone()
    }

    /// Build a `place_labels` input from `sizes` + an optional `strategy`.
    fn place_input(sizes: serde_json::Value, strategy: Option<serde_json::Value>) -> String {
        let mut input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": sizes,
        });
        if let Some(s) = strategy {
            input["strategy"] = s;
        }
        input.to_string()
    }

    #[test]
    fn place_labels_interior_when_small() {
        // Labels much smaller than every region fit inside: interior placement,
        // no leader geometry.
        let sizes = serde_json::json!({ "A": [0.1, 0.1], "B": [0.1, 0.1], "A&B": [0.05, 0.05] });
        let out = call(eunoia_place_labels, &place_input(sizes, None));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let p = &v["placements"];
        for combo in ["A", "B", "A&B"] {
            assert_eq!(p[combo]["kind"], "interior", "combo {combo}");
            assert_eq!(p[combo]["anchor"].as_array().unwrap().len(), 2);
            assert!(p[combo].get("tether").is_none(), "interior has no tether");
        }
    }

    #[test]
    fn place_labels_exterior_when_oversized() {
        // A label far larger than its region cannot fit inside, so it is pushed
        // outside with a leader line back to a tether point.
        let sizes = serde_json::json!({ "A&B": [10.0, 10.0] });
        let out = call(eunoia_place_labels, &place_input(sizes, None));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let placement = &v["placements"]["A&B"];
        let kind = placement["kind"].as_str().unwrap();
        assert!(kind.starts_with("exterior_"), "kind was {kind}");
        assert_eq!(placement["tether"].as_array().unwrap().len(), 2);
        assert_eq!(placement["leader_end"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn place_labels_force_directed_and_elbow_accepted() {
        let sizes = serde_json::json!({ "A": [10.0, 10.0], "B": [10.0, 10.0] });

        // Force-directed straight leaders.
        let strat =
            serde_json::json!({ "leader": { "placement": "force_directed", "iterations": 50 } });
        let out = call(
            eunoia_place_labels,
            &place_input(sizes.clone(), Some(strat)),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert!(
            v["placements"]["A"]["kind"]
                .as_str()
                .unwrap()
                .starts_with("exterior_")
        );

        // Elbow leaders route through a waypoint (the knee).
        let strat = serde_json::json!({ "leader": { "type": "elbow" } });
        let out = call(eunoia_place_labels, &place_input(sizes, Some(strat)));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let waypoints = v["placements"]["A"]["leader_waypoints"].as_array();
        assert!(
            waypoints.is_some_and(|w| !w.is_empty()),
            "elbow leader should carry waypoints, got {:?}",
            v["placements"]["A"]
        );
    }

    #[test]
    fn place_labels_matched_round_trips() {
        // The "matched" boundary-labeling placement parses and reports its
        // `exterior_matched` kind with a straight leader (no waypoints).
        let sizes = serde_json::json!({ "A": [10.0, 10.0], "B": [10.0, 10.0] });
        let strat = serde_json::json!({ "leader": { "placement": "matched" } });
        let out = call(eunoia_place_labels, &place_input(sizes, Some(strat)));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let a = &v["placements"]["A"];
        assert_eq!(a["kind"], "exterior_matched");
        assert_eq!(a["tether"].as_array().unwrap().len(), 2);
        assert!(
            a.get("leader_waypoints").is_none(),
            "matched leaders are straight (no waypoints), got {a:?}"
        );
    }

    #[test]
    fn place_labels_bad_tokens_are_reported() {
        let sizes = serde_json::json!({ "A": [0.1, 0.1] });
        for (strat, bad) in [
            (
                serde_json::json!({ "leader": { "placement": "bogus" } }),
                "bogus",
            ),
            (
                serde_json::json!({ "leader": { "type": "zigzag" } }),
                "zigzag",
            ),
            (serde_json::json!({ "tether": "middle" }), "middle"),
        ] {
            let out = call(
                eunoia_place_labels,
                &place_input(sizes.clone(), Some(strat)),
            );
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["ok"], false);
            assert!(
                v["error"].as_str().unwrap().contains(bad),
                "error should mention '{bad}', got {}",
                v["error"]
            );
        }
    }

    // ------------------------------------------------------------------------
    // Set-label placement
    // ------------------------------------------------------------------------

    /// Shape outlines from a default two-set fit — exactly the
    /// `{set: [[x, y], ...]}` shape `eunoia_place_set_labels` expects.
    fn shape_outlines() -> serde_json::Value {
        let out = call(eunoia_euler, &two_set(""));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        v["plot_data"]["shape_outlines"].clone()
    }

    #[test]
    fn place_set_labels_round_trips() {
        let input = serde_json::json!({
            "outlines": shape_outlines(),
            "sizes": { "A": [0.2, 0.1], "B": [0.2, 0.1] },
        })
        .to_string();
        let out = call(eunoia_place_set_labels, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);

        for set in ["A", "B"] {
            let p = &v["placements"][set];
            assert_eq!(p["kind"], "exterior_set", "set {set}");
            assert_eq!(p["anchor"].as_array().unwrap().len(), 2);
            // No leader geometry — the label is adjacent to its shape.
            assert!(p.get("tether").is_none(), "set {set} should have no tether");
            assert!(p.get("leader_end").is_none(), "set {set} has no leader end");
        }

        // The two labels must land on opposite sides: each rotates away from
        // its neighbour.
        let ax = v["placements"]["A"]["anchor"][0].as_f64().unwrap();
        let bx = v["placements"]["B"]["anchor"][0].as_f64().unwrap();
        assert!(ax < bx, "A's label ({ax}) should sit left of B's ({bx})");
    }

    #[test]
    fn place_set_labels_accepts_full_strategy() {
        let input = serde_json::json!({
            "outlines": shape_outlines(),
            "sizes": { "A": [0.2, 0.1], "B": [0.2, 0.1] },
            "strategy": {
                "margin": 0.05,
                "angular_steps": 64,
                "precision": 0.02,
                "obstacles": [{ "x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1 }],
            },
        })
        .to_string();
        let out = call(eunoia_place_set_labels, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["placements"].as_object().unwrap().len(), 2);
    }

    #[test]
    fn place_set_labels_reports_bad_input() {
        let out = call(eunoia_place_set_labels, r#"{"outlines": {}}"#);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["error"].as_str().unwrap().contains("sizes"));
    }

    // ------------------------------------------------------------------------
    // Glyph placement
    // ------------------------------------------------------------------------

    #[test]
    fn place_glyphs_round_trips() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 8, "B": 5, "A&B": 2 },
        })
        .to_string();
        let out = call(eunoia_place_glyphs, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert!(v["radius"].as_f64().unwrap() > 0.0);
        for (combo, n) in [("A", 8), ("B", 5), ("A&B", 2)] {
            let points = v["positions"][combo].as_array().unwrap();
            assert_eq!(points.len(), n, "combo {combo}");
            assert_eq!(points[0].as_array().unwrap().len(), 2);
        }
        assert!(v.get("unplaced").is_none(), "auto radius places everything");
    }

    #[test]
    fn place_glyphs_random_and_fixed_radius_accepted() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 4 },
            "options": { "arrangement": "random", "radius": 0.05, "seed": 3 },
        })
        .to_string();
        let out = call(eunoia_place_glyphs, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert_eq!(v["radius"].as_f64().unwrap(), 0.05);
        assert_eq!(v["positions"]["A"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn place_glyphs_obstacles_are_honored() {
        // The intended pipeline: place the labels, then hand their boxes to
        // the glyph packer as keep-out boxes.
        let boxes = [("A", 0.1, 0.1), ("B", 0.1, 0.1), ("A&B", 0.05, 0.05)];
        let sizes = serde_json::json!({ "A": [0.1, 0.1], "B": [0.1, 0.1], "A&B": [0.05, 0.05] });
        let labels = call(eunoia_place_labels, &place_input(sizes, None));
        let labels: serde_json::Value = serde_json::from_str(&labels).unwrap();
        let obstacles: Vec<serde_json::Value> = boxes
            .iter()
            .map(|(key, w, h)| {
                let anchor = labels["placements"][key]["anchor"].as_array().unwrap();
                serde_json::json!({
                    "x": anchor[0].as_f64().unwrap(),
                    "y": anchor[1].as_f64().unwrap(),
                    "width": w,
                    "height": h,
                })
            })
            .collect();

        let radius = 0.02;
        let input = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 8, "B": 5, "A&B": 2 },
            "options": { "radius": radius, "obstacles": obstacles },
        })
        .to_string();
        let out = call(eunoia_place_glyphs, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert!(v.get("unplaced").is_none(), "got {v}");

        let clearance = radius * 1.25;
        for (combo, n) in [("A", 8), ("B", 5), ("A&B", 2)] {
            let points = v["positions"][combo].as_array().unwrap();
            assert_eq!(points.len(), n, "combo {combo}");
            for point in points {
                let (px, py) = (point[0].as_f64().unwrap(), point[1].as_f64().unwrap());
                for (i, (_, w, h)) in boxes.iter().enumerate() {
                    let cx = obstacles[i]["x"].as_f64().unwrap();
                    let cy = obstacles[i]["y"].as_f64().unwrap();
                    let dx = ((px - cx).abs() - 0.5 * w).max(0.0);
                    let dy = ((py - cy).abs() - 0.5 * h).max(0.0);
                    assert!(
                        dx.hypot(dy) >= clearance - 1e-9,
                        "glyph in {combo} sits on a label box"
                    );
                }
            }
        }
    }

    #[test]
    fn place_glyphs_degenerate_obstacle_is_dropped() {
        // A 0 x 0 box (an empty label measures exactly that) is ignored, not
        // an error.
        let with_junk = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 4 },
            "options": {
                "obstacles": [{ "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0 }],
            },
        })
        .to_string();
        let plain = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 4 },
        })
        .to_string();

        let a: serde_json::Value =
            serde_json::from_str(&call(eunoia_place_glyphs, &with_junk)).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(&call(eunoia_place_glyphs, &plain)).unwrap();
        assert_eq!(a["ok"], true, "got {a}");
        assert_eq!(a, b);
    }

    #[test]
    fn place_glyphs_bad_arrangement_is_reported() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "counts": { "A": 1 },
            "options": { "arrangement": "spiral" },
        })
        .to_string();
        let out = call(eunoia_place_glyphs, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["error"].as_str().unwrap().contains("spiral"));
    }

    /// `n` copies of one measured box, as the wire format wants them.
    fn box_sizes(n: usize, w: f64, h: f64) -> serde_json::Value {
        serde_json::Value::Array(vec![serde_json::json!([w, h]); n])
    }

    #[test]
    fn place_glyph_boxes_round_trips() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": {
                "A": box_sizes(4, 0.12, 0.05),
                "B": box_sizes(2, 0.12, 0.05),
                "A&B": box_sizes(1, 0.08, 0.05),
            },
        })
        .to_string();
        let out = call(eunoia_place_glyph_boxes, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        let scale = v["scale"].as_f64().unwrap();
        assert!(scale > 0.0 && scale <= 1.0, "scale {scale}");
        for (combo, n) in [("A", 4), ("B", 2), ("A&B", 1)] {
            let boxes = v["boxes"][combo].as_array().unwrap();
            assert_eq!(boxes.len(), n, "combo {combo}");
            // Each entry is a `[cx, cy, w, h]` quad at the reported scale.
            let quad = boxes[0].as_array().unwrap();
            assert_eq!(quad.len(), 4);
            assert!((quad[3].as_f64().unwrap() - 0.05 * scale).abs() < 1e-12);
        }
        assert!(v.get("unplaced").is_none(), "auto scale places everything");
    }

    #[test]
    fn place_glyph_boxes_random_and_fixed_scale_accepted() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(3, 0.1, 0.04) },
            "options": { "arrangement": "random", "scale": 0.5, "seed": 3 },
        })
        .to_string();
        let out = call(eunoia_place_glyph_boxes, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert_eq!(v["scale"].as_f64().unwrap(), 0.5);
        let boxes = v["boxes"]["A"].as_array().unwrap();
        assert_eq!(boxes.len(), 3);
        assert!((boxes[0][2].as_f64().unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn place_glyph_boxes_obstacles_are_honored() {
        // Same pipeline as the disc version: place the labels, then hand
        // their boxes to the packer as keep-outs.
        let boxes = [("A", 0.1, 0.1), ("B", 0.1, 0.1), ("A&B", 0.05, 0.05)];
        let sizes = serde_json::json!({ "A": [0.1, 0.1], "B": [0.1, 0.1], "A&B": [0.05, 0.05] });
        let labels = call(eunoia_place_labels, &place_input(sizes, None));
        let labels: serde_json::Value = serde_json::from_str(&labels).unwrap();
        let obstacles: Vec<serde_json::Value> = boxes
            .iter()
            .map(|(key, w, h)| {
                let anchor = labels["placements"][key]["anchor"].as_array().unwrap();
                serde_json::json!({
                    "x": anchor[0].as_f64().unwrap(),
                    "y": anchor[1].as_f64().unwrap(),
                    "width": w,
                    "height": h,
                })
            })
            .collect();

        let input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(3, 0.1, 0.03), "B": box_sizes(2, 0.1, 0.03) },
            "options": { "scale": 0.5, "obstacles": obstacles },
        })
        .to_string();
        let out = call(eunoia_place_glyph_boxes, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert!(v.get("unplaced").is_none(), "got {v}");

        // halo = 0.5 * gap * row_height = 0.5 * 0.25 * (0.5 * 0.03).
        let halo = 0.5 * 0.25 * 0.5 * 0.03;
        for combo in ["A", "B"] {
            for quad in v["boxes"][combo].as_array().unwrap() {
                let (cx, cy) = (quad[0].as_f64().unwrap(), quad[1].as_f64().unwrap());
                let (w, h) = (quad[2].as_f64().unwrap(), quad[3].as_f64().unwrap());
                for (i, (_, bw, bh)) in boxes.iter().enumerate() {
                    let ox = obstacles[i]["x"].as_f64().unwrap();
                    let oy = obstacles[i]["y"].as_f64().unwrap();
                    let dx = ((cx - ox).abs() - 0.5 * w - 0.5 * bw).max(0.0);
                    let dy = ((cy - oy).abs() - 0.5 * h - 0.5 * bh).max(0.0);
                    assert!(
                        dx.hypot(dy) >= halo - 1e-9,
                        "box in {combo} sits on a label box"
                    );
                }
            }
        }
    }

    #[test]
    fn place_glyph_boxes_degenerate_obstacle_is_dropped() {
        let with_junk = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(3, 0.1, 0.04) },
            "options": {
                "obstacles": [{ "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0 }],
            },
        })
        .to_string();
        let plain = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(3, 0.1, 0.04) },
        })
        .to_string();

        let a: serde_json::Value =
            serde_json::from_str(&call(eunoia_place_glyph_boxes, &with_junk)).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(&call(eunoia_place_glyph_boxes, &plain)).unwrap();
        assert_eq!(a["ok"], true, "got {a}");
        assert_eq!(a, b);
    }

    #[test]
    fn place_glyph_boxes_reports_unplaced_below_min_scale() {
        // Boxes far wider than any region: the scale bottoms out at the
        // caller's readability floor and the shortfall is reported.
        let input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(3, 50.0, 0.05) },
            "options": { "min_scale": 0.5 },
        })
        .to_string();
        let out = call(eunoia_place_glyph_boxes, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true, "got {v}");
        assert!(
            (v["scale"].as_f64().unwrap() - 0.5).abs() < 1e-12,
            "got {v}"
        );
        assert_eq!(v["unplaced"]["A"].as_u64().unwrap(), 3);
    }

    #[test]
    fn place_glyph_boxes_bad_arrangement_is_reported() {
        let input = serde_json::json!({
            "regions": region_pieces(),
            "sizes": { "A": box_sizes(1, 0.1, 0.04) },
            "options": { "arrangement": "spiral" },
        })
        .to_string();
        let out = call(eunoia_place_glyph_boxes, &input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["error"].as_str().unwrap().contains("spiral"));
    }
}
