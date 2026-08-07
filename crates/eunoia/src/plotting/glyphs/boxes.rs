//! The rectangular footprint: [`place_glyph_boxes`] packs per-item `w × h`
//! boxes — member *names*, typically — instead of anonymous discs.
//!
//! See the [parent module](super) for the shared machinery. What differs
//! from the disc packer:
//!
//! * **Footprints are heterogeneous.** There is no single radius to bisect
//!   on, so the diagram-wide knob is a *scale factor* `k` applied to every
//!   measured box. The caller measures its text once at a reference font
//!   size and re-renders at `fontSize * scale`.
//! * **`k` only ever shrinks.** The bracket is `[min_scale, 1.0]`: the
//!   caller owns the font size, so the packer never enlarges past what was
//!   measured. This is deliberately asymmetric with
//!   [`place_glyphs`](super::place_glyphs), which grows to fill.
//! * **The uniform arrangement lays out rows**, not a lattice. Rows share
//!   one height (the diagram-wide maximum scaled box height); boxes fill a
//!   row left to right at their own widths, and the block of rows is
//!   centered on the region's pole of inaccessibility. Text reads in rows,
//!   and a max-cell grid would let one long name inflate every cell.
//! * **There is no spread refinement.** Row pitch is fixed by the row
//!   height, so [`PROBE`] and [`PACK`] differ only in how strictly they
//!   treat obstacles, and agree bit-for-bit when there are none. Whatever
//!   the feasibility bisection accepted, the final pass reproduces.
//!
//! Feasibility is not *provably* monotone in `k` — shrinking re-phases the
//! row block against the piece's concavities, so a smaller `k` can in
//! principle lose a row. The disc packer's radius bisection has the
//! identical exposure; in practice neither bites, and the bisection only
//! ever returns a `k` it probed with the same algorithm that will run
//! again for real.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::inscribed::rect_fits_in_piece;
use crate::plotting::regions::{RegionPiece, RegionPolygons, poi_with_holes};

use super::scan::{PieceScan, subtract_range};
use super::{
    GlyphArrangement, OBSTACLE_SHRINK_FLOOR, PACK, PROBE, PackMode, apportion,
    box_clear_of_obstacles, fnv1a, obstacles_near, sanitize_obstacles,
};

/// Configuration bundle for [`place_glyph_boxes`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GlyphBoxOptions {
    /// Arrangement of boxes within each region. Defaults to
    /// [`GlyphArrangement::Uniform`], here meaning row/shelf packing.
    pub arrangement: GlyphArrangement,
    /// Diagram-wide factor applied to every measured box. `None` (default)
    /// bisects for the largest factor in `[min_scale, 1.0]` at which every
    /// region holds all of its items. Set it explicitly to take control —
    /// values above `1.0` are then honoured — in which case regions that
    /// overflow report the shortfall in [`GlyphBoxPlacements::unplaced`]
    /// rather than erroring.
    pub scale: Option<f64>,
    /// Lower end of the auto-scale bracket: the readability floor below
    /// which shrinking the text is worse than dropping items. Clamped to
    /// `(0.0, 1.0]`; ignored when [`scale`](Self::scale) is set. Defaults
    /// to `0.35`.
    pub min_scale: f64,
    /// Breathing room around every box, as a fraction of the **row
    /// height** — the diagram-wide maximum scaled box height, which is this
    /// packer's analogue of the disc packer's radius. Each box carries a
    /// halo of `0.5 * gap * row_height`, so adjacent boxes end up
    /// `gap * row_height` apart and every box keeps `0.5 * gap * row_height`
    /// to region boundaries, holes, and obstacles. The arithmetic mirrors
    /// [`GlyphOptions::gap`](super::GlyphOptions::gap) term for term, so
    /// the shared default of `0.25` means the same thing in both modes:
    /// here, "a quarter of a line-height apart, half that from the edge".
    /// Negative values are clamped to `0.0`.
    pub gap: f64,
    /// Seed for the [`GlyphArrangement::Random`] arrangement. Ignored by
    /// [`GlyphArrangement::Uniform`]. Defaults to `0`.
    pub seed: u64,
    /// Polylabel-style search precision for the per-region pole of
    /// inaccessibility. Defaults to `0.01`.
    pub precision: f64,
    /// [`GlyphArrangement::Random`] only: dart throws attempted per box
    /// before the piece is declared full. Defaults to `300`.
    pub max_attempts: u32,
    /// Axis-aligned keep-out boxes, with the same semantics as
    /// [`GlyphOptions::obstacles`](super::GlyphOptions::obstacles): a
    /// diagram-wide list that boxes keep their halo clear of, best-effort,
    /// with the scale allowed to shrink only to half of what it would have
    /// been without them.
    pub obstacles: Vec<Rectangle>,
}

impl Default for GlyphBoxOptions {
    /// [`GlyphArrangement::Uniform`], auto scale, `min_scale = 0.35`,
    /// `gap = 0.25`, `seed = 0`, `precision = 0.01`, `max_attempts = 300`,
    /// no obstacles.
    fn default() -> Self {
        Self {
            arrangement: GlyphArrangement::default(),
            scale: None,
            min_scale: 0.35,
            gap: 0.25,
            seed: 0,
            precision: 0.01,
            max_attempts: 300,
            obstacles: Vec::new(),
        }
    }
}

impl GlyphBoxOptions {
    /// Sets [`arrangement`](Self::arrangement) and returns `self`.
    pub fn arrangement(mut self, arrangement: GlyphArrangement) -> Self {
        self.arrangement = arrangement;
        self
    }

    /// Sets [`scale`](Self::scale) and returns `self`. Accepts a bare `f64`
    /// or `None` (e.g. `.scale(0.8)` or `.scale(None)`).
    pub fn scale(mut self, scale: impl Into<Option<f64>>) -> Self {
        self.scale = scale.into();
        self
    }

    /// Sets [`min_scale`](Self::min_scale) and returns `self`.
    pub fn min_scale(mut self, min_scale: f64) -> Self {
        self.min_scale = min_scale;
        self
    }

    /// Sets [`gap`](Self::gap) and returns `self`.
    pub fn gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }

    /// Sets [`seed`](Self::seed) and returns `self`.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets [`precision`](Self::precision) and returns `self`.
    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = precision;
        self
    }

    /// Sets [`max_attempts`](Self::max_attempts) and returns `self`.
    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Sets [`obstacles`](Self::obstacles) and returns `self`.
    pub fn obstacles(mut self, obstacles: impl IntoIterator<Item = Rectangle>) -> Self {
        self.obstacles = obstacles.into_iter().collect();
        self
    }
}

/// Result of [`place_glyph_boxes`].
///
/// `#[non_exhaustive]`: an output type, not constructed downstream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GlyphBoxPlacements {
    /// The factor actually used — the auto-chosen one, or the caller's
    /// [`GlyphBoxOptions::scale`] echoed back. `0.0` when nothing could be
    /// placed (degenerate input). Multiply the reference font size by this
    /// to render the text at the size the boxes were packed at.
    pub scale: f64,
    /// Placed boxes per region, keyed by the canonical
    /// [`Combination`](crate::spec::Combination) string form (`""` for the
    /// complement region).
    ///
    /// **Index-aligned with the input**: `boxes[key]` is a *prefix* of
    /// `sizes[key]`, so `boxes[key][i]` belongs to `sizes[key][i]` (and
    /// hence to the caller's `labels[key][i]`). Each rectangle is that
    /// item's own `scale * (w, h)`; the halo is not included. Regions
    /// absent from the input, or with no items, are omitted.
    pub boxes: HashMap<String, Vec<Rectangle>>,
    /// Per-region item count that did **not** fit: exactly
    /// `sizes[key].len() - boxes[key].len()` when that is positive.
    pub unplaced: HashMap<String, usize>,
}

/// Pack the caller's measured `sizes[region]` boxes inside each region.
///
/// `regions` is typically obtained from [`crate::Layout::region_polygons`].
/// `sizes` is keyed by the canonical
/// [`Combination`](crate::spec::Combination) string form (use `""` for the
/// complement region) and holds one `(width, height)` per item, **in the
/// order the caller wants them placed**. Regions with no matching entry, or
/// an empty one, are skipped; size keys with no matching region are ignored
/// — both mirror [`place_glyphs`](super::place_glyphs).
///
/// A single diagram-wide [`scale`](GlyphBoxOptions::scale) multiplies every
/// box, so relative text sizes are preserved and the caller can recover the
/// font size to render at. Every placed box keeps a halo of
/// `0.5 * gap * row_height` to the region boundary (outer ring and holes),
/// to every other box in the region, and — best-effort — to
/// [`obstacles`](GlyphBoxOptions::obstacles).
///
/// # Dropped items
///
/// Text boxes are typically five to ten times wider than tall, so a region
/// with ample *area* can still fail to seat a row of them. The packer
/// consumes each region's items in order and stops at the first that fits
/// nowhere, so what comes back is a prefix: which items get dropped is
/// decided by the order you supply them in. Sort meaningfully, check
/// [`unplaced`](GlyphBoxPlacements::unplaced), and consider rendering a
/// "+n more" affordance — or measure wrapped, multi-line boxes, which this
/// packer handles with no special support.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::{GlyphBoxOptions, place_glyph_boxes};
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 5.0)
///     .set("B", 3.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Exclusive)
///     .build()
///     .unwrap();
///
/// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
/// let regions = layout.region_polygons(&spec, 64);
///
/// // Widths and heights as measured by the renderer, in layout units.
/// let mut sizes = HashMap::new();
/// sizes.insert("A".to_string(), vec![(0.30, 0.10), (0.24, 0.10)]);
/// sizes.insert("A&B".to_string(), vec![(0.20, 0.10)]);
///
/// let placed = place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default());
/// assert!(placed.scale > 0.0 && placed.scale <= 1.0);
/// assert_eq!(placed.boxes["A"].len(), 2);
/// // Boxes come back at `scale` times the measured size, in input order.
/// assert!(placed.boxes["A"][0].width() > placed.boxes["A"][1].width());
/// ```
pub fn place_glyph_boxes(
    regions: &RegionPolygons,
    sizes: &HashMap<String, Vec<(f64, f64)>>,
    options: &GlyphBoxOptions,
) -> GlyphBoxPlacements {
    let gap = options.gap.max(0.0);
    let min_scale = if options.min_scale.is_finite() {
        options.min_scale.clamp(MIN_SCALE_FLOOR, 1.0)
    } else {
        MIN_SCALE_FLOOR
    };

    // Deterministic work list: canonical region order, regions with items
    // only. Sizes are sanitized rather than rejected — an empty member name
    // legitimately measures 0 × 0.
    let work: Vec<RegionWork<'_>> = regions
        .iter()
        .filter_map(|(combo, pieces)| {
            let key = combo.to_string();
            let items: Vec<(f64, f64)> = sizes.get(&key)?.iter().map(|&wh| sanitize(wh)).collect();
            (!items.is_empty() && !pieces.is_empty()).then(|| RegionWork {
                scans: pieces.iter().map(PieceScan::new).collect(),
                areas: pieces.iter().map(RegionPiece::area).collect(),
                key,
                items,
            })
        })
        .collect();

    if work.is_empty() {
        return GlyphBoxPlacements {
            scale: options.scale.unwrap_or(0.0).max(0.0),
            boxes: HashMap::new(),
            unplaced: HashMap::new(),
        };
    }

    let obstacles = sanitize_obstacles(&options.obstacles);

    // The row height at `k = 1`. Everything downstream is linear in `k`, so
    // this one number carries the whole footprint model.
    let h1 = work
        .iter()
        .flat_map(|w| w.items.iter().map(|&(_, h)| h))
        .fold(0.0_f64, f64::max);
    if h1 <= 0.0 || !h1.is_finite() {
        return all_unplaced(&work);
    }

    let scale = match options.scale {
        Some(k) if k > 0.0 && k.is_finite() => k,
        Some(_) => return all_unplaced(&work),
        None => {
            let Some(k_hi) = analytic_scale_hi(&work, gap, h1) else {
                return all_unplaced(&work);
            };
            // The bracket is `[min_scale, k_hi]`. An inverted one means the
            // geometry cannot support even the readability floor the caller
            // declared — and below that floor, shrinking further buys
            // illegible text, so the answer is `min_scale` with the
            // shortfall reported in `unplaced`. (The disc packer's
            // `OBSTACLE_FREE_FALLBACK` has no analogue here for exactly that
            // reason: there is nothing to invent.)
            let search = |obstacles: &[Rectangle]| {
                (k_hi > min_scale)
                    .then(|| auto_scale(&work, options, gap, h1, obstacles, k_hi, min_scale))
                    .flatten()
            };
            let k_free = search(&[]).unwrap_or(min_scale);
            if obstacles.is_empty() {
                k_free
            } else {
                // Obstacles are hard walls for the feasibility search, so the
                // text really does shrink to make room — but only so far.
                search(&obstacles)
                    .unwrap_or(0.0)
                    .max(OBSTACLE_SHRINK_FLOOR * k_free)
                    .max(min_scale)
            }
        }
    };

    let ctx = BoxCtx {
        k: scale,
        row_h: scale * h1,
        gap,
        obstacles: &obstacles,
        mode: PACK,
    };
    let mut boxes = HashMap::new();
    let mut unplaced = HashMap::new();
    for region in &work {
        let placed = pack_boxes_region(region, options, ctx);
        if placed.len() < region.items.len() {
            unplaced.insert(region.key.clone(), region.items.len() - placed.len());
        }
        boxes.insert(region.key.clone(), placed);
    }

    GlyphBoxPlacements {
        scale,
        boxes,
        unplaced,
    }
}

/// Hard lower clamp on [`GlyphBoxOptions::min_scale`], so a zero or negative
/// floor cannot collapse the bracket onto `0.0`.
const MIN_SCALE_FLOOR: f64 = 1e-6;

/// How many consecutive row counts may fail to improve on the best packing
/// before the row-count search gives up. Capacity rises with the row count
/// until the block runs out of vertical room, but not strictly — adding a
/// row re-centers the whole block, so a concave piece can wobble. Three
/// barren steps is the wobble tolerance; without it an infeasible region
/// would replan every row count its bounding box admits.
const ROW_COUNT_STALL: usize = 3;

/// One region's packing problem, with the per-piece scan ladders built once
/// so the scale bisection can re-probe without rebuilding them.
struct RegionWork<'a> {
    key: String,
    /// `None` for a degenerate piece, keeping the index aligned with
    /// `areas` and the apportioned quotas.
    scans: Vec<Option<PieceScan<'a>>>,
    areas: Vec<f64>,
    /// Measured sizes at `k = 1`, in the caller's order.
    items: Vec<(f64, f64)>,
}

/// Clamps a measured size into something packable. Non-finite or negative
/// extents become `0.0`: an empty label really does measure nothing, and a
/// zero-width box still consumes its halo, which is the honest layout.
fn sanitize((w, h): (f64, f64)) -> (f64, f64) {
    let clean = |v: f64| if v.is_finite() && v > 0.0 { v } else { 0.0 };
    (clean(w), clean(h))
}

/// Degenerate outcome: nothing placeable, every item reported short.
fn all_unplaced(work: &[RegionWork<'_>]) -> GlyphBoxPlacements {
    GlyphBoxPlacements {
        scale: 0.0,
        boxes: HashMap::new(),
        unplaced: work
            .iter()
            .map(|w| (w.key.clone(), w.items.len()))
            .collect(),
    }
}

/// Everything a packer needs about the geometry it is packing at: the trial
/// scale, the row height it implies, the spacing fraction, the keep-out
/// boxes, and how strictly to treat them.
#[derive(Debug, Clone, Copy)]
struct BoxCtx<'a> {
    k: f64,
    row_h: f64,
    gap: f64,
    obstacles: &'a [Rectangle],
    mode: PackMode,
}

impl BoxCtx<'_> {
    /// Halo every box carries: half the box-to-box separation, and the full
    /// clearance to rings and obstacles.
    fn halo(&self) -> f64 {
        0.5 * self.gap * self.row_h
    }

    /// Vertical pitch between consecutive row centers. Bands abut, so rows
    /// end up `2 * halo` apart — the same separation boxes get horizontally.
    fn pitch(&self) -> f64 {
        self.row_h + 2.0 * self.halo()
    }
}

/// Analytic upper bound on the scale: no region can hold items whose
/// combined haloed footprint exceeds its area, nor an item wider than the
/// widest piece it might land in. Capped at `1.0` — this packer only ever
/// shrinks. `None` when the bound is degenerate.
///
/// Deliberately *not* transposed from
/// [`analytic_r_hi`](super::discs): the pole-of-inaccessibility clearance
/// caps a disc, but for a rectangle the inscribed disc is a bound from
/// below, so using it here would cut off feasible scales in exactly the
/// elongated regions where wide text fits best.
fn analytic_scale_hi(work: &[RegionWork<'_>], gap: f64, h1: f64) -> Option<f64> {
    let mut k_hi = 1.0_f64;
    for region in work {
        // Haloed footprint of item `i` at scale `k` is
        // `k²·(w + gap·h1)·(h + gap·h1)`; require the sum to fit the area.
        let area: f64 = region.areas.iter().sum();
        let footprint: f64 = region
            .items
            .iter()
            .map(|&(w, h)| (w + gap * h1) * (h + gap * h1))
            .sum();
        if area > 0.0 && footprint > 0.0 {
            k_hi = k_hi.min((area / footprint).sqrt());
        }

        // The widest item needs *some* piece to hold it, so this bound is
        // the max over pieces, not the min.
        let w_max = region.items.iter().map(|&(w, _)| w).fold(0.0_f64, f64::max);
        let mut k_bbox = 0.0_f64;
        for scan in region.scans.iter().flatten() {
            let (min_x, max_x, min_y, max_y) = scan.bbox();
            k_bbox = k_bbox.max(
                ((max_x - min_x) / (w_max + gap * h1)).min((max_y - min_y) / (h1 * (1.0 + gap))),
            );
        }
        k_hi = k_hi.min(k_bbox);
    }
    (k_hi.is_finite() && k_hi > 0.0).then_some(k_hi)
}

/// Largest scale in `[floor, k_hi]` at which every region holds all of its
/// items while honouring `obstacles`. `None` when even `floor` is
/// infeasible.
fn auto_scale(
    work: &[RegionWork<'_>],
    options: &GlyphBoxOptions,
    gap: f64,
    h1: f64,
    obstacles: &[Rectangle],
    k_hi: f64,
    floor: f64,
) -> Option<f64> {
    let feasible = |k: f64| {
        let ctx = BoxCtx {
            k,
            row_h: k * h1,
            gap,
            obstacles,
            mode: PROBE,
        };
        work.iter()
            .all(|region| pack_boxes_region(region, options, ctx).len() == region.items.len())
    };

    if feasible(k_hi) {
        return Some(k_hi);
    }
    if !feasible(floor) {
        return None;
    }
    let mut lo = floor;
    let mut hi = k_hi;
    let mut best = Some(floor);
    // `lo` stays feasible by construction, so the final `PACK` pass at the
    // returned scale cannot come up short.
    for _ in 0..24 {
        if hi - lo <= 1e-3 * k_hi {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if feasible(mid) {
            lo = mid;
            best = Some(mid);
        } else {
            hi = mid;
        }
    }
    best
}

/// Pack a region's items into its (possibly multi-piece) geometry,
/// apportioning the item count across pieces by net area.
///
/// Pieces are walked in order with a carry: whatever a piece fails to place
/// is offered to the next one. Since each piece places a prefix of what it
/// is offered, the region-wide result is a prefix of `items` — which is the
/// contract [`GlyphBoxPlacements::boxes`] advertises.
fn pack_boxes_region(
    region: &RegionWork<'_>,
    options: &GlyphBoxOptions,
    ctx: BoxCtx<'_>,
) -> Vec<Rectangle> {
    let scaled: Vec<(f64, f64)> = region
        .items
        .iter()
        .map(|&(w, h)| (ctx.k * w, ctx.k * h))
        .collect();
    let quotas = apportion(scaled.len(), &region.areas);
    // One RNG stream per region, threaded through its pieces in order, so a
    // region's scatter is independent of every other region's item count.
    let mut rng = StdRng::seed_from_u64(options.seed ^ fnv1a(region.key.as_bytes()));

    let mut out = Vec::with_capacity(scaled.len());
    let mut budget = 0usize;
    for (i, scan) in region.scans.iter().enumerate() {
        budget += quotas.get(i).copied().unwrap_or(0);
        if budget == 0 {
            continue;
        }
        // A degenerate piece places nothing; its quota carries onward.
        let Some(scan) = scan else { continue };
        let offered = &scaled[out.len()..(out.len() + budget).min(scaled.len())];
        let placed = match options.arrangement {
            GlyphArrangement::Uniform => pack_rows_piece(scan, offered, options.precision, ctx),
            GlyphArrangement::Random => {
                pack_random_boxes_piece(scan, offered, options.max_attempts, ctx, &mut rng)
            }
        };
        budget -= placed.len();
        out.extend(placed);
    }
    out
}

/// Row/shelf packer for one piece.
///
/// Rows share the height `ctx.row_h` and abut at `ctx.pitch()`; the block of
/// rows is centered vertically on the piece's pole of inaccessibility, so a
/// single row lands right on it. Row counts are tried from one upward,
/// returning as soon as everything fits and otherwise keeping the best
/// attempt — capacity generally rises with the row count but not strictly,
/// since adding a row re-centers the block against the piece's concavities.
///
/// Obstacles are honoured on a first pass; under a lenient [`PackMode`] a
/// second pass ignoring them takes over when the first cannot seat
/// everything, so a region buried under a label box packs into it rather
/// than dropping every item.
///
/// Consumes `items` in order and stops at the first that fits nowhere, so
/// the result is a prefix.
fn pack_rows_piece(
    scan: &PieceScan<'_>,
    items: &[(f64, f64)],
    precision: f64,
    ctx: BoxCtx<'_>,
) -> Vec<Rectangle> {
    if items.is_empty() {
        return Vec::new();
    }
    let pitch = ctx.pitch();
    if pitch <= 0.0 || !pitch.is_finite() {
        return Vec::new();
    }
    let piece = scan.piece();
    let Some((poi, _)) = poi_with_holes(std::slice::from_ref(piece), precision) else {
        return Vec::new();
    };
    let halo = ctx.halo();
    let local = obstacles_near(piece, ctx.obstacles, halo + 0.5 * pitch);
    let anchor_x = poi.x();

    let (_, _, min_y, max_y) = scan.bbox();
    // More rows than items is pointless, and no more rows fit than the
    // bounding box admits.
    let max_rows = ((max_y - min_y) / pitch).floor().max(0.0) as usize + 1;
    let max_rows = max_rows.min(items.len()).max(1);

    let mut best: Vec<Rectangle> = Vec::new();
    for honor_obstacles in [true, false] {
        if !honor_obstacles && (ctx.mode.strict_obstacles || local.is_empty()) {
            break;
        }
        let obstacles: &[Rectangle] = if honor_obstacles { &local } else { &[] };
        let anchor_y = escape_obstacles_y(poi.y(), scan, obstacles, ctx);
        let mut stalled = 0usize;
        for rows in 1..=max_rows {
            let attempt = fill_rows(scan, items, anchor_x, anchor_y, rows, obstacles, ctx);
            if attempt.len() == items.len() {
                return attempt;
            }
            if attempt.len() > best.len() {
                best = attempt;
                stalled = 0;
            } else {
                stalled += 1;
                if stalled >= ROW_COUNT_STALL {
                    break;
                }
            }
        }
    }
    best
}

/// Lays `items` into `rows` bands centered on `anchor_y`, taking them in
/// order and stopping at the first that fits in no remaining row.
fn fill_rows(
    scan: &PieceScan<'_>,
    items: &[(f64, f64)],
    anchor_x: f64,
    anchor_y: f64,
    rows: usize,
    obstacles: &[Rectangle],
    ctx: BoxCtx<'_>,
) -> Vec<Rectangle> {
    let halo = ctx.halo();
    let pitch = ctx.pitch();
    let half_pitch = 0.5 * pitch;
    let mut out: Vec<Rectangle> = Vec::with_capacity(items.len());

    for j in 0..rows {
        if out.len() == items.len() {
            break;
        }
        let cy = anchor_y + (j as f64 - 0.5 * (rows as f64 - 1.0)) * pitch;
        let (yb, yt) = (cy - half_pitch, cy + half_pitch);
        let mut intervals = scan.band_intervals(yb, yt);
        if intervals.is_empty() {
            continue;
        }
        if !obstacles.is_empty() {
            intervals = subtract_obstacles(&intervals, yb, yt, obstacles, halo);
        }
        // Nearest-to-the-anchor first, so a short row clusters at the pole
        // of inaccessibility instead of pinning to a region's leftmost lobe.
        // A no-op in the common single-interval case.
        intervals.sort_by(|a, b| {
            let da = (0.5 * (a.0 + a.1) - anchor_x).abs();
            let db = (0.5 * (b.0 + b.1) - anchor_x).abs();
            da.total_cmp(&db).then(a.0.total_cmp(&b.0))
        });

        for (lo, hi) in intervals {
            if out.len() == items.len() {
                break;
            }
            // How many of the remaining items this interval swallows, and
            // how much width they consume with their halos.
            let mut consumed = 0.0_f64;
            let mut taken = 0usize;
            while out.len() + taken < items.len() {
                let need = items[out.len() + taken].0 + 2.0 * halo;
                if lo + consumed + need > hi {
                    break;
                }
                consumed += need;
                taken += 1;
            }
            if taken == 0 {
                continue;
            }
            // Center the run within the interval.
            let mut x = lo + 0.5 * (hi - lo - consumed);
            for _ in 0..taken {
                let (w, h) = items[out.len()];
                let cx = x + halo + 0.5 * w;
                debug_assert!(
                    w <= 0.0
                        || h <= 0.0
                        || rect_fits_in_piece(
                            scan.piece(),
                            cx,
                            cy,
                            (0.5 * w + halo) * (1.0 - 1e-9),
                            (0.5 * h + halo) * (1.0 - 1e-9),
                        ),
                    "band oracle placed a box the containment predicate rejects"
                );
                out.push(Rectangle::new(Point::new(cx, cy), w, h));
                x += w + 2.0 * halo;
            }
        }
    }
    out
}

/// Removes from `intervals` the x-ranges blocked by obstacles overlapping
/// the band `[yb, yt]`.
///
/// Treating the box's vertical extent as the whole band is conservative —
/// a short box in a tall row is pushed aside by an obstacle it would just
/// clear — but it keeps the subtraction a pure 1-D operation, which is what
/// lets an axis-aligned keep-out compose with the interval model at all.
fn subtract_obstacles(
    intervals: &[(f64, f64)],
    yb: f64,
    yt: f64,
    obstacles: &[Rectangle],
    clearance: f64,
) -> Vec<(f64, f64)> {
    let mut out = intervals.to_vec();
    for rect in obstacles {
        let (cx, cy) = (rect.center().x(), rect.center().y());
        let half_h = 0.5 * rect.height() + clearance;
        if cy + half_h <= yb || cy - half_h >= yt {
            continue;
        }
        let half_w = 0.5 * rect.width() + clearance;
        out = subtract_range(&out, cx - half_w, cx + half_w);
        if out.is_empty() {
            break;
        }
    }
    out
}

/// Nudges the row block's vertical phase off an obstacle sitting on the
/// pole of inaccessibility, keeping the raw anchor when nothing helps.
///
/// The 1-D cousin of the disc packer's `escape_obstacles`, and it exists for
/// the same reason: the anchor is the block's phase, not merely a candidate
/// position, so leaving it buried under a label box costs the whole region
/// its best row. Only vertical, because the horizontal degree of freedom is
/// already handled by subtracting obstacles from each row's intervals.
/// Candidates clear each blocking obstacle by a full row.
fn escape_obstacles_y(
    anchor_y: f64,
    scan: &PieceScan<'_>,
    obstacles: &[Rectangle],
    ctx: BoxCtx<'_>,
) -> f64 {
    if obstacles.is_empty() {
        return anchor_y;
    }
    let halo = ctx.halo();
    let half_pitch = 0.5 * ctx.pitch();
    let usable = |y: f64| {
        let intervals = scan.band_intervals(y - half_pitch, y + half_pitch);
        !intervals.is_empty()
            && !subtract_obstacles(&intervals, y - half_pitch, y + half_pitch, obstacles, halo)
                .is_empty()
    };
    if usable(anchor_y) {
        return anchor_y;
    }
    let mut best: Option<(f64, f64)> = None;
    for rect in obstacles {
        let cy = rect.center().y();
        let offset = 0.5 * rect.height() + halo + half_pitch;
        for candidate in [cy - offset, cy + offset] {
            if !usable(candidate) {
                continue;
            }
            let d = (candidate - anchor_y).abs();
            let better =
                best.is_none_or(|(bd, bc)| d.total_cmp(&bd).then(candidate.total_cmp(&bc)).is_lt());
            if better {
                best = Some((d, candidate));
            }
        }
    }
    best.map_or(anchor_y, |(_, y)| y)
}

/// Random (dart-throwing) packer for one piece: uniform samples over the
/// piece's bounding box, accepted when the haloed box lies inside the piece
/// and clears every box already placed.
///
/// The pairwise test is per-axis AABB separation with the two halos summed —
/// the disc packer can compare one squared distance only because all its
/// glyphs are the same size.
///
/// Obstacles join the acceptance test for a first pass; under a lenient
/// [`PackMode`] a second pass without them tops up what the first could not
/// place. Stops at the first box that exhausts its darts, both to keep the
/// prefix contract and because an exhausted box means the piece is full.
fn pack_random_boxes_piece(
    scan: &PieceScan<'_>,
    items: &[(f64, f64)],
    max_attempts: u32,
    ctx: BoxCtx<'_>,
    rng: &mut StdRng,
) -> Vec<Rectangle> {
    if items.is_empty() {
        return Vec::new();
    }
    let piece = scan.piece();
    let (min_x, max_x, min_y, max_y) = scan.bbox();
    let halo = ctx.halo();
    let w_max = items.iter().map(|&(w, _)| w).fold(0.0_f64, f64::max);
    let local = obstacles_near(piece, ctx.obstacles, halo + 0.5 * (w_max + ctx.row_h));

    let mut placed: Vec<Rectangle> = Vec::with_capacity(items.len());
    for honor_obstacles in [true, false] {
        if !honor_obstacles && (ctx.mode.strict_obstacles || local.is_empty()) {
            break;
        }
        'items: while placed.len() < items.len() {
            let (w, h) = items[placed.len()];
            // `rect_fits_in_piece` rejects a zero half-extent outright, so a
            // degenerate item with no gap still needs a sliver to test.
            let hw = (0.5 * w + halo).max(1e-12);
            let hh = (0.5 * h + halo).max(1e-12);
            for _ in 0..max_attempts {
                let x = rng.random_range(min_x..=max_x);
                let y = rng.random_range(min_y..=max_y);
                let clear_of_placed = placed.iter().all(|r| {
                    (x - r.center().x()).abs() >= 0.5 * (w + r.width()) + 2.0 * halo
                        || (y - r.center().y()).abs() >= 0.5 * (h + r.height()) + 2.0 * halo
                });
                if clear_of_placed
                    && rect_fits_in_piece(piece, x, y, hw, hh)
                    && (!honor_obstacles || box_clear_of_obstacles(x, y, w, h, &local, halo))
                {
                    placed.push(Rectangle::new(Point::new(x, y), w, h));
                    continue 'items;
                }
            }
            break;
        }
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plotting::regions::classify_into_pieces;
    use crate::spec::Combination;

    use super::super::box_rect_separation;
    use super::super::test_utils::{boxed_region, rect_ring, two_set_regions};

    /// A region's sizes, as `n` copies of one `w × h` measurement.
    fn uniform_sizes(n: usize, w: f64, h: f64) -> Vec<(f64, f64)> {
        vec![(w, h); n]
    }

    fn sizes(entries: &[(&str, Vec<(f64, f64)>)]) -> HashMap<String, Vec<(f64, f64)>> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// Halo implied by a result: the packer's `0.5 * gap * row_height`.
    fn halo_of(result: &GlyphBoxPlacements, h1: f64, gap: f64) -> f64 {
        0.5 * gap * result.scale * h1
    }

    fn assert_inside_region(
        result: &GlyphBoxPlacements,
        regions: &RegionPolygons,
        halo: f64,
        context: &str,
    ) {
        for (key, rects) in &result.boxes {
            let combo: Combination = key.parse().unwrap();
            let pieces = regions.get(&combo).unwrap();
            for (i, r) in rects.iter().enumerate() {
                let (cx, cy) = (r.center().x(), r.center().y());
                let fits = pieces.iter().any(|piece| {
                    rect_fits_in_piece(
                        piece,
                        cx,
                        cy,
                        (0.5 * r.width() + halo) * (1.0 - 1e-6),
                        (0.5 * r.height() + halo) * (1.0 - 1e-6),
                    )
                });
                assert!(fits, "{context}: box {i} in {key:?} escapes its region");
            }
        }
    }

    fn assert_no_overlap(result: &GlyphBoxPlacements, halo: f64, context: &str) {
        for (key, rects) in &result.boxes {
            for (i, a) in rects.iter().enumerate() {
                for b in &rects[i + 1..] {
                    let dx = (a.center().x() - b.center().x()).abs()
                        - 0.5 * (a.width() + b.width())
                        - 2.0 * halo;
                    let dy = (a.center().y() - b.center().y()).abs()
                        - 0.5 * (a.height() + b.height())
                        - 2.0 * halo;
                    assert!(
                        dx >= -1e-9 || dy >= -1e-9,
                        "{context}: boxes overlap in {key:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn uniform_places_every_item() {
        let regions = two_set_regions();
        let sizes = sizes(&[
            ("A", uniform_sizes(6, 0.25, 0.10)),
            ("B", uniform_sizes(3, 0.25, 0.10)),
            ("A&B", uniform_sizes(1, 0.20, 0.10)),
        ]);
        let options = GlyphBoxOptions::default();
        let result = place_glyph_boxes(&regions, &sizes, &options);

        assert!(
            result.scale > 0.0 && result.scale <= 1.0,
            "{}",
            result.scale
        );
        assert!(result.unplaced.is_empty(), "{:?}", result.unplaced);
        for (key, items) in &sizes {
            assert_eq!(result.boxes[key].len(), items.len(), "{key}");
        }
        let halo = halo_of(&result, 0.10, options.gap);
        assert_inside_region(&result, &regions, halo, "uniform");
        assert_no_overlap(&result, halo, "uniform");
    }

    #[test]
    fn random_places_every_item() {
        let regions = two_set_regions();
        let sizes = sizes(&[
            ("A", uniform_sizes(4, 0.20, 0.08)),
            ("B", uniform_sizes(2, 0.20, 0.08)),
        ]);
        let options = GlyphBoxOptions::default().arrangement(GlyphArrangement::Random);
        let result = place_glyph_boxes(&regions, &sizes, &options);

        assert!(result.unplaced.is_empty(), "{:?}", result.unplaced);
        let halo = halo_of(&result, 0.08, options.gap);
        assert_inside_region(&result, &regions, halo, "random");
        assert_no_overlap(&result, halo, "random");
    }

    #[test]
    fn order_is_preserved() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let items = vec![(3.0, 1.0), (1.0, 1.0), (2.0, 0.5), (0.5, 1.0)];
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", items.clone())]),
            &GlyphBoxOptions::default(),
        );
        assert_eq!(result.boxes["A"].len(), items.len());
        for (r, (w, h)) in result.boxes["A"].iter().zip(&items) {
            assert!((r.width() - result.scale * w).abs() < 1e-12);
            assert!((r.height() - result.scale * h).abs() < 1e-12);
        }
    }

    #[test]
    fn uniform_rows_share_one_pitch() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let options = GlyphBoxOptions::default();
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(9, 3.0, 1.0))]),
            &options,
        );
        assert!(result.unplaced.is_empty());

        let mut ys: Vec<f64> = result.boxes["A"].iter().map(|r| r.center().y()).collect();
        ys.sort_by(f64::total_cmp);
        ys.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert!(ys.len() > 1, "9 wide boxes in a 10x10 region must wrap");
        // Consecutive distinct row centers are exactly one pitch apart.
        let pitch = result.scale * 1.0 * (1.0 + options.gap);
        for pair in ys.windows(2) {
            assert!(
                (pair[1] - pair[0] - pitch).abs() < 1e-6,
                "row spacing {} != pitch {pitch}",
                pair[1] - pair[0]
            );
        }
    }

    #[test]
    fn single_item_lands_on_the_poi() {
        let regions = boxed_region(0.0, 0.0, 2.0, 2.0);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", vec![(0.4, 0.2)])]),
            &GlyphBoxOptions::default(),
        );
        assert_eq!(result.boxes["A"].len(), 1);
        let r = &result.boxes["A"][0];
        assert!((r.center().x() - 1.0).abs() < 0.05, "{r:?}");
        assert!((r.center().y() - 1.0).abs() < 0.05, "{r:?}");
    }

    #[test]
    fn rows_are_centered_on_the_anchor() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(6, 4.0, 1.0))]),
            &GlyphBoxOptions::default(),
        );
        assert!(result.unplaced.is_empty());
        let ys: Vec<f64> = result.boxes["A"].iter().map(|r| r.center().y()).collect();
        let mid = 0.5
            * (ys.iter().cloned().fold(f64::MAX, f64::min)
                + ys.iter().cloned().fold(f64::MIN, f64::max));
        // The block straddles the POI of the square, at its center.
        assert!((mid - 5.0).abs() < 0.1, "block center {mid} is off the POI");
    }

    #[test]
    fn multi_piece_region_apportions_by_area() {
        // Areas 9 and 3 (3:1): 8 items split 6 / 2 by largest remainder.
        let pieces = classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 3.0, 3.0),
            rect_ring(10.0, 0.0, 13.0, 1.0),
        ]);
        assert_eq!(pieces.len(), 2);
        let big_first = pieces[0].area() > pieces[1].area();
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);

        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(8, 0.6, 0.25))]),
            &GlyphBoxOptions::default(),
        );
        assert!(result.unplaced.is_empty(), "{:?}", result.unplaced);
        let in_left = result.boxes["A"]
            .iter()
            .filter(|r| r.center().x() < 5.0)
            .count();
        let (expect_big, expect_small) = if big_first { (6, 2) } else { (2, 6) };
        assert_eq!(in_left, if big_first { expect_big } else { expect_small });
    }

    #[test]
    fn overflow_reports_a_prefix() {
        // Distinct widths, so a prefix is distinguishable from a subset.
        let regions = boxed_region(0.0, 0.0, 2.0, 2.0);
        let items: Vec<(f64, f64)> = (0..12).map(|i| (0.5 + 0.1 * i as f64, 0.4)).collect();
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", items.clone())]),
            &GlyphBoxOptions::default().scale(1.0),
        );
        let placed = &result.boxes["A"];
        assert!(placed.len() < items.len(), "the region must overflow");
        assert_eq!(result.unplaced["A"], items.len() - placed.len());
        for (r, (w, _)) in placed.iter().zip(&items) {
            assert!((r.width() - w).abs() < 1e-12, "not the input prefix");
        }
    }

    #[test]
    fn hole_region_keeps_boxes_out_of_the_hole() {
        let pieces = classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 10.0, 10.0),
            rect_ring(3.0, 3.0, 7.0, 7.0),
        ]);
        assert_eq!(pieces[0].holes.len(), 1);
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);

        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let options = GlyphBoxOptions::default().arrangement(arrangement);
            let result = place_glyph_boxes(
                &regions,
                &sizes(&[("A", uniform_sizes(8, 1.2, 0.5))]),
                &options,
            );
            for r in &result.boxes["A"] {
                let (cx, cy) = (r.center().x(), r.center().y());
                let (hw, hh) = (0.5 * r.width(), 0.5 * r.height());
                let overlaps_hole =
                    cx + hw > 3.0 && cx - hw < 7.0 && cy + hh > 3.0 && cy - hh < 7.0;
                assert!(!overlaps_hole, "{arrangement:?}: box {r:?} enters the hole");
            }
        }
    }

    #[test]
    fn auto_scale_never_exceeds_one() {
        // A huge region and tiny text: no reason to shrink, and no licence
        // to grow.
        let regions = boxed_region(0.0, 0.0, 100.0, 100.0);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(3, 0.5, 0.2))]),
            &GlyphBoxOptions::default(),
        );
        assert_eq!(result.scale, 1.0);
    }

    #[test]
    fn auto_scale_shrinks_for_a_crowded_region() {
        let items = uniform_sizes(12, 1.0, 0.4);
        let roomy = place_glyph_boxes(
            &boxed_region(0.0, 0.0, 20.0, 20.0),
            &sizes(&[("A", items.clone())]),
            &GlyphBoxOptions::default(),
        );
        let cramped = place_glyph_boxes(
            &boxed_region(0.0, 0.0, 3.0, 3.0),
            &sizes(&[("A", items)]),
            &GlyphBoxOptions::default(),
        );
        assert_eq!(roomy.scale, 1.0);
        assert!(cramped.scale < roomy.scale, "{} ", cramped.scale);
        assert!(cramped.scale >= GlyphBoxOptions::default().min_scale - 1e-12);
    }

    #[test]
    fn min_scale_is_a_hard_floor() {
        // Hopeless: one box far wider than the region at any allowed scale.
        let regions = boxed_region(0.0, 0.0, 1.0, 1.0);
        let options = GlyphBoxOptions::default().min_scale(0.5);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(4, 20.0, 1.0))]),
            &options,
        );
        assert!((result.scale - 0.5).abs() < 1e-12, "{}", result.scale);
        assert_eq!(result.unplaced["A"], 4);
        assert!(result.boxes["A"].is_empty());
    }

    #[test]
    fn fixed_scale_is_echoed_back() {
        let regions = boxed_region(0.0, 0.0, 4.0, 4.0);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(2, 1.0, 0.5))]),
            &GlyphBoxOptions::default().scale(1.0),
        );
        assert_eq!(result.scale, 1.0);
        assert!((result.boxes["A"][0].width() - 1.0).abs() < 1e-12);
        // Above 1.0 is honoured too: the caller took control.
        let big = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(2, 1.0, 0.5))]),
            &GlyphBoxOptions::default().scale(2.0),
        );
        assert_eq!(big.scale, 2.0);
        assert!((big.boxes["A"][0].width() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn non_positive_scale_reports_everything_unplaced() {
        let regions = boxed_region(0.0, 0.0, 4.0, 4.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let result = place_glyph_boxes(
                &regions,
                &sizes(&[("A", uniform_sizes(3, 1.0, 0.5))]),
                &GlyphBoxOptions::default().scale(bad),
            );
            assert_eq!(result.scale, 0.0, "scale = {bad}");
            assert_eq!(result.unplaced["A"], 3, "scale = {bad}");
            assert!(result.boxes.is_empty(), "scale = {bad}");
        }
    }

    #[test]
    fn empty_and_unknown_size_keys_are_omitted() {
        let regions = two_set_regions();
        let sizes = sizes(&[
            ("A", Vec::new()),
            ("Z", uniform_sizes(2, 0.2, 0.1)),
            ("B", uniform_sizes(2, 0.2, 0.1)),
        ]);
        let result = place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default());
        assert!(!result.boxes.contains_key("A"));
        assert!(!result.boxes.contains_key("Z"));
        assert_eq!(result.boxes["B"].len(), 2);
    }

    #[test]
    fn degenerate_item_sizes_are_tolerated() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let items = vec![
            (1.0, 0.5),
            (0.0, 0.0),
            (f64::NAN, 0.5),
            (-1.0, 0.5),
            (1.0, f64::INFINITY),
        ];
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", items.clone())]),
            &GlyphBoxOptions::default(),
        );
        // Degenerate extents sanitize to zero; nothing panics and nothing is
        // silently dropped.
        assert_eq!(result.boxes["A"].len(), items.len());
        assert!(result.unplaced.is_empty());
        assert!(result.boxes["A"].iter().all(|r| r.width().is_finite()));
    }

    #[test]
    fn all_zero_height_input_is_degenerate() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", uniform_sizes(3, 1.0, 0.0))]),
            &GlyphBoxOptions::default(),
        );
        assert_eq!(result.scale, 0.0);
        assert_eq!(result.unplaced["A"], 3);
    }

    #[test]
    fn empty_obstacles_match_no_obstacles() {
        let regions = two_set_regions();
        let sizes = sizes(&[
            ("A", uniform_sizes(4, 0.2, 0.08)),
            ("B", uniform_sizes(2, 0.2, 0.08)),
        ]);
        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let plain = GlyphBoxOptions::default().arrangement(arrangement);
            let empty = GlyphBoxOptions::default()
                .arrangement(arrangement)
                .obstacles([]);
            assert_eq!(
                place_glyph_boxes(&regions, &sizes, &plain),
                place_glyph_boxes(&regions, &sizes, &empty),
                "{arrangement:?}"
            );
        }
    }

    #[test]
    fn degenerate_and_distant_obstacles_are_ignored() {
        let regions = two_set_regions();
        let sizes = sizes(&[("A", uniform_sizes(4, 0.2, 0.08))]);
        let baseline = place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default());

        let junk = [
            Rectangle::new(Point::new(f64::NAN, 0.0), 1.0, 1.0),
            Rectangle::new(Point::new(0.0, 0.0), 0.0, 1.0),
            Rectangle::new(Point::new(0.0, 0.0), 1.0, -1.0),
            Rectangle::new(Point::new(0.0, 0.0), f64::INFINITY, 1.0),
            Rectangle::new(Point::new(1000.0, 1000.0), 1.0, 1.0),
        ];
        let result = place_glyph_boxes(
            &regions,
            &sizes,
            &GlyphBoxOptions::default().obstacles(junk),
        );
        assert_eq!(result, baseline);
    }

    #[test]
    fn obstacles_are_honored_for_both_arrangements() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let obstacles = [Rectangle::new(Point::new(5.0, 5.0), 3.0, 1.5)];
        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let options = GlyphBoxOptions::default()
                .arrangement(arrangement)
                .obstacles(obstacles);
            let result = place_glyph_boxes(
                &regions,
                &sizes(&[("A", uniform_sizes(6, 1.5, 0.6))]),
                &options,
            );
            assert!(result.unplaced.is_empty(), "{arrangement:?}");
            let halo = halo_of(&result, 0.6, options.gap);
            for (i, r) in result.boxes["A"].iter().enumerate() {
                let d = box_rect_separation(
                    r.center().x(),
                    r.center().y(),
                    r.width(),
                    r.height(),
                    &obstacles[0],
                );
                assert!(
                    d >= halo - 1e-9,
                    "{arrangement:?}: box {i} is {d} from the obstacle, want {halo}"
                );
            }
        }
    }

    #[test]
    fn auto_scale_shrinks_but_is_floored_by_obstacles() {
        let regions = boxed_region(0.0, 0.0, 6.0, 6.0);
        let items = uniform_sizes(8, 2.0, 0.8);
        let k_free = place_glyph_boxes(
            &regions,
            &sizes(&[("A", items.clone())]),
            &GlyphBoxOptions::default(),
        )
        .scale;

        let obstacles = [Rectangle::new(Point::new(3.0, 3.0), 4.0, 2.0)];
        let result = place_glyph_boxes(
            &regions,
            &sizes(&[("A", items)]),
            &GlyphBoxOptions::default().obstacles(obstacles),
        );
        assert!(
            result.scale <= k_free + 1e-12,
            "obstacles must not grow the text: {} > {k_free}",
            result.scale
        );
        assert!(
            result.scale >= OBSTACLE_SHRINK_FLOOR * k_free - 1e-12,
            "scale {} fell below the floor of {}",
            result.scale,
            OBSTACLE_SHRINK_FLOOR * k_free
        );
    }

    #[test]
    fn hopeless_obstacle_does_not_wreck_other_regions() {
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["A"]),
            classify_into_pieces(vec![rect_ring(0.0, 0.0, 1.2, 1.2)]),
        );
        map.insert(
            Combination::new(&["B"]),
            classify_into_pieces(vec![rect_ring(10.0, 0.0, 20.0, 10.0)]),
        );
        let regions = RegionPolygons::from_map(map);
        let sizes = sizes(&[
            ("A", uniform_sizes(2, 0.5, 0.2)),
            ("B", uniform_sizes(6, 2.0, 0.8)),
        ]);
        let k_free = place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default()).scale;

        // A box swallowing the tiny region whole.
        let obstacles = [Rectangle::new(Point::new(0.6, 0.6), 4.0, 4.0)];
        let result = place_glyph_boxes(
            &regions,
            &sizes,
            &GlyphBoxOptions::default().obstacles(obstacles),
        );

        assert!(
            result.scale >= OBSTACLE_SHRINK_FLOOR * k_free - 1e-12,
            "one hopeless region collapsed the diagram-wide scale"
        );
        // The roomy region keeps every item; the buried one packs under its
        // box rather than dropping marks.
        assert_eq!(result.boxes["B"].len(), 6);
        assert_eq!(result.boxes["A"].len(), 2);
    }

    #[test]
    fn random_is_seed_deterministic() {
        let regions = two_set_regions();
        let sizes = sizes(&[
            ("A", uniform_sizes(4, 0.2, 0.08)),
            ("B", uniform_sizes(2, 0.2, 0.08)),
        ]);
        let opts_a = GlyphBoxOptions::default()
            .arrangement(GlyphArrangement::Random)
            .seed(7);
        assert_eq!(
            place_glyph_boxes(&regions, &sizes, &opts_a),
            place_glyph_boxes(&regions, &sizes, &opts_a)
        );
        let other = place_glyph_boxes(&regions, &sizes, &opts_a.clone().seed(8));
        assert_ne!(
            place_glyph_boxes(&regions, &sizes, &opts_a).boxes,
            other.boxes
        );
    }

    #[test]
    fn random_with_obstacles_is_seed_deterministic() {
        let regions = boxed_region(0.0, 0.0, 10.0, 10.0);
        let options = GlyphBoxOptions::default()
            .arrangement(GlyphArrangement::Random)
            .seed(7)
            .obstacles([Rectangle::new(Point::new(5.0, 5.0), 3.0, 1.5)]);
        let sizes = sizes(&[("A", uniform_sizes(5, 1.5, 0.6))]);
        assert_eq!(
            place_glyph_boxes(&regions, &sizes, &options),
            place_glyph_boxes(&regions, &sizes, &options)
        );
    }

    #[test]
    fn uniform_ignores_the_seed() {
        let regions = two_set_regions();
        let sizes = sizes(&[("A", uniform_sizes(4, 0.2, 0.08))]);
        assert_eq!(
            place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default().seed(1)),
            place_glyph_boxes(&regions, &sizes, &GlyphBoxOptions::default().seed(2))
        );
    }

    #[test]
    fn box_rect_separation_matches_a_point_box() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 4.0);
        assert_eq!(box_rect_separation(0.0, 0.0, 0.0, 0.0, &rect), 0.0);
        assert!((box_rect_separation(4.0, 0.0, 0.0, 0.0, &rect) - 3.0).abs() < 1e-12);
        // A box of its own shrinks the gap by its half-extent.
        assert!((box_rect_separation(4.0, 0.0, 2.0, 0.0, &rect) - 2.0).abs() < 1e-12);
        assert_eq!(box_rect_separation(4.0, 0.0, 6.0, 0.0, &rect), 0.0);
    }
}
