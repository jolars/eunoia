//! The disc footprint: [`place_glyphs`] packs equally-sized circular marks.
//!
//! See the [parent module](super) for the feature overview. Everything here
//! is specific to the single-diagram-wide-radius model; the pieces shared
//! with other footprints (obstacle primitives, apportionment, the
//! [`PackMode`] split) live one level up.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::regions::{RegionPiece, RegionPolygons, poi_with_holes, signed_clearance};

use super::relax::relax_scatter;
use super::{
    GlyphArrangement, OBSTACLE_SHRINK_FLOOR, PACK, PROBE, PackMode, apportion, clear_of_obstacles,
    fnv1a, obstacles_near, ring_bounds, sanitize_obstacles,
};

/// Configuration bundle for [`place_glyphs`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GlyphOptions {
    /// Arrangement of glyph centers within each region. Defaults to
    /// [`GlyphArrangement::Uniform`].
    pub arrangement: GlyphArrangement,
    /// Glyph radius, in the same units as the region polygons. `None`
    /// (default) selects the largest radius at which every requested region
    /// can hold its full count, found by bisection. When set explicitly,
    /// regions that cannot hold their count at this radius place as many
    /// glyphs as fit and report the rest in [`GlyphPlacements::unplaced`].
    pub radius: Option<f64>,
    /// Extra breathing room around glyphs, as a fraction of the radius. It
    /// applies both between glyphs — the minimum center-to-center distance
    /// is `2r * (1 + gap)` — and against the region boundary: centers keep
    /// a clearance of `r * (1 + gap)` to every ring, so glyphs never sit
    /// tangent to (or under the stroke of) a region edge. Negative values
    /// are clamped to `0.0` (glyphs may then touch each other and the
    /// boundary). Defaults to `0.25`.
    pub gap: f64,
    /// Seed for the [`GlyphArrangement::Random`] arrangement. Ignored by
    /// [`GlyphArrangement::Uniform`]. Defaults to `0`.
    pub seed: u64,
    /// Polylabel-style search precision for the per-region pole of
    /// inaccessibility, in the same units as the region polygons. Defaults
    /// to `0.01` (matching
    /// [`PlacementStrategy`](crate::plotting::PlacementStrategy)).
    pub precision: f64,
    /// [`GlyphArrangement::Random`] only: dart throws attempted per glyph
    /// before the region is declared full. Defaults to `300`.
    pub max_attempts: u32,
    /// Axis-aligned keep-out boxes, in the same coordinates as the region
    /// polygons. Glyph centers keep the same clearance `r * (1 + gap)` to
    /// these as they do to region boundaries, so glyphs steer clear of
    /// whatever is drawn on top of them — most usefully the caller's measured
    /// label boxes, which [`label_boxes`](crate::plotting::label_boxes)
    /// builds from a [`place_labels`](crate::plotting::place_labels) result.
    /// The list is diagram-wide, not per region: occlusion is a question of
    /// what is painted over what, and an exterior label box can land on a
    /// region that is not its own.
    ///
    /// Clearance is a strong preference, not a guarantee. Boxes tighten the
    /// auto-radius bisection, but only down to half the radius that would
    /// have been chosen without them; below that floor a region packs into
    /// its boxes rather than let one cramped region shrink every glyph in the
    /// diagram. Degenerate boxes (non-finite, or with a non-positive extent)
    /// are ignored. Empty by default, in which case placement is bit-for-bit
    /// what it was before obstacles existed.
    pub obstacles: Vec<Rectangle>,
}

impl Default for GlyphOptions {
    /// [`GlyphArrangement::Uniform`], auto radius, `gap = 0.25`, `seed = 0`,
    /// `precision = 0.01`, `max_attempts = 300`, no obstacles.
    fn default() -> Self {
        Self {
            arrangement: GlyphArrangement::default(),
            radius: None,
            gap: 0.25,
            seed: 0,
            precision: 0.01,
            max_attempts: 300,
            obstacles: Vec::new(),
        }
    }
}

impl GlyphOptions {
    /// Sets [`arrangement`](Self::arrangement) and returns `self`.
    pub fn arrangement(mut self, arrangement: GlyphArrangement) -> Self {
        self.arrangement = arrangement;
        self
    }

    /// Sets [`radius`](Self::radius) and returns `self`. Accepts a bare
    /// `f64` or `None` (e.g. `.radius(0.05)` or `.radius(None)`).
    pub fn radius(mut self, radius: impl Into<Option<f64>>) -> Self {
        self.radius = radius.into();
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

/// Result of [`place_glyphs`].
///
/// `#[non_exhaustive]`: an output type; future glyph footprints (e.g. boxes
/// for member labels) add fields, so it's not constructed downstream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GlyphPlacements {
    /// The radius actually used — the auto-chosen feasible radius, or the
    /// caller's [`GlyphOptions::radius`] echoed back. `0.0` when no region
    /// could hold any glyph (degenerate input).
    pub radius: f64,
    /// Glyph center points per region, keyed by the canonical
    /// [`Combination`](crate::spec::Combination) string form (`""` for the
    /// complement region). Regions absent from the counts input, or with a
    /// count of zero, are omitted.
    pub positions: HashMap<String, Vec<Point>>,
    /// Per-region count that did **not** fit at the used radius. Empty in
    /// auto-radius mode unless the input is degenerate; populated in
    /// fixed-radius mode when a region overflows.
    pub unplaced: HashMap<String, usize>,
}

/// Place `counts[region]` equally-sized circular glyphs inside each region.
///
/// `regions` is typically obtained from [`crate::Layout::region_polygons`]
/// (or [`crate::plotting::decompose_regions`] directly). `counts` is keyed
/// by the canonical [`Combination`](crate::spec::Combination) string form
/// (use `""` for the complement region). Regions with no matching count,
/// or a count of zero, are skipped; count keys with no matching region are
/// ignored — both mirror [`place_labels`](crate::plotting::place_labels).
///
/// Every glyph center is guaranteed to have clearance ≥ `r * (1 + gap)` to
/// the region boundary (outer ring and holes), and every pair of centers
/// within a region is at least `2r * (1 + gap)` apart — the same `gap`
/// fraction pads both, so glyphs keep visible breathing room from edges as
/// well as from each other. When a region is split into several
/// disconnected pieces, its count is apportioned across the pieces
/// proportionally to their net areas (largest-remainder rounding).
///
/// [`GlyphOptions::obstacles`] adds keep-out boxes at the same clearance.
/// Unlike the boundary and spacing invariants above, that clearance is
/// best-effort: see the field docs for the radius floor that bounds how far
/// one cramped region may shrink the whole diagram.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::{GlyphOptions, place_glyphs};
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
/// let mut counts = HashMap::new();
/// counts.insert("A".to_string(), 10);
/// counts.insert("B".to_string(), 6);
/// counts.insert("A&B".to_string(), 2);
///
/// let glyphs = place_glyphs(&regions, &counts, &GlyphOptions::default());
/// assert!(glyphs.radius > 0.0);
/// assert_eq!(glyphs.positions["A"].len(), 10);
/// assert_eq!(glyphs.positions["A&B"].len(), 2);
/// assert!(glyphs.unplaced.is_empty());
/// ```
pub fn place_glyphs(
    regions: &RegionPolygons,
    counts: &HashMap<String, usize>,
    options: &GlyphOptions,
) -> GlyphPlacements {
    let gap = options.gap.max(0.0);

    // Deterministic work list: canonical region order, regions with a
    // positive count only.
    let work: Vec<(String, &Vec<RegionPiece>, usize)> = regions
        .iter()
        .filter_map(|(combo, pieces)| {
            let key = combo.to_string();
            let n = counts.get(&key).copied().unwrap_or(0);
            (n > 0 && !pieces.is_empty()).then_some((key, pieces, n))
        })
        .collect();

    if work.is_empty() {
        return GlyphPlacements {
            radius: options.radius.unwrap_or(0.0).max(0.0),
            positions: HashMap::new(),
            unplaced: HashMap::new(),
        };
    }

    let obstacles = sanitize_obstacles(&options.obstacles);

    let radius = match options.radius {
        Some(r) if r > 0.0 && r.is_finite() => r,
        Some(_) => {
            // Non-positive or non-finite explicit radius: nothing can be
            // placed; report every count as unplaced.
            return GlyphPlacements {
                radius: 0.0,
                positions: HashMap::new(),
                unplaced: work.into_iter().map(|(key, _, n)| (key, n)).collect(),
            };
        }
        None => {
            let Some(r_hi) = analytic_r_hi(&work, gap, options.precision) else {
                return GlyphPlacements {
                    radius: 0.0,
                    positions: HashMap::new(),
                    unplaced: work.into_iter().map(|(key, _, n)| (key, n)).collect(),
                };
            };
            // Nothing feasible even at tiny radii (e.g. a degenerate piece
            // that can never hold its quota): fall back to a
            // visible-but-small radius and let the caller see the shortfall
            // in `unplaced`.
            let r_free = auto_radius(&work, gap, options, &[], r_hi)
                .unwrap_or(OBSTACLE_FREE_FALLBACK * r_hi);
            if obstacles.is_empty() {
                r_free
            } else {
                // Obstacles are hard walls for the feasibility search, so the
                // radius really does shrink to make room — but only so far.
                auto_radius(&work, gap, options, &obstacles, r_hi)
                    .unwrap_or(0.0)
                    .max(OBSTACLE_SHRINK_FLOOR * r_free)
            }
        }
    };

    let mut positions = HashMap::new();
    let mut unplaced = HashMap::new();
    for (key, pieces, n) in &work {
        let ctx = PackCtx {
            r: radius,
            gap,
            obstacles: &obstacles,
            mode: PACK,
        };
        let placed = pack_region(pieces, *n, options, key, ctx);
        if placed.len() < *n {
            unplaced.insert(key.clone(), n - placed.len());
        }
        positions.insert(key.clone(), placed);
    }

    GlyphPlacements {
        radius,
        positions,
        unplaced,
    }
}

/// Radius used when no radius is feasible at all: visible but small, so the
/// caller sees the shortfall in [`GlyphPlacements::unplaced`] rather than
/// nothing at all. Disc-only — the box packer has the caller's `min_scale`
/// to fall back to instead.
const OBSTACLE_FREE_FALLBACK: f64 = 0.25;

/// Everything a packer needs about the geometry it is packing at: the trial
/// radius, the spacing fraction, the keep-out boxes, and how strictly to
/// treat them. Bundled because it is threaded unchanged from
/// [`place_glyphs`] down to every per-piece packer.
#[derive(Debug, Clone, Copy)]
struct PackCtx<'a> {
    r: f64,
    gap: f64,
    obstacles: &'a [Rectangle],
    mode: PackMode,
}

impl PackCtx<'_> {
    /// Clearance every glyph center keeps from region rings and obstacles.
    fn inset(&self) -> f64 {
        self.r * (1.0 + self.gap)
    }

    /// Minimum center-to-center distance between two glyphs.
    fn spacing(&self) -> f64 {
        2.0 * self.inset()
    }
}

/// Analytic upper bound on the glyph radius: no region can hold a glyph
/// larger than its pole-of-inaccessibility clearance, nor `n` glyphs whose
/// combined disc area exceeds its own. `None` when the bound is degenerate
/// (no region can hold any glyph at any radius).
fn analytic_r_hi(
    work: &[(String, &Vec<RegionPiece>, usize)],
    gap: f64,
    precision: f64,
) -> Option<f64> {
    let mut r_hi = f64::INFINITY;
    for (_, pieces, n) in work {
        let area: f64 = pieces.iter().map(|p| p.area()).sum();
        if let Some((_, clearance)) = poi_with_holes(pieces, precision) {
            // Centers must keep clearance r*(1+gap) to the boundary, so the
            // deepest point caps the radius at clearance/(1+gap).
            r_hi = r_hi
                .min(clearance / (1.0 + gap))
                .min((area / (*n as f64 * PI)).sqrt());
        }
    }
    (r_hi.is_finite() && r_hi > 0.0).then_some(r_hi)
}

/// Largest radius at which every region in `work` holds its full count while
/// honouring `obstacles`, found by bisection on `(0, r_hi]`. Returns `None`
/// when no radius in that range is feasible.
fn auto_radius(
    work: &[(String, &Vec<RegionPiece>, usize)],
    gap: f64,
    options: &GlyphOptions,
    obstacles: &[Rectangle],
    r_hi: f64,
) -> Option<f64> {
    let feasible = |r: f64| {
        let ctx = PackCtx {
            r,
            gap,
            obstacles,
            mode: PROBE,
        };
        work.iter()
            .all(|(key, pieces, n)| pack_region(pieces, *n, options, key, ctx).len() == *n)
    };

    if feasible(r_hi) {
        return Some(r_hi);
    }
    let mut lo = 0.0_f64;
    let mut hi = r_hi;
    let mut best = None;
    // 2^-24 relative resolution — far below the 1e-3 stopping tolerance,
    // this is a hard cap against pathological non-convergence.
    for _ in 0..24 {
        if hi - lo <= 1e-3 * r_hi {
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

/// Pack `n` glyphs of radius `r` into a (possibly multi-piece) region,
/// apportioning the count across pieces by net area. `spread` selects the
/// full uniform pack (spacing widened across the region) versus the cheap
/// feasibility probe at minimum spacing; the two agree on whether all `n`
/// fit. Returns the placed centers (fewer than `n` when the region is full).
fn pack_region(
    pieces: &[RegionPiece],
    n: usize,
    options: &GlyphOptions,
    region_key: &str,
    ctx: PackCtx<'_>,
) -> Vec<Point> {
    let quotas = apportion(n, &pieces.iter().map(|p| p.area()).collect::<Vec<_>>());
    // One RNG stream per region, threaded through its pieces in order, so a
    // region's scatter is independent of every other region's count.
    let mut rng = StdRng::seed_from_u64(options.seed ^ fnv1a(region_key.as_bytes()));
    let mut placed = Vec::with_capacity(n);
    for (piece, quota) in pieces.iter().zip(quotas) {
        if quota == 0 {
            continue;
        }
        match options.arrangement {
            GlyphArrangement::Uniform => {
                placed.extend(pack_uniform_piece(piece, quota, options.precision, ctx));
            }
            GlyphArrangement::Random => {
                placed.extend(pack_random_piece(
                    piece,
                    quota,
                    options.max_attempts,
                    ctx,
                    &mut rng,
                ));
            }
        }
    }
    placed
}

/// Moves a blocked lattice anchor to the nearest point that clears every
/// obstacle, keeping the raw anchor when nothing qualifies.
///
/// The anchor is the lattice's phase, not just a candidate position: leaving
/// it inside a label box means the spacing bisection has to walk *away* from
/// the anchor to find a free cell, and since it seeks the largest spacing
/// that still fits the quota, a lone glyph would be flung to the far edge of
/// the region. Candidates are the axis exits of each obstacle — a hair past
/// the clearance ring, so rounding cannot leave them a ulp inside.
fn escape_obstacles(
    anchor: Point,
    piece: &RegionPiece,
    obstacles: &[Rectangle],
    clearance: f64,
) -> Point {
    if clear_of_obstacles(anchor.x(), anchor.y(), obstacles, clearance) {
        return anchor;
    }
    let pad = clearance * (1.0 + 1e-9);
    let mut best: Option<(f64, Point)> = None;
    for rect in obstacles {
        let half_w = 0.5 * rect.width() + pad;
        let half_h = 0.5 * rect.height() + pad;
        let (cx, cy) = (rect.center().x(), rect.center().y());
        let candidates = [
            Point::new(cx - half_w, anchor.y()),
            Point::new(cx + half_w, anchor.y()),
            Point::new(anchor.x(), cy - half_h),
            Point::new(anchor.x(), cy + half_h),
        ];
        for candidate in candidates {
            let (x, y) = (candidate.x(), candidate.y());
            if !clear_of_obstacles(x, y, obstacles, clearance)
                || signed_clearance(x, y, piece) < clearance
            {
                continue;
            }
            let d = (x - anchor.x()).powi(2) + (y - anchor.y()).powi(2);
            let better = best.is_none_or(|(best_d, best_p)| {
                d.total_cmp(&best_d)
                    .then(x.total_cmp(&best_p.x()))
                    .then(y.total_cmp(&best_p.y()))
                    .is_lt()
            });
            if better {
                best = Some((d, candidate));
            }
        }
    }
    best.map_or(anchor, |(_, p)| p)
}

/// Safety valve on lattice enumeration: a sliver piece with a large
/// bounding box probed at a tiny spacing would otherwise enumerate an
/// unbounded number of cells. Hitting the cap under-reports the valid-cell
/// count, which at worst makes the feasibility bisection settle on a
/// slightly larger radius.
const HEX_ENUM_CAP: usize = 1_000_000;

/// Lattice cells inside a piece, split by whether they also clear the
/// obstacles. `blocked` is the fallback pool a piece dips into when it
/// cannot hold its quota without stepping on an obstacle.
#[derive(Debug, Default)]
struct HexCells {
    free: Vec<Point>,
    blocked: Vec<Point>,
}

/// Centers of a hexagonal lattice with spacing `s`, anchored at `anchor`,
/// restricted to points with clearance ≥ `min_clearance` inside `piece`
/// (the caller passes the gap-padded inset `r * (1 + gap)`), and partitioned
/// by the same clearance to `obstacles`. Stops early once `cap` free cells
/// are found (feasibility probing only needs to know whether the count
/// reaches the quota).
fn hex_valid_cells(
    piece: &RegionPiece,
    anchor: Point,
    s: f64,
    min_clearance: f64,
    obstacles: &[Rectangle],
    cap: Option<usize>,
) -> HexCells {
    let mut cells = HexCells::default();
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return cells;
    };
    if s <= 0.0 || !s.is_finite() {
        return cells;
    }
    let row_h = s * 3.0_f64.sqrt() / 2.0;
    let j_min = ((min_y - anchor.y()) / row_h).floor() as i64;
    let j_max = ((max_y - anchor.y()) / row_h).ceil() as i64;
    let mut visited = 0usize;
    for j in j_min..=j_max {
        let y = anchor.y() + j as f64 * row_h;
        let offset = if j.rem_euclid(2) == 1 { s / 2.0 } else { 0.0 };
        let i_min = ((min_x - offset - anchor.x()) / s).floor() as i64;
        let i_max = ((max_x - offset - anchor.x()) / s).ceil() as i64;
        for i in i_min..=i_max {
            visited += 1;
            if visited > HEX_ENUM_CAP {
                return cells;
            }
            let x = anchor.x() + offset + i as f64 * s;
            if signed_clearance(x, y, piece) >= min_clearance {
                if clear_of_obstacles(x, y, obstacles, min_clearance) {
                    cells.free.push(Point::new(x, y));
                    if cap.is_some_and(|c| cells.free.len() >= c) {
                        return cells;
                    }
                } else {
                    cells.blocked.push(Point::new(x, y));
                }
            }
        }
    }
    cells
}

/// Orders lattice cells by distance from the anchor, so truncating to the
/// quota keeps the glyphs centered. Deterministic: ties break on `y` then
/// `x`.
fn sort_by_anchor(cells: &mut [Point], anchor: Point) {
    cells.sort_by(|a, b| {
        let da = (a.x() - anchor.x()).powi(2) + (a.y() - anchor.y()).powi(2);
        let db = (b.x() - anchor.x()).powi(2) + (b.y() - anchor.y()).powi(2);
        da.total_cmp(&db)
            .then(a.y().total_cmp(&b.y()))
            .then(a.x().total_cmp(&b.x()))
    });
}

/// The `n` cells nearest `anchor`: obstacle-free ones first, topped up from
/// the blocked pool when the piece cannot honour its obstacles anyway.
fn take_cells(mut cells: HexCells, n: usize, anchor: Point, honoring: bool) -> Vec<Point> {
    sort_by_anchor(&mut cells.free, anchor);
    if honoring || cells.free.len() >= n {
        cells.free.truncate(n);
        return cells.free;
    }
    sort_by_anchor(&mut cells.blocked, anchor);
    let short = n - cells.free.len();
    cells.free.extend(cells.blocked.into_iter().take(short));
    cells.free
}

/// Uniform (hex-lattice) packer for one piece. At minimum spacing
/// `2r * (1 + gap)` the piece either holds `n` centers or it doesn't; when
/// it does and `spread` is set, the spacing is widened by bisection to the
/// largest value still yielding `n` valid cells, and the `n` cells nearest
/// the piece's pole of inaccessibility are kept (centered, deterministic).
/// Valid cells keep clearance ≥ `r * (1 + gap)` to the boundary, so glyphs
/// get the same breathing room from edges as from each other.
///
/// Obstacles are honoured whenever the piece can hold its quota without
/// them; a piece that cannot falls back to the full cell set — under a
/// strict [`PackMode`] by coming up short (which is what pushes the
/// auto-radius bisection down), otherwise by packing into the obstacle.
fn pack_uniform_piece(
    piece: &RegionPiece,
    n: usize,
    precision: f64,
    ctx: PackCtx<'_>,
) -> Vec<Point> {
    let single = std::slice::from_ref(piece);
    let Some((poi, _)) = poi_with_holes(single, precision) else {
        return Vec::new();
    };
    let inset = ctx.inset();
    let local = obstacles_near(piece, ctx.obstacles, inset);
    let anchor = if local.is_empty() {
        poi
    } else {
        escape_obstacles(poi, piece, &local, inset)
    };
    let s_min = ctx.spacing();
    let at_min = hex_valid_cells(piece, anchor, s_min, inset, &local, Some(n));
    let honoring = at_min.free.len() >= n;
    if !honoring && ctx.mode.strict_obstacles {
        // The probe's verdict: this piece cannot hold its quota clear of the
        // obstacles at this radius.
        return at_min.free;
    }
    // Cells the piece may actually use, given whether it is honouring.
    let available =
        |cells: &HexCells| cells.free.len() + if honoring { 0 } else { cells.blocked.len() };
    if available(&at_min) < n || !ctx.mode.spread {
        // Infeasible (best effort: whatever fit at minimum spacing), or a
        // feasibility probe that doesn't need the spread refinement.
        return take_cells(at_min, n, anchor, honoring);
    }
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return take_cells(at_min, n, anchor, honoring);
    };
    let s_max = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    let mut lo = s_min;
    let mut hi = s_max.max(s_min);
    if available(&hex_valid_cells(piece, anchor, hi, inset, &local, Some(n))) >= n {
        lo = hi;
    } else {
        // Largest spacing still fitting `n` cells; `lo` stays feasible by
        // construction, so the final enumeration below cannot come up short.
        for _ in 0..24 {
            if hi - lo <= 1e-3 * s_max {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if available(&hex_valid_cells(piece, anchor, mid, inset, &local, Some(n))) >= n {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let cells = hex_valid_cells(piece, anchor, lo, inset, &local, None);
    take_cells(cells, n, anchor, honoring)
}

/// Random (dart-throwing) packer for one piece: uniform samples over the
/// piece's bounding box, accepted when they keep clearance ≥ `r * (1 + gap)`
/// to the boundary and distance ≥ `2r * (1 + gap)` to every accepted center.
/// Gives up after `max_attempts` misses for a single glyph (a glyph that
/// exhausts its darts means the piece is effectively full).
///
/// Obstacles join the acceptance test for a first pass; under a lenient
/// [`PackMode`] a second pass without them tops up whatever the first pass
/// could not place.
///
/// Darts guarantee a minimum spacing and nothing more, which reads as lumpy,
/// so a spreading [`PackMode`] finishes with a [relaxation
/// pass](super::relax::relax_scatter). It moves centers but places none, so
/// the feasibility probes skip it.
fn pack_random_piece(
    piece: &RegionPiece,
    n: usize,
    max_attempts: u32,
    ctx: PackCtx<'_>,
    rng: &mut StdRng,
) -> Vec<Point> {
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return Vec::new();
    };
    let inset = ctx.inset();
    let local = obstacles_near(piece, ctx.obstacles, inset);
    let d_min2 = ctx.spacing().powi(2);
    let mut placed: Vec<Point> = Vec::with_capacity(n);
    for honor_obstacles in [true, false] {
        if !honor_obstacles && (ctx.mode.strict_obstacles || local.is_empty()) {
            break;
        }
        'glyphs: while placed.len() < n {
            for _ in 0..max_attempts {
                let x = rng.random_range(min_x..=max_x);
                let y = rng.random_range(min_y..=max_y);
                let clear_of_placed = placed
                    .iter()
                    .all(|q| (x - q.x()).powi(2) + (y - q.y()).powi(2) >= d_min2);
                if clear_of_placed
                    && signed_clearance(x, y, piece) >= inset
                    && (!honor_obstacles || clear_of_obstacles(x, y, &local, inset))
                {
                    placed.push(Point::new(x, y));
                    continue 'glyphs;
                }
            }
            break;
        }
    }
    if ctx.mode.spread {
        relax_scatter(&mut placed, piece, &local, ctx.spacing(), inset);
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plotting::regions::classify_into_pieces;
    use crate::spec::Combination;

    use super::super::dist_to_rect;
    use super::super::test_utils::{boxed_region, rect_ring, region_clearance, two_set_regions};

    fn assert_invariants(result: &GlyphPlacements, regions: &RegionPolygons, gap: f64) {
        let d_min = 2.0 * result.radius * (1.0 + gap);
        let inset = result.radius * (1.0 + gap);
        for (key, points) in &result.positions {
            let combo: Combination = key.parse().unwrap();
            let pieces = regions.get(&combo).unwrap();
            for (i, p) in points.iter().enumerate() {
                assert!(
                    region_clearance(p, pieces) >= inset - 1e-9,
                    "glyph {i} in {key:?} too close to the boundary"
                );
                for q in &points[i + 1..] {
                    let d = ((p.x() - q.x()).powi(2) + (p.y() - q.y()).powi(2)).sqrt();
                    assert!(
                        d >= d_min - 1e-9,
                        "glyphs overlap in {key:?}: {d} < {d_min}"
                    );
                }
            }
        }
    }

    fn counts(entries: &[(&str, usize)]) -> HashMap<String, usize> {
        entries.iter().map(|(k, n)| (k.to_string(), *n)).collect()
    }

    #[test]
    fn uniform_fitted_two_set_places_all_counts() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        let options = GlyphOptions::default();
        let result = place_glyphs(&regions, &counts, &options);
        assert!(result.radius > 0.0);
        assert!(result.unplaced.is_empty());
        for (key, n) in &counts {
            assert_eq!(result.positions[key].len(), *n);
        }
        assert_invariants(&result, &regions, options.gap);
    }

    #[test]
    fn random_fitted_two_set_places_all_counts() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        let options = GlyphOptions::default().arrangement(GlyphArrangement::Random);
        let result = place_glyphs(&regions, &counts, &options);
        assert!(result.radius > 0.0);
        assert!(result.unplaced.is_empty());
        for (key, n) in &counts {
            assert_eq!(result.positions[key].len(), *n);
        }
        assert_invariants(&result, &regions, options.gap);
    }

    #[test]
    fn multi_piece_region_apportions_by_area() {
        // Two disjoint pieces with areas 9 and 3 (3:1) under one region:
        // n = 8 splits 6 / 2 by largest remainder.
        let pieces = classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 3.0, 3.0),
            rect_ring(10.0, 0.0, 13.0, 1.0),
        ]);
        assert_eq!(pieces.len(), 2);
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);

        let result = place_glyphs(&regions, &counts(&[("A", 8)]), &GlyphOptions::default());
        assert!(result.unplaced.is_empty());
        let points = &result.positions["A"];
        assert_eq!(points.len(), 8);
        let combo = Combination::new(&["A"]);
        let pieces = regions.get(&combo).unwrap();
        let in_big = points
            .iter()
            .filter(|p| signed_clearance(p.x(), p.y(), &pieces[0]) > 0.0)
            .count();
        let in_small = points
            .iter()
            .filter(|p| signed_clearance(p.x(), p.y(), &pieces[1]) > 0.0)
            .count();
        // Piece order out of classify_into_pieces is deterministic but not
        // specified; identify the big piece by area.
        let (expect_big, expect_small) = if pieces[0].area() > pieces[1].area() {
            (6, 2)
        } else {
            (2, 6)
        };
        assert_eq!(in_big, expect_big);
        assert_eq!(in_small, expect_small);
    }

    #[test]
    fn hole_region_keeps_glyphs_out_of_the_hole() {
        let pieces = classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 10.0, 10.0),
            rect_ring(4.0, 4.0, 6.0, 6.0),
        ]);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].holes.len(), 1);
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);

        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let options = GlyphOptions::default().arrangement(arrangement);
            let result = place_glyphs(&regions, &counts(&[("A", 20)]), &options);
            assert!(result.unplaced.is_empty(), "{arrangement:?}");
            assert_eq!(result.positions["A"].len(), 20);
            for p in &result.positions["A"] {
                let inside_hole = p.x() > 4.0 && p.x() < 6.0 && p.y() > 4.0 && p.y() < 6.0;
                assert!(
                    !inside_hole,
                    "{arrangement:?}: glyph center inside the hole"
                );
            }
            assert_invariants(&result, &regions, options.gap);
        }
    }

    #[test]
    fn relaxation_evens_out_the_raw_darts() {
        // Relative spread of the nearest-neighbour distances: the lower, the
        // more even the scatter.
        fn nn_spread(points: &[Point]) -> f64 {
            let nns: Vec<f64> = points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    points
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, q)| (p.x() - q.x()).hypot(p.y() - q.y()))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let mean = nns.iter().sum::<f64>() / nns.len() as f64;
            let var = nns.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / nns.len() as f64;
            var.sqrt() / mean
        }

        let pieces = classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)]);
        let piece = &pieces[0];
        let ctx = |spread| PackCtx {
            r: 0.4,
            gap: 0.25,
            obstacles: &[],
            mode: PackMode {
                spread,
                strict_obstacles: false,
            },
        };
        // The same dart stream both times: the only difference is the pass
        // that runs after it.
        let raw = pack_random_piece(piece, 40, 300, ctx(false), &mut StdRng::seed_from_u64(11));
        let relaxed = pack_random_piece(piece, 40, 300, ctx(true), &mut StdRng::seed_from_u64(11));

        assert_eq!(relaxed.len(), raw.len(), "relaxation must place no glyphs");
        assert!(
            nn_spread(&relaxed) < nn_spread(&raw),
            "relaxation should even the scatter out: {} -> {}",
            nn_spread(&raw),
            nn_spread(&relaxed)
        );
    }

    #[test]
    fn random_is_seed_deterministic() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        let opts_a = GlyphOptions::default()
            .arrangement(GlyphArrangement::Random)
            .seed(7);
        let first = place_glyphs(&regions, &counts, &opts_a);
        let second = place_glyphs(&regions, &counts, &opts_a);
        assert_eq!(first, second);

        let opts_b = opts_a.seed(8);
        let other = place_glyphs(&regions, &counts, &opts_b);
        assert_ne!(first.positions, other.positions);
    }

    #[test]
    fn uniform_ignores_the_seed() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6)]);
        let first = place_glyphs(&regions, &counts, &GlyphOptions::default().seed(1));
        let second = place_glyphs(&regions, &counts, &GlyphOptions::default().seed(2));
        assert_eq!(first, second);
    }

    #[test]
    fn oversized_fixed_radius_reports_unplaced() {
        let pieces = classify_into_pieces(vec![rect_ring(0.0, 0.0, 4.0, 4.0)]);
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);

        let options = GlyphOptions::default().radius(1.5);
        let result = place_glyphs(&regions, &counts(&[("A", 50)]), &options);
        assert_eq!(result.radius, 1.5);
        let placed = result.positions["A"].len();
        assert!(placed < 50);
        assert_eq!(result.unplaced["A"], 50 - placed);
        assert!(placed >= 1, "at least one glyph fits a 4x4 square at r=1.5");
        assert_invariants(&result, &regions, options.gap);
    }

    #[test]
    fn zero_and_unknown_counts_are_omitted() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 0), ("Z", 5), ("B", 3)]);
        let result = place_glyphs(&regions, &counts, &GlyphOptions::default());
        assert!(!result.positions.contains_key("A"));
        assert!(!result.positions.contains_key("Z"));
        assert_eq!(result.positions["B"].len(), 3);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn apportion_is_exact_and_deterministic() {
        assert_eq!(apportion(8, &[9.0, 3.0]), vec![6, 2]);
        assert_eq!(apportion(0, &[1.0, 1.0]), vec![0, 0]);
        assert_eq!(apportion(5, &[0.0]), vec![5]);
        assert_eq!(apportion(3, &[1.0, 1.0, 1.0]).iter().sum::<usize>(), 3);
        // Ties broken by index: equal weights, one leftover unit -> index 0.
        assert_eq!(apportion(3, &[1.0, 1.0]), vec![2, 1]);
    }

    #[test]
    fn single_glyph_lands_on_the_poi() {
        let regions = boxed_region(0.0, 0.0, 2.0, 2.0);
        let result = place_glyphs(&regions, &counts(&[("A", 1)]), &GlyphOptions::default());
        assert_eq!(result.positions["A"].len(), 1);
        let p = &result.positions["A"][0];
        // The lattice is anchored at the POI, so a lone glyph sits on it.
        assert!((p.x() - 1.0).abs() < 0.05 && (p.y() - 1.0).abs() < 0.05);
    }

    fn assert_clear_of_obstacles(
        result: &GlyphPlacements,
        obstacles: &[Rectangle],
        gap: f64,
        context: &str,
    ) {
        let clearance = result.radius * (1.0 + gap);
        for (key, points) in &result.positions {
            for (i, p) in points.iter().enumerate() {
                for rect in obstacles {
                    let d = dist_to_rect(p.x(), p.y(), rect);
                    assert!(
                        d >= clearance - 1e-9,
                        "{context}: glyph {i} in {key:?} is {d} from an obstacle, want {clearance}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_obstacles_match_no_obstacles() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let plain = GlyphOptions::default().arrangement(arrangement);
            let empty = GlyphOptions::default()
                .arrangement(arrangement)
                .obstacles([]);
            assert_eq!(
                place_glyphs(&regions, &counts, &plain),
                place_glyphs(&regions, &counts, &empty),
                "{arrangement:?}"
            );
        }
    }

    #[test]
    fn degenerate_and_distant_obstacles_are_ignored() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        let baseline = place_glyphs(&regions, &counts, &GlyphOptions::default());

        let junk = [
            Rectangle::new(Point::new(f64::NAN, 0.0), 1.0, 1.0),
            Rectangle::new(Point::new(0.0, 0.0), 0.0, 1.0),
            Rectangle::new(Point::new(0.0, 0.0), 1.0, -1.0),
            Rectangle::new(Point::new(0.0, 0.0), f64::INFINITY, 1.0),
            // Far outside every region's bounding box: dropped by the
            // per-piece pre-filter.
            Rectangle::new(Point::new(1000.0, 1000.0), 1.0, 1.0),
        ];
        let result = place_glyphs(&regions, &counts, &GlyphOptions::default().obstacles(junk));
        assert_eq!(result, baseline);
    }

    #[test]
    fn single_glyph_avoids_a_label_box_at_the_poi() {
        let regions = boxed_region(0.0, 0.0, 2.0, 2.0);
        let obstacles = [Rectangle::new(Point::new(1.0, 1.0), 0.8, 0.4)];
        // A fixed radius: left to size itself, a lone glyph fills its whole
        // region (`r_hi` caps at the equal-area disc), so no radius above the
        // floor could clear a box at the center — that trade-off is
        // `auto_radius_shrinks_but_is_floored_by_obstacles`'s business.
        let options = GlyphOptions::default().radius(0.15).obstacles(obstacles);
        let result = place_glyphs(&regions, &counts(&[("A", 1)]), &options);

        assert_eq!(result.positions["A"].len(), 1);
        assert!(result.unplaced.is_empty());
        assert_clear_of_obstacles(&result, &obstacles, options.gap, "single glyph");
        // ...and it stays next to the label rather than being flung to the
        // region edge by the spacing bisection.
        let p = &result.positions["A"][0];
        let clearance = result.radius * (1.0 + options.gap);
        assert!(
            dist_to_rect(p.x(), p.y(), &obstacles[0]) <= clearance + 0.2,
            "glyph drifted to {p:?}, far from the label box"
        );
    }

    #[test]
    fn obstacles_are_honored_for_both_arrangements() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        // A box over each region's pole of inaccessibility, where an
        // interior label would sit.
        let obstacles: Vec<Rectangle> = regions
            .iter()
            .filter_map(|(_, pieces)| {
                poi_with_holes(pieces, 0.01).map(|(poi, _)| Rectangle::new(poi, 0.25, 0.12))
            })
            .collect();
        assert_eq!(obstacles.len(), 3);

        for arrangement in [GlyphArrangement::Uniform, GlyphArrangement::Random] {
            let options = GlyphOptions::default()
                .arrangement(arrangement)
                .obstacles(obstacles.clone());
            let result = place_glyphs(&regions, &counts, &options);
            assert!(result.unplaced.is_empty(), "{arrangement:?}");
            for (key, n) in &counts {
                assert_eq!(result.positions[key].len(), *n, "{arrangement:?}");
            }
            assert_invariants(&result, &regions, options.gap);
            assert_clear_of_obstacles(
                &result,
                &obstacles,
                options.gap,
                &format!("{arrangement:?}"),
            );
        }
    }

    #[test]
    fn auto_radius_shrinks_but_is_floored_by_obstacles() {
        let regions = boxed_region(0.0, 0.0, 4.0, 4.0);
        let counts = counts(&[("A", 12)]);
        let r_free = place_glyphs(&regions, &counts, &GlyphOptions::default()).radius;

        let obstacles = [Rectangle::new(Point::new(2.0, 2.0), 2.0, 1.0)];
        let options = GlyphOptions::default().obstacles(obstacles);
        let result = place_glyphs(&regions, &counts, &options);

        assert!(
            result.radius <= r_free + 1e-12,
            "obstacles must not grow the radius: {} > {r_free}",
            result.radius
        );
        assert!(
            result.radius >= OBSTACLE_SHRINK_FLOOR * r_free - 1e-12,
            "radius {} fell below the floor of {}",
            result.radius,
            OBSTACLE_SHRINK_FLOOR * r_free
        );
        assert_eq!(result.positions["A"].len(), 12);
        assert_clear_of_obstacles(&result, &obstacles, options.gap, "shrunk radius");
    }

    #[test]
    fn hopeless_obstacle_packs_under_it_without_wrecking_other_regions() {
        // Two disjoint pieces under one region: a tiny one buried under an
        // oversized box, and a roomy one that must keep usable glyphs.
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["A"]),
            classify_into_pieces(vec![rect_ring(0.0, 0.0, 0.6, 0.6)]),
        );
        map.insert(
            Combination::new(&["B"]),
            classify_into_pieces(vec![rect_ring(10.0, 0.0, 16.0, 6.0)]),
        );
        let regions = RegionPolygons::from_map(map);
        let counts = counts(&[("A", 3), ("B", 10)]);
        let r_free = place_glyphs(&regions, &counts, &GlyphOptions::default()).radius;

        let obstacles = [Rectangle::new(Point::new(0.3, 0.3), 2.0, 2.0)];
        let options = GlyphOptions::default().obstacles(obstacles);
        let result = place_glyphs(&regions, &counts, &options);

        assert!(result.unplaced.is_empty(), "the soft pack drops no glyphs");
        assert_eq!(result.positions["A"].len(), 3);
        assert_eq!(result.positions["B"].len(), 10);
        assert!(
            result.radius >= OBSTACLE_SHRINK_FLOOR * r_free - 1e-12,
            "one hopeless region collapsed the diagram-wide radius"
        );
        // The buried region packs into its box — that is the trade the floor
        // buys, and the boundary invariants still hold.
        assert!(
            result.positions["A"]
                .iter()
                .any(|p| dist_to_rect(p.x(), p.y(), &obstacles[0]) < result.radius),
            "the buried region should have given up on the obstacle"
        );
        assert_invariants(&result, &regions, options.gap);
    }

    #[test]
    fn fixed_radius_prefers_obstacle_free_positions() {
        let regions = boxed_region(0.0, 0.0, 4.0, 4.0);
        let obstacles = [Rectangle::new(Point::new(2.0, 2.0), 1.6, 1.6)];
        let options = GlyphOptions::default().radius(0.3).obstacles(obstacles);

        let modest = place_glyphs(&regions, &counts(&[("A", 4)]), &options);
        assert!(modest.unplaced.is_empty());
        assert_clear_of_obstacles(&modest, &obstacles, options.gap, "within free capacity");

        let greedy = place_glyphs(&regions, &counts(&[("A", 20)]), &options);
        assert!(greedy.positions["A"].len() > modest.positions["A"].len());
        assert!(
            greedy.positions["A"]
                .iter()
                .any(|p| dist_to_rect(p.x(), p.y(), &obstacles[0]) < 0.3),
            "beyond free capacity the packer should use blocked cells"
        );
        assert_invariants(&greedy, &regions, options.gap);
    }

    #[test]
    fn random_with_obstacles_is_seed_deterministic() {
        let regions = two_set_regions();
        let counts = counts(&[("A", 10), ("B", 6), ("A&B", 2)]);
        let options = GlyphOptions::default()
            .arrangement(GlyphArrangement::Random)
            .seed(7)
            .obstacles([Rectangle::new(Point::new(0.0, 0.0), 1.0, 0.5)]);
        assert_eq!(
            place_glyphs(&regions, &counts, &options),
            place_glyphs(&regions, &counts, &options)
        );
    }

    #[test]
    fn dist_to_rect_is_zero_inside_and_euclidean_outside() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 4.0);
        assert_eq!(dist_to_rect(0.0, 0.0, &rect), 0.0);
        assert_eq!(dist_to_rect(1.0, 2.0, &rect), 0.0);
        // Straight out of an edge.
        assert!((dist_to_rect(4.0, 0.0, &rect) - 3.0).abs() < 1e-12);
        assert!((dist_to_rect(0.0, -5.0, &rect) - 3.0).abs() < 1e-12);
        // Diagonally off a corner: (3, 5) is (2, 3) from the corner (1, 2).
        assert!((dist_to_rect(3.0, 5.0, &rect) - 13.0_f64.sqrt()).abs() < 1e-12);
    }
}
