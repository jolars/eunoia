//! Glyph placement.
//!
//! [`place_glyphs`] packs equally-sized circular glyphs — one mark per data
//! unit, in the style of eulerGlyphs (Micallef, Dragicevic & Fekete,
//! *Assessing the Effect of Visualizations on Bayesian Reasoning through
//! Crowdsourcing*, IEEE TVCG 2012) — inside the exclusive regions of a
//! fitted diagram. Callers supply a per-region count; the placer returns a
//! center point per glyph plus the single radius shared by every glyph in
//! the diagram.
//!
//! Two arrangements are offered via [`GlyphArrangement`]:
//!
//! * [`Uniform`](GlyphArrangement::Uniform) (default) — glyph centers sit on
//!   a hexagonal lattice anchored at the region's pole of inaccessibility,
//!   with the lattice spacing widened as far as the region allows so the
//!   glyphs spread across it instead of clumping. Fully deterministic.
//! * [`Random`](GlyphArrangement::Random) — seeded dart throwing with a
//!   minimum center-to-center spacing, giving the scattered look of the
//!   original eulerGlyphs tool. Deterministic for a fixed
//!   [`seed`](GlyphOptions::seed).
//!
//! The glyph radius is a single diagram-wide value — equal glyph size is
//! what makes counts comparable across regions. Leave
//! [`radius`](GlyphOptions::radius) unset to have the largest feasible
//! radius chosen automatically (a bisection over packing feasibility across
//! all requested regions); set it to take control, in which case regions
//! that cannot hold their count report the shortfall in
//! [`GlyphPlacements::unplaced`] rather than erroring.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::geometry::primitives::Point;
use crate::plotting::regions::{RegionPiece, RegionPolygons, poi_with_holes, signed_clearance};

/// How glyph centers are arranged within a region.
///
/// `#[non_exhaustive]`: further arrangements (e.g. phyllotaxis) are natural
/// follow-ups, so downstream matches must carry a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GlyphArrangement {
    /// Hexagonal lattice anchored at the region's pole of inaccessibility,
    /// spacing widened to spread the glyphs across the region. Deterministic
    /// — [`GlyphOptions::seed`] is ignored.
    #[default]
    Uniform,
    /// Seeded dart throwing with minimum center-to-center spacing
    /// `2r * (1 + gap)`. Deterministic for a fixed [`GlyphOptions::seed`];
    /// each region draws from its own seed-derived stream, so one region's
    /// scatter is stable when another region's count changes.
    Random,
}

/// Configuration bundle for [`place_glyphs`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl Default for GlyphOptions {
    /// [`GlyphArrangement::Uniform`], auto radius, `gap = 0.25`, `seed = 0`,
    /// `precision = 0.01`, `max_attempts = 300`.
    fn default() -> Self {
        Self {
            arrangement: GlyphArrangement::default(),
            radius: None,
            gap: 0.25,
            seed: 0,
            precision: 0.01,
            max_attempts: 300,
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
        None => match auto_radius(&work, gap, options) {
            Some(r) => r,
            None => {
                return GlyphPlacements {
                    radius: 0.0,
                    positions: HashMap::new(),
                    unplaced: work.into_iter().map(|(key, _, n)| (key, n)).collect(),
                };
            }
        },
    };

    let mut positions = HashMap::new();
    let mut unplaced = HashMap::new();
    for (key, pieces, n) in &work {
        let placed = pack_region(pieces, *n, radius, gap, options, key, true);
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

/// Largest radius at which every region in `work` holds its full count,
/// found by bisection on `(0, r_hi]`. `r_hi` is the analytic upper bound:
/// no region can hold a glyph larger than its pole-of-inaccessibility
/// clearance, nor `n` glyphs whose combined disc area exceeds its own.
/// Returns `None` when the bound is degenerate (no region can hold any
/// glyph at any radius).
fn auto_radius(
    work: &[(String, &Vec<RegionPiece>, usize)],
    gap: f64,
    options: &GlyphOptions,
) -> Option<f64> {
    let mut r_hi = f64::INFINITY;
    for (_, pieces, n) in work {
        let area: f64 = pieces.iter().map(|p| p.area()).sum();
        if let Some((_, clearance)) = poi_with_holes(pieces, options.precision) {
            // Centers must keep clearance r*(1+gap) to the boundary, so the
            // deepest point caps the radius at clearance/(1+gap).
            r_hi = r_hi
                .min(clearance / (1.0 + gap))
                .min((area / (*n as f64 * PI)).sqrt());
        }
    }
    if !r_hi.is_finite() || r_hi <= 0.0 {
        return None;
    }

    let feasible = |r: f64| {
        work.iter().all(|(key, pieces, n)| {
            pack_region(pieces, *n, r, gap, options, key, false).len() == *n
        })
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
    // Nothing feasible even at tiny radii (e.g. a degenerate piece that can
    // never hold its quota): fall back to a visible-but-small radius and let
    // the caller see the shortfall in `unplaced`.
    best.or(Some(r_hi * 0.25))
}

/// Pack `n` glyphs of radius `r` into a (possibly multi-piece) region,
/// apportioning the count across pieces by net area. `spread` selects the
/// full uniform pack (spacing widened across the region) versus the cheap
/// feasibility probe at minimum spacing; the two agree on whether all `n`
/// fit. Returns the placed centers (fewer than `n` when the region is full).
fn pack_region(
    pieces: &[RegionPiece],
    n: usize,
    r: f64,
    gap: f64,
    options: &GlyphOptions,
    region_key: &str,
    spread: bool,
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
                placed.extend(pack_uniform_piece(
                    piece,
                    quota,
                    r,
                    gap,
                    options.precision,
                    spread,
                ));
            }
            GlyphArrangement::Random => {
                placed.extend(pack_random_piece(
                    piece,
                    quota,
                    r,
                    gap,
                    options.max_attempts,
                    &mut rng,
                ));
            }
        }
    }
    placed
}

/// Largest-remainder apportionment of `n` over non-negative `weights`.
/// Floors each proportional share, then hands the leftover units to the
/// largest fractional remainders (ties broken by index) — fully
/// deterministic. A non-positive total weight sends everything to index 0.
fn apportion(n: usize, weights: &[f64]) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    if weights.is_empty() {
        return Vec::new();
    }
    if total.is_nan() || total <= 0.0 {
        let mut out = vec![0; weights.len()];
        out[0] = n;
        return out;
    }
    let mut base = Vec::with_capacity(weights.len());
    let mut rems = Vec::with_capacity(weights.len());
    for w in weights {
        let share = n as f64 * (w / total).max(0.0);
        let floor = share.floor();
        base.push(floor as usize);
        rems.push(share - floor);
    }
    let assigned: usize = base.iter().sum();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| rems[b].total_cmp(&rems[a]).then(a.cmp(&b)));
    for &i in order.iter().take(n.saturating_sub(assigned)) {
        base[i] += 1;
    }
    base
}

/// FNV-1a over `bytes`. Hand-rolled because the per-region RNG stream must
/// be stable across Rust releases and platforms, which
/// `std::collections::hash_map::DefaultHasher` does not guarantee.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Axis-aligned bounds of a ring. `None` for degenerate rings.
fn ring_bounds(ring: &[Point]) -> Option<(f64, f64, f64, f64)> {
    if ring.len() < 3 {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in ring {
        min_x = min_x.min(p.x());
        max_x = max_x.max(p.x());
        min_y = min_y.min(p.y());
        max_y = max_y.max(p.y());
    }
    (max_x > min_x && max_y > min_y).then_some((min_x, max_x, min_y, max_y))
}

/// Safety valve on lattice enumeration: a sliver piece with a large
/// bounding box probed at a tiny spacing would otherwise enumerate an
/// unbounded number of cells. Hitting the cap under-reports the valid-cell
/// count, which at worst makes the feasibility bisection settle on a
/// slightly larger radius.
const HEX_ENUM_CAP: usize = 1_000_000;

/// Centers of a hexagonal lattice with spacing `s`, anchored at `anchor`,
/// restricted to points with clearance ≥ `min_clearance` inside `piece`
/// (the caller passes the gap-padded inset `r * (1 + gap)`). Stops early
/// once `cap` valid cells are found (feasibility probing only needs to know
/// whether the count reaches the quota).
fn hex_valid_cells(
    piece: &RegionPiece,
    anchor: Point,
    s: f64,
    min_clearance: f64,
    cap: Option<usize>,
) -> Vec<Point> {
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return Vec::new();
    };
    if s <= 0.0 || !s.is_finite() {
        return Vec::new();
    }
    let row_h = s * 3.0_f64.sqrt() / 2.0;
    let j_min = ((min_y - anchor.y()) / row_h).floor() as i64;
    let j_max = ((max_y - anchor.y()) / row_h).ceil() as i64;
    let mut cells = Vec::new();
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
                cells.push(Point::new(x, y));
                if cap.is_some_and(|c| cells.len() >= c) {
                    return cells;
                }
            }
        }
    }
    cells
}

/// Uniform (hex-lattice) packer for one piece. At minimum spacing
/// `2r * (1 + gap)` the piece either holds `n` centers or it doesn't; when
/// it does and `spread` is set, the spacing is widened by bisection to the
/// largest value still yielding `n` valid cells, and the `n` cells nearest
/// the piece's pole of inaccessibility are kept (centered, deterministic).
/// Valid cells keep clearance ≥ `r * (1 + gap)` to the boundary, so glyphs
/// get the same breathing room from edges as from each other.
fn pack_uniform_piece(
    piece: &RegionPiece,
    n: usize,
    r: f64,
    gap: f64,
    precision: f64,
    spread: bool,
) -> Vec<Point> {
    let single = std::slice::from_ref(piece);
    let Some((anchor, _)) = poi_with_holes(single, precision) else {
        return Vec::new();
    };
    let inset = r * (1.0 + gap);
    let s_min = 2.0 * r * (1.0 + gap);
    let at_min = hex_valid_cells(piece, anchor, s_min, inset, Some(n));
    if at_min.len() < n || !spread {
        // Infeasible (best effort: whatever fit at minimum spacing), or a
        // feasibility probe that doesn't need the spread refinement.
        return at_min;
    }
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return at_min;
    };
    let s_max = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    let mut lo = s_min;
    let mut hi = s_max.max(s_min);
    if hex_valid_cells(piece, anchor, hi, inset, Some(n)).len() >= n {
        lo = hi;
    } else {
        // Largest spacing still fitting `n` cells; `lo` stays feasible by
        // construction, so the final enumeration below cannot come up short.
        for _ in 0..24 {
            if hi - lo <= 1e-3 * s_max {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if hex_valid_cells(piece, anchor, mid, inset, Some(n)).len() >= n {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let mut cells = hex_valid_cells(piece, anchor, lo, inset, None);
    cells.sort_by(|a, b| {
        let da = (a.x() - anchor.x()).powi(2) + (a.y() - anchor.y()).powi(2);
        let db = (b.x() - anchor.x()).powi(2) + (b.y() - anchor.y()).powi(2);
        da.total_cmp(&db)
            .then(a.y().total_cmp(&b.y()))
            .then(a.x().total_cmp(&b.x()))
    });
    cells.truncate(n);
    cells
}

/// Random (dart-throwing) packer for one piece: uniform samples over the
/// piece's bounding box, accepted when they keep clearance ≥ `r * (1 + gap)`
/// to the boundary and distance ≥ `2r * (1 + gap)` to every accepted center.
/// Gives up after `max_attempts` misses for a single glyph (a glyph that
/// exhausts its darts means the piece is effectively full).
fn pack_random_piece(
    piece: &RegionPiece,
    n: usize,
    r: f64,
    gap: f64,
    max_attempts: u32,
    rng: &mut StdRng,
) -> Vec<Point> {
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return Vec::new();
    };
    let inset = r * (1.0 + gap);
    let d_min2 = (2.0 * r * (1.0 + gap)).powi(2);
    let mut placed: Vec<Point> = Vec::with_capacity(n);
    'glyphs: for _ in 0..n {
        for _ in 0..max_attempts {
            let x = rng.random_range(min_x..=max_x);
            let y = rng.random_range(min_y..=max_y);
            let clear_of_placed = placed
                .iter()
                .all(|q| (x - q.x()).powi(2) + (y - q.y()).powi(2) >= d_min2);
            if clear_of_placed && signed_clearance(x, y, piece) >= inset {
                placed.push(Point::new(x, y));
                continue 'glyphs;
            }
        }
        break;
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::{Circle, Polygon};
    use crate::plotting::regions::classify_into_pieces;
    use crate::spec::Combination;
    use crate::{DiagramSpecBuilder, Fitter, InputType};

    fn rect_ring(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ])
    }

    /// Max signed clearance of `p` over the region's pieces — positive iff
    /// the point sits inside some piece, with that much room to spare.
    fn region_clearance(p: &Point, pieces: &[RegionPiece]) -> f64 {
        pieces
            .iter()
            .map(|piece| signed_clearance(p.x(), p.y(), piece))
            .fold(f64::NEG_INFINITY, f64::max)
    }

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

    fn two_set_regions() -> RegionPolygons {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();
        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        layout.region_polygons(&spec, 64)
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
        let pieces = classify_into_pieces(vec![rect_ring(0.0, 0.0, 2.0, 2.0)]);
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), pieces);
        let regions = RegionPolygons::from_map(map);
        let result = place_glyphs(&regions, &counts(&[("A", 1)]), &GlyphOptions::default());
        assert_eq!(result.positions["A"].len(), 1);
        let p = &result.positions["A"][0];
        // The lattice is anchored at the POI, so a lone glyph sits on it.
        assert!((p.x() - 1.0).abs() < 0.05 && (p.y() - 1.0).abs() < 0.05);
    }
}
