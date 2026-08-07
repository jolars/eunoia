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
//!
//! Glyphs are drawn under region labels, so
//! [`obstacles`](GlyphOptions::obstacles) accepts keep-out boxes — typically
//! the caller's measured label boxes, via
//! [`label_boxes`](crate::plotting::label_boxes) — that glyph centers steer
//! clear of.
//!
//! # Module layout
//!
//! The disc packer lives in [`discs`]; this file holds the pieces both it
//! and any future footprint mode share — the arrangement enum, the
//! obstacle primitives, and the apportionment of a region's count across
//! its disconnected pieces.

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::regions::RegionPiece;

mod boxes;
mod discs;
mod scan;

pub use boxes::{GlyphBoxOptions, GlyphBoxPlacements, place_glyph_boxes};
pub use discs::{GlyphOptions, GlyphPlacements, place_glyphs};

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

/// How much of the obstacle-blind glyph size the obstacle-aware bisection is
/// allowed to give away. A label box inscribed in a small region can leave
/// only a hairline annulus behind, and the feasibility predicate is a
/// conjunction over every region — without a floor, one cramped region would
/// shrink every glyph in the diagram to a dot. At the floor the cramped
/// region packs into its box instead (see [`GlyphOptions::obstacles`]).
pub(super) const OBSTACLE_SHRINK_FLOOR: f64 = 0.5;

/// Feasibility probing: minimum spacing (the spread refinement changes no
/// counts) and obstacles as hard walls, so blocked capacity really does push
/// the bisection down.
pub(super) const PROBE: PackMode = PackMode {
    spread: false,
    strict_obstacles: true,
};

/// Final placement: spread across the region, and obstacles as a preference
/// the packer abandons rather than drop glyphs.
pub(super) const PACK: PackMode = PackMode {
    spread: true,
    strict_obstacles: false,
};

/// How a pack treats spacing and obstacles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackMode {
    /// Widen the lattice spacing to spread glyphs across the piece. Off for
    /// feasibility probes, which only count what fits at minimum spacing.
    pub(super) spread: bool,
    /// Reject blocked positions outright instead of falling back to them
    /// when the piece cannot hold its quota without.
    pub(super) strict_obstacles: bool,
}

/// Largest-remainder apportionment of `n` over non-negative `weights`.
/// Floors each proportional share, then hands the leftover units to the
/// largest fractional remainders (ties broken by index) — fully
/// deterministic. A non-positive total weight sends everything to index 0.
pub(super) fn apportion(n: usize, weights: &[f64]) -> Vec<usize> {
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
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Distance from `(px, py)` to the nearest point of `rect`; `0.0` when the
/// point is inside it.
pub(super) fn dist_to_rect(px: f64, py: f64, rect: &Rectangle) -> f64 {
    let dx = (px - rect.center().x()).abs() - 0.5 * rect.width();
    let dy = (py - rect.center().y()).abs() - 0.5 * rect.height();
    dx.max(0.0).hypot(dy.max(0.0))
}

/// Whether `(px, py)` keeps `clearance` to every obstacle.
pub(super) fn clear_of_obstacles(
    px: f64,
    py: f64,
    obstacles: &[Rectangle],
    clearance: f64,
) -> bool {
    obstacles
        .iter()
        .all(|rect| dist_to_rect(px, py, rect) >= clearance)
}

/// Separation between the axis-aligned `w × h` box centered at `(cx, cy)`
/// and `rect`: the distance between the two boxes' nearest points, `0.0`
/// when they touch or overlap. The box analogue of [`dist_to_rect`], which
/// it reduces to exactly when `w` and `h` are zero.
pub(super) fn box_rect_separation(cx: f64, cy: f64, w: f64, h: f64, rect: &Rectangle) -> f64 {
    let dx = (cx - rect.center().x()).abs() - 0.5 * w - 0.5 * rect.width();
    let dy = (cy - rect.center().y()).abs() - 0.5 * h - 0.5 * rect.height();
    dx.max(0.0).hypot(dy.max(0.0))
}

/// Whether the `w × h` box centered at `(cx, cy)` keeps `clearance` to every
/// obstacle.
pub(super) fn box_clear_of_obstacles(
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    obstacles: &[Rectangle],
    clearance: f64,
) -> bool {
    obstacles
        .iter()
        .all(|rect| box_rect_separation(cx, cy, w, h, rect) >= clearance)
}

/// Drops obstacles that cannot describe a keep-out area. An empty label
/// string legitimately measures 0 × 0, so a degenerate box is a no-op rather
/// than an error.
pub(super) fn sanitize_obstacles(obstacles: &[Rectangle]) -> Vec<Rectangle> {
    obstacles
        .iter()
        .filter(|rect| {
            rect.center().x().is_finite()
                && rect.center().y().is_finite()
                && rect.width() > 0.0
                && rect.height() > 0.0
                && rect.width().is_finite()
                && rect.height().is_finite()
        })
        .copied()
        .collect()
}

/// The obstacles that can possibly affect `piece`, by bounding-box overlap.
/// Cheap to compute once per piece, and it keeps the per-candidate loop over
/// obstacles short for the common case of a label far from this region.
pub(super) fn obstacles_near(
    piece: &RegionPiece,
    obstacles: &[Rectangle],
    clearance: f64,
) -> Vec<Rectangle> {
    if obstacles.is_empty() {
        return Vec::new();
    }
    let Some((min_x, max_x, min_y, max_y)) = ring_bounds(piece.outer.vertices()) else {
        return Vec::new();
    };
    obstacles
        .iter()
        .filter(|rect| {
            let half_w = 0.5 * rect.width() + clearance;
            let half_h = 0.5 * rect.height() + clearance;
            let (cx, cy) = (rect.center().x(), rect.center().y());
            cx + half_w >= min_x
                && cx - half_w <= max_x
                && cy + half_h >= min_y
                && cy - half_h <= max_y
        })
        .copied()
        .collect()
}

/// Axis-aligned bounds of a ring. `None` for degenerate rings.
pub(super) fn ring_bounds(ring: &[Point]) -> Option<(f64, f64, f64, f64)> {
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

/// Fixtures shared by the packers' test modules. Private, so descendants of
/// this module reach it as `super::test_utils` and nothing else can.
#[cfg(test)]
mod test_utils {
    use std::collections::HashMap;

    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::{Circle, Polygon};
    use crate::plotting::regions::signed_clearance;
    use crate::plotting::regions::{RegionPiece, RegionPolygons, classify_into_pieces};
    use crate::spec::Combination;
    use crate::{DiagramSpecBuilder, Fitter, InputType};

    pub(super) fn rect_ring(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ])
    }

    /// Max signed clearance of `p` over the region's pieces — positive iff
    /// the point sits inside some piece, with that much room to spare.
    pub(super) fn region_clearance(p: &Point, pieces: &[RegionPiece]) -> f64 {
        pieces
            .iter()
            .map(|piece| signed_clearance(p.x(), p.y(), piece))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The `A` / `B` / `A&B` regions of a fitted two-set circle diagram.
    pub(super) fn two_set_regions() -> RegionPolygons {
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

    /// Single-region fixture over an axis-aligned rectangle.
    pub(super) fn boxed_region(x0: f64, y0: f64, x1: f64, y1: f64) -> RegionPolygons {
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["A"]),
            classify_into_pieces(vec![rect_ring(x0, y0, x1, y1)]),
        );
        RegionPolygons::from_map(map)
    }
}
