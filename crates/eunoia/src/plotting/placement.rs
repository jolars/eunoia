//! Strategy-driven label placement.
//!
//! [`fit_labels_in_regions`](super::fit_labels_in_regions) is a *predicate* —
//! it returns interior anchors for regions where the label fits and silently
//! omits the rest. That is occasionally what callers want, but as a default
//! placement API it leaves users without a position for any region whose
//! label is too big.
//!
//! [`place_labels`] is the high-level entry: every input region gets a
//! [`LabelPlacement`] back (or an explicit absence in the rare degenerate
//! case), with the placement [`PlacementKind`] telling the renderer whether
//! the anchor is inside the region or outside. The behaviour is configured
//! through a [`PlacementStrategy`] with two orthogonal axes — interior
//! policy and exterior fallback — so callers can opt into different
//! trade-offs without forking the implementation.
//!
//! v1 implements [`InteriorPolicy::Strict`] + [`ExteriorPolicy::Raycast`]
//! end-to-end (the default). The other variants are present in the enum so
//! the surface is forward-compatible, but selecting them returns
//! [`PlacementError::Unimplemented`] until the corresponding solver is built.

use std::collections::HashMap;
use std::fmt;

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::inscribed::{fit_label_in_region, principal_axis};
use crate::plotting::regions::{RegionPiece, RegionPolygons};
use crate::spec::Combination;

/// Result of placing one label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelPlacement {
    /// Centre of the label box, in the same coordinate space as the regions.
    pub anchor: Point,
    /// Where the placement landed (interior / overflow / exterior).
    pub kind: PlacementKind,
    /// Inside-region point to draw a leader line to. `None` for interior
    /// placements; `Some` for exterior. Renderers use this to draw the
    /// tether from `anchor` toward `tether`.
    pub tether: Option<Point>,
}

/// Discriminator on [`LabelPlacement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementKind {
    /// Box fits inside the region's polygon (Strict success).
    Interior,
    /// Loose policy: anchor is at the POI but the box overflows the region.
    /// Reserved for the future `Loose` interior policy; v1 never emits it.
    InteriorOverflow,
    /// Anchor is outside the diagram, ray-cast from centroid through POI.
    ExteriorRaycast,
    /// Anchor is outside the diagram, decided by the force-directed solver.
    /// Reserved for the future `ForceDirected` exterior policy; v1 never
    /// emits it.
    ExteriorForceDirected,
}

/// What to do when a label box would (or would not) fit inside its region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteriorPolicy {
    /// Anchor at the POI only when [`fit_label_in_region`] says yes;
    /// otherwise fall through to [`ExteriorPolicy`].
    Strict,
    /// Always anchor at the POI, even when the box overflows the polygon.
    /// `exterior` is ignored. **Not implemented in v1** — selecting this
    /// returns [`PlacementError::Unimplemented`].
    Loose,
}

/// What to do for regions where `Strict` says "doesn't fit".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExteriorPolicy {
    /// Omit the region from the result. Equivalent to the
    /// [`fit_labels_in_regions`](super::fit_labels_in_regions) predicate.
    /// **Not implemented in v1** — selecting this returns
    /// [`PlacementError::Unimplemented`]; callers wanting that behaviour
    /// should call `fit_labels_in_regions` directly.
    None,
    /// Deterministic ray from the diagram centroid through the region's POI.
    /// The anchor is placed outside the diagram bounding box (or container,
    /// when complement is set), padded by `margin`. `margin = None` selects
    /// a proportional default of `0.5 * max(label_w, label_h)` per region.
    Raycast { margin: Option<f64> },
    /// Iterative spring/repulsion solve, polygon-aware. **Not implemented in
    /// v1** — selecting this returns [`PlacementError::Unimplemented`].
    ForceDirected,
}

/// Configuration bundle for [`place_labels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementStrategy {
    pub interior: InteriorPolicy,
    pub exterior: ExteriorPolicy,
    /// Polylabel-style search precision, in the same units as the region
    /// polygons. Smaller values yield more accurate POIs at higher cost.
    pub precision: f64,
}

impl Default for PlacementStrategy {
    /// `Strict + Raycast` with proportional margin and `precision = 0.01`.
    fn default() -> Self {
        Self {
            interior: InteriorPolicy::Strict,
            exterior: ExteriorPolicy::Raycast { margin: None },
            precision: 0.01,
        }
    }
}

/// Errors returned by [`place_labels`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    /// The selected strategy variant is not implemented in this version.
    /// Callers can pattern-match on this and fall back to an alternative
    /// strategy or to a direct [`fit_labels_in_regions`] call.
    Unimplemented(&'static str),
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::Unimplemented(variant) => {
                write!(
                    f,
                    "placement strategy variant `{variant}` is not implemented yet"
                )
            }
        }
    }
}

impl std::error::Error for PlacementError {}

/// Place a label per region.
///
/// `regions` is typically obtained from [`crate::Layout::region_polygons`]
/// (or [`crate::plotting::decompose_regions`] directly). `sizes` is keyed by
/// the canonical [`Combination::to_string`] form (use `""` for the
/// complement region). `container` is the jointly-fitted complement
/// container, when the spec carried a complement — pass [`None`] otherwise.
///
/// Regions absent from `sizes`, regions whose key fails to parse, and
/// regions whose POI cannot be computed (degenerate input) are omitted from
/// the result map. Every present key has a real [`LabelPlacement`].
///
/// # Caveats
///
/// The `Strict` interior fit-check inherits the radial-conservative bound
/// from [`fit_label_in_region`]; very anisotropic regions may bounce a
/// fitting label out to the exterior fallback. Tighter bounds and the
/// `Loose` / `None` / `ForceDirected` strategy variants are planned
/// follow-ups.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::{place_labels, PlacementStrategy};
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
/// let mut sizes = HashMap::new();
/// sizes.insert("A".to_string(), (0.4, 0.2));
/// sizes.insert("B".to_string(), (0.4, 0.2));
/// sizes.insert("A&B".to_string(), (0.4, 0.2));
///
/// let strategy = PlacementStrategy::default();
/// let placements = place_labels(&regions, &sizes, None, &strategy).unwrap();
/// for placement in placements.values() {
///     // Every region that was asked for got a position back.
///     assert!(placement.anchor.x().is_finite());
/// }
/// ```
pub fn place_labels(
    regions: &RegionPolygons,
    sizes: &HashMap<String, (f64, f64)>,
    container: Option<&Rectangle>,
    strategy: &PlacementStrategy,
) -> Result<HashMap<String, LabelPlacement>, PlacementError> {
    if matches!(strategy.interior, InteriorPolicy::Loose) {
        return Err(PlacementError::Unimplemented("InteriorPolicy::Loose"));
    }
    let raycast_margin = match strategy.exterior {
        ExteriorPolicy::Raycast { margin } => margin,
        ExteriorPolicy::None => {
            return Err(PlacementError::Unimplemented("ExteriorPolicy::None"));
        }
        ExteriorPolicy::ForceDirected => {
            return Err(PlacementError::Unimplemented(
                "ExteriorPolicy::ForceDirected",
            ));
        }
    };

    // Diagram bounding box: container if set, else the union of every
    // region piece's outer-ring bbox. Returning `Ok(empty)` for an empty
    // input is fine — there's nothing to place.
    let diagram_bbox = match container {
        Some(rect) => Some(*rect),
        None => union_bbox(regions),
    };
    let centroid = diagram_bbox.map(|r| *r.center());

    let pois = regions.label_points(strategy.precision);
    let mut out: HashMap<String, LabelPlacement> = HashMap::with_capacity(sizes.len());
    let mut exteriors: Vec<ExteriorEntry> = Vec::new();

    for (key, &(w, h)) in sizes {
        if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let combo: Combination = match key.parse() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Some(pieces) = regions.get(&combo) else {
            continue;
        };

        // Strict: try interior first, fall through to raycast on miss.
        if let Some(anchor) = fit_label_in_region(pieces, w, h, strategy.precision) {
            out.insert(
                key.clone(),
                LabelPlacement {
                    anchor,
                    kind: PlacementKind::Interior,
                    tether: None,
                },
            );
            continue;
        }

        // Need exterior. Raycast requires a diagram bbox / centroid /
        // region POI; if any of those is missing the input is degenerate
        // and we silently skip the region (caller still sees a missing
        // entry, which they can fall back on however they choose).
        let Some(bbox) = diagram_bbox else { continue };
        let Some(centroid) = centroid else { continue };
        let Some(poi) = pois.get(&combo).copied() else {
            continue;
        };

        let direction = direction_from(&poi, &centroid, pieces);
        let margin = raycast_margin.unwrap_or_else(|| 0.5 * w.max(h));
        let anchor = raycast_anchor(&poi, w, h, &bbox, margin, direction);
        exteriors.push(ExteriorEntry {
            key: key.clone(),
            anchor,
            poi,
            direction,
            w,
            h,
        });
    }

    // Resolve overlaps between exterior labels. The raycast geometry pushes
    // every "doesn't fit" label out along (POI − centroid), so regions with
    // similar POI angles pile up at similar exterior positions. A handful
    // of tangential-push iterations is much cheaper than a full
    // force-directed solve and is enough for typical 3–5 label exteriors.
    resolve_exterior_collisions(&mut exteriors, 50);

    for entry in exteriors {
        out.insert(
            entry.key,
            LabelPlacement {
                anchor: entry.anchor,
                kind: PlacementKind::ExteriorRaycast,
                tether: Some(entry.poi),
            },
        );
    }

    Ok(out)
}

/// In-flight bookkeeping for an exterior label between raycast placement
/// and collision resolution. The fields after the resolution pass are
/// re-packed into a [`LabelPlacement`].
struct ExteriorEntry {
    key: String,
    anchor: Point,
    poi: Point,
    /// Unit-length raycast direction; tangent is its 90° rotation.
    direction: (f64, f64),
    w: f64,
    h: f64,
}

/// Iteratively push overlapping exterior label boxes apart along their
/// tangents (perpendicular to each label's raycast direction). Each
/// pairwise overlap moves both labels half the smaller AABB-overlap; the
/// loop terminates early when a pass makes no moves, and is capped at
/// `max_iters` to bound worst-case cost.
fn resolve_exterior_collisions(entries: &mut [ExteriorEntry], max_iters: usize) {
    if entries.len() < 2 {
        return;
    }
    let eps = 1e-9;
    for _ in 0..max_iters {
        let mut moved = false;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (left, right) = entries.split_at_mut(j);
                let a = &mut left[i];
                let b = &mut right[0];
                let dx = a.anchor.x() - b.anchor.x();
                let dy = a.anchor.y() - b.anchor.y();
                let half_sum_w = 0.5 * (a.w + b.w);
                let half_sum_h = 0.5 * (a.h + b.h);
                let ox = half_sum_w - dx.abs();
                let oy = half_sum_h - dy.abs();
                if ox <= 0.0 || oy <= 0.0 {
                    continue;
                }

                // Push axis = average of the two labels' tangents (the 90°
                // rotation of each raycast direction). Tangent is purely
                // perpendicular to `direction`, so a tangential push doesn't
                // pull the label back toward the diagram bbox.
                let tax = -a.direction.1;
                let tay = a.direction.0;
                let tbx = -b.direction.1;
                let tby = b.direction.0;
                let mut tx = 0.5 * (tax + tbx);
                let mut ty = 0.5 * (tay + tby);
                let tlen = (tx * tx + ty * ty).sqrt();
                if tlen < eps {
                    // Tangents are nearly opposite (labels on opposite sides
                    // of the diagram). Use a's tangent — the geometry won't
                    // realistically have them collide at the same point.
                    tx = tax;
                    ty = tay;
                } else {
                    tx /= tlen;
                    ty /= tlen;
                }

                // Sign: keep `a` on the side it's already on relative to the
                // pair midpoint, projected onto the push axis. Ties (`sa_t = 0`)
                // pick `+t` deterministically.
                let mid_x = 0.5 * (a.anchor.x() + b.anchor.x());
                let mid_y = 0.5 * (a.anchor.y() + b.anchor.y());
                let sa_t = (a.anchor.x() - mid_x) * tx + (a.anchor.y() - mid_y) * ty;
                let sign = if sa_t >= 0.0 { 1.0 } else { -1.0 };

                // Half each direction, plus a tiny epsilon to escape exact
                // ties (anchors at the same spot).
                let push = 0.5 * (ox.min(oy) + 1e-6);
                a.anchor = Point::new(
                    a.anchor.x() + sign * push * tx,
                    a.anchor.y() + sign * push * ty,
                );
                b.anchor = Point::new(
                    b.anchor.x() - sign * push * tx,
                    b.anchor.y() - sign * push * ty,
                );
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}

/// Bounding box of the union of every region piece's outer ring. Returns
/// `None` when `regions` has no pieces with at least one vertex.
fn union_bbox(regions: &RegionPolygons) -> Option<Rectangle> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;
    for (_, pieces) in regions.iter() {
        for piece in pieces {
            for v in piece.outer.vertices() {
                min_x = min_x.min(v.x());
                min_y = min_y.min(v.y());
                max_x = max_x.max(v.x());
                max_y = max_y.max(v.y());
                found = true;
            }
        }
    }
    if !found
        || !min_x.is_finite()
        || !max_x.is_finite()
        || !min_y.is_finite()
        || !max_y.is_finite()
        || max_x <= min_x
        || max_y <= min_y
    {
        return None;
    }
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    Some(Rectangle::new(
        Point::new(cx, cy),
        max_x - min_x,
        max_y - min_y,
    ))
}

/// Pick a unit-length ray direction from `centroid` through `poi`. When the
/// two coincide (typical for a centred Venn region), fall back to the
/// region's principal axis on its largest piece; if the region is
/// effectively isotropic (elongation < 1.05), use a fixed `+y` convention.
fn direction_from(poi: &Point, centroid: &Point, pieces: &[RegionPiece]) -> (f64, f64) {
    let dx = poi.x() - centroid.x();
    let dy = poi.y() - centroid.y();
    let mag = (dx * dx + dy * dy).sqrt();
    let eps = 1e-9;
    if mag > eps {
        return (dx / mag, dy / mag);
    }

    // POI ≈ centroid: principal-axis tiebreak.
    let largest = pieces.iter().max_by(|a, b| {
        a.area()
            .partial_cmp(&b.area())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(piece) = largest {
        let (angle, elongation) = principal_axis(piece);
        if elongation >= 1.05 {
            return (angle.cos(), angle.sin());
        }
    }
    (0.0, 1.0)
}

/// Closed-form anchor: walk along `direction` from `poi` until the
/// `w × h` label box (centred on the candidate) is AABB-separated from
/// `bbox` expanded by `margin`. Two axis-aligned rectangles are disjoint
/// iff *any* axis separates them — so we compute the per-axis `t` needed
/// to satisfy each side and take the **minimum** (the smallest `t` that
/// produces a separating axis), not the maximum (which over-constrains
/// and pushes the anchor wildly off-screen when `direction` is nearly
/// aligned with one axis).
fn raycast_anchor(
    poi: &Point,
    w: f64,
    h: f64,
    bbox: &Rectangle,
    margin: f64,
    direction: (f64, f64),
) -> Point {
    let (dx, dy) = direction;
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let bx = *bbox.center();
    let bw = 0.5 * bbox.width();
    let bh = 0.5 * bbox.height();
    let bbox_min_x = bx.x() - bw;
    let bbox_max_x = bx.x() + bw;
    let bbox_min_y = bx.y() - bh;
    let bbox_max_y = bx.y() + bh;

    // Per-axis `t` to push the label box past the bbox edge we're heading
    // toward by `margin`. We only consider sides we're moving away from —
    // a direction with `dx ≈ 0` doesn't try to clear the x-edges.
    let eps = 1e-9;
    let mut best_t = f64::INFINITY;
    if dx > eps {
        // Heading right: need anchor_left = poi.x + t*dx - half_w ≥ bbox_max_x + margin.
        let need = (bbox_max_x + margin + half_w - poi.x()) / dx;
        if need < best_t {
            best_t = need;
        }
    } else if dx < -eps {
        // Heading left: need anchor_right = poi.x + t*dx + half_w ≤ bbox_min_x - margin.
        let need = (bbox_min_x - margin - half_w - poi.x()) / dx;
        if need < best_t {
            best_t = need;
        }
    }
    if dy > eps {
        let need = (bbox_max_y + margin + half_h - poi.y()) / dy;
        if need < best_t {
            best_t = need;
        }
    } else if dy < -eps {
        let need = (bbox_min_y - margin - half_h - poi.y()) / dy;
        if need < best_t {
            best_t = need;
        }
    }

    let t = if best_t.is_finite() {
        best_t.max(0.0)
    } else {
        0.0
    };
    Point::new(poi.x() + t * dx, poi.y() + t * dy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::{Circle, Polygon};
    use crate::plotting::regions::RegionPiece;
    use crate::plotting::{decompose_regions, RegionPolygons};
    use crate::spec::{Combination, DiagramSpecBuilder, InputType};

    fn axis_aligned_square_piece(side: f64) -> RegionPiece {
        let s = side;
        RegionPiece {
            outer: Polygon::new(vec![
                Point::new(0.0, 0.0),
                Point::new(s, 0.0),
                Point::new(s, s),
                Point::new(0.0, s),
            ]),
            holes: vec![],
        }
    }

    fn single_region(combo: &[&str], pieces: Vec<RegionPiece>) -> RegionPolygons {
        let mut map = HashMap::new();
        map.insert(Combination::new(combo), pieces);
        RegionPolygons::from_map(map)
    }

    #[test]
    fn test_default_strategy_is_strict_raycast() {
        let s = PlacementStrategy::default();
        assert_eq!(s.interior, InteriorPolicy::Strict);
        assert!(matches!(
            s.exterior,
            ExteriorPolicy::Raycast { margin: None }
        ));
        assert!((s.precision - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_strict_raycast_interior_fit() {
        // 10×10 region; tiny label fits comfortably → Interior, no tether.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::Interior);
        assert!(p.tether.is_none());
        // Anchor sits inside the 10×10 region.
        assert!((0.0..=10.0).contains(&p.anchor.x()));
        assert!((0.0..=10.0).contains(&p.anchor.y()));
    }

    #[test]
    fn test_strict_raycast_exterior_fallback() {
        // 10×10 region; label too big → ExteriorRaycast with tether at POI.
        // Diagram bbox = the region itself (no container). POI sits at
        // (5, 5), centroid of the region's bbox is also (5, 5), so the
        // direction falls back to the principal axis tiebreak. The
        // axis-aligned square is isotropic (elongation ≈ 1), so direction
        // collapses to (0, +1) and the anchor lands above the bbox + margin.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        let tether = p.tether.expect("exterior placement should have a tether");
        assert!((tether.x() - 5.0).abs() < 0.5);
        assert!((tether.y() - 5.0).abs() < 0.5);
        // Anchor should be far above the bbox top (y = 10) by margin + half-h.
        let margin = 0.5 * 20.0;
        let half_h = 10.0;
        assert!(
            p.anchor.y() >= 10.0 + margin + half_h - 1e-6,
            "anchor y = {}",
            p.anchor.y()
        );
    }

    #[test]
    fn test_strict_raycast_uses_centroid_to_poi_direction() {
        // Two pieces: A is a small square in the bottom-left; B is a
        // (huge) region that drives the union bbox to be large. We place
        // a too-big label for A and confirm the anchor lands "outside"
        // the union bbox in the (POI − centroid) direction (which is
        // toward the bottom-left of the union bbox).
        let mut regions = RegionPolygons::new();
        // Small piece for A around (1, 1).
        regions.insert(
            Combination::new(&["A"]),
            vec![RegionPiece {
                outer: Polygon::new(vec![
                    Point::new(0.0, 0.0),
                    Point::new(2.0, 0.0),
                    Point::new(2.0, 2.0),
                    Point::new(0.0, 2.0),
                ]),
                holes: vec![],
            }],
        );
        // Bigger piece for B around (50, 50) — drives union bbox.
        regions.insert(
            Combination::new(&["B"]),
            vec![RegionPiece {
                outer: Polygon::new(vec![
                    Point::new(40.0, 40.0),
                    Point::new(60.0, 40.0),
                    Point::new(60.0, 60.0),
                    Point::new(40.0, 60.0),
                ]),
                holes: vec![],
            }],
        );

        let mut sizes = HashMap::new();
        // A's label is too big for the 2×2 piece but small relative to
        // the union bbox, so raycast pushes it outside the union bbox.
        sizes.insert("A".to_string(), (5.0, 5.0));

        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);

        // Union bbox is (0, 0)-(60, 60), centroid (30, 30); POI of A is
        // ≈ (1, 1). Direction is (1−30, 1−30) ≈ (-1, -1) normalised, so
        // the anchor should land below-left of the bbox.
        assert!(p.anchor.x() < 0.0, "anchor.x = {}", p.anchor.x());
        assert!(p.anchor.y() < 0.0, "anchor.y = {}", p.anchor.y());
    }

    #[test]
    fn test_strict_raycast_with_container() {
        // Same A region, but now the container drives the bbox out further;
        // the anchor must land outside the *container*, not just outside the
        // region's own bbox.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let container = Rectangle::new(Point::new(5.0, 5.0), 100.0, 100.0);

        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (50.0, 50.0));
        let placements = place_labels(
            &regions,
            &sizes,
            Some(&container),
            &PlacementStrategy::default(),
        )
        .unwrap();
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);

        // Container is (-45, -45)-(55, 55). Anchor must clear that by
        // margin (= 0.5 * max(50, 50) = 25) plus half the label dimension.
        let margin = 25.0;
        let half = 25.0;
        let outside_top = p.anchor.y() >= 55.0 + margin + half - 1e-6;
        let outside_bottom = p.anchor.y() <= -45.0 - margin - half + 1e-6;
        let outside_right = p.anchor.x() >= 55.0 + margin + half - 1e-6;
        let outside_left = p.anchor.x() <= -45.0 - margin - half + 1e-6;
        assert!(
            outside_top || outside_bottom || outside_right || outside_left,
            "anchor {:?} not outside container + margin",
            p.anchor
        );
    }

    #[test]
    fn test_strict_raycast_isotropic_centroid_tiebreak() {
        // Single region centred at (0, 0); POI ≈ centroid; isotropic shape
        // → fallback to +y direction.
        let outer = Polygon::new(vec![
            Point::new(-5.0, -5.0),
            Point::new(5.0, -5.0),
            Point::new(5.0, 5.0),
            Point::new(-5.0, 5.0),
        ]);
        let regions = single_region(
            &["A"],
            vec![RegionPiece {
                outer,
                holes: vec![],
            }],
        );
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        // +y fallback: anchor's x ≈ POI's x (≈ 0), y >> 5.
        assert!(p.anchor.x().abs() < 1e-6, "anchor x = {}", p.anchor.x());
        assert!(p.anchor.y() > 5.0, "anchor y = {}", p.anchor.y());
    }

    #[test]
    fn test_loose_returns_unimplemented() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 1.0));
        let strategy = PlacementStrategy {
            interior: InteriorPolicy::Loose,
            ..PlacementStrategy::default()
        };
        let err = place_labels(&regions, &sizes, None, &strategy).unwrap_err();
        assert!(matches!(err, PlacementError::Unimplemented(_)));
    }

    #[test]
    fn test_exterior_none_returns_unimplemented() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let sizes = HashMap::new();
        let strategy = PlacementStrategy {
            exterior: ExteriorPolicy::None,
            ..PlacementStrategy::default()
        };
        let err = place_labels(&regions, &sizes, None, &strategy).unwrap_err();
        assert!(matches!(err, PlacementError::Unimplemented(_)));
    }

    #[test]
    fn test_exterior_force_directed_returns_unimplemented() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let sizes = HashMap::new();
        let strategy = PlacementStrategy {
            exterior: ExteriorPolicy::ForceDirected,
            ..PlacementStrategy::default()
        };
        let err = place_labels(&regions, &sizes, None, &strategy).unwrap_err();
        assert!(matches!(err, PlacementError::Unimplemented(_)));
    }

    #[test]
    fn test_unknown_keys_skipped() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("Z".to_string(), (1.0, 1.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        assert!(placements.is_empty());
    }

    #[test]
    fn test_invalid_dimensions_skipped() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (0.0, 1.0));
        sizes.insert("B".to_string(), (f64::NAN, 1.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        assert!(placements.is_empty());
    }

    #[test]
    fn test_strict_raycast_resolves_collisions_between_exterior_labels() {
        // Three centred isotropic regions all fall back to the +y direction
        // and would otherwise pile up at the same exterior anchor. After
        // the resolution sweep, every pair of label boxes must be AABB-
        // disjoint.
        let outer = || {
            Polygon::new(vec![
                Point::new(-5.0, -5.0),
                Point::new(5.0, -5.0),
                Point::new(5.0, 5.0),
                Point::new(-5.0, 5.0),
            ])
        };
        let mut regions_map = HashMap::new();
        for combo in &[&["A"][..], &["B"][..], &["C"][..]] {
            regions_map.insert(
                Combination::new(combo),
                vec![RegionPiece {
                    outer: outer(),
                    holes: vec![],
                }],
            );
        }
        let regions = RegionPolygons::from_map(regions_map);

        let mut sizes = HashMap::new();
        // 20×20 labels are too big for the 10×10 region; all three fall
        // through to the centroid-tiebreak +y exterior placement.
        sizes.insert("A".to_string(), (20.0, 20.0));
        sizes.insert("B".to_string(), (20.0, 20.0));
        sizes.insert("C".to_string(), (20.0, 20.0));

        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        assert_eq!(placements.len(), 3);

        let entries: Vec<&LabelPlacement> = ["A", "B", "C"]
            .iter()
            .map(|k| placements.get(*k).expect("placed"))
            .collect();
        for p in &entries {
            assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        }

        // Pairwise AABB disjoint: |dx| ≥ w (= 20) OR |dy| ≥ h (= 20).
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let dx = (entries[i].anchor.x() - entries[j].anchor.x()).abs();
                let dy = (entries[i].anchor.y() - entries[j].anchor.y()).abs();
                assert!(
                    dx >= 20.0 - 1e-6 || dy >= 20.0 - 1e-6,
                    "pair ({}, {}) overlaps: dx = {}, dy = {}",
                    i,
                    j,
                    dx,
                    dy
                );
            }
        }
    }

    #[test]
    fn test_strict_raycast_no_collisions_leaves_anchors_unchanged() {
        // One exterior label → resolution sweep is a no-op; anchor stays
        // where the raycast geometry put it.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("A").unwrap();
        // Same expected anchor as test_strict_raycast_exterior_fallback:
        // POI ≈ (5, 5), isotropic tiebreak → +y, anchor.y ≥ bbox_top (10)
        // + margin (10) + half_h (10) = 30. The tangent-push resolution
        // must not perturb a single exterior label.
        assert!((p.anchor.x() - 5.0).abs() < 1e-6);
        assert!(p.anchor.y() >= 30.0 - 1e-6);
    }

    #[test]
    fn test_two_circle_decomposition_all_regions_placed() {
        // End-to-end through the real decomposer: every region for which
        // we ask gets a placement back, even if the label is too big.
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();
        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|n| *layout.shape_for_set(n).unwrap())
            .collect();
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, None, 64);

        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (0.2, 0.1));
        // Way too big for the intersection — forces ExteriorRaycast.
        sizes.insert("A&B".to_string(), (100.0, 100.0));

        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        assert!(placements.contains_key("A"));
        let ab = placements.get("A&B").expect("A&B should be placed");
        assert_eq!(ab.kind, PlacementKind::ExteriorRaycast);
        assert!(ab.tether.is_some());
    }
}
