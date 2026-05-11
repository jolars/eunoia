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
//! [`InteriorPolicy::Strict`] is implemented end-to-end with two exterior
//! solvers: [`ExteriorPolicy::Raycast`] (the default — closed-form anchor
//! along the centroid→POI ray) and [`ExteriorPolicy::ForceDirected`] (an
//! iterative spring-and-repulsion solve that's polygon-aware: each label
//! repels both other labels *and* foreign region pieces, so labels are
//! prevented from drifting across unrelated regions). The other variants
//! ([`InteriorPolicy::Loose`], [`ExteriorPolicy::None`]) are present in the
//! enum so the surface is forward-compatible, but selecting them returns
//! [`PlacementError::Unimplemented`].

use std::collections::HashMap;
use std::fmt;

use crate::geometry::primitives::Point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::plotting::clip::polygon_union_many;
use crate::plotting::inscribed::{fit_label_in_region, principal_axis};
use crate::plotting::regions::{
    classify_into_pieces, signed_clearance, RegionPiece, RegionPolygons,
};
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
    /// Anchor is outside the diagram, decided by the force-directed solver
    /// — emitted by [`ExteriorPolicy::ForceDirected`].
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
    /// The anchor is placed outside the union polygon of the fitted shapes
    /// (or the container, when complement is set), padded by `margin`.
    /// `margin = None` selects a proportional default of
    /// `0.5 * max(label_w, label_h)` per region.
    ///
    /// Clearance against the union polygon is checked per-vertex with the
    /// full label box footprint — width and height — so a long label on a
    /// diagonal ray doesn't dip back into a curving boundary. Falls back
    /// to AABB-based separation when the union is degenerate.
    Raycast { margin: Option<f64> },
    /// Iterative spring/repulsion solve. Initial positions come from the
    /// raycast geometry (so labels start in the right halfspace), then a
    /// damped relaxation balances three forces:
    ///
    /// * a soft spring pulling each label back toward its raycast home,
    /// * label-vs-label AABB repulsion (any pair of overlapping label boxes
    ///   gets pushed apart), and
    /// * label-vs-foreign-region repulsion — every label avoids every
    ///   region piece **except its own**, treating each foreign piece as a
    ///   no-go zone and pushing along the polygon boundary normal.
    ///
    /// The polygon-vs-label repulsion is what differentiates this from
    /// ggrepel-style point/box repulsion: labels can be constrained to not
    /// drift across unrelated regions, which the centroid-through-POI
    /// raycast can't enforce on its own.
    ///
    /// `margin = None` selects the same per-region proportional default as
    /// [`ExteriorPolicy::Raycast`]. `iterations = None` selects 200 — fine
    /// for typical 3–5 label exteriors, raise it for crowded diagrams that
    /// haven't converged.
    ForceDirected {
        margin: Option<f64>,
        iterations: Option<usize>,
    },
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
    let exterior_kind = match strategy.exterior {
        ExteriorPolicy::Raycast { margin } => ExteriorPlan::Raycast { margin },
        ExteriorPolicy::None => {
            return Err(PlacementError::Unimplemented("ExteriorPolicy::None"));
        }
        ExteriorPolicy::ForceDirected { margin, iterations } => ExteriorPlan::ForceDirected {
            margin,
            iterations: iterations.unwrap_or(DEFAULT_FORCE_DIRECTED_ITERATIONS),
        },
    };

    // Diagram boundary: when there's no container, we ray-cast against the
    // union of all region piece outer rings (a tight polygon-shaped
    // boundary). The AABB stays around as a fallback for degenerate
    // layouts where the union is empty or the union pass produces no
    // pieces; it's also what we use unmodified when a container rectangle
    // is provided, since the container is already axis-aligned by design.
    let union_pieces: Vec<RegionPiece> = if container.is_none() {
        build_diagram_union(regions)
    } else {
        Vec::new()
    };
    let diagram_bbox = match container {
        Some(rect) => Some(*rect),
        None => union_bbox(regions),
    };
    let centroid = diagram_bbox.map(|r| *r.center());

    let pois = regions.label_points(strategy.precision);
    let mut out: HashMap<String, LabelPlacement> = HashMap::with_capacity(sizes.len());
    let mut exteriors: Vec<ExteriorEntry> = Vec::new();

    let raycast_margin_opt = match exterior_kind {
        ExteriorPlan::Raycast { margin } => margin,
        ExteriorPlan::ForceDirected { margin, .. } => margin,
    };

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

        // Strict: try interior first, fall through to exterior on miss.
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

        // Need exterior. Both Raycast and ForceDirected require a diagram
        // bbox / centroid / region POI to compute initial positions; if any
        // is missing the input is degenerate and we silently skip the region
        // (caller still sees a missing entry, which they can fall back on
        // however they choose).
        let Some(bbox) = diagram_bbox else { continue };
        let Some(centroid) = centroid else { continue };
        let Some(poi) = pois.get(&combo).copied() else {
            continue;
        };

        let direction = direction_from(&poi, &centroid, pieces);
        let margin = raycast_margin_opt.unwrap_or_else(|| 0.5 * w.max(h));
        let anchor = raycast_anchor_union(&poi, w, h, &union_pieces, margin, direction)
            .unwrap_or_else(|| raycast_anchor(&poi, w, h, &bbox, margin, direction));
        exteriors.push(ExteriorEntry {
            key: key.clone(),
            combo,
            anchor,
            home: anchor,
            poi,
            direction,
            margin,
            w,
            h,
        });
    }

    // Resolve overlaps between exterior labels. Different solvers per
    // strategy: Raycast uses a cheap tangential-push collision sweep that's
    // ideal when labels share an exterior side; ForceDirected adds spring
    // and polygon-aware repulsion so labels avoid both other labels and
    // foreign region pieces.
    let exterior_kind_label = match exterior_kind {
        ExteriorPlan::Raycast { .. } => {
            resolve_exterior_collisions(&mut exteriors, 50);
            PlacementKind::ExteriorRaycast
        }
        ExteriorPlan::ForceDirected { iterations, .. } => {
            if let Some(bbox) = diagram_bbox {
                resolve_force_directed(&mut exteriors, regions, &union_pieces, &bbox, iterations);
            }
            PlacementKind::ExteriorForceDirected
        }
    };

    for entry in exteriors {
        out.insert(
            entry.key,
            LabelPlacement {
                anchor: entry.anchor,
                kind: exterior_kind_label,
                tether: Some(entry.poi),
            },
        );
    }

    Ok(out)
}

/// Axis-aligned bounding box of every placed label box.
///
/// For each entry in `placements`, expands the AABB by
/// `(anchor.x ± w/2, anchor.y ± h/2)` where `(w, h)` comes from the
/// matching entry in `sizes`. Placements with no matching size, or sizes
/// with non-finite or non-positive dimensions, are skipped.
///
/// Returns [`None`] when no placement contributed (empty input, or every
/// entry was skipped) — distinct from "zero-area bbox".
///
/// # Why callers want this
///
/// Renderers and resize loops need to extend the canvas so exterior
/// labels (which routinely sit well outside the diagram bbox) aren't
/// clipped. The naive walk is "for each placement, union with `anchor ±
/// half_label`"; this helper canonicalises it so every binding doesn't
/// reinvent the loop. Pair with [`crate::Layout::container`] and the
/// region polygons' own bbox to compute the full canvas extent.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use eunoia::plotting::{
///     placements_bbox, LabelPlacement, PlacementKind,
/// };
/// use eunoia::geometry::primitives::Point;
///
/// let mut placements = HashMap::new();
/// placements.insert("A".to_string(), LabelPlacement {
///     anchor: Point::new(0.0, 0.0),
///     kind: PlacementKind::Interior,
///     tether: None,
/// });
/// placements.insert("B".to_string(), LabelPlacement {
///     anchor: Point::new(10.0, 5.0),
///     kind: PlacementKind::ExteriorRaycast,
///     tether: Some(Point::new(8.0, 4.0)),
/// });
///
/// let mut sizes = HashMap::new();
/// sizes.insert("A".to_string(), (4.0, 2.0));
/// sizes.insert("B".to_string(), (4.0, 2.0));
///
/// let bbox = placements_bbox(&placements, &sizes).unwrap();
/// // A spans [-2, 2] × [-1, 1]; B spans [8, 12] × [4, 6].
/// // Union: [-2, 12] × [-1, 6] → centre (5, 2.5), 14 × 7.
/// assert!((bbox.center().x() - 5.0).abs() < 1e-9);
/// assert!((bbox.center().y() - 2.5).abs() < 1e-9);
/// assert!((bbox.width() - 14.0).abs() < 1e-9);
/// assert!((bbox.height() - 7.0).abs() < 1e-9);
/// ```
pub fn placements_bbox(
    placements: &HashMap<String, LabelPlacement>,
    sizes: &HashMap<String, (f64, f64)>,
) -> Option<Rectangle> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;
    for (key, placement) in placements {
        let Some(&(w, h)) = sizes.get(key) else {
            continue;
        };
        if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let half_w = 0.5 * w;
        let half_h = 0.5 * h;
        let cx = placement.anchor.x();
        let cy = placement.anchor.y();
        if !(cx.is_finite() && cy.is_finite()) {
            continue;
        }
        min_x = min_x.min(cx - half_w);
        max_x = max_x.max(cx + half_w);
        min_y = min_y.min(cy - half_h);
        max_y = max_y.max(cy + half_h);
        found = true;
    }
    if !found {
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

/// Iteratively place labels and remeasure on bbox change until the
/// canvas bbox stabilises.
///
/// Label sizes in user coordinates depend on the diagram bbox (font is
/// in physical units; user-coord size = `font_pt / scale`). Placement
/// extends the bbox by the exterior label boxes, which changes the
/// scale, which changes the sizes — a fixed-point loop. This helper
/// drives that loop for native Rust callers; FFI bindings (R, Python,
/// JS) typically iterate in their host language because passing a Rust
/// closure across the boundary buys nothing.
///
/// `measure(&bbox) -> sizes` is called at the start of each iteration
/// (after the first). The caller is responsible for the
/// `bbox → physical scale → user-coord size` mapping — eunoia has no
/// font/text knowledge.
///
/// # Convergence
///
/// Stops when the relative change in `bbox.short_side()` between
/// consecutive iterations drops below `bbox_tolerance`, or after
/// `max_iters` iterations (best-effort return). Typical diagrams
/// converge in 1–3 iterations.
///
/// # Returns
///
/// The placements from the final iteration. Any [`PlacementError`]
/// from the inner [`place_labels`] call short-circuits the loop.
pub fn place_labels_to_fixed_point<F>(
    regions: &RegionPolygons,
    container: Option<&Rectangle>,
    initial_sizes: HashMap<String, (f64, f64)>,
    strategy: &PlacementStrategy,
    mut measure: F,
    bbox_tolerance: f64,
    max_iters: usize,
) -> Result<HashMap<String, LabelPlacement>, PlacementError>
where
    F: FnMut(&Rectangle) -> HashMap<String, (f64, f64)>,
{
    let mut sizes = initial_sizes;
    let mut placements = place_labels(regions, &sizes, container, strategy)?;
    let mut prev_short =
        canvas_bbox(regions, container, &placements, &sizes).map(|r| r.width().min(r.height()));

    for _ in 0..max_iters {
        let Some(prev_bbox) = canvas_bbox(regions, container, &placements, &sizes) else {
            return Ok(placements);
        };
        let new_sizes = measure(&prev_bbox);
        let new_placements = place_labels(regions, &new_sizes, container, strategy)?;
        let new_short = canvas_bbox(regions, container, &new_placements, &new_sizes)
            .map(|r| r.width().min(r.height()));

        // Convergence check: relative change in short side. Falls back
        // to "converged" when either side has no measurable bbox (every
        // input was degenerate) — the alternative is an infinite loop.
        let converged = match (prev_short, new_short) {
            (Some(a), Some(b)) if a > 0.0 => (a - b).abs() / a <= bbox_tolerance,
            _ => true,
        };

        sizes = new_sizes;
        placements = new_placements;
        prev_short = new_short;

        if converged {
            return Ok(placements);
        }
    }
    Ok(placements)
}

/// Union AABB of `regions`, `container` (when set), and the placed
/// label boxes. Returns [`None`] only when every input is empty or
/// degenerate.
fn canvas_bbox(
    regions: &RegionPolygons,
    container: Option<&Rectangle>,
    placements: &HashMap<String, LabelPlacement>,
    sizes: &HashMap<String, (f64, f64)>,
) -> Option<Rectangle> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;

    let mut consume = |xmin: f64, xmax: f64, ymin: f64, ymax: f64| {
        if xmin.is_finite() && xmax.is_finite() && ymin.is_finite() && ymax.is_finite() {
            min_x = min_x.min(xmin);
            max_x = max_x.max(xmax);
            min_y = min_y.min(ymin);
            max_y = max_y.max(ymax);
            found = true;
        }
    };

    if let Some(r) = union_bbox(regions) {
        let (xmin, xmax, ymin, ymax) = r.bounds();
        consume(xmin, xmax, ymin, ymax);
    }
    if let Some(c) = container {
        let (xmin, xmax, ymin, ymax) = c.bounds();
        consume(xmin, xmax, ymin, ymax);
    }
    if let Some(b) = placements_bbox(placements, sizes) {
        let (xmin, xmax, ymin, ymax) = b.bounds();
        consume(xmin, xmax, ymin, ymax);
    }

    if !found {
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

/// Default iteration cap for [`ExteriorPolicy::ForceDirected`] when the
/// caller doesn't override it. Generous enough for crowded 4–5 set
/// diagrams to converge under the soft-spring schedule below; cheap
/// enough that the typical 3-set diagram runs in under a millisecond.
const DEFAULT_FORCE_DIRECTED_ITERATIONS: usize = 200;

/// Internal expansion of the user-facing [`ExteriorPolicy`] with defaults
/// resolved (e.g. `iterations = None` → 200).
#[derive(Clone, Copy)]
enum ExteriorPlan {
    Raycast {
        margin: Option<f64>,
    },
    ForceDirected {
        margin: Option<f64>,
        iterations: usize,
    },
}

/// In-flight bookkeeping for an exterior label between raycast placement
/// and collision resolution. The fields after the resolution pass are
/// re-packed into a [`LabelPlacement`].
struct ExteriorEntry {
    key: String,
    /// Combination this label belongs to — used by the force-directed
    /// solver to identify foreign region pieces.
    combo: Combination,
    /// Mutable: current label-box centre, updated by the resolver.
    anchor: Point,
    /// Immutable: the raycast position; the force-directed spring pulls
    /// the anchor back toward this point so labels don't drift unboundedly.
    home: Point,
    poi: Point,
    /// Unit-length raycast direction; tangent is its 90° rotation, also
    /// the natural "outward" axis the bbox containment force uses.
    direction: (f64, f64),
    /// Per-label margin around the diagram bbox. The force-directed
    /// solver re-uses this so the polygon-aware solve and the initial
    /// raycast agree on how far outside the bbox a label belongs.
    margin: f64,
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

/// Polygon-aware force-directed relaxation for exterior labels.
///
/// Each label starts at its raycast `home` and is pulled back toward it by
/// a soft spring (so labels don't drift unboundedly). Three repulsive
/// constraints push labels away from one another and from any region they
/// shouldn't overlap:
///
/// * **Label–label**: any two labels whose AABBs overlap get pushed apart
///   along their centre-to-centre direction by the smaller of the two
///   penetration depths (matching the cheap collision sweep used for the
///   pure-Raycast path).
/// * **Bbox containment**: if the label centre drifts inside
///   `bbox + margin + half_label`, push it back along the raycast
///   direction so it ends up just outside that envelope. The raycast
///   direction is the natural exit axis — perpendicular pushes would
///   trade one bbox-side overlap for another.
/// * **Foreign-region repulsion**: for every region piece that does *not*
///   belong to this label's combo, treat the polygon as a no-go zone with
///   a `buffer = 0.5 * max(label_w, label_h)` skin. When the label centre
///   sits inside the buffer (or, worse, inside the polygon itself), push
///   along the polygon-boundary normal until the buffer clears. This is
///   the polygon-awareness the strategy is named for: ggrepel-style
///   point/box repulsion can't see polygon geometry, so labels routinely
///   land on top of unrelated regions.
///
/// Convergence: the spring + repulsion fixed-point is linear in the worst
/// case, so we iterate until either the largest per-iteration displacement
/// drops below a tolerance proportional to the diagram's short side, or
/// `max_iters` is hit. We don't add velocity / momentum because the
/// repulsion forces are penetration-resolving (they apply exactly the
/// displacement needed to clear the violation) and the spring is small,
/// so under-relaxation isn't required.
fn resolve_force_directed(
    entries: &mut [ExteriorEntry],
    regions: &RegionPolygons,
    union_pieces: &[RegionPiece],
    bbox: &Rectangle,
    max_iters: usize,
) {
    if entries.is_empty() || max_iters == 0 {
        return;
    }

    let bbox_short = bbox.width().min(bbox.height()).max(1e-6);
    let tolerance = 1e-4 * bbox_short;
    let spring = 0.05_f64;

    // Cache the foreign-piece set for each label once: parsing the
    // combination map per iteration would dominate the inner loop. The
    // owning combo is *excluded* — labels are free to overlap their own
    // region's pieces (that's the whole point of an exterior label,
    // which originated because the label couldn't fit inside).
    let foreign_pieces: Vec<Vec<&RegionPiece>> = entries
        .iter()
        .map(|entry| {
            let mut foreign: Vec<&RegionPiece> = Vec::new();
            for (combo, pieces) in regions.iter() {
                if combo == &entry.combo {
                    continue;
                }
                for piece in pieces {
                    foreign.push(piece);
                }
            }
            foreign
        })
        .collect();

    for _ in 0..max_iters {
        let mut max_move: f64 = 0.0;

        // 1. Soft spring toward home + per-label constraint resolution.
        //    Done in a single pass per label so each constraint check
        //    sees the latest position from the previous label.
        for i in 0..entries.len() {
            let prev = entries[i].anchor;

            // Spring nudge.
            let dx = entries[i].home.x() - entries[i].anchor.x();
            let dy = entries[i].home.y() - entries[i].anchor.y();
            entries[i].anchor = Point::new(prev.x() + spring * dx, prev.y() + spring * dy);

            // Diagram-boundary containment (push along the raycast
            // direction). Uses the union polygon when available so labels
            // can settle into AABB corners that contain no actual shape;
            // falls back to the AABB envelope when the union is empty
            // (degenerate input, or when a container is in play).
            let (bx, by) = if union_pieces.is_empty() {
                bbox_push_along(
                    &entries[i].anchor,
                    0.5 * entries[i].w,
                    0.5 * entries[i].h,
                    bbox,
                    entries[i].margin,
                    entries[i].direction,
                )
            } else {
                union_push_along(
                    &entries[i].anchor,
                    0.5 * entries[i].w,
                    0.5 * entries[i].h,
                    union_pieces,
                    entries[i].margin,
                    entries[i].direction,
                )
            };
            entries[i].anchor = Point::new(entries[i].anchor.x() + bx, entries[i].anchor.y() + by);

            // Foreign-region repulsion.
            let buffer = 0.5 * entries[i].w.max(entries[i].h);
            for piece in &foreign_pieces[i] {
                let (px, py) = polygon_push(&entries[i].anchor, buffer, piece);
                entries[i].anchor =
                    Point::new(entries[i].anchor.x() + px, entries[i].anchor.y() + py);
            }

            let moved = ((entries[i].anchor.x() - prev.x()).powi(2)
                + (entries[i].anchor.y() - prev.y()).powi(2))
            .sqrt();
            max_move = max_move.max(moved);
        }

        // 2. Resolve label-label AABB overlaps to disjointness. Sub-
        //    iterating here matters: the spring step above can re-introduce
        //    an overlap the previous outer iteration just resolved, and a
        //    single pairwise pass only guarantees per-pair disjointness
        //    (not transitive — moving A out of B can shove A into C). The
        //    inner loop terminates as soon as a sweep makes no moves, so
        //    converged configurations cost one cheap O(n²) pass per outer
        //    iteration.
        const MAX_LABEL_COLLISION_SWEEPS: usize = 50;
        for _ in 0..MAX_LABEL_COLLISION_SWEEPS {
            let mut moved_any = false;
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
                    let len = (dx * dx + dy * dy).sqrt();
                    let (nx, ny) = if len > 1e-9 {
                        (dx / len, dy / len)
                    } else {
                        // Perfectly stacked labels — pick a deterministic
                        // separation axis so the next sweep has a usable
                        // gradient.
                        (1.0, 0.0)
                    };
                    let push = 0.5 * (ox.min(oy) + 1e-6);
                    a.anchor = Point::new(a.anchor.x() + push * nx, a.anchor.y() + push * ny);
                    b.anchor = Point::new(b.anchor.x() - push * nx, b.anchor.y() - push * ny);
                    max_move = max_move.max(push);
                    moved_any = true;
                }
            }
            if !moved_any {
                break;
            }
        }

        if max_move < tolerance {
            break;
        }
    }
}

/// Push the label box back outside `bbox + margin + half_label` along the
/// raycast direction `dir` if it currently violates that envelope.
///
/// Returns the displacement to apply to the anchor; `(0, 0)` when no
/// violation. We push along `dir` rather than the shortest exit axis
/// because the shortest axis can pop a label across the diagram into a
/// completely different exterior halfspace, which other forces would then
/// have to undo.
fn bbox_push_along(
    center: &Point,
    half_w: f64,
    half_h: f64,
    bbox: &Rectangle,
    margin: f64,
    dir: (f64, f64),
) -> (f64, f64) {
    let (xmin, xmax, ymin, ymax) = bbox.bounds();
    let xmin = xmin - margin - half_w;
    let xmax = xmax + margin + half_w;
    let ymin = ymin - margin - half_h;
    let ymax = ymax + margin + half_h;

    let inside = center.x() > xmin && center.x() < xmax && center.y() > ymin && center.y() < ymax;
    if !inside {
        return (0.0, 0.0);
    }

    // Smallest positive `t` along `dir` that exits the envelope.
    let eps = 1e-9;
    let mut t_min = f64::INFINITY;
    if dir.0 > eps {
        let t = (xmax - center.x()) / dir.0;
        if t > 0.0 && t < t_min {
            t_min = t;
        }
    } else if dir.0 < -eps {
        let t = (xmin - center.x()) / dir.0;
        if t > 0.0 && t < t_min {
            t_min = t;
        }
    }
    if dir.1 > eps {
        let t = (ymax - center.y()) / dir.1;
        if t > 0.0 && t < t_min {
            t_min = t;
        }
    } else if dir.1 < -eps {
        let t = (ymin - center.y()) / dir.1;
        if t > 0.0 && t < t_min {
            t_min = t;
        }
    }

    if t_min.is_finite() {
        let push = t_min + eps;
        (push * dir.0, push * dir.1)
    } else {
        (0.0, 0.0)
    }
}

/// Push the label centre out of (or away from) a foreign region piece.
///
/// `buffer` is a "skin" the label wants around itself; the function applies
/// a displacement whenever `signed_clearance(centre, piece) + buffer > 0`,
/// i.e. whenever the centre is closer to the polygon than `buffer` (or
/// inside it). The push direction is the polygon-boundary outward normal
/// at the closest boundary point, with the sign flipped when the centre
/// lies inside the piece — which keeps the gradient pointing toward the
/// nearest exit instead of deeper in.
fn polygon_push(center: &Point, buffer: f64, piece: &RegionPiece) -> (f64, f64) {
    let signed = signed_clearance(center.x(), center.y(), piece);
    let penetration = buffer + signed;
    if penetration <= 0.0 {
        return (0.0, 0.0);
    }
    let (qx, qy) = closest_point_on_piece(center.x(), center.y(), piece);
    let dx = center.x() - qx;
    let dy = center.y() - qy;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        // Centre coincides with a polygon vertex — fall back to a fixed
        // offset so the next iteration has a usable gradient.
        return (penetration, 0.0);
    }
    // `(center - closest)` points outward when the centre is outside the
    // piece, inward when it's inside; flip in the inside case so the push
    // always points toward (then beyond) the boundary.
    let sign = if signed > 0.0 { -1.0 } else { 1.0 };
    let nx = sign * dx / len;
    let ny = sign * dy / len;
    let push = penetration + 1e-9;
    (push * nx, push * ny)
}

/// Closest point to `(px, py)` on any ring of `piece` (outer + holes).
fn closest_point_on_piece(px: f64, py: f64, piece: &RegionPiece) -> (f64, f64) {
    let mut best_d2 = f64::INFINITY;
    let mut best = (px, py);
    let mut update_with = |ring: &crate::geometry::shapes::Polygon| {
        let v = ring.vertices();
        let n = v.len();
        if n < 2 {
            return;
        }
        for i in 0..n {
            let a = v[i];
            let b = v[(i + 1) % n];
            let ex = b.x() - a.x();
            let ey = b.y() - a.y();
            let len2 = ex * ex + ey * ey;
            let t = if len2 > 0.0 {
                (((px - a.x()) * ex + (py - a.y()) * ey) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let qx = a.x() + t * ex;
            let qy = a.y() + t * ey;
            let dx = px - qx;
            let dy = py - qy;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = (qx, qy);
            }
        }
    };
    update_with(&piece.outer);
    for h in &piece.holes {
        update_with(h);
    }
    best
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

/// Build the diagram's outer-boundary polygon as a list of pieces
/// (outer + holes), in the same coordinate space as `regions`.
///
/// Used by the polygon-aware raycast / force-directed paths. Returns an
/// empty vec when every region is empty or every union ring is degenerate
/// — callers fall back to the AABB envelope in that case.
fn build_diagram_union(regions: &RegionPolygons) -> Vec<RegionPiece> {
    let mut outers: Vec<Polygon> = Vec::new();
    for (_, pieces) in regions.iter() {
        for piece in pieces {
            if !piece.outer.vertices().is_empty() {
                outers.push(piece.outer.clone());
            }
        }
    }
    if outers.is_empty() {
        return Vec::new();
    }
    let rings = polygon_union_many(&outers);
    classify_into_pieces(rings)
}

/// Smallest `t > 0` such that, for every union-polygon vertex, the
/// `(half_w_m, half_h_m)`-expanded label AABB centred at
/// `origin + t·dir` no longer contains the vertex.
///
/// Returns [`None`] when no vertex is ever inside the expanded box on the
/// forward ray (degenerate input, or a POI that's already clear of every
/// vertex on the outgoing side) — callers fall back to the AABB envelope.
///
/// Per-vertex SAT — for each vertex `(px, py)`:
///   * `|px - cx| <= half_w_m` is satisfied on a t-interval `[tx_lo, tx_hi]`
///     (or always / never when `dx ≈ 0`).
///   * `|py - cy| <= half_h_m` likewise gives `[ty_lo, ty_hi]`.
///   * The vertex is inside the expanded box iff `t` is in the intersection
///     of these two intervals. The smallest `t` at which the box has
///     fully passed the vertex along `+dir` is the upper end of the
///     intersection (`t_in_hi`).
///
/// Taking the max of `t_in_hi` over every vertex gives a tight clearance
/// that accounts for the box's full footprint — width and height — on
/// curving boundaries. Only outer rings contribute; hole rings sit inside
/// the polygon and can't constrain an exterior placement.
fn last_vertex_clearance_t(
    origin: &Point,
    dir: (f64, f64),
    half_w_m: f64,
    half_h_m: f64,
    union_pieces: &[RegionPiece],
) -> Option<f64> {
    let (ox, oy) = (origin.x(), origin.y());
    let (dx, dy) = dir;
    let eps = 1e-12;
    let mut t_max = f64::NEG_INFINITY;
    let mut found = false;

    for piece in union_pieces {
        for v in piece.outer.vertices() {
            let px = v.x();
            let py = v.y();

            let (tx_lo, tx_hi) = if dx.abs() < eps {
                if (px - ox).abs() <= half_w_m {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    continue;
                }
            } else {
                let a = (px - ox - half_w_m) / dx;
                let b = (px - ox + half_w_m) / dx;
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            };
            let (ty_lo, ty_hi) = if dy.abs() < eps {
                if (py - oy).abs() <= half_h_m {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    continue;
                }
            } else {
                let a = (py - oy - half_h_m) / dy;
                let b = (py - oy + half_h_m) / dy;
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            };

            let t_in_lo = tx_lo.max(ty_lo);
            let t_in_hi = tx_hi.min(ty_hi);
            if t_in_lo < t_in_hi && t_in_hi > 0.0 {
                if t_in_hi > t_max {
                    t_max = t_in_hi;
                }
                found = true;
            }
        }
    }

    if found {
        Some(t_max)
    } else {
        None
    }
}

/// Polygon-aware analog of [`raycast_anchor`]: place the `w × h` label
/// box's centre at the smallest `t > 0` past `poi` along `direction` such
/// that the expanded box (with `margin` skin) doesn't contain any union
/// polygon vertex.
///
/// Returns [`None`] when no vertex was ever inside the expanded box on
/// the outgoing ray (degenerate input, or POI already clear of every
/// vertex on the `+direction` side) — caller falls back to the AABB
/// path.
///
/// Width and height of the label are folded into the clearance test
/// directly (via the expanded-box check), so on boundaries that curve
/// obliquely to the ray — most importantly ellipses — the label's
/// perpendicular extent can't dip back into the polygon. For a centred
/// circle/ellipse the controlling vertex coincides with the ray-exit
/// point, so the result is still tight; for off-centre or bulging
/// boundaries the perpendicular slab catches the closer vertex.
fn raycast_anchor_union(
    poi: &Point,
    w: f64,
    h: f64,
    union_pieces: &[RegionPiece],
    margin: f64,
    direction: (f64, f64),
) -> Option<Point> {
    if union_pieces.is_empty() {
        return None;
    }
    let half_w_m = 0.5 * w + margin;
    let half_h_m = 0.5 * h + margin;
    let t = last_vertex_clearance_t(poi, direction, half_w_m, half_h_m, union_pieces)?;
    let t = t.max(0.0);
    Some(Point::new(
        poi.x() + t * direction.0,
        poi.y() + t * direction.1,
    ))
}

/// Polygon-aware analog of [`bbox_push_along`]: if any union-polygon
/// vertex is still inside the label's expanded box on the forward ray,
/// push along `+dir` to the smallest `t` that clears every vertex.
/// Returns `(0, 0)` when no vertex constrains the forward ray — the
/// label is already outside.
fn union_push_along(
    center: &Point,
    half_w: f64,
    half_h: f64,
    union_pieces: &[RegionPiece],
    margin: f64,
    dir: (f64, f64),
) -> (f64, f64) {
    let half_w_m = half_w + margin;
    let half_h_m = half_h + margin;
    let Some(t) = last_vertex_clearance_t(center, dir, half_w_m, half_h_m, union_pieces) else {
        return (0.0, 0.0);
    };
    if t <= 0.0 {
        return (0.0, 0.0);
    }
    (t * dir.0, t * dir.1)
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

    /// Diamond (rotated square) centered at `(cx, cy)` with vertex offset
    /// `r`. Vertices: `(cx±r, cy)`, `(cx, cy±r)`. The AABB is
    /// `[cx-r, cx+r] × [cy-r, cy+r]` but the diagonal-direction boundary
    /// sits at distance `r/√2` from the center — exactly the "AABB is
    /// generous on the diagonal" pathology the union-polygon raycast is
    /// meant to handle.
    fn diamond_piece(cx: f64, cy: f64, r: f64) -> RegionPiece {
        RegionPiece {
            outer: Polygon::new(vec![
                Point::new(cx + r, cy),
                Point::new(cx, cy + r),
                Point::new(cx - r, cy),
                Point::new(cx, cy - r),
            ]),
            holes: vec![],
        }
    }

    #[test]
    fn test_raycast_uses_union_polygon_on_diagonal_direction() {
        // Two regions arranged so the centroid → POI direction for B is
        // diagonal (+45°), not axis-aligned. Otherwise the AABB and the
        // union polygon would give the same anchor and the test wouldn't
        // distinguish the new path from the old one.
        //
        // Region A: rectangle filling the SW quadrant — its job is to drag
        // the union AABB centre away from B's POI.
        // Region B: diamond at NE with center (10, 10), vertex offset 5.
        //   Diamond NE edge crosses the +45° ray from (10, 10) at distance
        //   5/√2 ≈ 3.54 — much tighter than the AABB corner at (15, 15),
        //   which is at distance 5√2 ≈ 7.07 from (10, 10).
        let mut map = HashMap::new();
        let a_piece = RegionPiece {
            outer: Polygon::new(vec![
                Point::new(-10.0, -10.0),
                Point::new(0.0, -10.0),
                Point::new(0.0, 0.0),
                Point::new(-10.0, 0.0),
            ]),
            holes: vec![],
        };
        map.insert(Combination::new(&["A"]), vec![a_piece]);
        map.insert(
            Combination::new(&["B"]),
            vec![diamond_piece(10.0, 10.0, 5.0)],
        );
        let regions = RegionPolygons::from_map(map);
        // Union AABB: [-10, 15] × [-10, 15]; centre ≈ (2.5, 2.5).
        // B's POI ≈ (10, 10) (diamond centre). Direction from centre→POI
        // ≈ (+1, +1)/√2.
        //
        // Label 10×10 doesn't fit inside the diamond (the largest
        // axis-aligned rect inscribed in |x|+|y|≤5 is 5×5 at most), so
        // the placement falls through to ExteriorRaycast.
        //
        // AABB-only path: smallest t such that the label clears both the
        // x and y sides at (15, 15). With margin = 0.5·10 = 5 and half_w
        // = 5: per-axis t = ((15 + 5 + 5) − 10)/(1/√2) = 15√2 ≈ 21.2
        // along the diagonal. Anchor distance from POI ≈ 21.2.
        //
        // Union path (per-vertex clearance): the controlling diamond
        // vertex on the +45° outgoing ray is (15, 10) or (10, 15). For
        // either, the expanded box (margin 5) leaves the vertex at the t
        // returned by the SAT check — strictly tighter than the AABB.
        let mut sizes = HashMap::new();
        sizes.insert("B".to_string(), (10.0, 10.0));
        let placements =
            place_labels(&regions, &sizes, None, &PlacementStrategy::default()).unwrap();
        let p = placements.get("B").expect("B should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);

        let dx = p.anchor.x() - 10.0;
        let dy = p.anchor.y() - 10.0;
        let dist_from_poi = (dx * dx + dy * dy).sqrt();
        // AABB-only path distance ≈ 15√2 ≈ 21.2 — assert the union path
        // gets noticeably tighter than that.
        let aabb_only = 15.0 * std::f64::consts::SQRT_2;
        assert!(
            dist_from_poi < aabb_only - 1.0,
            "expected union-polygon raycast to be tighter than AABB ({aabb_only}); got {dist_from_poi}",
        );
        // The real correctness check: the label AABB (10×10 centred at
        // the anchor) must be at least `margin = 5` away from every
        // diamond vertex. Vertices: (15, 10), (10, 15), (5, 10), (10, 5).
        let half = 5.0;
        let margin = 5.0;
        let cx = p.anchor.x();
        let cy = p.anchor.y();
        for (vx, vy) in [(15.0_f64, 10.0_f64), (10.0, 15.0), (5.0, 10.0), (10.0, 5.0)] {
            let qx = vx.clamp(cx - half, cx + half);
            let qy = vy.clamp(cy - half, cy + half);
            let d = ((vx - qx).powi(2) + (vy - qy).powi(2)).sqrt();
            assert!(
                d >= margin - 1e-6,
                "diamond vertex ({vx}, {vy}) too close to label box at ({cx}, {cy}): \
                 distance {d} < margin {margin}",
            );
        }
    }

    fn force_directed_strategy() -> PlacementStrategy {
        PlacementStrategy {
            interior: InteriorPolicy::Strict,
            exterior: ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: None,
            },
            precision: 0.01,
        }
    }

    #[test]
    fn test_force_directed_interior_fit_unchanged() {
        // A label that fits inside its region must stay at its POI under
        // ForceDirected — the solver only runs on exteriors.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy()).unwrap();
        let p = placements.get("A").unwrap();
        assert_eq!(p.kind, PlacementKind::Interior);
        assert!(p.tether.is_none());
    }

    #[test]
    fn test_force_directed_exterior_kind_and_tether() {
        // A too-big label still gets placed exteriorly; kind should be
        // ExteriorForceDirected and tether should be the region's POI
        // (mirroring the Raycast contract).
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy()).unwrap();
        let p = placements.get("A").unwrap();
        assert_eq!(p.kind, PlacementKind::ExteriorForceDirected);
        let tether = p.tether.expect("force-directed exterior must have tether");
        // Region POI is ≈ (5, 5).
        assert!((tether.x() - 5.0).abs() < 0.5);
        assert!((tether.y() - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_force_directed_resolves_label_label_overlap() {
        // Three centred isotropic regions — same warm-start collision as
        // the Raycast collision test, but routed through the force-directed
        // solver. Pairwise AABBs must end up disjoint.
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
        sizes.insert("A".to_string(), (20.0, 20.0));
        sizes.insert("B".to_string(), (20.0, 20.0));
        sizes.insert("C".to_string(), (20.0, 20.0));

        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy()).unwrap();
        let entries: Vec<&LabelPlacement> = ["A", "B", "C"]
            .iter()
            .map(|k| placements.get(*k).expect("placed"))
            .collect();
        for p in &entries {
            assert_eq!(p.kind, PlacementKind::ExteriorForceDirected);
        }
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let dx = (entries[i].anchor.x() - entries[j].anchor.x()).abs();
                let dy = (entries[i].anchor.y() - entries[j].anchor.y()).abs();
                assert!(
                    dx >= 20.0 - 1e-3 || dy >= 20.0 - 1e-3,
                    "pair ({}, {}) overlaps after ForceDirected: dx = {}, dy = {}",
                    i,
                    j,
                    dx,
                    dy
                );
            }
        }
    }

    #[test]
    fn test_force_directed_avoids_foreign_polygon() {
        // Two regions arranged side-by-side: A on the left, B on the right.
        // A's label is too big for A but a *deliberately small* label box
        // (smaller than A's bbox short side) — under Raycast the centroid
        // of A and B together is between them, and the (POI_A − centroid)
        // ray for A points LEFT (away from B). Good. Now we use a label
        // size that would naturally overlap B if we forced the anchor to
        // sit between the two regions; we then assert that the
        // force-directed anchor does NOT overlap B's polygon.
        let mut regions_map = HashMap::new();
        // A: 4×4 square at (0..4, 0..4).
        regions_map.insert(
            Combination::new(&["A"]),
            vec![RegionPiece {
                outer: Polygon::new(vec![
                    Point::new(0.0, 0.0),
                    Point::new(4.0, 0.0),
                    Point::new(4.0, 4.0),
                    Point::new(0.0, 4.0),
                ]),
                holes: vec![],
            }],
        );
        // B: 4×4 square at (10..14, 0..4).
        regions_map.insert(
            Combination::new(&["B"]),
            vec![RegionPiece {
                outer: Polygon::new(vec![
                    Point::new(10.0, 0.0),
                    Point::new(14.0, 0.0),
                    Point::new(14.0, 4.0),
                    Point::new(10.0, 4.0),
                ]),
                holes: vec![],
            }],
        );
        let regions = RegionPolygons::from_map(regions_map);

        // Label too big to fit inside A, but not absurdly so. With margin=0
        // and the union bbox = (0..14, 0..4), the (POI_A − centroid)
        // direction for A points up-and-left, and for B points up-and-right.
        // Either way the force-directed solver must keep A's anchor box
        // out of B's polygon.
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (5.0, 5.0));
        let strategy = PlacementStrategy {
            interior: InteriorPolicy::Strict,
            exterior: ExteriorPolicy::ForceDirected {
                margin: Some(0.5),
                iterations: Some(300),
            },
            precision: 0.05,
        };
        let placements = place_labels(&regions, &sizes, None, &strategy).unwrap();
        let p = placements.get("A").unwrap();
        assert_eq!(p.kind, PlacementKind::ExteriorForceDirected);

        // A's label box (centred on `anchor`, 5×5) must not overlap B's
        // polygon (axis-aligned rectangle [10..14] × [0..4]).
        let half = 2.5;
        let label_xmin = p.anchor.x() - half;
        let label_xmax = p.anchor.x() + half;
        let label_ymin = p.anchor.y() - half;
        let label_ymax = p.anchor.y() + half;
        let overlap_x = label_xmax > 10.0 && label_xmin < 14.0;
        let overlap_y = label_ymax > 0.0 && label_ymin < 4.0;
        assert!(
            !(overlap_x && overlap_y),
            "A's label box at ({}, {}) overlaps B's region",
            p.anchor.x(),
            p.anchor.y()
        );
    }

    #[test]
    fn test_force_directed_two_circle_decomposition() {
        // End-to-end through the real decomposer with the force-directed
        // exterior: every region asked for gets a placement back, with
        // the right kind discriminator.
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
        sizes.insert("A&B".to_string(), (100.0, 100.0));

        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy()).unwrap();
        assert!(placements.contains_key("A"));
        let ab = placements.get("A&B").expect("A&B should be placed");
        assert_eq!(ab.kind, PlacementKind::ExteriorForceDirected);
        assert!(ab.tether.is_some());
    }

    #[test]
    fn test_force_directed_zero_iterations_keeps_raycast_anchor() {
        // iterations = 0 short-circuits the solver, so the anchor stays at
        // the raycast warm-start position — confirms initialisation is
        // correct independent of the solver's relaxation.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let strategy = PlacementStrategy {
            interior: InteriorPolicy::Strict,
            exterior: ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: Some(0),
            },
            precision: 0.01,
        };
        let placements = place_labels(&regions, &sizes, None, &strategy).unwrap();
        let p = placements.get("A").unwrap();
        assert_eq!(p.kind, PlacementKind::ExteriorForceDirected);
        // Same expected anchor as the raycast variant: POI ≈ (5, 5),
        // isotropic +y tiebreak, anchor.y ≥ bbox_top + margin + half_h = 30.
        assert!((p.anchor.x() - 5.0).abs() < 1e-6);
        assert!(p.anchor.y() >= 30.0 - 1e-6);
    }

    // ----- placements_bbox -----------------------------------------------

    fn placement(x: f64, y: f64, kind: PlacementKind) -> LabelPlacement {
        LabelPlacement {
            anchor: Point::new(x, y),
            kind,
            tether: None,
        }
    }

    #[test]
    fn test_placements_bbox_unions_label_boxes() {
        let mut placements = HashMap::new();
        placements.insert(
            "A".to_string(),
            placement(0.0, 0.0, PlacementKind::Interior),
        );
        placements.insert(
            "B".to_string(),
            placement(10.0, 5.0, PlacementKind::ExteriorRaycast),
        );
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (4.0, 2.0));
        sizes.insert("B".to_string(), (4.0, 2.0));
        let bbox = placements_bbox(&placements, &sizes).expect("bbox");
        // A: [-2, 2] × [-1, 1]; B: [8, 12] × [4, 6]; union [-2, 12] × [-1, 6].
        assert!((bbox.center().x() - 5.0).abs() < 1e-9);
        assert!((bbox.center().y() - 2.5).abs() < 1e-9);
        assert!((bbox.width() - 14.0).abs() < 1e-9);
        assert!((bbox.height() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_placements_bbox_single_placement() {
        let mut placements = HashMap::new();
        placements.insert(
            "A".to_string(),
            placement(3.0, 4.0, PlacementKind::Interior),
        );
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (2.0, 2.0));
        let bbox = placements_bbox(&placements, &sizes).expect("bbox");
        assert!((bbox.center().x() - 3.0).abs() < 1e-9);
        assert!((bbox.center().y() - 4.0).abs() < 1e-9);
        assert!((bbox.width() - 2.0).abs() < 1e-9);
        assert!((bbox.height() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_placements_bbox_empty_returns_none() {
        let placements: HashMap<String, LabelPlacement> = HashMap::new();
        let sizes: HashMap<String, (f64, f64)> = HashMap::new();
        assert!(placements_bbox(&placements, &sizes).is_none());
    }

    #[test]
    fn test_placements_bbox_skips_missing_sizes() {
        // A has a placement but no measured size — skip it. B provides
        // both, so the bbox covers only B.
        let mut placements = HashMap::new();
        placements.insert(
            "A".to_string(),
            placement(0.0, 0.0, PlacementKind::Interior),
        );
        placements.insert(
            "B".to_string(),
            placement(10.0, 0.0, PlacementKind::Interior),
        );
        let mut sizes = HashMap::new();
        sizes.insert("B".to_string(), (4.0, 2.0));
        let bbox = placements_bbox(&placements, &sizes).expect("bbox");
        assert!((bbox.center().x() - 10.0).abs() < 1e-9);
        assert!((bbox.width() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_placements_bbox_skips_invalid_dimensions() {
        let mut placements = HashMap::new();
        placements.insert(
            "A".to_string(),
            placement(0.0, 0.0, PlacementKind::Interior),
        );
        placements.insert(
            "B".to_string(),
            placement(5.0, 5.0, PlacementKind::Interior),
        );
        placements.insert(
            "C".to_string(),
            placement(8.0, 8.0, PlacementKind::Interior),
        );
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (0.0, 1.0)); // zero width — skip
        sizes.insert("B".to_string(), (f64::NAN, 1.0)); // NaN — skip
        sizes.insert("C".to_string(), (2.0, 2.0)); // valid
        let bbox = placements_bbox(&placements, &sizes).expect("bbox");
        assert!((bbox.center().x() - 8.0).abs() < 1e-9);
        assert!((bbox.center().y() - 8.0).abs() < 1e-9);
    }

    // ----- place_labels_to_fixed_point -----------------------------------

    #[test]
    fn test_fixed_point_converges_with_inverse_scaling() {
        // Single region; the "measure" closure pretends label size is
        // proportional to canvas short side (the realistic case where
        // font is in physical units and `user_coord_size = font_pt /
        // scale`, with scale = canvas_px / bbox_extent — so label size
        // in user coords scales with bbox extent). The fixed-point
        // loop should converge in a handful of iterations and the
        // returned sizes should be self-consistent: re-running
        // `measure` on the converged bbox should produce the same sizes
        // again (within the tolerance).
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);

        // Initial size: too big for the 10×10 region → exterior on the
        // first iteration. As the loop runs, the inverse coupling will
        // either drop the label below the fit threshold (interior) or
        // settle into a steady-state exterior bbox.
        let mut initial = HashMap::new();
        initial.insert("A".to_string(), (20.0, 20.0));

        let mut iter_count = 0usize;
        let mut last_size: f64 = 0.0;
        let placements = place_labels_to_fixed_point(
            &regions,
            None,
            initial,
            &PlacementStrategy::default(),
            |bbox| {
                iter_count += 1;
                let s = bbox.width().min(bbox.height()) * 0.6;
                last_size = s;
                let mut out = HashMap::new();
                out.insert("A".to_string(), (s, s));
                out
            },
            1e-3,
            10,
        )
        .unwrap();

        let p = placements.get("A").expect("A placed");
        // Either interior or exterior is fine — the property under test
        // is convergence, not the discriminator.
        assert!(matches!(
            p.kind,
            PlacementKind::Interior | PlacementKind::ExteriorRaycast
        ));
        assert!(iter_count >= 1, "measure closure was never called");
        assert!(iter_count < 10, "loop didn't converge within 10 iters");
        assert!(last_size > 0.0, "measure produced a zero size");
    }

    #[test]
    fn test_fixed_point_returns_after_max_iters() {
        // Pathological: every iteration grows the canvas by at least
        // `bbox_tolerance + ε`. The loop must return after `max_iters`
        // rather than spin forever.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut initial = HashMap::new();
        initial.insert("A".to_string(), (20.0, 20.0));

        let mut iter_count = 0usize;
        let placements = place_labels_to_fixed_point(
            &regions,
            None,
            initial,
            &PlacementStrategy::default(),
            |bbox| {
                iter_count += 1;
                let grow = bbox.width().max(bbox.height()) * 1.5;
                let mut out = HashMap::new();
                out.insert("A".to_string(), (grow, grow));
                out
            },
            1e-3,
            5,
        )
        .unwrap();

        assert_eq!(iter_count, 5, "expected exactly max_iters measure calls");
        assert!(placements.contains_key("A"));
    }

    #[test]
    fn test_fixed_point_propagates_placement_error() {
        // Loose interior policy still returns Unimplemented from
        // place_labels — the fixed-point helper must propagate that
        // error rather than swallow it.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut initial = HashMap::new();
        initial.insert("A".to_string(), (1.0, 1.0));
        let strategy = PlacementStrategy {
            interior: InteriorPolicy::Loose,
            ..PlacementStrategy::default()
        };
        let err = place_labels_to_fixed_point(
            &regions,
            None,
            initial,
            &strategy,
            |_| HashMap::new(),
            1e-3,
            5,
        )
        .unwrap_err();
        assert!(matches!(err, PlacementError::Unimplemented(_)));
    }
}
