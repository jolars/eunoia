//! Label placement.
//!
//! [`place_labels`] is the single entry point: every requested region gets
//! a [`LabelPlacement`] back, with [`PlacementKind`] telling the renderer
//! whether the anchor lies inside the region or outside. When a label
//! doesn't fit inside its region's polygon, the leader strategy selected
//! by the [`PlacementStrategy`] takes over and positions the label outside
//! the diagram, returning a `tether` point (and any intermediate
//! `leader_waypoints`) so the caller can draw a leader line back to the
//! region.
//!
//! The leader strategy ties the *edge type* to the *placement algorithm*
//! that suits it. Today the only edge type is [`LeaderStrategy::Straight`]
//! — straight leader lines — placed by one of two exterior solvers:
//! [`ExteriorPolicy::Raycast`] (the default — closed-form anchor along the
//! centroid→POI ray, with collision resolution) and
//! [`ExteriorPolicy::ForceDirected`] (an iterative spring-and-repulsion
//! solve that's polygon-aware: each label repels both other labels *and*
//! foreign region pieces, so labels are prevented from drifting across
//! unrelated regions). Elbow (orthogonal) leaders with their own placement
//! algorithm are a planned follow-up — see `LABEL_PLACEMENT_PLAN.md`.

use std::collections::HashMap;

use crate::geometry::primitives::Point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::plotting::clip::polygon_union_many;
use crate::plotting::inscribed::{fit_label_in_region, principal_axis};
use crate::plotting::regions::{
    RegionPiece, RegionPolygons, classify_into_pieces, signed_clearance,
};
use crate::spec::Combination;

/// Result of placing one label.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelPlacement {
    /// Centre of the label box, in the same coordinate space as the regions.
    pub anchor: Point,
    /// Where the placement landed (interior / overflow / exterior).
    pub kind: PlacementKind,
    /// Inside-region point to draw a leader line to. `None` for interior
    /// placements; `Some` for exterior. Renderers use this to draw the
    /// tether from `anchor` toward `tether`.
    pub tether: Option<Point>,
    /// Point on the label's bounding box where the leader line should
    /// terminate, so the line stops at the box edge instead of continuing
    /// through the rendered text. `None` for interior placements (no
    /// leader); `Some` for exterior. Sits on the AABB of size `(w, h)`
    /// centred on `anchor` — the AABB the caller supplied in `sizes`, so
    /// any half-gap padding the caller added is preserved as the visible
    /// gap between the leader tip and the text.
    pub leader_end: Option<Point>,
    /// Intermediate vertices of the leader polyline, in draw order, running
    /// between `tether` and `leader_end`. **Empty** for interior placements
    /// (no leader) and for straight leaders ([`LeaderStrategy::Straight`]),
    /// where the leader is the single segment `tether → leader_end`. Future
    /// edge types (e.g. elbow/orthogonal leaders) populate this with their
    /// bend joints, so a renderer always draws the leader as the polyline
    /// `tether → waypoints… → leader_end`.
    pub leader_waypoints: Vec<Point>,
}

/// Discriminator on [`LabelPlacement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementKind {
    /// Box fits inside the region's polygon — anchor at the region's POI.
    Interior,
    /// Anchor is outside the diagram, ray-cast from centroid through POI.
    ExteriorRaycast,
    /// Anchor is outside the diagram, decided by the force-directed solver
    /// — emitted by [`ExteriorPolicy::ForceDirected`].
    ExteriorForceDirected,
    /// Anchor is outside the diagram in a left/right column, reached by an
    /// orthogonal (elbow) leader — emitted by [`LeaderStrategy::Elbow`].
    ExteriorElbow,
}

/// Exterior fallback solver to use when a label doesn't fit inside its
/// region's polygon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExteriorPolicy {
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

/// Tuning for d3-pie style orthogonal (elbow) leaders — [`LeaderStrategy::Elbow`].
///
/// Both fields take a proportional default consistent with the
/// `0.5 * max(w, h)` margin convention used by [`ExteriorPolicy::Raycast`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElbowOptions {
    /// Horizontal gap between the diagram bounding box edge and the column's
    /// vertical bend rail (the shared `bend_x`). `None` selects a column-wide
    /// default of `0.5 * max(w, h)` taken over the largest label in the
    /// column — the same proportional convention as
    /// [`ExteriorPolicy::Raycast`], applied per side so every leader in a
    /// column bends on the same vertical line.
    pub margin: Option<f64>,
    /// Minimum vertical centre-to-centre spacing between stacked label boxes
    /// in a column. `None` selects `1.5 * max(h)` for each adjacent pair
    /// (one full box height plus a half-height of padding), computed from the
    /// taller of the two neighbours so the stacked boxes are collision-free
    /// by construction.
    pub min_gap: Option<f64>,
}

impl Default for ElbowOptions {
    /// Both knobs unset (`None`), selecting the proportional defaults.
    fn default() -> Self {
        Self {
            margin: None,
            min_gap: None,
        }
    }
}

/// Where the exterior-leader tether attaches to the source region.
///
/// Only consulted for exterior placements
/// ([`PlacementKind::ExteriorRaycast`] / [`PlacementKind::ExteriorForceDirected`]
/// / [`PlacementKind::ExteriorElbow`]); interior placements always carry
/// `tether: None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TetherSource {
    /// Tether is the region's pole of inaccessibility (deep inside the
    /// region). The rendered leader line runs from a point inside the
    /// polygon out to the exterior anchor. Default — safe for any
    /// rendering style, including stroke-less fills.
    #[default]
    Poi,
    /// Tether is the first intersection of the `(poi → anchor)` ray with
    /// the source region's outer polygon ring — i.e. the point where the
    /// outgoing ray exits the polygon. The rendered leader starts on the
    /// polygon boundary, matching the standard labeling convention; opt
    /// in for stroked rendering styles where the polygon edge is drawn.
    /// Falls back to the POI if no exit intersection is found (degenerate
    /// input).
    Boundary,
}

/// Leader strategy: the *edge type* drawn between a region and its exterior
/// label, coupled with the *placement algorithm* that suits that edge type.
///
/// This is the user's runtime choice. Picking the edge type picks a placement
/// algorithm appropriate for it — raycasting, for instance, is a straight-line
/// construction, so it's only offered for straight leaders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeaderStrategy {
    /// Straight leader lines (a single `tether → leader_end` segment). The
    /// label is positioned by the chosen exterior solver: [`ExteriorPolicy`]
    /// — [`ExteriorPolicy::Raycast`] (default) or
    /// [`ExteriorPolicy::ForceDirected`].
    Straight(ExteriorPolicy),
    /// d3-pie style orthogonal (elbow) leaders. Exterior labels are assigned
    /// to a left or right column, stacked vertically without overlap, and
    /// reached by an orthogonal three-segment polyline. A standalone
    /// column-based placement algorithm — it does *not* take an
    /// [`ExteriorPolicy`]. Tuned by [`ElbowOptions`].
    Elbow(ElbowOptions),
}

impl Default for LeaderStrategy {
    /// [`LeaderStrategy::Straight`] placed by [`ExteriorPolicy::Raycast`] with
    /// a proportional margin — straight leaders, raycast placement.
    fn default() -> Self {
        LeaderStrategy::Straight(ExteriorPolicy::Raycast { margin: None })
    }
}

/// Configuration bundle for [`place_labels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementStrategy {
    /// Leader strategy — the edge type and the placement algorithm that
    /// positions exterior labels. Defaults to straight leaders placed by
    /// raycasting ([`LeaderStrategy::default`]).
    pub leader: LeaderStrategy,
    /// Polylabel-style search precision, in the same units as the region
    /// polygons. Smaller values yield more accurate POIs at higher cost.
    pub precision: f64,
    /// Where the leader tether attaches on the source region for exterior
    /// placements. Defaults to [`TetherSource::Poi`].
    pub tether: TetherSource,
    /// Visible gap (in the same coordinate units as the label sizes)
    /// between the leader-line tip and the label's bounding box. Inflates
    /// the box used to compute [`LabelPlacement::leader_end`] by `leader_gap`
    /// on every side, so the leader stops `leader_gap` units short of the
    /// rendered text edge. Negative values are clamped to `0.0`. Defaults
    /// to `0.0` (leader ends exactly at the box edge).
    ///
    /// Use this when your renderer hands raw measured text bboxes to the
    /// placer and you want visible breathing room between the leader tip
    /// and the glyphs. If you instead pre-pad the sizes you pass in (per
    /// the label-vs-label gap convention), keep `leader_gap = 0.0` — the
    /// padding you already added shows up as the visible gap.
    pub leader_gap: f64,
}

impl Default for PlacementStrategy {
    /// [`LeaderStrategy::default`] (straight leaders, raycast placement with
    /// a proportional margin), `precision = 0.01`, POI tether (the rendered
    /// leader runs from the region's POI to the exterior anchor — safe for
    /// any rendering style), and `leader_gap = 0.0`.
    fn default() -> Self {
        Self {
            leader: LeaderStrategy::default(),
            precision: 0.01,
            tether: TetherSource::Poi,
            leader_gap: 0.0,
        }
    }
}

/// Place a label per region.
///
/// `regions` is typically obtained from [`crate::Layout::region_polygons`]
/// (or [`crate::plotting::decompose_regions`] directly). `sizes` is keyed by
/// the canonical [`Combination`] string form (use `""` for the
/// complement region). `container` is the jointly-fitted complement
/// container, when the spec carried a complement — pass [`None`] otherwise.
///
/// Every requested region for which a position can be computed gets a
/// [`LabelPlacement`] back. Regions absent from `sizes`, regions whose key
/// fails to parse, and regions whose POI cannot be computed (degenerate
/// input) are omitted from the result map.
///
/// # Caveats
///
/// The interior fit-check inherits the radial-conservative bound from
/// [`fit_label_in_region`]; very anisotropic regions may bounce a fitting
/// label out to the exterior fallback. A tighter directional-clearance
/// solver is a planned follow-up.
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
/// let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
/// for placement in placements.values() {
///     assert!(placement.anchor.x().is_finite());
/// }
/// ```
pub fn place_labels(
    regions: &RegionPolygons,
    sizes: &HashMap<String, (f64, f64)>,
    container: Option<&Rectangle>,
    strategy: &PlacementStrategy,
) -> HashMap<String, LabelPlacement> {
    // Elbow leaders use a standalone, column-based placement that ignores
    // `ExteriorPolicy`; dispatch to it early. The straight path (raycast /
    // force-directed) continues below.
    let exterior = match strategy.leader {
        LeaderStrategy::Straight(exterior) => exterior,
        LeaderStrategy::Elbow(opts) => {
            return place_elbow_labels(regions, sizes, container, strategy, opts);
        }
    };
    let exterior_kind = match exterior {
        ExteriorPolicy::Raycast { margin } => ExteriorPlan::Raycast { margin },
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

    // Below this POI-to-centroid distance the radial exit direction is
    // unreliable and `direction_from` falls back to the region's principal
    // axis. Scaled to the diagram (a fraction of the bbox diagonal) so it
    // fires consistently whatever the canvas units: a POI within this radius
    // of the centre is "central". The key case is a thin region straddling
    // the gap between disjoint clusters — its POI lands near the global
    // centroid, so the radial direction points straight back into the
    // neighbouring shapes; the principal axis instead sends the label out
    // along the sliver, into open space. `0.0` when the bbox is missing
    // (degenerate), which keeps the old `> 1e-9` floor in `direction_from`.
    let degenerate_dir_threshold = diagram_bbox
        .map(|r| 0.05 * r.width().hypot(r.height()))
        .unwrap_or(0.0);

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
                    leader_end: None,
                    leader_waypoints: Vec::new(),
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

        let mut direction = direction_from(&poi, &centroid, pieces, degenerate_dir_threshold);
        let margin = raycast_margin_opt.unwrap_or_else(|| 0.5 * w.max(h));
        let mut anchor = raycast_anchor_union(&poi, w, h, &union_pieces, margin, direction)
            .unwrap_or_else(|| raycast_anchor(&poi, w, h, &bbox, margin, direction));

        // Backstop: if the placement still sits on a region this label doesn't
        // belong to, the exit ray pointed into occupied space — the typical
        // case being a thin region on the gap-facing edge of its cluster whose
        // POI is too far from the global centroid for the degeneracy fallback
        // above to catch (so the radial ray points straight into a neighbour).
        // Re-cast along the region's principal axis, which runs out along the
        // sliver into the open space at its tips; try both senses (+axis then
        // −axis) and keep the first that actually clears. If neither clears we
        // leave the original placement untouched, so this can only improve.
        if box_overlaps_foreign(&anchor, w, h, regions, &combo) {
            let pa = principal_axis_direction(pieces);
            for cand in [pa, (-pa.0, -pa.1)] {
                let alt = raycast_anchor_union(&poi, w, h, &union_pieces, margin, cand)
                    .unwrap_or_else(|| raycast_anchor(&poi, w, h, &bbox, margin, cand));
                if !box_overlaps_foreign(&alt, w, h, regions, &combo) {
                    direction = cand;
                    anchor = alt;
                    break;
                }
            }
        }

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

    // Interior placements are finalised in `out` before any exterior
    // entries; cache their AABBs once so both exterior solvers can keep
    // leader lines (tether → anchor) from crossing them. Labels whose
    // size was missing from `sizes` or whose dimensions are non-finite /
    // non-positive contribute no AABB.
    let interior_aabbs: Vec<InteriorAabb> = out
        .iter()
        .filter(|(_, p)| p.kind == PlacementKind::Interior)
        .filter_map(|(k, p)| {
            let (w, h) = *sizes.get(k)?;
            if !(w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0) {
                return None;
            }
            Some(InteriorAabb {
                xmin: p.anchor.x() - 0.5 * w,
                ymin: p.anchor.y() - 0.5 * h,
                xmax: p.anchor.x() + 0.5 * w,
                ymax: p.anchor.y() + 0.5 * h,
            })
        })
        .collect();

    // Resolve overlaps between exterior labels. Different solvers per
    // strategy: Raycast uses a cheap tangential-push collision sweep that's
    // ideal when labels share an exterior side; ForceDirected adds spring
    // and polygon-aware repulsion so labels avoid both other labels and
    // foreign region pieces.
    let exterior_kind_label = match exterior_kind {
        ExteriorPlan::Raycast { .. } => {
            resolve_exterior_collisions(&mut exteriors, &interior_aabbs, 50);
            PlacementKind::ExteriorRaycast
        }
        ExteriorPlan::ForceDirected { iterations, .. } => {
            if let Some(bbox) = diagram_bbox {
                resolve_force_directed(
                    &mut exteriors,
                    regions,
                    &union_pieces,
                    &bbox,
                    &interior_aabbs,
                    iterations,
                );
            }
            PlacementKind::ExteriorForceDirected
        }
    };

    for entry in exteriors {
        // `direction = (anchor - poi)` is what the renderer actually draws,
        // and the resolver may have moved `anchor` from its raycast warm-start,
        // so recompute the ray direction here instead of reusing
        // `entry.direction`. This keeps the boundary tether geometrically
        // consistent with the rendered leader.
        let pieces = regions
            .get(&entry.combo)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let tether_pt = tether_point(
            strategy.tether,
            &entry.poi,
            (
                entry.anchor.x() - entry.poi.x(),
                entry.anchor.y() - entry.poi.y(),
            ),
            pieces,
        );
        let leader_end = leader_end_on_label_box(
            &tether_pt,
            &entry.anchor,
            entry.w,
            entry.h,
            strategy.leader_gap,
        );
        out.insert(
            entry.key,
            LabelPlacement {
                anchor: entry.anchor,
                kind: exterior_kind_label,
                tether: Some(tether_pt),
                leader_end: Some(leader_end),
                // Straight leaders are a single `tether → leader_end`
                // segment, so no intermediate waypoints. Elbow leaders will
                // populate these with their bend joints.
                leader_waypoints: Vec::new(),
            },
        );
    }

    out
}

/// Resolve the tether attachment for an exterior label given an explicit
/// exit-ray direction.
///
/// [`TetherSource::Poi`] returns the POI unchanged. [`TetherSource::Boundary`]
/// casts the (normalised) `dir` from the POI and returns the first outer-ring
/// exit — the point where the ray leaves the region — falling back to the POI
/// on a degenerate ray or when the ray hits nothing.
///
/// The ray direction is the one knob that differs between leader strategies:
/// the straight path passes the live `(anchor − poi)` direction; the elbow
/// path passes the horizontal `(±1, 0)` toward its column side, so each
/// leader's boundary tether lines up with the segment the renderer draws.
fn tether_point(
    source: TetherSource,
    poi: &Point,
    dir: (f64, f64),
    pieces: &[RegionPiece],
) -> Point {
    match source {
        TetherSource::Poi => *poi,
        TetherSource::Boundary => {
            let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
            if len < 1e-12 {
                return *poi;
            }
            let unit = (dir.0 / len, dir.1 / len);
            ray_first_edge_exit(poi, unit, pieces).unwrap_or(*poi)
        }
    }
}

/// Place labels with d3-pie style orthogonal (elbow) leaders.
///
/// Interior-fitting labels anchor at their region POI, exactly as the straight
/// path does. Labels that don't fit are assigned to a left or right column by
/// the sign of `(poi.x − diagram_centroid.x)`, stacked vertically without
/// overlap, and reached by an orthogonal two-leg polyline
/// `tether → (tether.x, row_y) → leader_end`.
///
/// The leader is routed **vertical-first**: it drops/rises from the source to
/// the label's row height before running horizontally out to the label, which
/// sits in a column with near-edges aligned at `bend_x`. Doing the vertical leg
/// first keeps the horizontal run at `row_y` clear of the diagram interior, so
/// it doesn't sweep across content the way a horizontal-first leg at the tether
/// height would for centrally-placed regions.
///
/// Row heights are chosen so the horizontal legs clear the interior labels: the
/// interior labels' combined y-extent forms a **keep-out band**, and exterior
/// rows stack outward from its edges (sources below the band's centre below it,
/// sources above it above), each side spaced by the minimum gap. With no
/// interior labels the rows instead spread by a two-pass min-gap sweep centred
/// on the source cluster.
fn place_elbow_labels(
    regions: &RegionPolygons,
    sizes: &HashMap<String, (f64, f64)>,
    container: Option<&Rectangle>,
    strategy: &PlacementStrategy,
    opts: ElbowOptions,
) -> HashMap<String, LabelPlacement> {
    let diagram_bbox = match container {
        Some(rect) => Some(*rect),
        None => union_bbox(regions),
    };
    let centroid = diagram_bbox.map(|r| *r.center());
    let pois = regions.label_points(strategy.precision);

    let mut out: HashMap<String, LabelPlacement> = HashMap::with_capacity(sizes.len());

    // Exterior candidate, before the vertical-distribution sweep. Its
    // unconstrained slot is `poi.y`.
    struct ElbowCand {
        key: String,
        combo: Combination,
        poi: Point,
        w: f64,
        h: f64,
    }
    let mut left: Vec<ElbowCand> = Vec::new();
    let mut right: Vec<ElbowCand> = Vec::new();

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

        // Interior first — identical to the straight path.
        if let Some(anchor) = fit_label_in_region(pieces, w, h, strategy.precision) {
            out.insert(
                key.clone(),
                LabelPlacement {
                    anchor,
                    kind: PlacementKind::Interior,
                    tether: None,
                    leader_end: None,
                    leader_waypoints: Vec::new(),
                },
            );
            continue;
        }

        // Exterior. Needs a centroid (for column assignment) and a POI;
        // skip the region if either is missing (degenerate input), mirroring
        // the straight path's skip behaviour.
        let Some(centroid) = centroid else { continue };
        let Some(poi) = pois.get(&combo).copied() else {
            continue;
        };

        let cand = ElbowCand {
            key: key.clone(),
            combo,
            poi,
            w,
            h,
        };
        // Centre tiebreak (poi.x == centroid.x, typical for centred Venn
        // regions) goes RIGHT — deterministic, matching the positive-axis
        // convention used by `direction_from`.
        if poi.x() >= centroid.x() {
            right.push(cand);
        } else {
            left.push(cand);
        }
    }

    // No diagram extent → no exterior placement possible; return whatever
    // interior placements we found.
    let Some(bbox) = diagram_bbox else {
        return out;
    };
    let (xmin, xmax, _, _) = bbox.bounds();

    // Interior labels are obstacles: an exterior leader's horizontal leg,
    // routed at its row height, must not cross them. Treat their combined
    // y-extent as a keep-out band and stack exterior rows outside it.
    let interior_band: Option<(f64, f64)> = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (k, p) in out.iter() {
            if p.kind != PlacementKind::Interior {
                continue;
            }
            let Some(&(_, ih)) = sizes.get(k) else {
                continue;
            };
            if !(ih.is_finite() && ih > 0.0) {
                continue;
            }
            lo = lo.min(p.anchor.y() - 0.5 * ih);
            hi = hi.max(p.anchor.y() + 0.5 * ih);
        }
        (lo.is_finite() && hi.is_finite() && hi >= lo).then_some((lo, hi))
    };

    for (side_right, mut col) in [(true, right), (false, left)] {
        if col.is_empty() {
            continue;
        }
        // Stable sort by (ideal_y, key) so the result is independent of the
        // `sizes` HashMap iteration order.
        col.sort_by(|a, b| {
            a.poi
                .y()
                .partial_cmp(&b.poi.y())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.key.cmp(&b.key))
        });

        // Column-wide bend rail just outside the diagram on this side.
        let col_margin = opts
            .margin
            .unwrap_or_else(|| 0.5 * col.iter().map(|c| c.w.max(c.h)).fold(0.0_f64, f64::max));
        let bend_x = if side_right {
            xmax + col_margin
        } else {
            xmin - col_margin
        };
        let sign = if side_right { 1.0 } else { -1.0 };

        // Vertical distribution → a row height per candidate (in `col` order).
        let n = col.len();
        let rows: Vec<f64> = if let Some((band_lo, band_hi)) = interior_band {
            // Keep-out band: stack each candidate's row outside the interior
            // labels' combined y-extent so its leader's horizontal leg never
            // crosses an interior label. The column splits into a lower group
            // (stacked below the band) and an upper group (above it); within
            // each side the candidate nearest the band sits closest to it and
            // the rest stack outward with the minimum gap.
            let band_centre = 0.5 * (band_lo + band_hi);
            // Balance the column across the band: the lower-source half stacks
            // below it, the upper half above. Because `col` is sorted by poi.y
            // the split is also position-respecting; a lone label takes the
            // side its own source favours.
            let split = if n == 1 {
                usize::from(col[0].poi.y() < band_centre)
            } else {
                n / 2
            };
            let mut rows = vec![0.0_f64; n];
            // Below the band: indices [0, split), the one nearest the band
            // (highest poi.y) first, stacking downward.
            let mut cursor = 0.0;
            for k in 0..split {
                let i = split - 1 - k;
                cursor = if k == 0 {
                    band_lo - 0.5 * col[i].h
                } else {
                    cursor - opts.min_gap.unwrap_or(1.5 * col[i].h.max(col[i + 1].h))
                };
                rows[i] = cursor;
            }
            // Above the band: indices [split, n), the one nearest the band
            // (lowest poi.y) first, stacking upward.
            for k in 0..(n - split) {
                let i = split + k;
                cursor = if k == 0 {
                    band_hi + 0.5 * col[i].h
                } else {
                    cursor + opts.min_gap.unwrap_or(1.5 * col[i].h.max(col[i - 1].h))
                };
                rows[i] = cursor;
            }
            rows
        } else {
            // No interior labels to avoid: spread by a two-pass sweep enforcing
            // the minimum gap, then centre the stack on the cluster's ideal-y
            // midpoint (removes downward bias). `col` is sorted, so the ends
            // carry the min/max ideal and laid-out y.
            let mut y: Vec<f64> = col.iter().map(|c| c.poi.y()).collect();
            for i in 1..n {
                let g = opts.min_gap.unwrap_or(1.5 * col[i - 1].h.max(col[i].h));
                if y[i] < y[i - 1] + g {
                    y[i] = y[i - 1] + g;
                }
            }
            for i in (0..n - 1).rev() {
                let g = opts.min_gap.unwrap_or(1.5 * col[i].h.max(col[i + 1].h));
                if y[i] > y[i + 1] - g {
                    y[i] = y[i + 1] - g;
                }
            }
            let shift = 0.5 * (col[0].poi.y() + col[n - 1].poi.y()) - 0.5 * (y[0] + y[n - 1]);
            for v in &mut y {
                *v += shift;
            }
            y
        };
        let gap = strategy.leader_gap.max(0.0);

        for (i, c) in col.into_iter().enumerate() {
            let row_y = rows[i];
            // Label box: near (inflated) edge sits at `bend_x`, so the leader
            // terminates `gap` short of the text on the column's rail line.
            let anchor = Point::new(bend_x + sign * (gap + 0.5 * c.w), row_y);
            let pieces = regions.get(&c.combo).map(|v| v.as_slice()).unwrap_or(&[]);
            // Vertical-first routing: the leader drops/rises from the source
            // to its (already-spread) row height *before* travelling sideways,
            // then runs horizontally out to the label. Routing the horizontal
            // leg at `row_y` — pushed clear of the central cluster by the
            // vertical distribution — keeps it from sweeping across interior
            // labels and regions, which a horizontal-first leg at the tether
            // height would do for centrally-placed regions. The tether
            // therefore exits *vertically* toward the row, matching the first
            // (vertical) leg.
            let dy = row_y - c.poi.y();
            let tether = tether_point(strategy.tether, &c.poi, (0.0, dy), pieces);
            let knee = Point::new(tether.x(), row_y);
            let leader_end = leader_end_on_label_box(&knee, &anchor, c.w, c.h, gap);
            out.insert(
                c.key,
                LabelPlacement {
                    anchor,
                    kind: PlacementKind::ExteriorElbow,
                    tether: Some(tether),
                    leader_end: Some(leader_end),
                    // One bend joint: the corner where the vertical leg from
                    // the tether meets the horizontal run to the label. The
                    // renderer draws `tether → knee → leader_end`.
                    leader_waypoints: vec![knee],
                },
            );
        }
    }

    out
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
///     leader_end: None,
///     leader_waypoints: Vec::new(),
/// });
/// placements.insert("B".to_string(), LabelPlacement {
///     anchor: Point::new(10.0, 5.0),
///     kind: PlacementKind::ExteriorRaycast,
///     tether: Some(Point::new(8.0, 4.0)),
///     leader_end: Some(Point::new(8.0, 5.0)),
///     leader_waypoints: Vec::new(),
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
/// The placements from the final iteration.
pub fn place_labels_to_fixed_point<F>(
    regions: &RegionPolygons,
    container: Option<&Rectangle>,
    initial_sizes: HashMap<String, (f64, f64)>,
    strategy: &PlacementStrategy,
    mut measure: F,
    bbox_tolerance: f64,
    max_iters: usize,
) -> HashMap<String, LabelPlacement>
where
    F: FnMut(&Rectangle) -> HashMap<String, (f64, f64)>,
{
    let mut sizes = initial_sizes;
    let mut placements = place_labels(regions, &sizes, container, strategy);
    let mut prev_short =
        canvas_bbox(regions, container, &placements, &sizes).map(|r| r.width().min(r.height()));

    for _ in 0..max_iters {
        let Some(prev_bbox) = canvas_bbox(regions, container, &placements, &sizes) else {
            return placements;
        };
        let new_sizes = measure(&prev_bbox);
        let new_placements = place_labels(regions, &new_sizes, container, strategy);
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
            return placements;
        }
    }
    placements
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

/// Axis-aligned bounding box of an already-placed interior label, used by
/// the exterior solvers to keep leader lines from visually crossing it.
struct InteriorAabb {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
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
/// pairwise overlap moves both labels half the smaller AABB-overlap.
///
/// Also pushes the anchor tangentially when the leader segment (POI →
/// anchor) would visually cross an interior label's AABB, so leaders
/// don't sweep across the diagram's interior text. The two passes share
/// the outer iteration: the loop terminates early when neither pass
/// moves anything, and is capped at `max_iters` to bound worst-case
/// cost.
fn resolve_exterior_collisions(
    entries: &mut [ExteriorEntry],
    interior_aabbs: &[InteriorAabb],
    max_iters: usize,
) {
    if entries.is_empty() {
        return;
    }
    let eps = 1e-9;
    for _ in 0..max_iters {
        let mut moved = false;

        // Leader-vs-interior-label avoidance: for each exterior entry,
        // check whether the POI→anchor segment intersects any interior
        // label's AABB. On a hit, push the anchor along the tangent
        // (perpendicular to the raycast direction) far enough that the
        // segment, pivoted around the fixed POI, clears the AABB.
        for entry in entries.iter_mut() {
            for aabb in interior_aabbs {
                let Some((t_enter, t_exit)) = segment_vs_aabb(
                    &entry.poi,
                    &entry.anchor,
                    aabb.xmin,
                    aabb.ymin,
                    aabb.xmax,
                    aabb.ymax,
                ) else {
                    continue;
                };
                let push = leader_avoidance_push(entry, aabb, t_enter, t_exit);
                if push.is_none() {
                    continue;
                }
                let (dx, dy) = push.unwrap();
                entry.anchor = Point::new(entry.anchor.x() + dx, entry.anchor.y() + dy);
                moved = true;
            }
        }

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
    interior_aabbs: &[InteriorAabb],
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

            // Leader-vs-interior-label avoidance: push the anchor
            // tangentially when the POI→anchor segment crosses an
            // interior label's AABB. Penetration-resolving (each pass
            // moves the anchor just enough that the segment pivots
            // around the POI past the AABB), so the spring + repulsion
            // fixed-point stays well-behaved.
            for aabb in interior_aabbs {
                let Some((t_enter, t_exit)) = segment_vs_aabb(
                    &entries[i].poi,
                    &entries[i].anchor,
                    aabb.xmin,
                    aabb.ymin,
                    aabb.xmax,
                    aabb.ymax,
                ) else {
                    continue;
                };
                if let Some((dx, dy)) = leader_avoidance_push(&entries[i], aabb, t_enter, t_exit) {
                    entries[i].anchor =
                        Point::new(entries[i].anchor.x() + dx, entries[i].anchor.y() + dy);
                }
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
/// two are closer than `degenerate_threshold`, the radial direction is
/// unreliable — either the POI coincides with the centroid (a centred Venn
/// region) or it sits near the global centroid because the region straddles
/// the gap between disjoint clusters, where "away from the centroid" points
/// straight back into the neighbouring shapes. In that case fall back to the
/// region's principal axis on its largest piece (sending the label out along
/// an elongated sliver, into open space); if the region is effectively
/// isotropic (elongation < 1.05), use a fixed `+y` convention.
///
/// `degenerate_threshold` is scaled to the diagram by [`place_labels`] so the
/// fallback fires consistently regardless of canvas units. A hard `1e-9`
/// floor guards the division below when the threshold is zero.
fn direction_from(
    poi: &Point,
    centroid: &Point,
    pieces: &[RegionPiece],
    degenerate_threshold: f64,
) -> (f64, f64) {
    let dx = poi.x() - centroid.x();
    let dy = poi.y() - centroid.y();
    let mag = (dx * dx + dy * dy).sqrt();
    if mag > degenerate_threshold.max(1e-9) {
        return (dx / mag, dy / mag);
    }
    // POI ≈ centroid: principal-axis tiebreak.
    principal_axis_direction(pieces)
}

/// Unit direction along the region's principal axis — the long axis of its
/// largest piece — used as the exit direction when the centroid ray is
/// unreliable (a centred or gap-straddling region) or when the centroid ray
/// lands the label on a foreign region (see [`place_labels`]'s backstop). For
/// an elongated sliver this points out along its length, into the open space
/// at its tips; an effectively isotropic region (elongation < 1.05) has no
/// meaningful long axis, so fall back to a fixed `+y` convention.
fn principal_axis_direction(pieces: &[RegionPiece]) -> (f64, f64) {
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

/// Does the `w × h` label box centred at `center` overlap any region this
/// label doesn't belong to? "Foreign" is every region whose combination
/// differs from `own` — the label may freely sit on its own region (an
/// exterior label exists precisely because it didn't fit inside it), but
/// landing on a *neighbour's* shape is the failure the placement backstop
/// catches. Cheap and robust enough: tests the box centre and corners for
/// containment in each piece, plus each piece outer vertex for containment in
/// the box.
fn box_overlaps_foreign(
    center: &Point,
    w: f64,
    h: f64,
    regions: &RegionPolygons,
    own: &Combination,
) -> bool {
    let (cx, cy) = (center.x(), center.y());
    let (hw, hh) = (0.5 * w, 0.5 * h);
    let probes = [
        (cx, cy),
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx - hw, cy + hh),
        (cx + hw, cy + hh),
    ];
    for (combo, pieces) in regions.iter() {
        if combo == own {
            continue;
        }
        for piece in pieces {
            if probes
                .iter()
                .any(|&(px, py)| signed_clearance(px, py, piece) > 0.0)
            {
                return true;
            }
            if piece.outer.vertices().iter().any(|v| {
                v.x() >= cx - hw && v.x() <= cx + hw && v.y() >= cy - hh && v.y() <= cy + hh
            }) {
                return true;
            }
        }
    }
    false
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
                if a < b { (a, b) } else { (b, a) }
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
                if a < b { (a, b) } else { (b, a) }
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

    if found { Some(t_max) } else { None }
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

/// First intersection of the ray `origin + t * dir` (t > 0) with any outer
/// ring of the source region, returned as a [`Point`]. Used by
/// [`TetherSource::Boundary`] to anchor the rendered leader on the polygon
/// edge instead of at the POI.
///
/// Scans every edge of every outer ring and keeps the smallest forward `t`.
/// Holes are intentionally ignored — the visual goal is to start the
/// leader where the ray leaves the *region*, and the outer ring is what
/// renders as the region's silhouette.
///
/// Returns [`None`] when the ray hits nothing (degenerate input, origin
/// outside the polygon and aimed away, or `dir` is the zero vector).
/// Callers fall back to the POI in that case.
fn ray_first_edge_exit(origin: &Point, dir: (f64, f64), pieces: &[RegionPiece]) -> Option<Point> {
    let (ox, oy) = (origin.x(), origin.y());
    let (rx, ry) = dir;
    if rx.abs() < f64::EPSILON && ry.abs() < f64::EPSILON {
        return None;
    }
    let eps = 1e-9;
    let mut best_t = f64::INFINITY;
    for piece in pieces {
        let verts = piece.outer.vertices();
        let n = verts.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let a = &verts[i];
            let b = &verts[(i + 1) % n];
            let sx = b.x() - a.x();
            let sy = b.y() - a.y();
            // r × s
            let denom = rx * sy - ry * sx;
            if denom.abs() < eps {
                continue; // parallel
            }
            let wx = a.x() - ox;
            let wy = a.y() - oy;
            // t = (w × s) / (r × s); s = (w × r) / (r × s)
            let t = (wx * sy - wy * sx) / denom;
            let u = (wx * ry - wy * rx) / denom;
            if t > eps && (-eps..=1.0 + eps).contains(&u) && t < best_t {
                best_t = t;
            }
        }
    }
    if best_t.is_finite() {
        Some(Point::new(ox + best_t * rx, oy + best_t * ry))
    } else {
        None
    }
}

/// Tangential displacement to apply to `entry.anchor` so the leader
/// segment `poi → anchor`, pivoting around the fixed POI, clears the
/// AABB.
///
/// Geometry: a perpendicular push `delta` at the anchor moves the
/// segment by `t * delta` at parameter `t ∈ [0, 1]`. To clear an AABB
/// whose midline along the segment sits at `t_mid = (t_enter + t_exit) /
/// 2` and whose half-extent projected onto the tangent direction is
/// `half_tan`, the anchor must move by at least `(half_tan + eps) /
/// t_mid` perpendicular to the segment direction. Side is chosen so the
/// push moves the anchor away from the AABB centre, projected onto the
/// tangent. Returns [`None`] when the displacement isn't well-defined
/// (degenerate `t_mid`).
fn leader_avoidance_push(
    entry: &ExteriorEntry,
    aabb: &InteriorAabb,
    t_enter: f64,
    t_exit: f64,
) -> Option<(f64, f64)> {
    let t_mid = 0.5 * (t_enter + t_exit);
    if !t_mid.is_finite() || t_mid <= 1e-6 {
        return None;
    }
    // Tangent = 90° rotation of the raycast direction. The raycast
    // direction is preserved as a stable "outward" axis even when the
    // resolver moves the anchor; using it (instead of the live
    // anchor−poi direction) keeps the push axis fixed across iterations
    // and avoids feedback when several AABBs collide on the same label.
    let tx = -entry.direction.1;
    let ty = entry.direction.0;
    let cx = 0.5 * (aabb.xmin + aabb.xmax);
    let cy = 0.5 * (aabb.ymin + aabb.ymax);
    let half_tan =
        0.5 * (aabb.xmax - aabb.xmin) * tx.abs() + 0.5 * (aabb.ymax - aabb.ymin) * ty.abs();
    let proj = (entry.anchor.x() - cx) * tx + (entry.anchor.y() - cy) * ty;
    let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
    let mag = (half_tan + 1e-6) / t_mid;
    Some((sign * mag * tx, sign * mag * ty))
}

/// Where the leader from `tether` to `anchor` first enters the label box of
/// size `(w, h)` centred on `anchor`, inflated by `gap` on every side — the
/// point a renderer should use as the leader's terminus so the line stops
/// at the (possibly padded) box edge instead of continuing through the
/// rendered text.
///
/// `gap` is the configured `leader_gap` from [`PlacementStrategy`]. It's
/// clamped to `[0, ∞)`: negative values behave like `0.0`. With
/// `gap = 0.0` the leader terminates exactly at the visible text box
/// edge; with `gap > 0.0` it stops `gap` units short of every edge.
///
/// Falls back to `anchor` when the tether sits inside the (inflated)
/// label box. In that case there is no meaningful "edge entry" and
/// stopping at the centre is the least-bad option; the renderer simply
/// draws a zero-length leader.
fn leader_end_on_label_box(tether: &Point, anchor: &Point, w: f64, h: f64, gap: f64) -> Point {
    let gap = gap.max(0.0);
    let half_w = 0.5 * w + gap;
    let half_h = 0.5 * h + gap;
    let xmin = anchor.x() - half_w;
    let xmax = anchor.x() + half_w;
    let ymin = anchor.y() - half_h;
    let ymax = anchor.y() + half_h;
    let Some((t_enter, _)) = segment_vs_aabb(tether, anchor, xmin, ymin, xmax, ymax) else {
        return *anchor;
    };
    // t_enter == 0 means the tether is already on/inside the (inflated)
    // box edge: no meaningful entry point exists, so fall back to the
    // anchor and let the renderer draw a zero-length leader.
    if t_enter <= 0.0 {
        return *anchor;
    }
    let dx = anchor.x() - tether.x();
    let dy = anchor.y() - tether.y();
    Point::new(tether.x() + t_enter * dx, tether.y() + t_enter * dy)
}

/// Segment vs axis-aligned box intersection (Liang-Barsky slab method).
///
/// Returns `Some((t_enter, t_exit))` with both clamped to `[0, 1]` when the
/// segment from `p0` to `p1` intersects the closed AABB `[xmin, xmax] ×
/// [ymin, ymax]`; [`None`] otherwise. Endpoints touching the box edges
/// count as a hit. Used by the interior-label leader-avoidance pass in
/// both exterior resolvers.
fn segment_vs_aabb(
    p0: &Point,
    p1: &Point,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
) -> Option<(f64, f64)> {
    let dx = p1.x() - p0.x();
    let dy = p1.y() - p0.y();
    let mut t_min = 0.0_f64;
    let mut t_max = 1.0_f64;

    for &(o, d, lo, hi) in &[(p0.x(), dx, xmin, xmax), (p0.y(), dy, ymin, ymax)] {
        if d.abs() < f64::EPSILON {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let t1 = (lo - o) / d;
            let t2 = (hi - o) / d;
            let (t_lo, t_hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            if t_lo > t_min {
                t_min = t_lo;
            }
            if t_hi < t_max {
                t_max = t_hi;
            }
            if t_min > t_max {
                return None;
            }
        }
    }
    Some((t_min, t_max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::{Circle, Polygon};
    use crate::plotting::regions::RegionPiece;
    use crate::plotting::{RegionPolygons, decompose_regions};
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
    fn test_default_strategy_is_raycast() {
        let s = PlacementStrategy::default();
        assert!(matches!(
            s.leader,
            LeaderStrategy::Straight(ExteriorPolicy::Raycast { margin: None })
        ));
        assert!((s.precision - 0.01).abs() < 1e-12);
        assert_eq!(s.tether, TetherSource::Poi);
    }

    #[test]
    fn test_strict_raycast_interior_fit() {
        // 10×10 region; tiny label fits comfortably → Interior, no tether.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
    fn test_interior_placement_has_no_leader_end() {
        // Interior placements never carry a leader endpoint — the renderer
        // draws no leader line, so there's nothing to clip.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::Interior);
        assert!(p.leader_end.is_none());
    }

    #[test]
    fn test_exterior_raycast_leader_end_on_box_edge() {
        // Same setup as test_strict_raycast_exterior_fallback: exterior
        // raycast places A's label above the region. leader_end should
        // land on the bottom edge of the label box (the edge facing the
        // tether) and sit on the tether→anchor segment.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let (w, h) = (20.0, 20.0);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (w, h));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("A").expect("A should be placed");
        let tether = p.tether.expect("exterior placement should have a tether");
        let le = p
            .leader_end
            .expect("exterior placement should have leader_end");

        // leader_end on the label AABB boundary.
        let half_w = 0.5 * w;
        let half_h = 0.5 * h;
        let xmin = p.anchor.x() - half_w;
        let xmax = p.anchor.x() + half_w;
        let ymin = p.anchor.y() - half_h;
        let ymax = p.anchor.y() + half_h;
        let on_left = (le.x() - xmin).abs() < 1e-6;
        let on_right = (le.x() - xmax).abs() < 1e-6;
        let on_bottom = (le.y() - ymin).abs() < 1e-6;
        let on_top = (le.y() - ymax).abs() < 1e-6;
        let in_x = le.x() >= xmin - 1e-6 && le.x() <= xmax + 1e-6;
        let in_y = le.y() >= ymin - 1e-6 && le.y() <= ymax + 1e-6;
        assert!(
            (on_left || on_right || on_bottom || on_top) && in_x && in_y,
            "leader_end {:?} should lie on the label AABB edge ({:?} × {:?})",
            (le.x(), le.y()),
            (xmin, xmax),
            (ymin, ymax),
        );

        // leader_end on the tether→anchor segment: parameterise the
        // segment and check t ∈ [0, 1] and the point matches.
        let dx = p.anchor.x() - tether.x();
        let dy = p.anchor.y() - tether.y();
        let denom = dx * dx + dy * dy;
        assert!(denom > 0.0, "anchor and tether should not coincide");
        let t = ((le.x() - tether.x()) * dx + (le.y() - tether.y()) * dy) / denom;
        assert!(
            (0.0..=1.0).contains(&t),
            "leader_end should sit on the tether→anchor segment (t = {})",
            t
        );
        let reconstructed_x = tether.x() + t * dx;
        let reconstructed_y = tether.y() + t * dy;
        assert!((reconstructed_x - le.x()).abs() < 1e-6);
        assert!((reconstructed_y - le.y()).abs() < 1e-6);
    }

    #[test]
    fn test_exterior_force_directed_emits_leader_end() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let strategy = PlacementStrategy {
            leader: LeaderStrategy::Straight(ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: Some(10),
            }),
            ..PlacementStrategy::default()
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorForceDirected);
        assert!(p.leader_end.is_some());
    }

    #[test]
    fn test_leader_end_on_label_box_basic() {
        // Tether directly below the box centre → leader_end on the bottom edge.
        let tether = Point::new(5.0, -5.0);
        let anchor = Point::new(5.0, 10.0);
        let le = leader_end_on_label_box(&tether, &anchor, 4.0, 2.0, 0.0);
        // Box is [3, 7] × [9, 11]; the leader from (5, -5) to (5, 10) enters
        // the box at y = 9.
        assert!((le.x() - 5.0).abs() < 1e-9);
        assert!((le.y() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_leader_end_with_gap_sits_on_inflated_box_edge() {
        // gap = 0.5 inflates a 4×2 box to 5×3; the perpendicular leader
        // from below enters the inflated box at y = 10 - 1.5 = 8.5
        // instead of the raw-edge y = 9.0.
        let tether = Point::new(5.0, -5.0);
        let anchor = Point::new(5.0, 10.0);
        let le = leader_end_on_label_box(&tether, &anchor, 4.0, 2.0, 0.5);
        assert!((le.x() - 5.0).abs() < 1e-9);
        assert!((le.y() - 8.5).abs() < 1e-9);
    }

    #[test]
    fn test_leader_end_negative_gap_clamped_to_zero() {
        // Negative gap should behave the same as gap = 0 — the contract
        // says "visible breathing room", and a negative gap would extend
        // the leader past the box edge into the text which is never
        // desirable.
        let tether = Point::new(5.0, -5.0);
        let anchor = Point::new(5.0, 10.0);
        let le_neg = leader_end_on_label_box(&tether, &anchor, 4.0, 2.0, -1.0);
        let le_zero = leader_end_on_label_box(&tether, &anchor, 4.0, 2.0, 0.0);
        assert!((le_neg.x() - le_zero.x()).abs() < 1e-9);
        assert!((le_neg.y() - le_zero.y()).abs() < 1e-9);
    }

    #[test]
    fn test_leader_end_falls_back_when_tether_inside_box() {
        // Pathological case: tether sits inside the label box. There is no
        // meaningful edge entry, so leader_end falls back to the anchor.
        let anchor = Point::new(5.0, 5.0);
        let tether = Point::new(5.0, 5.0);
        let le = leader_end_on_label_box(&tether, &anchor, 4.0, 2.0, 0.0);
        assert!((le.x() - anchor.x()).abs() < 1e-9);
        assert!((le.y() - anchor.y()).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_leader_gap_inflates_leader_end() {
        // End-to-end: a non-zero leader_gap on the strategy should shift
        // the exterior leader_end inward by `gap` units along the leader.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let (w, h) = (20.0, 20.0);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (w, h));
        let base_strategy = PlacementStrategy::default();
        let gap = 1.5;
        let gap_strategy = PlacementStrategy {
            leader_gap: gap,
            ..base_strategy
        };

        let base = place_labels(&regions, &sizes, None, &base_strategy)
            .remove("A")
            .expect("A placed");
        let gapped = place_labels(&regions, &sizes, None, &gap_strategy)
            .remove("A")
            .expect("A placed");

        let le0 = base.leader_end.expect("leader_end");
        let le1 = gapped.leader_end.expect("leader_end");
        // Direction of the perpendicular leader (anchor above POI) is +y,
        // so the gapped endpoint sits `gap` units below the base endpoint.
        assert!((le0.x() - le1.x()).abs() < 1e-6);
        assert!(
            (le0.y() - le1.y() - gap).abs() < 1e-6,
            "gapped leader_end should sit `gap` units before the raw edge: \
             base.y = {}, gapped.y = {}, gap = {}",
            le0.y(),
            le1.y(),
            gap,
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

        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
        );
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
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        // +y fallback: anchor's x ≈ POI's x (≈ 0), y >> 5.
        assert!(p.anchor.x().abs() < 1e-6, "anchor x = {}", p.anchor.x());
        assert!(p.anchor.y() > 5.0, "anchor y = {}", p.anchor.y());
    }

    #[test]
    fn test_gap_straddling_sliver_deflects_to_principal_axis() {
        // Regression: a thin region straddling the gap between two disjoint
        // clusters has its POI near the global centroid, so the radial exit
        // direction (`poi − centroid`) points back into a neighbouring
        // cluster's shapes. The scale-aware degeneracy fallback must deflect
        // along the sliver's principal axis (vertical here), placing the label
        // in the open space above/below the gap rather than on the neighbours.
        //
        // Left cluster (A) fills x[0,48]; a tall 1-wide B sliver sits on its
        // gap-facing edge at x[48,49]; right cluster (C) is x[52,100]. The
        // union centroid is ≈(50, 24), so B's POI (≈48.5, 24) is only ~1.5
        // units away — well inside the scaled degeneracy radius — and the raw
        // radial direction would be ≈(−1, 0), straight into A.
        let rect = |x0: f64, y0: f64, x1: f64, y1: f64| RegionPiece {
            outer: Polygon::new(vec![
                Point::new(x0, y0),
                Point::new(x1, y0),
                Point::new(x1, y1),
                Point::new(x0, y1),
            ]),
            holes: vec![],
        };
        let mut map = HashMap::new();
        map.insert(Combination::new(&["A"]), vec![rect(0.0, 0.0, 48.0, 48.0)]);
        map.insert(Combination::new(&["B"]), vec![rect(48.0, 0.0, 49.0, 48.0)]);
        map.insert(Combination::new(&["C"]), vec![rect(52.0, 0.0, 100.0, 48.0)]);
        let regions = RegionPolygons::from_map(map);

        // B's label is far too wide for the 1-unit sliver, so it goes exterior.
        let mut sizes = HashMap::new();
        sizes.insert("B".to_string(), (20.0, 10.0));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("B").expect("B placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);

        // Deflected vertically: the anchor stays near the sliver's x (it is
        // NOT cast left into A) and is pushed clear of the diagram body in y.
        assert!(
            p.anchor.x() > 40.0,
            "B should not be cast left into A: anchor.x = {}",
            p.anchor.x()
        );
        assert!(
            p.anchor.y() < 0.0 || p.anchor.y() > 48.0,
            "B should be pushed vertically clear of the diagram body [0,48]: anchor.y = {}",
            p.anchor.y()
        );
    }

    #[test]
    fn test_offcenter_edge_sliver_backstop_clears_neighbour() {
        // Regression for the *backstop* path (not the centroid-degeneracy
        // threshold): a thin sliver on the gap-facing edge of a cluster that
        // sits well away from the global centroid. The radial direction then
        // points into the adjacent cluster's shape AND the sliver's POI is too
        // far from the centroid for the degeneracy fallback to fire, so only
        // the "placement overlaps a foreign region → re-cast along the
        // principal axis" backstop can rescue it.
        //
        // Layout: L fills x[0,40]; the right cluster's big region C fills
        // x[60,100]; D is a tall 1-wide sliver on C's gap-facing (left) edge
        // at x[59,60]. The union centroid is ≈(50, 25), so D's POI (≈59.5) is
        // ~9.5 units away — past the ~5.4 (5% of diagonal) degeneracy radius —
        // and the raw radial direction is ≈(+1, 0), straight into C.
        let rect = |verts: &[(f64, f64)]| RegionPiece {
            outer: Polygon::new(verts.iter().map(|&(x, y)| Point::new(x, y)).collect()),
            holes: vec![],
        };
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["L"]),
            vec![rect(&[(0.0, 5.0), (40.0, 5.0), (40.0, 45.0), (0.0, 45.0)])],
        );
        // C's left edge carries an extra vertex at y=25 (in D's y-slab) so the
        // rightward raycast actually lands the label *on* C rather than
        // flying past it — reproducing the real polygonised-square geometry.
        map.insert(
            Combination::new(&["C"]),
            vec![rect(&[
                (60.0, 5.0),
                (100.0, 5.0),
                (100.0, 45.0),
                (60.0, 45.0),
                (60.0, 25.0),
            ])],
        );
        map.insert(
            Combination::new(&["D"]),
            vec![rect(&[
                (59.0, 20.0),
                (60.0, 20.0),
                (60.0, 30.0),
                (59.0, 30.0),
            ])],
        );
        let regions = RegionPolygons::from_map(map);

        let mut sizes = HashMap::new();
        sizes.insert("D".to_string(), (8.0, 6.0));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("D").expect("D placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);

        // Deflected along its (vertical) principal axis: pushed clear of C's
        // y-range [5,45] rather than parked on C at y≈25, and kept near the
        // sliver's x rather than driven deep into C.
        assert!(
            p.anchor.y() < 5.0 || p.anchor.y() > 45.0,
            "D should be deflected vertically clear of C's body: anchor.y = {}",
            p.anchor.y()
        );
        assert!(
            p.anchor.x() < 65.0,
            "D should not be driven right into C: anchor.x = {}",
            p.anchor.x()
        );
        // And the final box must not sit on the foreign C region.
        assert!(
            !box_overlaps_foreign(&p.anchor, 8.0, 6.0, &regions, &Combination::new(&["D"])),
            "D's box still overlaps a foreign region at ({}, {})",
            p.anchor.x(),
            p.anchor.y(),
        );
    }

    #[test]
    fn test_unknown_keys_skipped() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("Z".to_string(), (1.0, 1.0));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        assert!(placements.is_empty());
    }

    #[test]
    fn test_invalid_dimensions_skipped() {
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (0.0, 1.0));
        sizes.insert("B".to_string(), (f64::NAN, 1.0));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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

        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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

        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
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
            leader: LeaderStrategy::Straight(ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: None,
            }),
            ..PlacementStrategy::default()
        }
    }

    #[test]
    fn test_force_directed_interior_fit_unchanged() {
        // A label that fits inside its region must stay at its POI under
        // ForceDirected — the solver only runs on exteriors.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy());
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
        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy());
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

        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy());
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
            leader: LeaderStrategy::Straight(ExteriorPolicy::ForceDirected {
                margin: Some(0.5),
                iterations: Some(300),
            }),
            precision: 0.05,
            tether: TetherSource::Poi,
            leader_gap: 0.0,
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
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

        let placements = place_labels(&regions, &sizes, None, &force_directed_strategy());
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
            leader: LeaderStrategy::Straight(ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: Some(0),
            }),
            precision: 0.01,
            tether: TetherSource::Poi,
            leader_gap: 0.0,
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
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
            leader_end: None,
            leader_waypoints: Vec::new(),
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
        );

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
        );

        assert_eq!(iter_count, 5, "expected exactly max_iters measure calls");
        assert!(placements.contains_key("A"));
    }

    fn rect_piece(x: f64, y: f64, w: f64, h: f64) -> RegionPiece {
        RegionPiece {
            outer: Polygon::new(vec![
                Point::new(x, y),
                Point::new(x + w, y),
                Point::new(x + w, y + h),
                Point::new(x, y + h),
            ]),
            holes: vec![],
        }
    }

    #[test]
    fn test_segment_vs_aabb_hit_and_miss() {
        // Hit along the +x axis.
        let p0 = Point::new(0.0, 1.0);
        let p1 = Point::new(10.0, 1.0);
        let hit = segment_vs_aabb(&p0, &p1, 3.0, 0.5, 5.0, 1.5);
        let (te, tx) = hit.expect("segment should hit");
        assert!((te - 0.3).abs() < 1e-9);
        assert!((tx - 0.5).abs() < 1e-9);
        // Miss: segment far above the AABB.
        let p2 = Point::new(0.0, 10.0);
        let p3 = Point::new(10.0, 10.0);
        assert!(segment_vs_aabb(&p2, &p3, 3.0, 0.5, 5.0, 1.5).is_none());
    }

    #[test]
    fn test_ray_first_edge_exit_centred() {
        // Ray from the centre of a 10×10 square pointing +x must exit at
        // x = 10 on the right edge.
        let pieces = vec![axis_aligned_square_piece(10.0)];
        let exit = ray_first_edge_exit(&Point::new(5.0, 5.0), (1.0, 0.0), &pieces)
            .expect("ray should exit through the right edge");
        assert!((exit.x() - 10.0).abs() < 1e-9);
        assert!((exit.y() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ray_first_edge_exit_no_hit() {
        // Origin outside the polygon and pointing away — no forward
        // intersection. Caller falls back to the POI.
        let pieces = vec![axis_aligned_square_piece(10.0)];
        assert!(ray_first_edge_exit(&Point::new(-1.0, 5.0), (-1.0, 0.0), &pieces).is_none());
    }

    #[test]
    fn test_boundary_tether_lands_on_polygon_edge() {
        // 10×10 region, label too big to fit interior. With
        // `TetherSource::Boundary`, the tether must lie on the region's
        // boundary, not at the POI (5, 5).
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let strategy = PlacementStrategy {
            tether: TetherSource::Boundary,
            ..PlacementStrategy::default()
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        let tether = p.tether.expect("exterior placement should have a tether");
        // Tether is on the square's outer ring (one of x ∈ {0, 10} or
        // y ∈ {0, 10}, with the other coordinate ∈ [0, 10]).
        let on_left = (tether.x()).abs() < 1e-6;
        let on_right = (tether.x() - 10.0).abs() < 1e-6;
        let on_bottom = (tether.y()).abs() < 1e-6;
        let on_top = (tether.y() - 10.0).abs() < 1e-6;
        assert!(
            (on_left || on_right || on_bottom || on_top)
                && (0.0..=10.0).contains(&tether.x())
                && (0.0..=10.0).contains(&tether.y()),
            "tether ({}, {}) should lie on the 10×10 boundary",
            tether.x(),
            tether.y(),
        );
    }

    #[test]
    fn test_poi_tether_is_default() {
        // Default strategy keeps the tether at the region's POI so that
        // rendered leaders stay anchored well inside the visible region
        // regardless of whether the caller draws shape strokes.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("A").expect("A should be placed");
        let tether = p.tether.expect("exterior placement should have a tether");
        assert!((tether.x() - 5.0).abs() < 0.5);
        assert!((tether.y() - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_boundary_tether_falls_back_to_poi_on_degenerate_ray() {
        // Zero-length raycast (anchor coincides with POI). The boundary
        // helper short-circuits to the POI rather than panicking on
        // `1 / 0` in the direction normalisation.
        let entry = ExteriorEntry {
            key: "A".to_string(),
            combo: Combination::new(&["A"]),
            anchor: Point::new(5.0, 5.0),
            home: Point::new(5.0, 5.0),
            poi: Point::new(5.0, 5.0),
            direction: (1.0, 0.0),
            margin: 0.0,
            w: 1.0,
            h: 1.0,
        };
        // Sanity: with a zero-length displacement the boundary tether
        // code path falls back to the POI (covered via
        // `place_labels` in the test below).
        let pieces = vec![axis_aligned_square_piece(10.0)];
        // No forward intersection from the centre pointing into a zero
        // direction → None → caller falls back to POI.
        assert!(ray_first_edge_exit(&entry.poi, (0.0, 0.0), &pieces).is_none());
    }

    #[test]
    fn test_leader_avoidance_raycast_pushes_anchor_clear_of_interior_label() {
        // Three-region layout where, by construction, region A's natural
        // raycast direction sweeps the leader segment over region C's
        // interior label AABB. With the always-on avoidance pass, the
        // resolved anchor must move so the segment misses C's AABB.
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["A"]),
            vec![rect_piece(4.0, 0.0, 2.0, 2.0)],
        );
        map.insert(
            Combination::new(&["B"]),
            vec![rect_piece(0.0, 0.0, 3.0, 2.0)],
        );
        map.insert(
            Combination::new(&["C"]),
            vec![rect_piece(7.0, 0.0, 2.0, 2.0)],
        );
        let regions = RegionPolygons::from_map(map);

        let mut sizes = HashMap::new();
        // A's label is huge — forces exterior, the natural raycast
        // direction is (+1, 0) (poi_A ≈ (5,1), centroid ≈ (4.5, 1)).
        sizes.insert("A".to_string(), (20.0, 10.0));
        // B's label fits comfortably interior.
        sizes.insert("B".to_string(), (1.0, 0.5));
        // C's label fits interior and sits squarely on A's ray. Keep
        // it small enough that the radial-conservative inscribed-rect
        // bound in `fit_label_in_region` accepts it.
        sizes.insert("C".to_string(), (1.0, 0.5));

        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let a = placements.get("A").expect("A should be placed");
        let c = placements.get("C").expect("C should be placed");
        assert_eq!(a.kind, PlacementKind::ExteriorRaycast);
        assert_eq!(c.kind, PlacementKind::Interior);

        let tether = a.tether.expect("A should have a tether");
        let c_xmin = c.anchor.x() - 0.5 * 1.0;
        let c_xmax = c.anchor.x() + 0.5 * 1.0;
        let c_ymin = c.anchor.y() - 0.5 * 0.5;
        let c_ymax = c.anchor.y() + 0.5 * 0.5;
        // The rendered leader runs from `tether` to `anchor`. The
        // avoidance pass must have pushed `anchor` enough that the
        // segment no longer crosses C's AABB.
        assert!(
            segment_vs_aabb(&tether, &a.anchor, c_xmin, c_ymin, c_xmax, c_ymax).is_none(),
            "leader segment ({}, {}) → ({}, {}) still crosses C's AABB ({}, {}, {}, {})",
            tether.x(),
            tether.y(),
            a.anchor.x(),
            a.anchor.y(),
            c_xmin,
            c_ymin,
            c_xmax,
            c_ymax,
        );
    }

    #[test]
    fn test_leader_avoidance_force_directed_pushes_anchor_clear_of_interior_label() {
        // Same setup as the Raycast avoidance test, with the
        // ForceDirected exterior solver instead.
        let mut map = HashMap::new();
        map.insert(
            Combination::new(&["A"]),
            vec![rect_piece(4.0, 0.0, 2.0, 2.0)],
        );
        map.insert(
            Combination::new(&["B"]),
            vec![rect_piece(0.0, 0.0, 3.0, 2.0)],
        );
        map.insert(
            Combination::new(&["C"]),
            vec![rect_piece(7.0, 0.0, 2.0, 2.0)],
        );
        let regions = RegionPolygons::from_map(map);

        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 10.0));
        sizes.insert("B".to_string(), (1.0, 0.5));
        sizes.insert("C".to_string(), (1.0, 0.5));

        let strategy = PlacementStrategy {
            leader: LeaderStrategy::Straight(ExteriorPolicy::ForceDirected {
                margin: None,
                iterations: Some(300),
            }),
            precision: 0.01,
            tether: TetherSource::Poi,
            leader_gap: 0.0,
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
        let a = placements.get("A").expect("A should be placed");
        let c = placements.get("C").expect("C should be placed");
        assert_eq!(a.kind, PlacementKind::ExteriorForceDirected);
        assert_eq!(c.kind, PlacementKind::Interior);

        let tether = a.tether.expect("A should have a tether");
        let c_xmin = c.anchor.x() - 0.5 * 1.0;
        let c_xmax = c.anchor.x() + 0.5 * 1.0;
        let c_ymin = c.anchor.y() - 0.5 * 0.5;
        let c_ymax = c.anchor.y() + 0.5 * 0.5;
        assert!(
            segment_vs_aabb(&tether, &a.anchor, c_xmin, c_ymin, c_xmax, c_ymax).is_none(),
            "leader segment ({}, {}) → ({}, {}) still crosses C's AABB ({}, {}, {}, {})",
            tether.x(),
            tether.y(),
            a.anchor.x(),
            a.anchor.y(),
            c_xmin,
            c_ymin,
            c_xmax,
            c_ymax,
        );
    }

    #[test]
    fn test_straight_leader_has_no_waypoints() {
        // Straight leaders are a single tether → leader_end segment, so an
        // exterior placement carries no intermediate waypoints. (Interior
        // placements never have a leader; see the test above.)
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0)); // too big to fit interior
        let placements = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
        let p = placements.get("A").expect("A should be placed");
        assert_eq!(p.kind, PlacementKind::ExteriorRaycast);
        assert!(p.tether.is_some());
        assert!(p.leader_end.is_some());
        assert!(
            p.leader_waypoints.is_empty(),
            "straight leaders carry no intermediate waypoints"
        );
    }

    // ---- Elbow leaders ----

    fn elbow_strategy() -> PlacementStrategy {
        PlacementStrategy {
            leader: LeaderStrategy::Elbow(ElbowOptions::default()),
            ..PlacementStrategy::default()
        }
    }

    fn square_piece_at(x0: f64, y0: f64, side: f64) -> RegionPiece {
        RegionPiece {
            outer: Polygon::new(vec![
                Point::new(x0, y0),
                Point::new(x0 + side, y0),
                Point::new(x0 + side, y0 + side),
                Point::new(x0, y0 + side),
            ]),
            holes: vec![],
        }
    }

    fn regions_from(entries: Vec<(&[&str], Vec<RegionPiece>)>) -> RegionPolygons {
        let mut map = HashMap::new();
        for (combo, pieces) in entries {
            map.insert(Combination::new(combo), pieces);
        }
        RegionPolygons::from_map(map)
    }

    #[test]
    fn test_elbow_default_options() {
        let opts = ElbowOptions::default();
        assert!(opts.margin.is_none());
        assert!(opts.min_gap.is_none());
        assert!(matches!(elbow_strategy().leader, LeaderStrategy::Elbow(_)));
    }

    #[test]
    fn test_elbow_column_assignment_left_right() {
        // A at x∈[0,10], B at x∈[20,30]; union bbox x∈[0,30], centroid x = 15.
        // A's POI (≈5) < 15 → left column; B's POI (≈25) > 15 → right column.
        let regions = regions_from(vec![
            (&["A"][..], vec![square_piece_at(0.0, 0.0, 10.0)]),
            (&["B"][..], vec![square_piece_at(20.0, 0.0, 10.0)]),
        ]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        sizes.insert("B".to_string(), (20.0, 20.0));
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let a = placements.get("A").expect("A placed");
        let b = placements.get("B").expect("B placed");
        assert_eq!(a.kind, PlacementKind::ExteriorElbow);
        assert_eq!(b.kind, PlacementKind::ExteriorElbow);
        assert!(
            a.anchor.x() < 0.0,
            "A should sit left of bbox: {}",
            a.anchor.x()
        );
        assert!(
            b.anchor.x() > 30.0,
            "B should sit right of bbox: {}",
            b.anchor.x()
        );
    }

    #[test]
    fn test_elbow_centre_tiebreak_goes_right() {
        // Single centred square: POI.x == centroid.x → right column.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let a = placements.get("A").expect("A placed");
        assert_eq!(a.kind, PlacementKind::ExteriorElbow);
        assert!(
            a.anchor.x() > 10.0,
            "centre tiebreak should go right: {}",
            a.anchor.x()
        );
    }

    #[test]
    fn test_elbow_vertical_distribution_no_overlap() {
        // Three identical regions → identical POIs (5, 5) → maximum vertical
        // spreading. Required gap = 1.5 * 20 = 30, so adjacent anchor centres
        // must be ≥ 30 apart (the h=20 boxes are then strictly disjoint).
        let regions = regions_from(vec![
            (&["A"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["B"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["C"][..], vec![axis_aligned_square_piece(10.0)]),
        ]);
        let mut sizes = HashMap::new();
        for k in ["A", "B", "C"] {
            sizes.insert(k.to_string(), (20.0, 20.0));
        }
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let mut ys: Vec<f64> = ["A", "B", "C"]
            .iter()
            .map(|k| {
                let p = placements.get(*k).expect("placed");
                assert_eq!(p.kind, PlacementKind::ExteriorElbow);
                p.anchor.y()
            })
            .collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(ys[1] - ys[0] >= 30.0 - 1e-6, "gap 0-1 = {}", ys[1] - ys[0]);
        assert!(ys[2] - ys[1] >= 30.0 - 1e-6, "gap 1-2 = {}", ys[2] - ys[1]);
    }

    #[test]
    fn test_elbow_distribution_centred_on_cluster() {
        // Same three-identical-region setup: ideal y = 5 for all, so the
        // centred stack must straddle y = 5 (mean anchor y ≈ 5), not drift
        // downward.
        let regions = regions_from(vec![
            (&["A"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["B"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["C"][..], vec![axis_aligned_square_piece(10.0)]),
        ]);
        let mut sizes = HashMap::new();
        for k in ["A", "B", "C"] {
            sizes.insert(k.to_string(), (20.0, 20.0));
        }
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let mean = ["A", "B", "C"]
            .iter()
            .map(|k| placements.get(*k).unwrap().anchor.y())
            .sum::<f64>()
            / 3.0;
        assert!(
            (mean - 5.0).abs() < 1e-6,
            "stack not centred: mean y = {mean}"
        );
    }

    #[test]
    fn test_elbow_keep_out_band_clears_interior_labels() {
        // A big region carries a small interior label near the centre; a
        // central region with an oversized label routes exterior. The keep-out
        // band must push the exterior row clear of the interior label's y-span,
        // so the leader's horizontal leg doesn't cross it.
        let regions = regions_from(vec![
            (&["A"][..], vec![square_piece_at(0.0, 0.0, 30.0)]),
            (&["B"][..], vec![square_piece_at(13.0, 13.0, 4.0)]),
        ]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (2.0, 1.0)); // fits inside A → interior
        sizes.insert("B".to_string(), (20.0, 20.0)); // too big → exterior elbow
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let a = placements.get("A").expect("A placed");
        let b = placements.get("B").expect("B placed");
        assert_eq!(a.kind, PlacementKind::Interior);
        assert_eq!(b.kind, PlacementKind::ExteriorElbow);
        // A's interior label spans [anchor.y − 0.5, anchor.y + 0.5] (h = 1).
        let (band_lo, band_hi) = (a.anchor.y() - 0.5, a.anchor.y() + 0.5);
        // B's row (its horizontal leg height) must sit outside that band.
        assert!(
            b.anchor.y() <= band_lo + 1e-9 || b.anchor.y() >= band_hi - 1e-9,
            "B row {} not clear of interior band [{}, {}]",
            b.anchor.y(),
            band_lo,
            band_hi
        );
    }

    #[test]
    fn test_elbow_waypoints_form_orthogonal_polyline() {
        // Three identical regions so at least one label's row_y differs from
        // its POI y (a genuine vertical first leg). Region A is sorted first
        // and pushed to the top of the stack, so A.row_y ≠ A.poi.y.
        let regions = regions_from(vec![
            (&["A"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["B"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["C"][..], vec![axis_aligned_square_piece(10.0)]),
        ]);
        let mut sizes = HashMap::new();
        for k in ["A", "B", "C"] {
            sizes.insert(k.to_string(), (20.0, 20.0));
        }
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let p = placements.get("A").expect("A placed");
        // Vertical-first leader has a single bend joint (the knee).
        assert_eq!(p.leader_waypoints.len(), 1);
        let tether = p.tether.expect("exterior has tether");
        let knee = p.leader_waypoints[0];
        let end = p.leader_end.expect("exterior has leader_end");
        // First leg vertical: tether → knee shares x.
        assert!(
            (knee.x() - tether.x()).abs() < 1e-9,
            "first leg not vertical"
        );
        // Knee sits at the row height (= anchor.y).
        assert!(
            (knee.y() - p.anchor.y()).abs() < 1e-9,
            "knee not at row height"
        );
        // Second leg horizontal: knee → leader_end shares y.
        assert!(
            (knee.y() - end.y()).abs() < 1e-9,
            "second leg not horizontal"
        );
        // Genuinely non-degenerate for the spread label.
        assert!(
            (knee.y() - tether.y()).abs() > 1e-6,
            "expected a real vertical leg"
        );
    }

    #[test]
    fn test_elbow_leader_end_on_box_edge() {
        // Single centred region → right column. leader_end lands on the near
        // (left) vertical edge of the box at the row height; with gap = 0 that
        // edge is anchor.x − w/2.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let (w, h) = (20.0, 20.0);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (w, h));
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let p = placements.get("A").expect("A placed");
        let le = p.leader_end.expect("exterior has leader_end");
        assert!(
            (le.x() - (p.anchor.x() - 0.5 * w)).abs() < 1e-6,
            "leader_end x = {} not on near edge",
            le.x()
        );
        assert!(
            (le.y() - p.anchor.y()).abs() < 1e-6,
            "leader_end y = {}",
            le.y()
        );
    }

    #[test]
    fn test_elbow_boundary_tether_vertical_exit() {
        // Vertical-first routing exits the region on a *vertical* ray toward
        // the label's row. Three identical squares stack into one column; the
        // top label (A, pushed to the lowest row by the centred sweep) exits
        // the bottom edge (y = 0) at the POI x (≈ 5); the bottom label (C,
        // pushed up) exits the top edge (y = 10).
        let regions = regions_from(vec![
            (&["A"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["B"][..], vec![axis_aligned_square_piece(10.0)]),
            (&["C"][..], vec![axis_aligned_square_piece(10.0)]),
        ]);
        let mut sizes = HashMap::new();
        for k in ["A", "B", "C"] {
            sizes.insert(k.to_string(), (20.0, 20.0));
        }
        let strategy = PlacementStrategy {
            leader: LeaderStrategy::Elbow(ElbowOptions::default()),
            tether: TetherSource::Boundary,
            ..PlacementStrategy::default()
        };
        let placements = place_labels(&regions, &sizes, None, &strategy);
        let a = placements.get("A").expect("A placed");
        let c = placements.get("C").expect("C placed");
        let ta = a.tether.expect("A has tether");
        let tc = c.tether.expect("C has tether");
        // Vertical exit ray keeps x at the POI (≈ 5).
        assert!((ta.x() - 5.0).abs() < 0.5, "A tether x = {}", ta.x());
        assert!((tc.x() - 5.0).abs() < 0.5, "C tether x = {}", tc.x());
        // A's row sits below its POI → exits the bottom edge (y = 0); C's row
        // sits above → exits the top edge (y = 10).
        assert!(a.anchor.y() < c.anchor.y(), "A row should be below C row");
        assert!(
            (ta.y() - 0.0).abs() < 1e-6,
            "A tether y = {} not bottom edge",
            ta.y()
        );
        assert!(
            (tc.y() - 10.0).abs() < 1e-6,
            "C tether y = {} not top edge",
            tc.y()
        );
    }

    #[test]
    fn test_elbow_interior_fit_unchanged() {
        // Small label still fits inside → Interior, no leader of any kind.
        let regions = single_region(&["A"], vec![axis_aligned_square_piece(10.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (1.0, 0.5));
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        let p = placements.get("A").expect("A placed");
        assert_eq!(p.kind, PlacementKind::Interior);
        assert!(p.tether.is_none());
        assert!(p.leader_end.is_none());
        assert!(p.leader_waypoints.is_empty());
    }

    #[test]
    fn test_elbow_two_circle_decomposition_all_regions_placed() {
        // End-to-end through the real decomposer with the elbow strategy.
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
        sizes.insert("A&B".to_string(), (100.0, 100.0)); // forces exterior

        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        assert!(placements.contains_key("A"));
        let ab = placements.get("A&B").expect("A&B placed");
        assert_eq!(ab.kind, PlacementKind::ExteriorElbow);
        assert!(ab.tether.is_some());
        assert_eq!(ab.leader_waypoints.len(), 1);
    }

    #[test]
    fn test_elbow_degenerate_region_skipped() {
        // Region with no pieces → no POI / no bbox → omitted from the result,
        // mirroring the straight path's skip behaviour.
        let regions = regions_from(vec![(&["A"][..], vec![])]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (20.0, 20.0));
        let placements = place_labels(&regions, &sizes, None, &elbow_strategy());
        assert!(!placements.contains_key("A"));
    }
}
