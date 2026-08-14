//! Exterior **set**-label placement — labels adjacent to their shape, with
//! no leader line.
//!
//! [`place_set_labels`] is a sibling of
//! [`place_labels`](crate::plotting::place_labels), and the difference is
//! what each one labels. `place_labels` is keyed by *region*, so a label
//! that doesn't fit goes to the diagram exterior with a leader line back to
//! the region it names. This one is keyed by *set*, and the label never
//! leaves its shape's side: it sits just outside the set's own outline, at
//! the angle around that outline with the most free space, close enough
//! that no leader is needed to say what it names.
//!
//! The algorithm is a rotation:
//!
//! 1. **Hug** — for a candidate angle `θ`, the anchor comes from
//!    [`raycast_anchor_union`] against the set's *own* outline, so the label
//!    box clears the boundary by `margin` and no more. Because that helper
//!    tests the full box footprint (width *and* height) rather than a point,
//!    a wide label on a diagonal can't dip back into a curving boundary.
//! 2. **Rotate** — sweep `θ` over the whole circle and keep the angle whose
//!    label box has the largest clearance to everything else: the other
//!    sets' outlines, the set labels already placed, and any caller-supplied
//!    keep-out boxes. This is the "as far from other shapes as possible"
//!    objective, and it is why a set buried in a cluster puts its name on
//!    the one side that happens to be open.
//! 3. **Guard** — if two labels still overlap after the sweep (a crowded
//!    diagram where no angle was free), the same tangential-push sweep the
//!    raycast exterior policy uses,
//!    [`resolve_exterior_collisions`], separates them. Its push axis is
//!    perpendicular to each label's outward direction, so for this mode
//!    "push tangentially" *is* "rotate further around the shape"; each pass
//!    is followed by a re-projection back onto the hug ring so a slid label
//!    doesn't drift into the shape along the chord.
//!
//! Sets are placed one at a time in descending shape-area order (ties by
//! name), each seeing the ones before it as obstacles, so the result is
//! independent of the caller's `HashMap` iteration order.

use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::geometry::primitives::{Bounds, Point};
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::geometry::traits::BoundingBox;
use crate::plotting::placement::{
    ExteriorEntry, LabelPlacement, PlacementKind, closest_point_on_piece, raycast_anchor_union,
    resolve_exterior_collisions,
};
use crate::plotting::regions::{
    RegionPiece, classify_into_pieces, poi_with_holes, signed_clearance,
};
use crate::spec::Combination;

/// Default number of candidate angles swept around each shape — 2°
/// resolution, fine enough that the chosen slot is visually
/// indistinguishable from the continuous optimum while keeping the sweep
/// well under a millisecond for typical diagrams.
const DEFAULT_ANGULAR_STEPS: usize = 180;

/// Fewer than this many candidate angles can't express "rotate around the
/// shape" at all, so [`SetLabelStrategy::angular_steps`] is clamped up to it.
const MIN_ANGULAR_STEPS: usize = 8;

/// Outer iterations of the collision-guard / re-projection loop. Each pass
/// is a full tangential-push sweep followed by a re-hug; two or three
/// resolve every crowded case we've seen, and the cap bounds the worst case.
const GUARD_PASSES: usize = 6;

/// Inner tangential-push sweeps per guard pass. Kept short because the
/// re-projection between passes changes the geometry the sweep is resolving
/// against, so running it to convergence in one pass is wasted work.
const GUARD_SWEEPS: usize = 12;

/// Configuration bundle for [`place_set_labels`].
///
/// `#[non_exhaustive]`: construct with [`SetLabelStrategy::default`] and the
/// builder methods, so new knobs can land without a breaking change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SetLabelStrategy {
    /// Gap between the set's outline and the near edge of the label box.
    /// `None` selects a per-set proportional default of `0.5 * label_h` —
    /// half a line of text.
    ///
    /// Note this scales with the label's **height**, unlike the
    /// `0.5 * max(w, h)` convention the leader-line exterior policies use
    /// (e.g. [`ExteriorPolicy::Raycast`](crate::plotting::ExteriorPolicy::Raycast)).
    /// There, the margin sets how far past the whole diagram a label sits,
    /// and scaling with the label's longest side keeps a wide label from
    /// crowding the outline it was pushed off. Here the label is meant to
    /// read as *attached* to its shape, and a width-scaled gap would fling a
    /// long set name several radii away — far enough that the reader loses
    /// which shape it names, which is exactly what the mode exists to avoid.
    pub margin: Option<f64>,

    /// Number of candidate angles swept around each shape. Defaults to
    /// `180` (2° steps). Values below `8` are clamped up; raising it buys
    /// finer slot selection on crowded diagrams at linear cost.
    pub angular_steps: usize,

    /// Extra axis-aligned keep-out boxes the labels must avoid, in the same
    /// coordinate space as the outlines. This is the hook for everything
    /// eunoia doesn't know about: region quantity labels
    /// ([`label_boxes`](crate::plotting::label_boxes) turns a
    /// [`place_labels`](crate::plotting::place_labels) result into exactly
    /// this list), a title, a legend. Same convention as
    /// [`GlyphOptions::obstacles`](crate::plotting::GlyphOptions::obstacles).
    pub obstacles: Vec<Rectangle>,

    /// Polylabel-style search precision for the ray origin inside each
    /// shape, in the same units as the outlines. Defaults to `0.01`.
    pub precision: f64,
}

impl Default for SetLabelStrategy {
    /// Proportional margin, `angular_steps = 180`, no extra obstacles,
    /// `precision = 0.01`.
    fn default() -> Self {
        Self {
            margin: None,
            angular_steps: DEFAULT_ANGULAR_STEPS,
            obstacles: Vec::new(),
            precision: 0.01,
        }
    }
}

impl SetLabelStrategy {
    /// Sets [`margin`](Self::margin) and returns `self`. Accepts a bare
    /// `f64` or `None` (e.g. `.margin(0.5)` or `.margin(None)`).
    pub fn margin(mut self, margin: impl Into<Option<f64>>) -> Self {
        self.margin = margin.into();
        self
    }

    /// Sets [`angular_steps`](Self::angular_steps) and returns `self`.
    pub fn angular_steps(mut self, angular_steps: usize) -> Self {
        self.angular_steps = angular_steps;
        self
    }

    /// Sets [`obstacles`](Self::obstacles) and returns `self`.
    pub fn obstacles(mut self, obstacles: Vec<Rectangle>) -> Self {
        self.obstacles = obstacles;
        self
    }

    /// Sets [`precision`](Self::precision) and returns `self`.
    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = precision;
        self
    }
}

/// Place one label per set, just outside that set's own shape.
///
/// `outlines` is typically [`PlotData::shape_outlines`](crate::plotting::PlotData::shape_outlines)
/// — the polygonised shape per set name. `sizes` is keyed by set name and
/// carries each label's measured `(width, height)`; eunoia has no font
/// knowledge, so the caller owns text measurement. `container` is the
/// jointly-fitted complement container when the spec carried one: label
/// boxes that fit inside it are preferred over ones that spill out. Pass
/// [`None`] otherwise.
///
/// Every set present in **both** `outlines` and `sizes` with a usable size
/// and a non-degenerate outline gets a [`LabelPlacement`] back, with
/// [`PlacementKind::ExteriorSet`] and no leader (`tether` and `leader_end`
/// are `None`, `leader_waypoints` empty). Sets missing from either map, or
/// with a non-finite / non-positive size, are omitted.
///
/// # Choosing the side
///
/// The chosen angle maximises the label box's clearance to the other sets'
/// outlines, the set labels already placed, and
/// [`SetLabelStrategy::obstacles`]. When several angles tie — the common
/// case for an isolated set, where every side is equally free — the
/// tie-break points the label away from the diagram's centre, so labels
/// fan outward instead of piling into the middle. A lone set in a
/// single-set diagram has no outward direction either, and falls back to
/// `+x`.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::{PlacementKind, PlotOptions, SetLabelStrategy, place_set_labels};
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
/// let plot = layout.plot_data(&spec, PlotOptions::default());
///
/// let mut sizes = HashMap::new();
/// sizes.insert("A".to_string(), (0.4, 0.2));
/// sizes.insert("B".to_string(), (0.4, 0.2));
///
/// let placements = place_set_labels(
///     &plot.shape_outlines,
///     &sizes,
///     None,
///     &SetLabelStrategy::default(),
/// );
///
/// // Every set gets an exterior, leader-less placement.
/// for name in ["A", "B"] {
///     let p = &placements[name];
///     assert_eq!(p.kind, PlacementKind::ExteriorSet);
///     assert!(p.tether.is_none());
/// }
/// ```
pub fn place_set_labels(
    outlines: &HashMap<String, Polygon>,
    sizes: &HashMap<String, (f64, f64)>,
    container: Option<&Rectangle>,
    strategy: &SetLabelStrategy,
) -> HashMap<String, LabelPlacement> {
    let steps = strategy.angular_steps.max(MIN_ANGULAR_STEPS);

    // One piece per set. `classify_into_pieces` normalises ring orientation
    // (which `signed_clearance` relies on) and drops degenerate outlines.
    let mut candidates: Vec<Candidate> = Vec::with_capacity(sizes.len());
    for (name, outline) in outlines {
        let Some(&(w, h)) = sizes.get(name) else {
            continue;
        };
        if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let pieces = classify_into_pieces(vec![outline.clone()]);
        if pieces.is_empty() {
            continue;
        }
        // Ray origin: the shape's own pole of inaccessibility. For the
        // convex shapes eunoia fits this is the centre in all but name; it
        // is used rather than the bbox centre so the sweep stays sane on a
        // heavily faceted or off-centre outline.
        let Some((origin, _)) = poi_with_holes(&pieces, strategy.precision) else {
            continue;
        };
        let area: f64 = pieces.iter().map(|p| p.area()).sum();
        candidates.push(Candidate {
            name: name.clone(),
            pieces,
            origin,
            area,
            w,
            h,
        });
    }
    if candidates.is_empty() {
        return HashMap::new();
    }

    // Descending area, name as tie-break: deterministic regardless of the
    // `HashMap` iteration order, and the visually dominant sets get first
    // pick of the free space.
    candidates.sort_by(|a, b| {
        b.area
            .partial_cmp(&a.area)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    let diagram_centre = outlines_centre(&candidates);

    // Greedy sweep: each set picks its best angle seeing the shapes of
    // every other set plus the label boxes already committed.
    let mut placed_boxes: Vec<Rectangle> = Vec::with_capacity(candidates.len());
    let mut entries: Vec<ExteriorEntry> = Vec::with_capacity(candidates.len());
    for (i, cand) in candidates.iter().enumerate() {
        let margin = strategy.margin.unwrap_or(0.5 * cand.h);
        let foreign: Vec<&RegionPiece> = candidates
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .flat_map(|(_, other)| other.pieces.iter())
            .collect();

        let (anchor, direction) = best_angle(
            cand,
            margin,
            steps,
            &foreign,
            &placed_boxes,
            &strategy.obstacles,
            container,
            &diagram_centre,
        );

        placed_boxes.push(Rectangle::new(anchor, cand.w, cand.h));
        entries.push(ExteriorEntry {
            key: cand.name.clone(),
            combo: Combination::new(&[&cand.name]),
            anchor,
            home: anchor,
            poi: cand.origin,
            direction,
            margin,
            w: cand.w,
            h: cand.h,
        });
    }

    // Guard: separate any label boxes the greedy sweep couldn't, then
    // re-hug. `resolve_exterior_collisions` pushes along each label's
    // tangent — perpendicular to `direction` — which slides a label around
    // its shape but along the chord, so it has to be re-projected back onto
    // the hug ring afterwards or it ends up creeping inside the outline.
    for _ in 0..GUARD_PASSES {
        let before: Vec<Point> = entries.iter().map(|e| e.anchor).collect();
        resolve_exterior_collisions(&mut entries, &[], GUARD_SWEEPS);
        for (entry, cand) in entries.iter_mut().zip(candidates.iter()) {
            let dx = entry.anchor.x() - entry.poi.x();
            let dy = entry.anchor.y() - entry.poi.y();
            let mag = dx.hypot(dy);
            if mag < 1e-12 {
                continue;
            }
            let dir = (dx / mag, dy / mag);
            entry.direction = dir;
            entry.anchor = hug_anchor(cand, dir, entry.margin);
        }
        let settled = entries
            .iter()
            .zip(before.iter())
            .all(|(e, p)| (e.anchor.x() - p.x()).hypot(e.anchor.y() - p.y()) < 1e-9);
        if settled {
            break;
        }
    }

    // Separation has the last word. The re-hug above snaps each anchor back
    // onto its hug ring, which can re-introduce an overlap the sweep just
    // resolved — so on a diagram crowded enough that the loop never settled,
    // the loop's final state is a *hugged* one, not a *disjoint* one. Running
    // one more sweep here inverts that priority: labels that can't both hug
    // and stay clear give up a little tightness rather than collide. A no-op
    // (single pass, nothing moves) whenever the loop did settle.
    resolve_exterior_collisions(&mut entries, &[], GUARD_SWEEPS);

    entries
        .into_iter()
        .map(|entry| {
            (
                entry.key,
                LabelPlacement {
                    anchor: entry.anchor,
                    kind: PlacementKind::ExteriorSet,
                    // Adjacent to the shape it names — no leader, so
                    // nothing for a renderer to draw between the two.
                    tether: None,
                    leader_end: None,
                    leader_waypoints: Vec::new(),
                },
            )
        })
        .collect()
}

/// A set that survived validation, with everything the sweep needs.
struct Candidate {
    name: String,
    /// The set's own outline, orientation-normalised. One piece for every
    /// shape eunoia fits; a `Vec` because that's what the clearance helpers
    /// take.
    pieces: Vec<RegionPiece>,
    /// Ray origin — the outline's pole of inaccessibility.
    origin: Point,
    area: f64,
    w: f64,
    h: f64,
}

/// Anchor for `cand`'s label pushed out along `dir` until its box clears the
/// set's own outline by `margin`.
///
/// This is the "hug" step, and it is [`raycast_anchor_union`] applied to a
/// single shape rather than to the diagram silhouette: the label box's full
/// footprint has to clear the boundary, so a wide label placed diagonally
/// can't clip a curving edge with a corner. Falls back to a circumradius
/// offset when the ray degenerates (an origin already clear of every vertex
/// on the outgoing side), which the convex outlines eunoia fits never hit.
fn hug_anchor(cand: &Candidate, dir: (f64, f64), margin: f64) -> Point {
    raycast_anchor_union(&cand.origin, cand.w, cand.h, &cand.pieces, margin, dir).unwrap_or_else(
        || {
            let mut r: f64 = 0.0;
            for piece in &cand.pieces {
                for v in piece.outer.vertices() {
                    r = r.max((v.x() - cand.origin.x()).hypot(v.y() - cand.origin.y()));
                }
            }
            let reach = r + margin + 0.5 * (dir.0.abs() * cand.w + dir.1.abs() * cand.h);
            Point::new(
                cand.origin.x() + reach * dir.0,
                cand.origin.y() + reach * dir.1,
            )
        },
    )
}

/// Sweep every candidate angle around `cand` and return the
/// `(anchor, direction)` with the most room.
///
/// Ranking is lexicographic: candidates whose box fits inside `container`
/// beat ones that spill out of it (no-op when there is no container), then
/// the largest clearance to everything else wins, and near-ties fall back to
/// pointing away from `diagram_centre`. The near-tie band matters more than
/// it looks: with no other shapes in range every angle scores `f64::MAX`, so
/// without it the winner would be whichever angle the sweep happened to
/// visit first.
#[allow(clippy::too_many_arguments)]
fn best_angle(
    cand: &Candidate,
    margin: f64,
    steps: usize,
    foreign: &[&RegionPiece],
    placed: &[Rectangle],
    obstacles: &[Rectangle],
    container: Option<&Rectangle>,
    diagram_centre: &Point,
) -> (Point, (f64, f64)) {
    // Outward reference: from the diagram centre toward this shape. Zero
    // for a set sitting on the centre (or a single-set diagram), in which
    // case the outward tie-break contributes nothing and the first angle
    // in the sweep — `+x` — wins.
    let ox = cand.origin.x() - diagram_centre.x();
    let oy = cand.origin.y() - diagram_centre.y();
    let omag = ox.hypot(oy);
    let outward = if omag > 1e-9 {
        (ox / omag, oy / omag)
    } else {
        (0.0, 0.0)
    };

    // Tie band, scaled to the shape so it behaves the same whatever the
    // canvas units: clearances within this of the best count as equal and
    // are decided by the outward preference.
    let scale = (cand.w.max(cand.h)).max(1e-9);
    let tie = 1e-3 * scale;

    /// One angle's score, ranked lexicographically by
    /// `(fits_container, clearance, outwardness)`.
    struct Slot {
        fits: bool,
        clearance: f64,
        outwardness: f64,
        anchor: Point,
        dir: (f64, f64),
    }

    let mut best: Option<Slot> = None;
    for k in 0..steps {
        let theta = TAU * (k as f64) / (steps as f64);
        let dir = (theta.cos(), theta.sin());
        let anchor = hug_anchor(cand, dir, margin);
        let slot = Slot {
            fits: container.is_none_or(|c| box_inside(&anchor, cand.w, cand.h, c)),
            clearance: min_clearance(&anchor, cand.w, cand.h, foreign, placed, obstacles),
            outwardness: dir.0 * outward.0 + dir.1 * outward.1,
            anchor,
            dir,
        };

        let better = match &best {
            None => true,
            Some(best) => {
                if slot.fits != best.fits {
                    // Staying inside the container outranks any clearance
                    // gain: a label outside it reads as belonging to
                    // nothing.
                    slot.fits
                } else if (slot.clearance - best.clearance).abs() > tie {
                    slot.clearance > best.clearance
                } else {
                    slot.outwardness > best.outwardness
                }
            }
        };
        if better {
            best = Some(slot);
        }
    }

    // `steps >= MIN_ANGULAR_STEPS > 0`, so the loop always sets `best`.
    let best = best.expect("angular sweep visits at least one candidate");
    (best.anchor, best.dir)
}

/// Smallest clearance from the `w × h` box centred at `anchor` to any
/// obstacle — foreign shape outlines, already-placed label boxes, and
/// caller-supplied keep-outs. Negative when the box overlaps something;
/// [`f64::MAX`] when there is nothing to avoid.
fn min_clearance(
    anchor: &Point,
    w: f64,
    h: f64,
    foreign: &[&RegionPiece],
    placed: &[Rectangle],
    obstacles: &[Rectangle],
) -> f64 {
    let mut worst = f64::MAX;
    for piece in foreign {
        worst = worst.min(box_polygon_clearance(anchor, w, h, piece));
    }
    for rect in placed.iter().chain(obstacles.iter()) {
        worst = worst.min(box_box_clearance(anchor, w, h, rect));
    }
    worst
}

/// Distance from the `w × h` box centred at `anchor` to `piece`, negative
/// when they overlap.
///
/// The exact box-to-polygon distance would need a full SAT sweep over every
/// edge; instead the centre's signed distance is reduced by the box's own
/// half-extent *along the direction of the nearest boundary point* — the
/// box's support function in that direction. That is exact when the nearest
/// feature is axis-aligned and conservative (it under-reports clearance)
/// otherwise, which is the safe direction to err for a keep-out test.
fn box_polygon_clearance(anchor: &Point, w: f64, h: f64, piece: &RegionPiece) -> f64 {
    // `signed_clearance` is positive inside the piece, so negate it to get
    // "distance to the shape" with overlap coming out negative.
    let outside = -signed_clearance(anchor.x(), anchor.y(), piece);
    let (qx, qy) = closest_point_on_piece(anchor.x(), anchor.y(), piece);
    let dx = anchor.x() - qx;
    let dy = anchor.y() - qy;
    let len = dx.hypot(dy);
    let half_extent = if len > 1e-12 {
        0.5 * ((dx / len).abs() * w + (dy / len).abs() * h)
    } else {
        // Centre sits on the boundary — no usable direction, so charge the
        // box's larger half-extent.
        0.5 * w.max(h)
    };
    outside - half_extent
}

/// Gap between the `w × h` box centred at `anchor` and `other`. Positive
/// when disjoint (the Euclidean gap), negative when they overlap (the
/// shallower of the two axis penetrations, i.e. the depth the separating
/// push would have to undo).
fn box_box_clearance(anchor: &Point, w: f64, h: f64, other: &Rectangle) -> f64 {
    let gap_x = (anchor.x() - other.center().x()).abs() - 0.5 * (w + other.width());
    let gap_y = (anchor.y() - other.center().y()).abs() - 0.5 * (h + other.height());
    if gap_x > 0.0 || gap_y > 0.0 {
        gap_x.max(0.0).hypot(gap_y.max(0.0))
    } else {
        gap_x.max(gap_y)
    }
}

/// Is the `w × h` box centred at `anchor` fully inside `container`?
fn box_inside(anchor: &Point, w: f64, h: f64, container: &Rectangle) -> bool {
    let Bounds {
        x_min,
        x_max,
        y_min,
        y_max,
    } = container.bounds();
    anchor.x() - 0.5 * w >= x_min
        && anchor.x() + 0.5 * w <= x_max
        && anchor.y() - 0.5 * h >= y_min
        && anchor.y() + 0.5 * h <= y_max
}

/// Centre of the bounding box of every candidate's outline — the reference
/// point the outward tie-break fans labels away from.
fn outlines_centre(candidates: &[Candidate]) -> Point {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for cand in candidates {
        for piece in &cand.pieces {
            for v in piece.outer.vertices() {
                min_x = min_x.min(v.x());
                min_y = min_y.min(v.y());
                max_x = max_x.max(v.x());
                max_y = max_y.max(v.y());
            }
        }
    }
    if min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite() {
        Point::new(0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
    } else {
        Point::new(0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::plotting::PlotOptions;
    use crate::spec::{DiagramSpecBuilder, InputType};

    /// Build outlines for a hand-placed set of circles, bypassing the fitter
    /// so a test can control the geometry exactly.
    fn outlines_of(circles: &[(&str, f64, f64, f64)]) -> HashMap<String, Polygon> {
        use crate::geometry::traits::Polygonize;
        circles
            .iter()
            .map(|&(name, x, y, r)| {
                (
                    name.to_string(),
                    Circle::new(Point::new(x, y), r).polygonize(200),
                )
            })
            .collect()
    }

    fn sizes_of(names: &[&str], w: f64, h: f64) -> HashMap<String, (f64, f64)> {
        names
            .iter()
            .map(|n| (n.to_string(), (w, h)))
            .collect::<HashMap<_, _>>()
    }

    /// Every label must sit outside its own shape — that is the whole point
    /// of the mode — and no farther out than the boundary plus the margin
    /// plus its own half-extent, which is the "close to the boundary" half
    /// of the contract.
    #[test]
    fn test_labels_hug_their_own_shape() {
        let outlines = outlines_of(&[("A", 0.0, 0.0, 1.0), ("B", 1.5, 0.0, 1.0)]);
        let sizes = sizes_of(&["A", "B"], 0.4, 0.2);
        let margin = 0.1;
        let placements = place_set_labels(
            &outlines,
            &sizes,
            None,
            &SetLabelStrategy::default().margin(margin),
        );

        assert_eq!(placements.len(), 2);
        for (name, centre) in [("A", (0.0, 0.0)), ("B", (1.5, 0.0))] {
            let p = &placements[name];
            let d = (p.anchor.x() - centre.0).hypot(p.anchor.y() - centre.1);
            // Outside the unit circle, by at least the margin plus the
            // smaller half-extent of the box.
            assert!(
                d > 1.0 + margin,
                "{name} label at distance {d} is not outside its shape + margin"
            );
            // And not flung out: the loosest bound is the box's half-diagonal.
            let half_diag = 0.5 * 0.4_f64.hypot(0.2);
            assert!(
                d <= 1.0 + margin + half_diag + 1e-6,
                "{name} label at distance {d} drifted past the hug ring"
            );
        }
    }

    /// Two circles side by side: each label must end up on the *far* side of
    /// its own circle, away from the neighbour. This is the rotation
    /// objective — with no other shapes, both would sit at `+x`.
    #[test]
    fn test_label_rotates_away_from_neighbour() {
        let outlines = outlines_of(&[("A", 0.0, 0.0, 1.0), ("B", 1.5, 0.0, 1.0)]);
        let sizes = sizes_of(&["A", "B"], 0.4, 0.2);
        let placements = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());

        // A is the left circle → its label goes left of A's centre;
        // B is the right circle → right of B's centre.
        assert!(
            placements["A"].anchor.x() < 0.0,
            "A's label should sit left of A, got x = {}",
            placements["A"].anchor.x()
        );
        assert!(
            placements["B"].anchor.x() > 1.5,
            "B's label should sit right of B, got x = {}",
            placements["B"].anchor.x()
        );
    }

    /// The label boxes must not overlap each other, even when the shapes are
    /// packed tightly enough that the greedy sweep alone can't separate them.
    #[test]
    fn test_label_boxes_do_not_overlap() {
        let outlines = outlines_of(&[
            ("A", 0.0, 0.0, 1.0),
            ("B", 1.2, 0.0, 1.0),
            ("C", 0.6, 1.0, 1.0),
            ("D", 0.6, -1.0, 1.0),
        ]);
        let sizes = sizes_of(&["A", "B", "C", "D"], 0.5, 0.25);
        let placements = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());
        assert_eq!(placements.len(), 4);

        let keys: Vec<&String> = placements.keys().collect();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let a = &placements[keys[i]].anchor;
                let b = &placements[keys[j]].anchor;
                let (aw, ah) = sizes[keys[i]];
                let (bw, bh) = sizes[keys[j]];
                let overlap_x = (a.x() - b.x()).abs() < 0.5 * (aw + bw) - 1e-9;
                let overlap_y = (a.y() - b.y()).abs() < 0.5 * (ah + bh) - 1e-9;
                assert!(
                    !(overlap_x && overlap_y),
                    "labels {} and {} overlap",
                    keys[i],
                    keys[j]
                );
            }
        }
    }

    /// The non-overlap guarantee has to survive the case it was written for:
    /// shapes packed tightly enough, with labels wide enough, that no set can
    /// both hug its outline and stay clear of its neighbours. Separation wins;
    /// the labels give up some tightness instead of colliding.
    #[test]
    fn test_crowded_labels_still_separate() {
        let names = ["A", "B", "C", "D", "E", "F"];
        let circles: Vec<(&str, f64, f64, f64)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let theta = TAU * (i as f64) / (names.len() as f64);
                (*name, 0.9 * theta.cos(), 0.9 * theta.sin(), 0.7)
            })
            .collect();
        let outlines = outlines_of(&circles);
        // Labels wide relative to the shapes: no angle is free.
        let sizes = sizes_of(&names, 1.4, 0.3);
        let placements = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());
        assert_eq!(placements.len(), names.len());

        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let a = &placements[names[i]].anchor;
                let b = &placements[names[j]].anchor;
                let overlap_x = (a.x() - b.x()).abs() < 1.4 - 1e-9;
                let overlap_y = (a.y() - b.y()).abs() < 0.3 - 1e-9;
                assert!(
                    !(overlap_x && overlap_y),
                    "labels {} and {} overlap at ({:.3}, {:.3}) / ({:.3}, {:.3})",
                    names[i],
                    names[j],
                    a.x(),
                    a.y(),
                    b.x(),
                    b.y()
                );
            }
        }
    }

    /// A caller-supplied keep-out box has to steer the sweep, the same way a
    /// neighbouring shape does. Blocking the whole left half of an isolated
    /// circle must push its label to the right.
    #[test]
    fn test_obstacles_steer_the_sweep() {
        let outlines = outlines_of(&[("A", 0.0, 0.0, 1.0)]);
        let sizes = sizes_of(&["A"], 0.4, 0.2);
        // A tall wall covering everything left of the circle.
        let wall = Rectangle::new(Point::new(-2.0, 0.0), 2.0, 8.0);
        let placements = place_set_labels(
            &outlines,
            &sizes,
            None,
            &SetLabelStrategy::default().obstacles(vec![wall]),
        );
        assert!(
            placements["A"].anchor.x() > 0.0,
            "label should avoid the keep-out wall, got x = {}",
            placements["A"].anchor.x()
        );
    }

    /// With a container in play, a label that fits inside it beats one that
    /// spills out, even when the outside slot has more raw clearance.
    #[test]
    fn test_container_keeps_labels_inside() {
        let outlines = outlines_of(&[("A", 0.0, 0.0, 1.0)]);
        let sizes = sizes_of(&["A"], 0.4, 0.2);
        // Container hugging the circle on the right but roomy on the left,
        // so only left-hand slots fit inside it.
        let container = Rectangle::new(Point::new(-0.6, 0.0), 3.2, 4.0);
        let placements = place_set_labels(
            &outlines,
            &sizes,
            Some(&container),
            &SetLabelStrategy::default(),
        );
        let p = &placements["A"];
        assert!(
            box_inside(&p.anchor, 0.4, 0.2, &container),
            "label at ({}, {}) escaped the container",
            p.anchor.x(),
            p.anchor.y()
        );
    }

    /// Result must not depend on `HashMap` iteration order.
    #[test]
    fn test_placement_is_deterministic() {
        let outlines = outlines_of(&[
            ("A", 0.0, 0.0, 1.0),
            ("B", 1.2, 0.0, 0.8),
            ("C", 0.6, 1.1, 0.6),
        ]);
        let sizes = sizes_of(&["A", "B", "C"], 0.4, 0.2);
        let first = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());
        for _ in 0..5 {
            let again = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());
            assert_eq!(first, again);
        }
    }

    /// Sets missing a size, or carrying a degenerate one, are skipped rather
    /// than placed somewhere arbitrary.
    #[test]
    fn test_skips_missing_and_degenerate_sizes() {
        let outlines = outlines_of(&[("A", 0.0, 0.0, 1.0), ("B", 3.0, 0.0, 1.0)]);
        let mut sizes = HashMap::new();
        sizes.insert("A".to_string(), (0.4, 0.0));
        // B has no entry at all.
        let placements = place_set_labels(&outlines, &sizes, None, &SetLabelStrategy::default());
        assert!(placements.is_empty());
    }

    /// End-to-end against a real fit: labels clear every *other* set's
    /// circle, which is the property a renderer actually depends on.
    #[test]
    fn test_fitted_diagram_labels_clear_other_shapes() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 6.0)
            .set("B", 4.0)
            .set("C", 3.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "C"], 1.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let plot = layout.plot_data(&spec, PlotOptions::default());
        let extent = {
            let mut r: f64 = 0.0;
            for name in spec.set_names() {
                r = r.max(layout.shape_for_set(name).unwrap().radius());
            }
            r
        };
        let sizes = sizes_of(&["A", "B", "C"], 0.35 * extent, 0.18 * extent);

        let placements = place_set_labels(
            &plot.shape_outlines,
            &sizes,
            None,
            &SetLabelStrategy::default(),
        );
        assert_eq!(placements.len(), 3);

        for name in spec.set_names() {
            let anchor = placements[name].anchor;
            for other in spec.set_names() {
                if other == name {
                    continue;
                }
                let c = layout.shape_for_set(other).unwrap();
                let d = (anchor.x() - c.center().x()).hypot(anchor.y() - c.center().y());
                assert!(
                    d > c.radius(),
                    "label {name} at ({:.3}, {:.3}) landed inside set {other}",
                    anchor.x(),
                    anchor.y()
                );
            }
        }
    }
}
