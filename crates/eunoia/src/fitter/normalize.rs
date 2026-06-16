//! Diagram layout normalization.
//!
//! This module provides functionality to normalize diagram layouts by:
//! 1. Rotating clusters to a canonical orientation
//! 2. Centering the overall layout
//! 3. Packing disjoint clusters compactly

use crate::fitter::clustering::{find_clusters, find_clusters_from_exclusive_regions};
use crate::fitter::packing::skyline_pack;
use crate::geometry::diagram::RegionMask;
use crate::geometry::primitives::{Bounds, Point};
use crate::geometry::shapes::Rectangle;
use crate::geometry::traits::DiagramShape;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Fraction of the internal-tangency distance by which a free contained child
/// is offset from its parent's center.
///
/// Strictly between `0` (concentric — a thin uniform annulus, the worst case
/// for labelling the parent's exclusive lune) and `1` (internally tangent — the
/// child jammed against the parent's edge). `0.5` leaves a near-side gap of
/// `0.5·d_max` and opens a far-side lune of `1.5·d_max`: clearly off-center
/// without touching.
const CONTAINED_CHILD_OFFSET_FRACTION: f64 = 0.5;

/// Normalize a collection of diagram shapes.
///
/// This function:
/// 1. Identifies disjoint clusters of shapes
/// 2. Rotates each cluster to a canonical orientation
/// 3. Centers each cluster
/// 4. Packs multiple clusters together compactly
/// 5. Centers the final layout
///
/// # Arguments
///
/// * `shapes` - Mutable slice of shapes to normalize
/// * `padding_factor` - Padding between clusters as a fraction of total width (default: 0.015)
pub fn normalize_layout<S>(shapes: &mut [S], padding_factor: f64)
where
    S: DiagramShape + Clone,
{
    normalize_layout_with_clusters::<S>(shapes, padding_factor, None);
}

/// Container-aware normalisation for complement (universe) layouts.
///
/// The container is **always axis-aligned** (a hard design invariant), so
/// rotating or mirroring the layout would either break that invariant or
/// require flipping the container — not useful for the visual frame the
/// container represents. Multi-cluster + complement is rejected upfront
/// (`spec_is_multi_cluster`), so packing is moot.
///
/// What we *do* want, in two pure-translation steps:
///
/// 1. Move the whole scene (shapes + container) so the container's center sits
///    at the origin. This is always loss-preserving — it translates everything
///    together — and puts the diagram in a predictable coordinate frame for
///    downstream consumers (renderers, bindings).
/// 2. Center the diagram *within* the container: slide the shapes so the center
///    of their joint bounding box coincides with the container center. The
///    optimiser is free to leave the union parked off-center inside the frame
///    (only the clipped region areas enter the loss), which looks lopsided;
///    this recenters it.
///
/// Step 2 is applied **only when every shape already lies fully inside the
/// container**. Region areas are clipped to the container
/// ([`DiagramShape::compute_exclusive_regions_clipped`]), so once a shape pokes
/// past an edge its optimiser-chosen offset is load-bearing — sliding it would
/// change the clipped complement and silently degrade the fit. In that case we
/// keep step 1 only.
pub fn normalize_layout_with_container<S>(shapes: &mut [S], container: &mut Rectangle)
where
    S: DiagramShape + Clone,
{
    // Step 1: park the container at the origin, taking the shapes along.
    let cx = container.center().x();
    let cy = container.center().y();
    if cx.abs() > 1e-12 || cy.abs() > 1e-12 {
        for shape in shapes.iter_mut() {
            *shape = translate_shape(shape, -cx, -cy);
        }
        *container = Rectangle::new(Point::new(0.0, 0.0), container.width(), container.height());
    }

    // Step 2: center the diagram within the (now origin-centered) container.
    center_shapes_in_container(shapes, container);
}

/// Translate `shapes` so the center of their joint bounding box coincides with
/// the container center, but only when the whole diagram already fits inside
/// the container so the move cannot change any container-clipped region area.
///
/// Assumes `container` is centered at the origin (the caller's step 1).
fn center_shapes_in_container<S>(shapes: &mut [S], container: &Rectangle)
where
    S: DiagramShape + Clone,
{
    if shapes.is_empty() {
        return;
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for shape in shapes.iter() {
        let Bounds {
            x_min: bx0,
            x_max: bx1,
            y_min: by0,
            y_max: by1,
        } = shape.bounds();
        x_min = x_min.min(bx0);
        x_max = x_max.max(bx1);
        y_min = y_min.min(by0);
        y_max = y_max.max(by1);
    }

    // Bail out if any shape extends past the container: recentering would alter
    // the clipped complement. A small tolerance keeps edge-touching fits (where
    // clipping is inactive) on the centering path.
    let half_w = container.width() / 2.0;
    let half_h = container.height() / 2.0;
    let tol = 1e-9 * half_w.max(half_h).max(1.0);
    if x_min < -half_w - tol
        || x_max > half_w + tol
        || y_min < -half_h - tol
        || y_max > half_h + tol
    {
        return;
    }

    let dx = (x_min + x_max) / 2.0;
    let dy = (y_min + y_max) / 2.0;
    if dx.abs() < 1e-12 && dy.abs() < 1e-12 {
        return;
    }
    for shape in shapes.iter_mut() {
        *shape = translate_shape(shape, -dx, -dy);
    }
}

/// Variant of [`normalize_layout`] that uses the caller-supplied
/// exclusive-region area map for cluster detection instead of the geometric
/// `Closed::intersects` test.
///
/// Routing clustering through `find_clusters_from_exclusive_regions` keeps
/// `normalize_layout`'s notion of "which shapes overlap" consistent with the
/// inclusion-exclusion math the optimizer minimised against — so packing
/// can't translate apart shapes the optimizer believes overlap (which is
/// what trips the post-normalize debug_assert on near-coincident ellipse
/// fits).
///
/// `exclusive_areas` should be `S::compute_exclusive_regions(shapes)` from
/// the caller. Passing `None` falls back to the geometric `find_clusters`
/// path for backwards compatibility.
pub fn normalize_layout_with_clusters<S>(
    shapes: &mut [S],
    padding_factor: f64,
    exclusive_areas: Option<&HashMap<RegionMask, f64>>,
) where
    S: DiagramShape + Clone,
{
    if shapes.is_empty() {
        return;
    }

    // Step 0: give any free contained child a canonical moderate offset from
    // its parent before the orientation/centering steps run (which then
    // standardise the overall frame). Loss-preserving — only shapes fully
    // inside a single parent and clear of everything else are moved, and never
    // past the parent's edge — so it doesn't disturb cluster detection below.
    offset_contained_children(shapes, exclusive_areas);

    // Step 1: Find disjoint clusters. Prefer the area-based path when the
    // caller provides exclusive-region areas — that's the same exact-conic
    // math the optimizer just minimised, so it's strictly more reliable on
    // near-coincident geometry than the geometric `Closed::intersects` /
    // `contains` check.
    let clusters = match exclusive_areas {
        Some(areas) => {
            // Tolerance scales with the largest fitted region: roundoff in
            // `compute_exclusive_regions` is bounded by `eps * max_region`,
            // so a region of size `1e-12 * max_region` is below the
            // numerical floor and shouldn't connect a cluster. `1e-10`
            // gives a few orders of margin above that floor while still
            // catching genuine small overlaps.
            let max_region = areas
                .values()
                .copied()
                .fold(0.0_f64, |a, b| a.max(b.abs()))
                .max(1.0);
            let tol = 1e-10 * max_region;
            find_clusters_from_exclusive_regions(shapes.len(), areas, tol)
        }
        None => find_clusters(shapes),
    };

    if clusters.len() == 1 {
        // Single cluster - just rotate and center
        let cluster = &clusters[0];
        if cluster.len() > 1 {
            rotate_cluster(shapes, cluster);
        }
        center_layout(shapes);
    } else {
        // Multiple clusters - rotate, pack, and center
        for cluster in &clusters {
            if cluster.len() > 1 {
                rotate_cluster(shapes, cluster);
            }
        }

        pack_clusters(shapes, &clusters, padding_factor);
        center_layout(shapes);
    }
}

/// Give each *free single contained child* a canonical, moderate offset from
/// its parent's center.
///
/// When a shape sits entirely inside exactly one other shape and touches
/// nothing else, its position is a loss-invariant free degree of freedom: the
/// optimiser leaves it parked arbitrarily and non-reproducibly. A concentric
/// child turns the parent's exclusive region into a thin uniform annulus (poor
/// for label placement); pushing the child off-center opens a wide lune on the
/// far side. This nudges each such child to [`CONTAINED_CHILD_OFFSET_FRACTION`]
/// of the way to internal tangency.
///
/// Only the single-child case is handled. A parent containing two or more
/// disjoint children is a constrained-packing problem and is left untouched —
/// moving either child could overlap the other and change the loss. Nested
/// chains (a child with more than one container, e.g. a russian-doll) are
/// likewise skipped.
///
/// Every move is verified to keep the child fully inside the parent and clear
/// of all other shapes, so the pass is loss-preserving by construction.
///
/// Containment is detected from the **exclusive-region areas**, not the
/// geometric `contains` predicate: the optimiser routinely parks a subset at
/// exact internal tangency, where `contains` is floating-point false even
/// though the set is geometrically a subset. The area view is position
/// independent — a free child of `p` is a set whose solo region is ~0 and which
/// co-occurs with exactly one other set (`p`). When the caller doesn't supply
/// the areas (the public `normalize_layout` entry) they are recomputed.
fn offset_contained_children<S>(
    shapes: &mut [S],
    exclusive_areas: Option<&HashMap<RegionMask, f64>>,
) where
    S: DiagramShape + Clone,
{
    let n = shapes.len();
    if n < 2 {
        return;
    }

    let owned;
    let areas = match exclusive_areas {
        Some(a) => a,
        None => {
            owned = S::compute_exclusive_regions(shapes);
            &owned
        }
    };
    let max_region = areas
        .values()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1.0);
    let tol = 1e-10 * max_region;

    // Total area of each set: the sum of every region it participates in. Used
    // to require a child be strictly smaller than its parent, which rules out
    // two coincident equal sets (each then looks like the other's free child,
    // but neither is contained — pushing them apart would destroy the overlap).
    let mut set_total = vec![0.0_f64; n];
    for (&mask, &area) in areas {
        if area <= 0.0 {
            continue;
        }
        let mut bits = mask;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            if b < n {
                set_total[b] += area;
            }
            bits &= bits - 1;
        }
    }

    // Identify every free contained child and its parent, then count children
    // per parent so we only act on the single-child case (a parent with two or
    // more free children is the deferred constrained-packing problem).
    let mut parent_of: Vec<Option<usize>> = vec![None; n];
    let mut child_count: Vec<usize> = vec![0; n];
    for (i, slot) in parent_of.iter_mut().enumerate() {
        if let Some(p) = contained_child_parent(i, n, areas, tol) {
            // Require the child to be strictly smaller than the parent.
            if set_total[i] < set_total[p] * (1.0 - 1e-6) {
                *slot = Some(p);
                child_count[p] += 1;
            }
        }
    }

    for (i, &maybe_parent) in parent_of.iter().enumerate() {
        if let Some(p) = maybe_parent {
            if child_count[p] == 1 {
                reposition_contained_child(shapes, p, i);
            }
        }
    }
}

/// From the exclusive-region areas, decide whether set `i` is a *free contained
/// child* and, if so, return its parent set's index.
///
/// `i` qualifies when its solo region (`mask == 1 << i`) is ~0 and every
/// significant region containing bit `i` also contains exactly one common other
/// bit `p` — i.e. `i` is entirely inside `p` and shares area with nothing else.
/// Returns `None` for sets with their own region, sets sharing area with two or
/// more others (nesting / russian-doll, multi-way overlaps), or empties.
fn contained_child_parent(
    i: usize,
    n: usize,
    areas: &HashMap<RegionMask, f64>,
    tol: f64,
) -> Option<usize> {
    let bit_i: RegionMask = 1 << i;

    // A non-trivial solo region means `i` is not wholly contained.
    if areas.get(&bit_i).copied().unwrap_or(0.0) > tol {
        return None;
    }

    let mut other_bits: RegionMask = 0;
    let mut saw_region = false;
    for (&mask, &area) in areas {
        if area <= tol || (mask & bit_i) == 0 {
            continue;
        }
        saw_region = true;
        other_bits |= mask & !bit_i;
    }

    // Must co-occur with exactly one other set, which is the parent.
    if !saw_region || other_bits.count_ones() != 1 {
        return None;
    }
    let p = other_bits.trailing_zeros() as usize;
    if p >= n || p == i {
        return None;
    }
    Some(p)
}

/// Move free child `i` to [`CONTAINED_CHILD_OFFSET_FRACTION`] of the way to
/// internal tangency with parent `p`, along a canonical direction. Leaves the
/// child untouched if the resulting move can't be verified loss-preserving.
fn reposition_contained_child<S>(shapes: &mut [S], p: usize, i: usize)
where
    S: DiagramShape + Clone,
{
    let pc = shapes[p].centroid();
    let cc = shapes[i].centroid();

    // Direction: keep the optimiser's parent→child direction when it carries
    // signal; fall back to a fixed canonical axis when (near-)concentric (the
    // direction is then noise, and `rotate_cluster` canonicalises the final
    // orientation regardless). The threshold scales with the parent so it is
    // coordinate-scale invariant.
    let Bounds {
        x_min,
        x_max,
        y_min,
        y_max,
    } = shapes[p].bounds();
    let parent_scale = ((x_max - x_min).powi(2) + (y_max - y_min).powi(2)).sqrt();
    let dx = cc.x() - pc.x();
    let dy = cc.y() - pc.y();
    let mag = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = if mag > 1e-9 * parent_scale.max(1.0) {
        (dx / mag, dy / mag)
    } else {
        (1.0, 0.0)
    };

    let d_max = max_contained_offset(&shapes[p], &shapes[i], ux, uy);
    if d_max <= 0.0 {
        return;
    }
    let d = CONTAINED_CHILD_OFFSET_FRACTION * d_max;

    // Target center = parent center + u·d; translate from the current center.
    let target_x = pc.x() + ux * d;
    let target_y = pc.y() + uy * d;
    let move_x = target_x - cc.x();
    let move_y = target_y - cc.y();

    // Already at the canonical offset (within tolerance): nothing to do. Keeps
    // re-normalisation a stable no-op despite binary-search round-off.
    if (move_x * move_x + move_y * move_y).sqrt() <= 1e-9 * d_max.max(1.0) {
        return;
    }
    let moved = translate_shape(&shapes[i], move_x, move_y);

    // Loss guard: the move must keep the child inside the parent and clear of
    // every other shape. Bail out (keep the original) on any FP surprise.
    if !shapes[p].contains(&moved) {
        return;
    }
    for (k, other) in shapes.iter().enumerate() {
        if k == i || k == p {
            continue;
        }
        if moved.intersects(other) || moved.contains(other) || other.contains(&moved) {
            return;
        }
    }

    shapes[i] = moved;
}

/// Binary-search the largest distance `d` such that translating `child` so its
/// center sits at `parent_center + (ux, uy)·d` keeps it fully inside `parent`.
///
/// Shape-agnostic: it only probes [`crate::geometry::traits::Closed::contains`].
/// For circles this recovers `R − r`. Assumes `d = 0` (child centered on the
/// parent) is contained, which holds for the library's convex shapes.
fn max_contained_offset<S>(parent: &S, child: &S, ux: f64, uy: f64) -> f64
where
    S: DiagramShape + Clone,
{
    let pc = parent.centroid();
    let cc = child.centroid();

    // Is `child` still inside `parent` once recentred at `parent_center + u·d`?
    let contains_at = |d: f64| -> bool {
        let tx = pc.x() + ux * d - cc.x();
        let ty = pc.y() + uy * d - cc.y();
        parent.contains(&translate_shape(child, tx, ty))
    };

    if !contains_at(0.0) {
        return 0.0;
    }

    // Upper bound: the parent's bbox diagonal is guaranteed beyond tangency for
    // a contained child of nonzero size. Expand defensively just in case.
    let Bounds {
        x_min,
        x_max,
        y_min,
        y_max,
    } = parent.bounds();
    let mut hi = ((x_max - x_min).powi(2) + (y_max - y_min).powi(2)).sqrt();
    if hi <= 0.0 {
        return 0.0;
    }
    let mut guard = 0;
    while contains_at(hi) && guard < 8 {
        hi *= 2.0;
        guard += 1;
    }

    let mut lo = 0.0;
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if contains_at(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Rotate a cluster to a canonical orientation.
///
/// For clusters with 2+ shapes:
/// 1. Rotate so the line between first two shapes is horizontal
/// 2. Mirror if needed so first shape is bottom-left
fn rotate_cluster<S>(shapes: &mut [S], cluster: &[usize])
where
    S: DiagramShape + Clone,
{
    if cluster.len() < 2 {
        return;
    }

    let idx0 = cluster[0];
    let idx1 = cluster[1];

    let c0 = shapes[idx0].centroid();
    let c1 = shapes[idx1].centroid();

    // Compute rotation angle to make the line horizontal
    let dx = c1.x() - c0.x();
    let dy = c1.y() - c0.y();
    let theta = -dy.atan2(dx); // Negative because we rotate the other way

    // Rotate all shapes in cluster around first shape's centroid
    if theta.abs() > 1e-10 {
        let pivot = c0;
        for &idx in cluster {
            shapes[idx] = rotate_shape(&shapes[idx], theta, &pivot);
        }
    }

    // Recompute centroids after rotation
    let c0 = shapes[idx0].centroid();
    let _c1 = shapes[idx1].centroid();

    // Compute cluster bounding box for mirroring decisions
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &idx in cluster {
        let c = shapes[idx].centroid();
        x_min = x_min.min(c.x());
        x_max = x_max.max(c.x());
        y_min = y_min.min(c.y());
        y_max = y_max.max(c.y());
    }

    let x_center = (x_min + x_max) / 2.0;
    let y_center = (y_min + y_max) / 2.0;

    // Mirror across y-axis if first shape is right of center
    if c0.x() > x_center {
        for &idx in cluster {
            shapes[idx] = mirror_x_shape(&shapes[idx], x_center);
        }
    }

    // Recompute after potential x-mirror
    let c0 = shapes[idx0].centroid();

    // Mirror across x-axis if first shape is above center
    if c0.y() > y_center {
        for &idx in cluster {
            shapes[idx] = mirror_y_shape(&shapes[idx], y_center);
        }
    }
}

/// Rotate a shape around a pivot point.
fn rotate_shape<S>(shape: &S, theta: f64, pivot: &Point) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    // For most shapes, first two params are (x, y) and last is rotation
    if n >= 2 {
        let x = params[0];
        let y = params[1];

        // Rotate point around pivot
        let dx = x - pivot.x();
        let dy = y - pivot.y();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let new_x = pivot.x() + dx * cos_t - dy * sin_t;
        let new_y = pivot.y() + dx * sin_t + dy * cos_t;

        let mut new_params = params.clone();
        new_params[0] = new_x;
        new_params[1] = new_y;

        // Update rotation parameter if it exists (last parameter). When the
        // cluster rotates by `theta` (CCW), an ellipse's semi-major axis,
        // previously at angle `phi` from world x-axis, now points at
        // `phi + theta`. The orientation parameter must follow the cluster
        // rotation, not oppose it — using `-theta` here was a bug that
        // de-synced ellipse orientations from their (correctly-rotated)
        // centers and silently destroyed near-perfect fits during normalize.
        if n >= 5 {
            new_params[n - 1] = params[n - 1] + theta;
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Mirror a shape across a vertical line x = x_center.
fn mirror_x_shape<S>(shape: &S, x_center: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 1 {
        let mut new_params = params.clone();
        new_params[0] = 2.0 * x_center - params[0]; // Mirror x coordinate

        // For shapes with rotation, also mirror the rotation
        if n >= 5 {
            new_params[n - 1] = PI - params[n - 1]; // Mirror rotation
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Mirror a shape across a horizontal line y = y_center.
fn mirror_y_shape<S>(shape: &S, y_center: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 2 {
        let mut new_params = params.clone();
        new_params[1] = 2.0 * y_center - params[1]; // Mirror y coordinate

        // For shapes with rotation, also mirror the rotation
        if n >= 5 {
            new_params[n - 1] = PI - params[n - 1]; // Mirror rotation
        }

        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Pack disjoint clusters using the skyline algorithm.
fn pack_clusters<S>(shapes: &mut [S], clusters: &[Vec<usize>], padding_factor: f64)
where
    S: DiagramShape + Clone,
{
    if clusters.len() <= 1 {
        return;
    }

    // Compute bounding boxes for each cluster
    let mut cluster_boxes = Vec::new();

    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }

        // Compute overall bounding box for this cluster
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for &idx in cluster {
            let Bounds {
                x_min: bx_min,
                x_max: bx_max,
                y_min: by_min,
                y_max: by_max,
            } = shapes[idx].bounds();
            x_min = x_min.min(bx_min);
            x_max = x_max.max(bx_max);
            y_min = y_min.min(by_min);
            y_max = y_max.max(by_max);
        }

        let width = x_max - x_min;
        let height = y_max - y_min;
        let center = Point::new((x_min + x_max) / 2.0, (y_min + y_max) / 2.0);

        cluster_boxes.push((Rectangle::new(center, width, height), x_min, y_min));
    }

    // Compute padding based on total area (to match new packing algorithm)
    let total_area: f64 = cluster_boxes
        .iter()
        .map(|(r, _, _)| r.width() * r.height())
        .sum();
    let padding = total_area.sqrt() * padding_factor;

    // Pack the bounding boxes
    let rectangles: Vec<Rectangle> = cluster_boxes.iter().map(|(r, _, _)| *r).collect();
    let packed = skyline_pack(&rectangles, padding);

    // Update shape positions based on new bounding box positions
    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.is_empty() {
            continue;
        }

        let (old_box, _old_x_min, _old_y_min) = cluster_boxes[i];
        let new_box = packed[i];

        // Compute translation
        let old_center = old_box.center();
        let new_center = new_box.center();
        let dx = new_center.x() - old_center.x();
        let dy = new_center.y() - old_center.y();

        // Translate all shapes in cluster
        for &idx in cluster {
            shapes[idx] = translate_shape(&shapes[idx], dx, dy);
        }
    }
}

/// Translate a shape by (dx, dy).
fn translate_shape<S>(shape: &S, dx: f64, dy: f64) -> S
where
    S: DiagramShape + Clone,
{
    let params = shape.to_params();
    let n = S::n_params();

    if n >= 2 {
        let mut new_params = params.clone();
        new_params[0] += dx;
        new_params[1] += dy;
        S::from_params(&new_params)
    } else {
        shape.clone()
    }
}

/// Center the entire layout around the origin.
fn center_layout<S>(shapes: &mut [S])
where
    S: DiagramShape + Clone,
{
    if shapes.is_empty() {
        return;
    }

    // Find bounding box of all shapes
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for shape in shapes.iter() {
        let Bounds {
            x_min: bx_min,
            x_max: bx_max,
            y_min: by_min,
            y_max: by_max,
        } = shape.bounds();
        x_min = x_min.min(bx_min);
        x_max = x_max.max(bx_max);
        y_min = y_min.min(by_min);
        y_max = y_max.max(by_max);
    }

    // Compute center
    let center_x = (x_min + x_max) / 2.0;
    let center_y = (y_min + y_max) / 2.0;

    // Translate all shapes to center the layout
    for shape in shapes.iter_mut() {
        *shape = translate_shape(shape, -center_x, -center_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::Circle;
    use crate::geometry::traits::Centroid;
    use crate::geometry::traits::Closed;
    use crate::geometry::traits::DiagramShape;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_center_single_shape() {
        let mut shapes = vec![Circle::new(Point::new(5.0, 3.0), 2.0)];
        center_layout(&mut shapes);

        let centroid = shapes[0].centroid();
        assert!(approx_eq(centroid.x(), 0.0));
        assert!(approx_eq(centroid.y(), 0.0));
    }

    #[test]
    fn test_center_two_shapes() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        center_layout(&mut shapes);

        // Center should be at x=5, so after centering, shapes should be at x=-5 and x=5
        assert!(approx_eq(shapes[0].centroid().x(), -5.0));
        assert!(approx_eq(shapes[1].centroid().x(), 5.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));
        assert!(approx_eq(shapes[1].centroid().y(), 0.0));
    }

    #[test]
    fn test_translate_shape() {
        let shape = Circle::new(Point::new(1.0, 2.0), 3.0);
        let translated = translate_shape(&shape, 4.0, 5.0);

        assert!(approx_eq(translated.centroid().x(), 5.0));
        assert!(approx_eq(translated.centroid().y(), 7.0));
        assert_eq!(translated.radius(), 3.0);
    }

    #[test]
    fn test_normalize_single_shape() {
        let mut shapes = vec![Circle::new(Point::new(5.0, 3.0), 2.0)];
        normalize_layout(&mut shapes, 0.015);

        // Should just be centered
        let centroid = shapes[0].centroid();
        assert!(approx_eq(centroid.x(), 0.0));
        assert!(approx_eq(centroid.y(), 0.0));
    }

    #[test]
    fn test_normalize_two_overlapping() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 2.0), 2.0),
            Circle::new(Point::new(3.0, 2.0), 2.0),
        ];
        normalize_layout(&mut shapes, 0.015);

        // Should be rotated to horizontal and centered
        let c0 = shapes[0].centroid();
        let c1 = shapes[1].centroid();

        // Y coordinates should be equal (horizontal alignment)
        assert!(
            approx_eq(c0.y(), c1.y()),
            "Expected y coords to be equal, got {} and {}",
            c0.y(),
            c1.y()
        );

        // Layout should be centered
        let center_x = (c0.x() + c1.x()) / 2.0;
        let center_y = (c0.y() + c1.y()) / 2.0;
        assert!(approx_eq(center_x, 0.0));
        assert!(approx_eq(center_y, 0.0));

        // First shape should be to the left
        assert!(c0.x() < c1.x());
    }

    #[test]
    fn test_normalize_disjoint_clusters() {
        // Two disjoint pairs
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 1.5),
            Circle::new(Point::new(2.0, 0.0), 1.5),
            Circle::new(Point::new(20.0, 0.0), 1.0),
            Circle::new(Point::new(22.0, 0.0), 1.0),
        ];

        normalize_layout(&mut shapes, 0.015);

        // Both clusters should be rotated and packed together
        let c0 = shapes[0].centroid();
        let c1 = shapes[1].centroid();
        let c2 = shapes[2].centroid();
        let c3 = shapes[3].centroid();

        // First cluster should be horizontally aligned (was already horizontal)
        assert!(
            (c0.y() - c1.y()).abs() < 1e-3,
            "Cluster 1 should be horizontal, got y diff: {}",
            (c0.y() - c1.y()).abs()
        );

        // Clusters should be closer together than original (packed)
        // Original separation was 20 units between cluster centers
        let max_x = c0.x().max(c1.x()).max(c2.x()).max(c3.x());
        let min_x = c0.x().min(c1.x()).min(c2.x()).min(c3.x());
        let max_y = c0.y().max(c1.y()).max(c2.y()).max(c3.y());
        let min_y = c0.y().min(c1.y()).min(c2.y()).min(c3.y());

        let packed_width = max_x - min_x;
        let _packed_height = max_y - min_y;

        // The bounding box should be smaller than if they were 20 units apart horizontally
        let original_bbox_width = 22.0 + 1.0 - (0.0 - 1.5); // rightmost + radius - leftmost + radius

        assert!(
            packed_width < original_bbox_width,
            "Packed width {packed_width} should be less than original width {original_bbox_width}"
        );

        // Verify layout is centered around origin using bounding boxes
        use crate::geometry::traits::BoundingBox;

        let mut bb_x_min = f64::INFINITY;
        let mut bb_x_max = f64::NEG_INFINITY;
        let mut bb_y_min = f64::INFINITY;
        let mut bb_y_max = f64::NEG_INFINITY;

        for shape in &shapes {
            let Bounds {
                x_min: bx_min,
                x_max: bx_max,
                y_min: by_min,
                y_max: by_max,
            } = shape.bounds();
            bb_x_min = bb_x_min.min(bx_min);
            bb_x_max = bb_x_max.max(bx_max);
            bb_y_min = bb_y_min.min(by_min);
            bb_y_max = bb_y_max.max(by_max);
        }

        let bb_center_x = (bb_x_max + bb_x_min) / 2.0;
        let bb_center_y = (bb_y_max + bb_y_min) / 2.0;

        assert!(
            (bb_center_x).abs() < 1e-6,
            "Should be centered in x, got {bb_center_x}"
        );
        assert!(
            (bb_center_y).abs() < 1e-6,
            "Should be centered in y, got {bb_center_y}"
        );
    }

    #[test]
    fn test_normalize_with_container_centres_container() {
        let mut shapes = vec![
            Circle::new(Point::new(11.0, 6.0), 1.0),
            Circle::new(Point::new(13.0, 6.0), 1.0),
        ];
        let mut container = Rectangle::new(Point::new(12.0, 5.0), 6.0, 4.0);

        normalize_layout_with_container(&mut shapes, &mut container);

        // Container is centred at origin; size preserved.
        assert!(approx_eq(container.center().x(), 0.0));
        assert!(approx_eq(container.center().y(), 0.0));
        assert!(approx_eq(container.width(), 6.0));
        assert!(approx_eq(container.height(), 4.0));

        // The diagram (bounding box of the two circles) is centred in the
        // container: its bbox center lands at the origin. Both circles fit
        // inside the container, so centering is loss-preserving here.
        let bb_cx = (shapes[0].centroid().x() + shapes[1].centroid().x()) / 2.0;
        let bb_cy = shapes[0].centroid().y(); // both circles share a y, r equal
        assert!(approx_eq(bb_cx, 0.0), "diagram not x-centred: {bb_cx}");
        assert!(approx_eq(bb_cy, 0.0), "diagram not y-centred: {bb_cy}");
        // Horizontal separation is preserved (pure translation).
        assert!(approx_eq(shapes[0].centroid().x(), -1.0));
        assert!(approx_eq(shapes[1].centroid().x(), 1.0));
    }

    #[test]
    fn test_normalize_with_container_centres_offset_diagram() {
        // Diagram parked in the lower-left corner of the container; both
        // circles fit inside, so it should be slid to the container center.
        let mut shapes = vec![Circle::new(Point::new(-2.0, -2.0), 1.0)];
        let mut container = Rectangle::new(Point::new(0.0, 0.0), 10.0, 10.0);

        normalize_layout_with_container(&mut shapes, &mut container);

        assert!(approx_eq(shapes[0].centroid().x(), 0.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));
        assert!(approx_eq(container.center().x(), 0.0));
        assert!(approx_eq(container.center().y(), 0.0));
    }

    #[test]
    fn test_normalize_with_container_keeps_offset_when_shape_pokes_out() {
        // The circle (radius 2) extends past the right edge of the 4-wide
        // container, so its clipped area depends on position. Centering must be
        // skipped — only the container is parked at the origin.
        let mut shapes = vec![Circle::new(Point::new(3.0, 0.0), 2.0)];
        let mut container = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);

        normalize_layout_with_container(&mut shapes, &mut container);

        // Step 1 translated everything by (-2, 0); step 2 was skipped, so the
        // circle keeps its optimiser-chosen offset relative to the container.
        assert!(approx_eq(shapes[0].centroid().x(), 1.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));
        assert!(approx_eq(container.center().x(), 0.0));
        assert!(approx_eq(container.center().y(), 0.0));
    }

    #[test]
    fn test_normalize_with_container_idempotent() {
        // Container at origin, single circle already centred in it — a fixed
        // point of normalisation: re-running must not move anything.
        let mut shapes = vec![Circle::new(Point::new(0.0, 0.0), 1.0)];
        let mut container = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);

        normalize_layout_with_container(&mut shapes, &mut container);
        assert!(approx_eq(shapes[0].centroid().x(), 0.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));

        // Second pass is a no-op.
        normalize_layout_with_container(&mut shapes, &mut container);
        assert!(approx_eq(shapes[0].centroid().x(), 0.0));
        assert!(approx_eq(shapes[0].centroid().y(), 0.0));
        assert!(approx_eq(container.width(), 4.0));
        assert!(approx_eq(container.height(), 3.0));
        assert!(approx_eq(container.center().x(), 0.0));
        assert!(approx_eq(container.center().y(), 0.0));
    }

    /// Distance between two shape centroids.
    fn centroid_dist<S: DiagramShape>(a: &S, b: &S) -> f64 {
        let (ca, cb) = (a.centroid(), b.centroid());
        ((ca.x() - cb.x()).powi(2) + (ca.y() - cb.y()).powi(2)).sqrt()
    }

    /// Assert two exclusive-region maps are equal within tolerance.
    fn assert_regions_eq(before: &HashMap<RegionMask, f64>, after: &HashMap<RegionMask, f64>) {
        let mut masks: Vec<_> = before.keys().chain(after.keys()).copied().collect();
        masks.sort_unstable();
        masks.dedup();
        for mask in masks {
            let lhs = before.get(&mask).copied().unwrap_or(0.0);
            let rhs = after.get(&mask).copied().unwrap_or(0.0);
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                (lhs - rhs).abs() <= 1e-8_f64.max(1e-6 * scale),
                "region {mask} changed: {lhs} -> {rhs}"
            );
        }
    }

    #[test]
    fn test_contained_child_offset_moderate() {
        // Small circle (r=1) parked off-center inside a big one (R=5). Internal
        // tangency distance is R-r=4, so it should land at 0.5*4 = 2 from the
        // parent center, along the existing +x direction: clearly off-center,
        // comfortably inside (2+1 < 5), never tangent.
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 5.0),
            Circle::new(Point::new(1.0, 0.0), 1.0),
        ];
        offset_contained_children(&mut shapes, None);

        assert!(approx_eq(shapes[1].centroid().x(), 2.0));
        assert!(approx_eq(shapes[1].centroid().y(), 0.0));
        let d = centroid_dist(&shapes[0], &shapes[1]);
        assert!(d > 1e-3, "should not be concentric, got {d}");
        assert!(
            shapes[0].contains(&shapes[1]),
            "child must stay inside parent"
        );
    }

    #[test]
    fn test_contained_child_offset_preserves_regions() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 5.0),
            Circle::new(Point::new(-1.5, 0.7), 1.0),
        ];
        let before = Circle::compute_exclusive_regions(&shapes);
        offset_contained_children(&mut shapes, None);
        let after = Circle::compute_exclusive_regions(&shapes);
        assert_regions_eq(&before, &after);
        // And it genuinely moved (off-center by 0.5*(R-r)=2).
        assert!(approx_eq(centroid_dist(&shapes[0], &shapes[1]), 2.0));
    }

    #[test]
    fn test_contained_child_near_concentric_fallback() {
        // Child exactly at the parent center: the parent->child direction is
        // pure noise, so it's pushed out along the canonical +x fallback.
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 5.0),
            Circle::new(Point::new(0.0, 0.0), 1.0),
        ];
        offset_contained_children(&mut shapes, None);
        assert!(approx_eq(shapes[1].centroid().x(), 2.0));
        assert!(approx_eq(shapes[1].centroid().y(), 0.0));
    }

    #[test]
    fn test_two_disjoint_children_not_moved() {
        // Parent with two disjoint children: the constrained-packing case we
        // defer. Neither child is moved (moving one could overlap the other).
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 10.0),
            Circle::new(Point::new(-3.0, 0.0), 1.0),
            Circle::new(Point::new(3.0, 0.0), 1.0),
        ];
        let snapshot = shapes.clone();
        offset_contained_children(&mut shapes, None);
        for (a, b) in shapes.iter().zip(&snapshot) {
            assert!(approx_eq(a.centroid().x(), b.centroid().x()));
            assert!(approx_eq(a.centroid().y(), b.centroid().y()));
        }
    }

    #[test]
    fn test_nested_child_not_moved() {
        // Russian-doll: C is inside both B and A (two containers), and B
        // contains C so B is not a leaf. Neither is repositioned in v1.
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 5.0), // A
            Circle::new(Point::new(0.5, 0.0), 2.5), // B inside A
            Circle::new(Point::new(0.6, 0.0), 0.5), // C inside B and A
        ];
        let snapshot = shapes.clone();
        offset_contained_children(&mut shapes, None);
        for (a, b) in shapes.iter().zip(&snapshot) {
            assert!(approx_eq(a.centroid().x(), b.centroid().x()));
            assert!(approx_eq(a.centroid().y(), b.centroid().y()));
        }
    }

    #[test]
    fn test_coincident_equal_sets_not_moved() {
        // Two identical coincident circles (the `two_overlapping_completely`
        // corpus case: A=0, B=0, A&B=area). Each looks like the other's free
        // child by the solo-region test, but neither is strictly contained —
        // pushing them apart would destroy the complete overlap. Both must stay.
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 2.0),
            Circle::new(Point::new(0.0, 0.0), 2.0),
        ];
        let before = Circle::compute_exclusive_regions(&shapes);
        offset_contained_children(&mut shapes, None);
        let after = Circle::compute_exclusive_regions(&shapes);
        assert_regions_eq(&before, &after);
        for s in &shapes {
            assert!(approx_eq(s.centroid().x(), 0.0));
            assert!(approx_eq(s.centroid().y(), 0.0));
        }
    }

    #[test]
    fn test_plain_overlap_unaffected() {
        // Two partially overlapping circles, neither contained: nothing moves.
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 0.0), 2.0),
            Circle::new(Point::new(2.0, 0.0), 2.0),
        ];
        let snapshot = shapes.clone();
        offset_contained_children(&mut shapes, None);
        for (a, b) in shapes.iter().zip(&snapshot) {
            assert!(approx_eq(a.centroid().x(), b.centroid().x()));
            assert!(approx_eq(a.centroid().y(), b.centroid().y()));
        }
    }

    #[test]
    fn test_normalize_preserves_exclusive_regions() {
        let mut shapes = vec![
            Circle::new(Point::new(0.0, 1.0), 3.0),
            Circle::new(Point::new(2.5, -0.5), 2.5),
            Circle::new(Point::new(12.0, 8.0), 1.5),
        ];

        let before = Circle::compute_exclusive_regions(&shapes);
        normalize_layout(&mut shapes, 0.015);
        let after = Circle::compute_exclusive_regions(&shapes);

        let mut all_masks: Vec<_> = before.keys().chain(after.keys()).copied().collect();
        all_masks.sort_unstable();
        all_masks.dedup();

        for mask in all_masks {
            let lhs = before.get(&mask).copied().unwrap_or(0.0);
            let rhs = after.get(&mask).copied().unwrap_or(0.0);
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                (lhs - rhs).abs() <= 1e-8_f64.max(1e-6 * scale),
                "mask {mask:b}: before={lhs:e}, after={rhs:e}"
            );
        }
    }
}
