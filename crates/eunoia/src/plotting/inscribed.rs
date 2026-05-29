//! Hole-aware inscribed-rectangle and principal-axis primitives for label
//! placement.
//!
//! Renderers usually want more than a single label point: they need to know
//! whether a region can fit a label rectangle of a given aspect ratio, and
//! which way the region is oriented. These two helpers expose that geometry
//! without committing to font metrics, viewport size, or leader-line
//! routing — those concerns stay in the renderer.
//!
//! Both helpers consume the [`RegionPiece`] hole-aware representation
//! produced by [`crate::plotting::decompose_regions`] (or
//! [`crate::Layout::region_polygons`]).

use crate::geometry::primitives::Point;
use crate::geometry::shapes::{Polygon, Rectangle};
use crate::plotting::regions::{RegionPiece, poi_with_holes, point_in_polygon, signed_clearance};

/// Best-effort largest axis-aligned rectangle of the given `aspect_ratio`
/// (width / height) inscribed in the union of `pieces` (each piece's outer
/// boundary minus its holes). Returns the rectangle plus a fit score in
/// `[0, 1]`.
///
/// # Aspect ratio
///
/// `aspect_ratio = 1.0` is a square; `aspect_ratio = 2.0` is "twice as wide
/// as tall". Values `<= 0.0` return `None`.
///
/// # Algorithm (directional clearance, POI-centred)
///
/// The rectangle is centred at the region's hole-aware pole of
/// inaccessibility (POI) — the same anchor every other label-position in
/// the crate uses, including [`crate::plotting::PlotData::region_anchors`].
/// With the centre fixed, the rectangle's short side `s` is found by
/// binary search up to `precision`: at each step a hole-aware containment
/// predicate decides whether an axis-aligned `a·s × s` rectangle (with
/// `a = aspect_ratio`) fits inside the winning piece's outer boundary and
/// outside every hole. The search is monotone (fits at `s` ⇒ fits at any
/// smaller `s`) so bisection is sound.
///
/// The search is seeded with the radial-disc result
/// (`s = 2r/√(1+a²)` from the POI's signed clearance `r`) as its
/// guaranteed-fitting lower bound, so the returned rectangle is never
/// smaller than the inscribed-disc bound — and is typically much larger
/// in anisotropic regions (thin lenses, wide-and-short slabs, crescents).
///
/// # Fit score
///
/// `score = (achieved short side) / min(outer_bbox_width, outer_bbox_height)`
/// of the *winning piece* (the piece whose interior contains the centre),
/// clamped to `[0, 1]`. A value near `1.0` means the rectangle saturates
/// the region's short dimension; a value near `0.0` means the region
/// won't comfortably hold a rectangular label.
///
/// # Returns
///
/// `None` when `aspect_ratio <= 0.0`, `pieces` is empty, or every piece
/// yields zero or negative signed clearance (degenerate input).
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::shapes::Polygon;
/// use eunoia::plotting::{largest_inscribed_rect, RegionPiece};
///
/// let outer = Polygon::new(vec![
///     Point::new(0.0, 0.0),
///     Point::new(10.0, 0.0),
///     Point::new(10.0, 10.0),
///     Point::new(0.0, 10.0),
/// ]);
/// let pieces = vec![RegionPiece { outer, holes: vec![] }];
/// let (rect, score) = largest_inscribed_rect(&pieces, 1.0, 0.01).unwrap();
/// assert!(score > 0.95); // directional bound ≈ 1.0 for the bbox-aligned square
/// assert!(rect.width() > 0.0 && rect.height() > 0.0);
/// ```
pub fn largest_inscribed_rect(
    pieces: &[RegionPiece],
    aspect_ratio: f64,
    precision: f64,
) -> Option<(Rectangle, f64)> {
    if aspect_ratio <= 0.0 || pieces.is_empty() {
        return None;
    }

    let (centre, r) = poi_with_holes(pieces, precision)?;
    if r <= 0.0 {
        return None;
    }

    // Identify the winning piece — the one whose interior contains the POI
    // centre. Pieces are disjoint by construction, so at most one matches;
    // if numerical noise leaves several with positive clearance we pick the
    // one with the largest.
    let mut winning: Option<&RegionPiece> = None;
    let mut best_d = 0.0;
    for piece in pieces {
        let d = signed_clearance(centre.x(), centre.y(), piece);
        if d > best_d {
            best_d = d;
            winning = Some(piece);
        }
    }
    let winning = winning?;

    // Outer-bbox of the winning piece, used both for the score denominator
    // and as the upper cap on the binary-search range (no rect can exceed
    // the bbox of the polygon it's inscribed in).
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in winning.outer.vertices() {
        min_x = min_x.min(p.x());
        max_x = max_x.max(p.x());
        min_y = min_y.min(p.y());
        max_y = max_y.max(p.y());
    }
    let bbox_w = max_x - min_x;
    let bbox_h = max_y - min_y;
    let bbox_short = bbox_w.min(bbox_h);
    if bbox_short <= 0.0 {
        return None;
    }

    // Binary search the short side `s`. With aspect `a = aspect_ratio`,
    // width = a·s and height = s, so the bbox caps `s` at
    // `min(bbox_h, bbox_w / a)`.
    let cx = centre.x();
    let cy = centre.y();
    let a = aspect_ratio;
    let inv = 1.0 / (1.0 + a * a).sqrt();
    let s_radial = 2.0 * r * inv;
    let s_cap = bbox_h.min(bbox_w / a);
    let mut lo = s_radial.min(s_cap);
    let mut hi = s_cap;
    let precision_clamped = precision.max(1e-12);
    let mut iters = 0;
    while hi - lo > precision_clamped && iters < 64 {
        let mid = 0.5 * (lo + hi);
        let hw = 0.5 * a * mid;
        let hh = 0.5 * mid;
        if rect_fits_in_piece(winning, cx, cy, hw, hh) {
            lo = mid;
        } else {
            hi = mid;
        }
        iters += 1;
    }
    let s = lo;
    let half_w = 0.5 * a * s;
    let half_h = 0.5 * s;
    let rect = Rectangle::new(centre, 2.0 * half_w, 2.0 * half_h);

    let achieved_short = (2.0 * half_w).min(2.0 * half_h);
    let score = (achieved_short / bbox_short).clamp(0.0, 1.0);

    Some((rect, score))
}

/// Hole-aware test: does the axis-aligned rectangle centred at `(cx, cy)`
/// with half-extents `(hw, hh)` fit entirely inside `piece` (inside the
/// outer ring and outside every hole)?
///
/// Combines four cheap checks, any failing returns false:
///
/// 1. All 4 corners lie inside `piece.outer`.
/// 2. No corner lies inside any hole.
/// 3. No outer-ring edge crosses any of the rect's 4 axis-aligned edges.
/// 4. No hole-ring edge crosses any rect edge.
/// 5. No hole lies entirely inside the rect (any single hole vertex inside
///    the rect AABB while no hole edge crossed any rect edge ⇒ contained).
///
/// Edge-vs-axis-aligned-edge intersection uses strict inequalities so
/// touching the boundary from inside counts as fitting (which is what a
/// label inscribed up to the boundary should be).
fn rect_fits_in_piece(piece: &RegionPiece, cx: f64, cy: f64, hw: f64, hh: f64) -> bool {
    if hw <= 0.0 || hh <= 0.0 {
        return false;
    }
    let xl = cx - hw;
    let xr = cx + hw;
    let yb = cy - hh;
    let yt = cy + hh;
    let corners = [
        Point::new(xl, yb),
        Point::new(xr, yb),
        Point::new(xr, yt),
        Point::new(xl, yt),
    ];
    for c in &corners {
        if !point_in_polygon(c, &piece.outer) {
            return false;
        }
    }
    for hole in &piece.holes {
        for c in &corners {
            if point_in_polygon(c, hole) {
                return false;
            }
        }
    }
    if ring_crosses_rect(piece.outer.vertices(), xl, xr, yb, yt) {
        return false;
    }
    for hole in &piece.holes {
        if ring_crosses_rect(hole.vertices(), xl, xr, yb, yt) {
            return false;
        }
        // Step 5: if a hole's edges never cross the rect (caught above) and
        // no rect corner sits inside the hole (caught above), the hole is
        // either entirely inside the rect or entirely outside. A single hole
        // vertex inside the rect AABB tells us which.
        if let Some(p) = hole.vertices().first()
            && p.x() > xl
            && p.x() < xr
            && p.y() > yb
            && p.y() < yt
        {
            return false;
        }
    }
    true
}

/// Does any segment of `verts` (treated as a closed ring) cross any of the
/// 4 axis-aligned edges of the rectangle `[xl, xr] × [yb, yt]`?
///
/// For each ring segment `(a, b)` and each rect edge, we use the
/// axis-aligned specialisation of segment-vs-segment intersection: a
/// segment crosses a horizontal line `y = y0` iff `(a.y - y0)` and
/// `(b.y - y0)` have strictly opposite signs, and the x-intercept lies
/// strictly inside `[xl, xr]`. Strict inequalities mean a touching ring
/// (endpoint on a rect edge) is not flagged — touching from inside is the
/// boundary case for a strictly-inscribed rectangle.
fn ring_crosses_rect(verts: &[Point], xl: f64, xr: f64, yb: f64, yt: f64) -> bool {
    let n = verts.len();
    if n < 2 {
        return false;
    }
    for i in 0..n {
        let a = &verts[i];
        let b = &verts[(i + 1) % n];
        let (ax, ay) = (a.x(), a.y());
        let (bx, by) = (b.x(), b.y());

        // Bottom edge y = yb.
        if (ay - yb) * (by - yb) < 0.0 {
            let x = ax + (yb - ay) * (bx - ax) / (by - ay);
            if x > xl && x < xr {
                return true;
            }
        }
        // Top edge y = yt.
        if (ay - yt) * (by - yt) < 0.0 {
            let x = ax + (yt - ay) * (bx - ax) / (by - ay);
            if x > xl && x < xr {
                return true;
            }
        }
        // Left edge x = xl.
        if (ax - xl) * (bx - xl) < 0.0 {
            let y = ay + (xl - ax) * (by - ay) / (bx - ax);
            if y > yb && y < yt {
                return true;
            }
        }
        // Right edge x = xr.
        if (ax - xr) * (bx - xr) < 0.0 {
            let y = ay + (xr - ax) * (by - ay) / (bx - ax);
            if y > yb && y < yt {
                return true;
            }
        }
    }
    false
}

/// Principal axis (radians, in `[-π/2, π/2]`) and elongation
/// (`sqrt(λ_max / λ_min)` of the area-weighted central second-moment
/// matrix) of a region piece.
///
/// Holes contribute negative-weight moments via the CW orientation contract
/// on [`RegionPiece`], so summing all rings yields net moments that
/// correctly reflect the piece's shape.
///
/// For isotropic or degenerate input (circle polygon, equilateral triangle,
/// near-zero net area) the function returns `(0.0, 1.0)`. **Callers should
/// gate on elongation before relying on the angle**: when elongation is
/// close to `1.0` the principal direction is mathematically undefined and
/// the returned angle carries no information.
pub fn principal_axis(piece: &RegionPiece) -> (f64, f64) {
    /// Raw moments `(A, M_x, M_y, M_xx, M_yy, M_xy)` of a single ring,
    /// signed by winding (CCW positive, CW negative). Closed-form Green's
    /// theorem expansion.
    fn moments(ring: &Polygon) -> (f64, f64, f64, f64, f64, f64) {
        let v = ring.vertices();
        let n = v.len();
        if n < 3 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let mut a2 = 0.0; // 2 * A
        let mut mx6 = 0.0; // 6 * M_x
        let mut my6 = 0.0; // 6 * M_y
        let mut mxx12 = 0.0; // 12 * M_xx
        let mut myy12 = 0.0; // 12 * M_yy
        let mut mxy24 = 0.0; // 24 * M_xy
        for i in 0..n {
            let j = (i + 1) % n;
            let (xi, yi) = (v[i].x(), v[i].y());
            let (xj, yj) = (v[j].x(), v[j].y());
            let cross = xi * yj - xj * yi;
            a2 += cross;
            mx6 += (xi + xj) * cross;
            my6 += (yi + yj) * cross;
            mxx12 += (xi * xi + xi * xj + xj * xj) * cross;
            myy12 += (yi * yi + yi * yj + yj * yj) * cross;
            mxy24 += (xi * yj + 2.0 * xi * yi + 2.0 * xj * yj + xj * yi) * cross;
        }
        (
            a2 / 2.0,
            mx6 / 6.0,
            my6 / 6.0,
            mxx12 / 12.0,
            myy12 / 12.0,
            mxy24 / 24.0,
        )
    }

    let (mut a, mut mx, mut my, mut mxx, mut myy, mut mxy) = moments(&piece.outer);
    for h in &piece.holes {
        let (a_h, mx_h, my_h, mxx_h, myy_h, mxy_h) = moments(h);
        a += a_h;
        mx += mx_h;
        my += my_h;
        mxx += mxx_h;
        myy += myy_h;
        mxy += mxy_h;
    }

    if a <= 1e-12 {
        return (0.0, 1.0);
    }

    let cx = mx / a;
    let cy = my / a;

    // Central second moments, normalised by area so the eigenvalues have
    // units of length^2 — eigenvalue ratios then directly give elongation^2.
    let mu_xx = mxx / a - cx * cx;
    let mu_yy = myy / a - cy * cy;
    let mu_xy = mxy / a - cx * cy;

    let trace = mu_xx + mu_yy;
    let disc = ((mu_xx - mu_yy).powi(2) + 4.0 * mu_xy * mu_xy)
        .max(0.0)
        .sqrt();
    let lambda_max = 0.5 * (trace + disc);
    let lambda_min = 0.5 * (trace - disc);

    if lambda_max <= 0.0 || lambda_min <= 0.0 {
        return (0.0, 1.0);
    }

    let elongation = (lambda_max / lambda_min).sqrt();
    let angle = 0.5 * (2.0 * mu_xy).atan2(mu_xx - mu_yy);
    (angle, elongation)
}

/// Tries to inscribe an axis-aligned `w × h` rectangle inside `pieces` and
/// returns its centre on success, `None` on failure.
///
/// This is a thin predicate over [`largest_inscribed_rect`]: we ask for the
/// largest rectangle of aspect ratio `w / h` and accept the answer when its
/// width (and therefore, given the preserved aspect ratio, its height) meets
/// the request.
///
/// # Anchor convention
///
/// The returned point is the **centre** of the inscribed rectangle, matching
/// every other label-anchor in the crate (e.g.
/// [`crate::plotting::regions::RegionPolygons::label_points`]). Renderers using SVG
/// `text-anchor="middle" dominant-baseline="central"` or grid
/// `gpar(hjust=0.5, vjust=0.5)` can use the point directly without offsetting
/// by the box dimensions.
///
/// # Failure modes
///
/// Returns `None` when:
/// * `w` or `h` is non-positive or non-finite.
/// * `pieces` is empty or every piece has non-positive signed clearance
///   (degenerate input).
/// * The largest inscribed rectangle of aspect `w / h` is strictly smaller
///   than the request.
///
/// # Caveat (POI-centred)
///
/// [`largest_inscribed_rect`] keeps the rectangle centre fixed at the
/// region's POI and grows directionally up to the configured `precision`,
/// so the predicate is sound (a `Some` rectangle is always strictly
/// inscribed) and tight for that centre. Callers wanting a non-POI anchor
/// — e.g. shifting toward one end of a strongly anisotropic region —
/// should compose their own centre search on top of `largest_inscribed_rect`.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Point;
/// use eunoia::geometry::shapes::Polygon;
/// use eunoia::plotting::{fit_label_in_region, RegionPiece};
///
/// let outer = Polygon::new(vec![
///     Point::new(0.0, 0.0),
///     Point::new(10.0, 0.0),
///     Point::new(10.0, 10.0),
///     Point::new(0.0, 10.0),
/// ]);
/// let pieces = vec![RegionPiece { outer, holes: vec![] }];
///
/// // A 2×1 label fits comfortably inside a 10×10 region.
/// assert!(fit_label_in_region(&pieces, 2.0, 1.0, 0.01).is_some());
///
/// // An 11×1 label is wider than the region's bounding box.
/// assert!(fit_label_in_region(&pieces, 11.0, 1.0, 0.01).is_none());
/// ```
pub fn fit_label_in_region(
    pieces: &[RegionPiece],
    w: f64,
    h: f64,
    precision: f64,
) -> Option<Point> {
    if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (rect, _score) = largest_inscribed_rect(pieces, w / h, precision)?;
    // Aspect is preserved by `largest_inscribed_rect`, so checking width
    // suffices; we still check height for FP-safety against the tiny
    // rounding wobble in `1 / sqrt(1 + a^2)`.
    let eps = 1e-9 * w.max(h);
    if rect.width() + eps >= w && rect.height() + eps >= h {
        Some(*rect.center())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::{Circle, Ellipse, Polygon};
    use crate::geometry::traits::Polygonize;

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

    #[test]
    fn test_inscribed_square_aspect_1() {
        // 10×10 square, aspect = 1. Directional clearance from the POI at
        // (5, 5) finds the full 10×10 fit (limited by the bbox cap, not
        // the disc radius). Result tolerated to `precision` (0.01) below.
        let pieces = vec![axis_aligned_square_piece(10.0)];
        let (rect, score) = largest_inscribed_rect(&pieces, 1.0, 0.01).unwrap();
        assert!(
            (rect.width() - 10.0).abs() < 0.05,
            "width = {}",
            rect.width()
        );
        assert!(
            (rect.height() - 10.0).abs() < 0.05,
            "height = {}",
            rect.height()
        );
        assert!(score > 0.99, "score = {}", score);
        assert!((rect.center().x() - 5.0).abs() < 0.2);
        assert!((rect.center().y() - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_inscribed_square_aspect_2() {
        // 10×10 square, aspect = 2. Width is bbox-capped at 10, so the
        // rectangle is 10×5 (height = width/aspect). The radial bound used
        // to give ≈ 8.94×4.47; the directional bound saturates the bbox.
        let pieces = vec![axis_aligned_square_piece(10.0)];
        let (rect, score) = largest_inscribed_rect(&pieces, 2.0, 0.01).unwrap();
        assert!(
            (rect.width() - 10.0).abs() < 0.05,
            "width = {}",
            rect.width()
        );
        assert!(
            (rect.height() - 5.0).abs() < 0.05,
            "height = {}",
            rect.height()
        );
        // Score = achieved short side / bbox short side; both bbox dims are 10.
        assert!((score - 0.5).abs() < 0.02, "score = {}", score);
    }

    #[test]
    fn test_inscribed_l_shape() {
        // Same L-shape as `polygon::test_pole_of_inaccessibility_l_shape`.
        let l_shape = RegionPiece {
            outer: Polygon::new(vec![
                Point::new(0.0, 0.0),
                Point::new(4.0, 0.0),
                Point::new(4.0, 1.0),
                Point::new(1.0, 1.0),
                Point::new(1.0, 4.0),
                Point::new(0.0, 4.0),
            ]),
            holes: vec![],
        };
        let pieces = vec![l_shape];
        let (rect, score) = largest_inscribed_rect(&pieces, 1.0, 0.01).unwrap();
        assert!(rect.width() > 0.0 && rect.height() > 0.0);
        // POI of the L sits in the corner arm near (~0.5, ~0.5) with
        // clearance ~0.5. Directional growth fills the arm thickness
        // (~1×1 square) inside a 4×4 bbox — score ≈ 0.25.
        assert!(rect.center().x() < 1.5);
        assert!(rect.center().y() < 1.5);
        assert!(score > 0.2, "score = {}", score);
        // The rectangle is bounded by the arm width (1.0) — neither
        // dimension can exceed the arm thickness.
        assert!(rect.width() <= 1.05, "width = {}", rect.width());
        assert!(rect.height() <= 1.05, "height = {}", rect.height());
    }

    #[test]
    fn test_inscribed_with_hole() {
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        // 2×2 hole centred at (5, 5), CW per the RegionPiece contract.
        let hole = Polygon::new(vec![
            Point::new(4.0, 4.0),
            Point::new(4.0, 6.0),
            Point::new(6.0, 6.0),
            Point::new(6.0, 4.0),
        ]);
        let piece = RegionPiece {
            outer,
            holes: vec![hole],
        };

        let baseline = vec![axis_aligned_square_piece(10.0)];
        let (_, baseline_score) = largest_inscribed_rect(&baseline, 1.0, 0.01).unwrap();

        let pieces = vec![piece];
        let (rect, score) = largest_inscribed_rect(&pieces, 1.0, 0.01).unwrap();

        let cx = rect.center().x();
        let cy = rect.center().y();
        assert!(
            !(4.0..=6.0).contains(&cx) || !(4.0..=6.0).contains(&cy),
            "centre ({}, {}) lies inside the hole",
            cx,
            cy
        );
        // Hole carves out clearance, so the rectangle and score shrink
        // relative to the unholed baseline (which saturates at ≈1.0 under
        // the directional bound).
        assert!(
            score < baseline_score - 0.1,
            "score {} not lower than baseline {}",
            score,
            baseline_score
        );
        // Rect must not overlap the hole — every corner strictly outside
        // the open hole interior (4, 6)². Touching the hole boundary at a
        // corner is acceptable (the rectangle is still strictly inscribed).
        let half_w = rect.width() * 0.5;
        let half_h = rect.height() * 0.5;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let x = cx + dx * half_w;
            let y = cy + dy * half_h;
            let strictly_in_hole = x > 4.0 && x < 6.0 && y > 4.0 && y < 6.0;
            assert!(!strictly_in_hole, "corner ({}, {}) overlaps hole", x, y);
        }
    }

    #[test]
    fn test_inscribed_wide_and_short_anisotropic() {
        // 10×2 outer, aspect = 5. Radial bound would give a tiny rect
        // (inscribed disc radius = 1 → ~1.96×0.39); the directional bound
        // saturates the bbox to 10×2.
        let piece = RegionPiece {
            outer: Polygon::new(vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 2.0),
                Point::new(0.0, 2.0),
            ]),
            holes: vec![],
        };
        let pieces = vec![piece];
        let (rect, _score) = largest_inscribed_rect(&pieces, 5.0, 0.01).unwrap();
        assert!(
            (rect.width() - 10.0).abs() < 0.05,
            "width = {}",
            rect.width()
        );
        assert!(
            (rect.height() - 2.0).abs() < 0.05,
            "height = {}",
            rect.height()
        );
    }

    #[test]
    fn test_inscribed_concave_notch_outer() {
        // 10×10 outer with a triangular notch on the right side. The
        // notch makes the region non-convex; the directional solver must
        // respect outer-edge crossings (not just hole boundaries) so the
        // returned rectangle's corners all stay inside the polygon and no
        // outer-edge slices through it.
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 4.0),
            Point::new(3.0, 5.0),
            Point::new(10.0, 6.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let piece = RegionPiece {
            outer: outer.clone(),
            holes: vec![],
        };
        let pieces = vec![piece];
        let (rect, _score) = largest_inscribed_rect(&pieces, 1.0, 0.01).unwrap();
        // Soundness: all 4 corners lie inside the outer polygon — this is
        // the guarantee the outer-edge crossing check exists to enforce.
        let cx = rect.center().x();
        let cy = rect.center().y();
        let half_w = rect.width() * 0.5;
        let half_h = rect.height() * 0.5;
        // Probe just inside each corner so point_in_polygon isn't on the
        // boundary FP-ambiguous knife edge.
        let inset = 1e-6;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let x = cx + dx * (half_w - inset);
            let y = cy + dy * (half_h - inset);
            assert!(
                crate::plotting::regions::point_in_polygon(&Point::new(x, y), &outer),
                "near-corner ({}, {}) escaped outer polygon",
                x,
                y
            );
        }
    }

    #[test]
    fn test_inscribed_thin_triangle() {
        let piece = RegionPiece {
            outer: Polygon::new(vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(5.0, 0.1),
            ]),
            holes: vec![],
        };
        let pieces = vec![piece];
        // Either Some with a tiny rectangle, or None at this precision —
        // both are documented as legal degenerate behaviour. The score is
        // relative to the bbox short side, so a well-inscribed disc gives
        // a score near the radial bound (~0.707) even when the absolute
        // size is tiny — the assertion targets absolute size, not score.
        if let Some((rect, _score)) = largest_inscribed_rect(&pieces, 1.0, 0.001) {
            assert!(rect.height() < 0.1, "height = {}", rect.height());
            assert!(rect.width() < 0.1, "width = {}", rect.width());
        }
    }

    #[test]
    fn test_inscribed_zero_aspect_returns_none() {
        let pieces = vec![axis_aligned_square_piece(10.0)];
        assert!(largest_inscribed_rect(&pieces, 0.0, 0.01).is_none());
        assert!(largest_inscribed_rect(&pieces, -1.0, 0.01).is_none());
    }

    #[test]
    fn test_inscribed_empty_pieces_returns_none() {
        let pieces: Vec<RegionPiece> = vec![];
        assert!(largest_inscribed_rect(&pieces, 1.0, 0.01).is_none());
    }

    #[test]
    fn test_principal_axis_circle_polygon() {
        let circle = Circle::new(Point::new(3.0, 4.0), 5.0);
        let polygon = circle.polygonize(64);
        let piece = RegionPiece {
            outer: polygon,
            holes: vec![],
        };
        let (_angle, elongation) = principal_axis(&piece);
        // Angle is mathematically undefined for an isotropic shape — only
        // pin elongation. A 64-vertex circle has very small numerical
        // anisotropy.
        assert!(
            (elongation - 1.0).abs() < 0.05,
            "elongation = {}",
            elongation
        );
    }

    #[test]
    fn test_principal_axis_rotated_ellipse() {
        // Semi-axes 4, 1 → expected elongation ≈ 4. Rotation θ = 30°
        // → expected angle ≈ π/6 (modulo π — principal axes are lines).
        let theta = std::f64::consts::PI / 6.0;
        let ellipse = Ellipse::new(Point::new(0.0, 0.0), 4.0, 1.0, theta);
        let polygon = ellipse.polygonize(128);
        let piece = RegionPiece {
            outer: polygon,
            holes: vec![],
        };
        let (angle, elongation) = principal_axis(&piece);
        // Wrap to compare modulo π.
        let mut diff = (angle - theta).abs();
        if diff > std::f64::consts::FRAC_PI_2 {
            diff = std::f64::consts::PI - diff;
        }
        assert!(diff < 0.05, "angle = {}, expected ≈ {}", angle, theta);
        assert!(elongation > 3.5, "elongation = {}", elongation);
    }

    #[test]
    fn test_principal_axis_with_hole() {
        // 10×10 outer, 1×1 CW hole at (1, 1)-(2, 2). The hole is off-centre,
        // so it breaks rotational symmetry; elongation rises strictly above 1.
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let hole = Polygon::new(vec![
            Point::new(1.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 2.0),
            Point::new(2.0, 1.0),
        ]);
        let piece = RegionPiece {
            outer,
            holes: vec![hole],
        };
        let (_angle, elongation) = principal_axis(&piece);
        assert!(elongation > 1.0, "elongation = {}", elongation);
    }

    #[test]
    fn test_fit_label_comfortably_fits_square() {
        // 10×10 region, label (2, 1). Aspect 2 → max-inscribed (8.94, 4.47),
        // both ≫ request, so the predicate returns the centre near (5, 5).
        let pieces = vec![axis_aligned_square_piece(10.0)];
        let anchor = fit_label_in_region(&pieces, 2.0, 1.0, 0.01).unwrap();
        assert!((anchor.x() - 5.0).abs() < 0.2, "x = {}", anchor.x());
        assert!((anchor.y() - 5.0).abs() < 0.2, "y = {}", anchor.y());
    }

    #[test]
    fn test_fit_label_too_wide_returns_none() {
        let pieces = vec![axis_aligned_square_piece(10.0)];
        // 11.0 > region's bbox short side (10.0); even at aspect 11 the
        // radial-conservative fit can't span 11 units of width.
        assert!(fit_label_in_region(&pieces, 11.0, 1.0, 0.01).is_none());
    }

    #[test]
    fn test_fit_label_too_tall_returns_none() {
        let pieces = vec![axis_aligned_square_piece(10.0)];
        assert!(fit_label_in_region(&pieces, 1.0, 11.0, 0.01).is_none());
    }

    #[test]
    fn test_fit_label_larger_than_bbox_returns_none() {
        let pieces = vec![axis_aligned_square_piece(10.0)];
        assert!(fit_label_in_region(&pieces, 20.0, 20.0, 0.01).is_none());
    }

    #[test]
    fn test_fit_label_with_hole_shrinks_fit() {
        // Same outer + hole as `test_inscribed_with_hole`. A square label
        // that fits in the unholed region should be rejected when the hole
        // carves out clearance; a small enough label still fits and lands
        // outside the hole.
        let outer = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let hole = Polygon::new(vec![
            Point::new(4.0, 4.0),
            Point::new(4.0, 6.0),
            Point::new(6.0, 6.0),
            Point::new(6.0, 4.0),
        ]);
        let pieces = vec![RegionPiece {
            outer,
            holes: vec![hole],
        }];

        // (5, 5) square: fits in the unholed 10×10 (max inscribed ≈ 7.07²)
        // but doesn't fit when the hole truncates clearance. Hard-pin the
        // contrast against the unholed baseline so the test rejects any
        // future regression that drops hole-awareness.
        let unholed = vec![axis_aligned_square_piece(10.0)];
        assert!(fit_label_in_region(&unholed, 5.0, 5.0, 0.01).is_some());
        assert!(fit_label_in_region(&pieces, 5.0, 5.0, 0.01).is_none());

        // A 0.5×0.5 label fits and the centre must lie outside the hole.
        let small = fit_label_in_region(&pieces, 0.5, 0.5, 0.01).unwrap();
        let inside_hole = (4.0..=6.0).contains(&small.x()) && (4.0..=6.0).contains(&small.y());
        assert!(
            !inside_hole,
            "small-label centre ({}, {}) lies inside the hole",
            small.x(),
            small.y()
        );
    }

    #[test]
    fn test_fit_label_complement_region() {
        // Synthetic complement: 10×10 container, two unit holes far from
        // the edges and from each other. The decomposed complement region
        // has one outer + two holes (mirrors
        // test_complement_with_strictly_nested_shapes_yields_outer_plus_holes).
        let outer = Polygon::new(vec![
            Point::new(-5.0, -5.0),
            Point::new(5.0, -5.0),
            Point::new(5.0, 5.0),
            Point::new(-5.0, 5.0),
        ]);
        // Two CW unit "holes" at (-3, 0) and (3, 0).
        let hole_a = Polygon::new(vec![
            Point::new(-2.5, -0.5),
            Point::new(-2.5, 0.5),
            Point::new(-3.5, 0.5),
            Point::new(-3.5, -0.5),
        ]);
        let hole_b = Polygon::new(vec![
            Point::new(2.5, -0.5),
            Point::new(2.5, 0.5),
            Point::new(3.5, 0.5),
            Point::new(3.5, -0.5),
        ]);
        let pieces = vec![RegionPiece {
            outer,
            holes: vec![hole_a, hole_b],
        }];

        // A small label fits and the anchor lands inside the container,
        // outside both holes.
        let anchor = fit_label_in_region(&pieces, 1.0, 0.4, 0.01).unwrap();
        assert!(anchor.x() >= -5.0 && anchor.x() <= 5.0);
        assert!(anchor.y() >= -5.0 && anchor.y() <= 5.0);
        let in_hole_a = (-3.5..=-2.5).contains(&anchor.x()) && (-0.5..=0.5).contains(&anchor.y());
        let in_hole_b = (2.5..=3.5).contains(&anchor.x()) && (-0.5..=0.5).contains(&anchor.y());
        assert!(!in_hole_a && !in_hole_b);
    }

    #[test]
    fn test_fit_label_invalid_dimensions() {
        let pieces = vec![axis_aligned_square_piece(10.0)];
        assert!(fit_label_in_region(&pieces, 0.0, 1.0, 0.01).is_none());
        assert!(fit_label_in_region(&pieces, 1.0, 0.0, 0.01).is_none());
        assert!(fit_label_in_region(&pieces, -1.0, 1.0, 0.01).is_none());
        assert!(fit_label_in_region(&pieces, 1.0, -1.0, 0.01).is_none());
        assert!(fit_label_in_region(&pieces, f64::NAN, 1.0, 0.01).is_none());
        assert!(fit_label_in_region(&pieces, 1.0, f64::INFINITY, 0.01).is_none());
    }
}
