//! Scan-line intervals over a region piece.
//!
//! A packer that places *axis-aligned boxes* needs to know, for a horizontal
//! band `[yb, yt]`, which x-ranges are wide open all the way through — not
//! just at one sample height. [`PieceScan::band_intervals`] answers that
//! exactly, and it is the containment oracle the row packer runs in its
//! inner loop instead of the general
//! [`rect_fits_in_piece`](crate::plotting::inscribed::rect_fits_in_piece).
//!
//! # Why it is exact
//!
//! Between two consecutive *vertex* y-values the piece's free x-set is a
//! union of trapezoids, so every interval endpoint moves linearly in `y`.
//! The narrowest slice over such a slab is therefore attained at one of the
//! slab's two boundaries. Intersecting the scan-line intervals at `yb`, at
//! `yt`, and at every vertex y strictly between them thus yields precisely
//! the x-set for which the *whole* band slice is interior — no sampling
//! error, and no dependence on how finely the polygon is tessellated.
//!
//! The vertex-y ladder does not depend on the band, so [`PieceScan::new`]
//! precomputes it once per piece; the box packer's scale bisection re-probes
//! the same piece dozens of times and must not rebuild it each pass.

use crate::geometry::primitives::Point;
use crate::plotting::regions::RegionPiece;

use super::ring_bounds;

/// A region piece prepared for repeated band queries: the sorted, unique
/// ladder of ring-vertex y-values plus the outer bounding box.
#[derive(Debug)]
pub(super) struct PieceScan<'a> {
    piece: &'a RegionPiece,
    ys: Vec<f64>,
    bbox: (f64, f64, f64, f64),
}

impl<'a> PieceScan<'a> {
    /// Prepares `piece` for band queries. `None` for a degenerate piece
    /// (fewer than three outer vertices, or a zero-extent bounding box).
    pub(super) fn new(piece: &'a RegionPiece) -> Option<Self> {
        let bbox = ring_bounds(piece.outer.vertices())?;
        let mut ys: Vec<f64> = piece
            .outer
            .vertices()
            .iter()
            .chain(piece.holes.iter().flat_map(|h| h.vertices()))
            .map(Point::y)
            .filter(|y| y.is_finite())
            .collect();
        ys.sort_by(f64::total_cmp);
        ys.dedup();
        Some(Self { piece, ys, bbox })
    }

    /// The piece this scan was built from.
    pub(super) fn piece(&self) -> &'a RegionPiece {
        self.piece
    }

    /// Axis-aligned bounds of the outer ring, as `(min_x, max_x, min_y, max_y)`.
    pub(super) fn bbox(&self) -> (f64, f64, f64, f64) {
        self.bbox
    }

    /// Free x-intervals available to a box spanning the horizontal band
    /// `[yb, yt]`: the set of x for which the entire band slice lies inside
    /// the outer ring and outside every hole. Sorted and disjoint; empty for
    /// a degenerate or fully blocked band.
    pub(super) fn band_intervals(&self, yb: f64, yt: f64) -> Vec<(f64, f64)> {
        if !(yb.is_finite() && yt.is_finite()) || yt <= yb {
            return Vec::new();
        }
        let mut acc = scan_intervals(self.piece, yb);
        if acc.is_empty() {
            return acc;
        }
        // The vertex ladder is sorted, so the events inside the band are a
        // contiguous slice.
        let start = self.ys.partition_point(|&y| y <= yb);
        for &y in &self.ys[start..] {
            if y >= yt {
                break;
            }
            acc = intersect_intervals(&acc, &scan_intervals(self.piece, y));
            if acc.is_empty() {
                return acc;
            }
        }
        intersect_intervals(&acc, &scan_intervals(self.piece, yt))
    }
}

/// Interior x-intervals of `piece` along the horizontal line `y`.
///
/// Uses the same half-open crossing rule as
/// [`point_in_polygon`](crate::plotting::regions::point_in_polygon) —
/// `(a.y > y) != (b.y > y)` — over the outer ring and every hole, then pairs
/// the sorted crossings even-odd. Sharing the rule is what makes this agree
/// with the crate's other containment tests; sharing it across rings is what
/// makes holes subtract themselves.
pub(super) fn scan_intervals(piece: &RegionPiece, y: f64) -> Vec<(f64, f64)> {
    let mut xs = Vec::new();
    ring_crossings(piece.outer.vertices(), y, &mut xs);
    for hole in &piece.holes {
        ring_crossings(hole.vertices(), y, &mut xs);
    }
    if xs.len() < 2 {
        return Vec::new();
    }
    xs.sort_by(f64::total_cmp);
    // An odd crossing count can only come from a degenerate ring; drop the
    // unpaired tail rather than inventing an unbounded interval.
    xs.chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .filter(|(a, b)| b > a)
        .collect()
}

/// Appends the x-intercepts where the closed ring crosses the line `y`.
fn ring_crossings(ring: &[Point], y: f64, out: &mut Vec<f64>) {
    let n = ring.len();
    if n < 3 {
        return;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i].x(), ring[i].y());
        let (xj, yj) = (ring[j].x(), ring[j].y());
        // The half-open test guarantees `yj != yi`, so the division is safe.
        if (yi > y) != (yj > y) {
            out.push(xi + (y - yi) * (xj - xi) / (yj - yi));
        }
        j = i;
    }
}

/// Intersection of two sorted, disjoint interval lists.
pub(super) fn intersect_intervals(a: &[(f64, f64)], b: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if hi > lo {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// `intervals` minus the open range `(lo, hi)`. An interval either survives
/// untouched, shrinks on one side, splits in two, or vanishes.
pub(super) fn subtract_range(intervals: &[(f64, f64)], lo: f64, hi: f64) -> Vec<(f64, f64)> {
    if lo.is_nan() || hi.is_nan() || lo >= hi {
        return intervals.to_vec();
    }
    let mut out = Vec::with_capacity(intervals.len() + 1);
    for &(a, b) in intervals {
        if hi <= a || lo >= b {
            out.push((a, b));
            continue;
        }
        if lo > a {
            out.push((a, lo));
        }
        if hi < b {
            out.push((hi, b));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::{Circle, Polygon};
    use crate::geometry::traits::Polygonize;
    use crate::plotting::inscribed::rect_fits_in_piece;
    use crate::plotting::regions::classify_into_pieces;

    use super::super::test_utils::rect_ring;

    fn square_piece() -> RegionPiece {
        classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])
            .pop()
            .unwrap()
    }

    fn holed_piece() -> RegionPiece {
        classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 10.0, 10.0),
            rect_ring(4.0, 4.0, 6.0, 6.0),
        ])
        .pop()
        .unwrap()
    }

    fn assert_close(got: &[(f64, f64)], want: &[(f64, f64)]) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g.0 - w.0).abs() < 1e-9 && (g.1 - w.1).abs() < 1e-9,
                "got {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn scan_intervals_of_a_square() {
        let piece = square_piece();
        assert_close(&scan_intervals(&piece, 5.0), &[(0.0, 10.0)]);
        // Above and below the ring there are no crossings at all.
        assert!(scan_intervals(&piece, -1.0).is_empty());
        assert!(scan_intervals(&piece, 11.0).is_empty());
    }

    #[test]
    fn scan_intervals_of_a_holed_square_splits_in_two() {
        let piece = holed_piece();
        assert_eq!(piece.holes.len(), 1);
        assert_close(&scan_intervals(&piece, 5.0), &[(0.0, 4.0), (6.0, 10.0)]);
        // Clear of the hole's y-range the row is whole again.
        assert_close(&scan_intervals(&piece, 2.0), &[(0.0, 10.0)]);
    }

    #[test]
    fn band_intervals_narrow_to_the_slab_minimum() {
        // A triangle narrowing upward: 0..10 wide at y = 0, a point at
        // y = 10. A band's interval must equal the interval at its *top*,
        // the narrow edge — not the midpoint sample.
        let piece = classify_into_pieces(vec![Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 10.0),
        ])])
        .pop()
        .unwrap();
        let scan = PieceScan::new(&piece).unwrap();

        let band = scan.band_intervals(2.0, 6.0);
        let at_top = scan_intervals(&piece, 6.0);
        assert_close(&band, &at_top);
        // ...and strictly narrower than the wide bottom edge.
        let at_bottom = scan_intervals(&piece, 2.0);
        assert!(band[0].1 - band[0].0 < at_bottom[0].1 - at_bottom[0].0);
    }

    #[test]
    fn band_intervals_see_a_hole_the_edges_miss() {
        // A hole strictly inside the band: neither `yb` nor `yt` crosses it,
        // so only the vertex-ladder passes can find it. This is the case a
        // two-sample implementation gets wrong.
        let piece = classify_into_pieces(vec![
            rect_ring(0.0, 0.0, 10.0, 10.0),
            rect_ring(4.0, 4.0, 6.0, 6.0),
        ])
        .pop()
        .unwrap();
        let scan = PieceScan::new(&piece).unwrap();

        assert_close(&scan_intervals(&piece, 3.0), &[(0.0, 10.0)]);
        assert_close(&scan_intervals(&piece, 7.0), &[(0.0, 10.0)]);
        assert_close(&scan.band_intervals(3.0, 7.0), &[(0.0, 4.0), (6.0, 10.0)]);
    }

    #[test]
    fn band_intervals_reject_degenerate_bands() {
        let piece = square_piece();
        let scan = PieceScan::new(&piece).unwrap();
        assert!(scan.band_intervals(5.0, 5.0).is_empty());
        assert!(scan.band_intervals(6.0, 5.0).is_empty());
        assert!(scan.band_intervals(f64::NAN, 5.0).is_empty());
        // Entirely outside the piece.
        assert!(scan.band_intervals(20.0, 21.0).is_empty());
    }

    #[test]
    fn band_intervals_agree_with_rect_fits_in_piece() {
        // An L-shape (concave) plus a circle (finely tessellated): for a
        // sweep of candidate boxes, "the band oracle says it fits" must
        // imply "the general predicate says it fits". The converse can fail
        // on knife edges only, so the implication is asserted one way and a
        // tiny inward margin keeps floating-point ties out of it.
        let l_shape = classify_into_pieces(vec![Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(8.0, 0.0),
            Point::new(8.0, 2.0),
            Point::new(2.0, 2.0),
            Point::new(2.0, 8.0),
            Point::new(0.0, 8.0),
        ])])
        .pop()
        .unwrap();
        let disc =
            classify_into_pieces(vec![Circle::new(Point::new(0.0, 0.0), 5.0).polygonize(64)])
                .pop()
                .unwrap();

        for piece in [&l_shape, &disc] {
            let scan = PieceScan::new(piece).unwrap();
            let (_, _, min_y, max_y) = scan.bbox();
            let mut checked = 0usize;
            for hh_step in 1..=4 {
                let hh = 0.1 * hh_step as f64;
                for row in 0..20 {
                    let cy = min_y + (max_y - min_y) * (row as f64 + 0.5) / 20.0;
                    for (lo, hi) in scan.band_intervals(cy - hh, cy + hh) {
                        for hw_step in 1..=4 {
                            let hw = 0.1 * hw_step as f64;
                            if hi - lo < 2.0 * hw {
                                continue;
                            }
                            for col in 0..5 {
                                let cx = lo + hw + (hi - lo - 2.0 * hw) * col as f64 / 4.0;
                                checked += 1;
                                assert!(
                                    rect_fits_in_piece(piece, cx, cy, hw * 0.999, hh * 0.999),
                                    "band oracle accepted ({cx}, {cy}) ± ({hw}, {hh}) \
                                     but rect_fits_in_piece rejects it"
                                );
                            }
                        }
                    }
                }
            }
            assert!(checked > 100, "cross-check exercised only {checked} boxes");
        }
    }

    #[test]
    fn subtract_range_splits_shrinks_and_drops() {
        let base = [(0.0, 10.0)];
        // Straddling the middle: splits.
        assert_close(&subtract_range(&base, 4.0, 6.0), &[(0.0, 4.0), (6.0, 10.0)]);
        // Overhanging one end: shrinks.
        assert_close(&subtract_range(&base, -5.0, 3.0), &[(3.0, 10.0)]);
        assert_close(&subtract_range(&base, 7.0, 20.0), &[(0.0, 7.0)]);
        // Covering everything: drops.
        assert!(subtract_range(&base, -1.0, 11.0).is_empty());
        // Disjoint or degenerate: untouched.
        assert_close(&subtract_range(&base, 20.0, 30.0), &base);
        assert_close(&subtract_range(&base, 5.0, 5.0), &base);
        // Multiple intervals are handled independently.
        let two = [(0.0, 4.0), (6.0, 10.0)];
        assert_close(&subtract_range(&two, 3.0, 7.0), &[(0.0, 3.0), (7.0, 10.0)]);
    }

    #[test]
    fn intersect_intervals_meets_on_overlap_only() {
        let a = [(0.0, 5.0), (7.0, 10.0)];
        let b = [(3.0, 8.0)];
        assert_close(&intersect_intervals(&a, &b), &[(3.0, 5.0), (7.0, 8.0)]);
        assert!(intersect_intervals(&a, &[(5.0, 7.0)]).is_empty());
        assert!(intersect_intervals(&a, &[]).is_empty());
    }
}
