//! Post-pass that evens out a dart-thrown scatter.
//!
//! Dart throwing guarantees a *minimum* spacing and nothing more, so the
//! result is locally lumpy: pairs sitting exactly at the minimum next to
//! voids two glyphs wide. [`relax_scatter`] runs a fixed number of
//! force-directed sweeps that push near neighbours apart and push points off
//! the region boundary, which spreads the scatter out without turning it
//! into a lattice.
//!
//! Two properties matter as much as the look:
//!
//! * **Determinism.** No randomness and a fixed sweep count, so the pass is
//!   a pure function of the scatter it is handed — the seeded packer stays
//!   reproducible.
//! * **Invariance.** Every move is proposed and then *accepted or rejected*
//!   against the packer's own predicates (boundary clearance, pairwise
//!   spacing, obstacle clearance), with a couple of halvings before the
//!   point stays put. The relaxation can therefore only improve the
//!   arrangement — it can never break the guarantees
//!   [`place_glyphs`](super::place_glyphs) advertises, and it never changes
//!   how many glyphs are placed. That is what lets it run only on the final
//!   pack while the feasibility probes skip it.
//!
//! Points are updated in place, one at a time (Gauss-Seidel), so each
//! acceptance test sees the moves already made in this sweep.

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;
use crate::plotting::regions::{RegionPiece, signed_clearance};

use super::clear_of_obstacles;

/// Sweeps over the scatter. Fixed rather than convergence-tested, so the
/// cost is bounded and the output is reproducible; the step size cools to
/// zero across them, which is what settles the arrangement.
const SWEEPS: usize = 8;

/// Neighbour influence radius, in multiples of the minimum spacing. Beyond
/// one spacing lies the second ring of neighbours; including part of it is
/// what lets a point drift into a nearby void rather than only shoving its
/// touching partner away.
const INFLUENCE: f64 = 1.5;

/// Boundary influence radius, in multiples of the clearance inset. Repelling
/// particles in a confined region pile up along its rim, so the boundary
/// pushes back over a band a little wider than the inset itself.
const BOUNDARY_INFLUENCE: f64 = 2.0;

/// Weight of the boundary push relative to one neighbour at full strength.
/// Below one, since the boundary is a single force competing with however
/// many neighbours a crowded point has.
const BOUNDARY_WEIGHT: f64 = 0.75;

/// Largest single move, in multiples of the minimum spacing, before cooling.
/// Small enough that a point cannot hop over a neighbour in one step.
const MAX_STEP: f64 = 0.35;

/// How many times a rejected move is halved before the point stays put.
/// A blocked point is usually blocked by geometry it cannot escape, so
/// backtracking far is wasted work.
const BACKTRACKS: usize = 2;

/// Finite-difference step for the clearance gradient, in multiples of the
/// inset.
const PROBE: f64 = 0.25;

/// Even out `points` in place: neighbours within [`INFLUENCE`] push each
/// other apart, the boundary pushes inward within [`BOUNDARY_INFLUENCE`],
/// and every resulting move must still keep `inset` clearance to the piece
/// (and to `obstacles`, for points that had it to begin with) and `spacing`
/// to every other point.
///
/// `spacing` and `inset` are the packer's own
/// [`PackCtx`](super::discs) values, so what comes out satisfies exactly
/// what went in.
pub(super) fn relax_scatter(
    points: &mut [Point],
    piece: &RegionPiece,
    obstacles: &[Rectangle],
    spacing: f64,
    inset: f64,
) {
    if points.is_empty() || !spacing.is_finite() || spacing <= 0.0 {
        return;
    }
    let field = Field {
        piece,
        obstacles,
        spacing,
        inset,
        influence: INFLUENCE * spacing,
        boundary_reach: BOUNDARY_INFLUENCE * inset,
        probe: PROBE * inset,
    };

    // A point placed by the lenient second dart pass sits on an obstacle
    // already; holding it to a clearance it never had would freeze it in
    // place. Points that do have the clearance must keep it.
    let honoring: Vec<bool> = points
        .iter()
        .map(|p| clear_of_obstacles(p.x(), p.y(), obstacles, inset))
        .collect();

    // `signed_clearance` walks every ring vertex, so it dominates the cost
    // of the whole pass. Keeping each point's value alongside it means the
    // sweeps pay for it only where a point actually moves: the acceptance
    // test has to evaluate the candidate anyway, and hands the value back.
    let mut clearance: Vec<f64> = points
        .iter()
        .map(|p| signed_clearance(p.x(), p.y(), piece))
        .collect();

    for sweep in 0..SWEEPS {
        let cooling = 1.0 - sweep as f64 / SWEEPS as f64;
        let max_step = MAX_STEP * spacing * cooling;
        for i in 0..points.len() {
            let p = points[i];
            let (fx, fy) = field.force_on(i, points, clearance[i]);
            let magnitude = fx.hypot(fy);
            if !magnitude.is_finite() || magnitude <= 0.0 {
                continue;
            }
            // Saturating at one neighbour's worth of force: past that the
            // direction is what matters, and the cap keeps the step honest.
            let mut step = max_step * magnitude.min(1.0);
            let (ux, uy) = (fx / magnitude, fy / magnitude);
            for _ in 0..=BACKTRACKS {
                let candidate = Point::new(p.x() + ux * step, p.y() + uy * step);
                if let Some(c) = field.accepts(candidate, i, points, honoring[i]) {
                    points[i] = candidate;
                    clearance[i] = c;
                    break;
                }
                step *= 0.5;
            }
        }
    }
}

/// What the sweeps relax within: the piece and its keep-out boxes, the two
/// distances the packer guaranteed, and the reaches derived from them.
/// Constant for the whole pass, hence a bundle rather than an argument list.
struct Field<'a> {
    piece: &'a RegionPiece,
    obstacles: &'a [Rectangle],
    /// Minimum center-to-center distance between two glyphs.
    spacing: f64,
    /// Clearance every center keeps from the piece's rings and from the
    /// obstacles it started clear of.
    inset: f64,
    /// Range of the neighbour repulsion.
    influence: f64,
    /// Range of the inward boundary push.
    boundary_reach: f64,
    /// Finite-difference step for the clearance gradient.
    probe: f64,
}

impl Field<'_> {
    /// Net push on `points[i]`, whose clearance to the piece is `clearance`:
    /// a linear-falloff repulsion from every neighbour inside
    /// [`influence`](Self::influence), plus an inward push once the point is
    /// within [`boundary_reach`](Self::boundary_reach) of the boundary.
    fn force_on(&self, i: usize, points: &[Point], clearance: f64) -> (f64, f64) {
        let p = points[i];
        let (mut fx, mut fy) = (0.0, 0.0);
        let influence2 = self.influence * self.influence;
        for (j, q) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            let (dx, dy) = (p.x() - q.x(), p.y() - q.y());
            let d2 = dx * dx + dy * dy;
            if d2 <= 0.0 || d2 >= influence2 {
                continue;
            }
            let d = d2.sqrt();
            let w = (self.influence - d) / self.influence;
            fx += w * dx / d;
            fy += w * dy / d;
        }

        if clearance < self.boundary_reach && self.boundary_reach > 0.0 {
            if let Some((gx, gy)) = self.clearance_gradient(p) {
                let w = BOUNDARY_WEIGHT * (self.boundary_reach - clearance) / self.boundary_reach;
                fx += w * gx;
                fy += w * gy;
            }
        }
        (fx, fy)
    }

    /// Unit direction of steepest increase in clearance, by central
    /// differences. `None` where the field is flat to numerical precision —
    /// the midline between two rings, say, where there is no inward to move.
    fn clearance_gradient(&self, p: Point) -> Option<(f64, f64)> {
        let (x, y, h) = (p.x(), p.y(), self.probe);
        let gx = signed_clearance(x + h, y, self.piece) - signed_clearance(x - h, y, self.piece);
        let gy = signed_clearance(x, y + h, self.piece) - signed_clearance(x, y - h, self.piece);
        let norm = gx.hypot(gy);
        (norm > 1e-12).then(|| (gx / norm, gy / norm))
    }

    /// Whether moving `points[i]` to `candidate` keeps every guarantee the
    /// packer made: clearance to the piece, spacing to every other point,
    /// and — for a point that had it — clearance to the obstacles. Returns
    /// the candidate's clearance on acceptance, which becomes the caller's
    /// cached value for the point's new position.
    fn accepts(&self, candidate: Point, i: usize, points: &[Point], honoring: bool) -> Option<f64> {
        let (x, y) = (candidate.x(), candidate.y());
        let clearance = signed_clearance(x, y, self.piece);
        if clearance < self.inset {
            return None;
        }
        if honoring && !clear_of_obstacles(x, y, self.obstacles, self.inset) {
            return None;
        }
        let spacing2 = self.spacing * self.spacing;
        let clear_of_placed = points
            .iter()
            .enumerate()
            .all(|(j, q)| j == i || (x - q.x()).powi(2) + (y - q.y()).powi(2) >= spacing2);
        clear_of_placed.then_some(clearance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plotting::regions::classify_into_pieces;

    use super::super::test_utils::rect_ring;

    /// Mean and standard deviation of every point's nearest-neighbour
    /// distance — the evenness measure the pass is trying to improve.
    fn nn_stats(points: &[Point]) -> (f64, f64) {
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
        (mean, var.sqrt())
    }

    fn assert_invariants(points: &[Point], piece: &RegionPiece, spacing: f64, inset: f64) {
        for (i, p) in points.iter().enumerate() {
            assert!(
                signed_clearance(p.x(), p.y(), piece) >= inset - 1e-9,
                "point {i} left the piece"
            );
            for q in &points[i + 1..] {
                let d = (p.x() - q.x()).hypot(p.y() - q.y());
                assert!(d >= spacing - 1e-9, "points overlap: {d} < {spacing}");
            }
        }
    }

    /// A clumped-but-legal scatter: two tight rows in one corner of a roomy
    /// piece, every pair exactly at the minimum spacing.
    fn clumped(spacing: f64) -> Vec<Point> {
        (0..4)
            .flat_map(|row| {
                (0..4).map(move |col| {
                    Point::new(1.0 + col as f64 * spacing, 1.0 + row as f64 * spacing)
                })
            })
            .collect()
    }

    #[test]
    fn spreads_a_clump_without_breaking_invariants() {
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let (spacing, inset) = (1.0, 0.5);
        let mut points = clumped(spacing);
        let before = nn_stats(&points);

        relax_scatter(&mut points, piece, &[], spacing, inset);

        let after = nn_stats(&points);
        assert!(
            after.0 > before.0 + 1e-6,
            "relaxation should open the scatter up: {} -> {}",
            before.0,
            after.0
        );
        assert_invariants(&points, piece, spacing, inset);
    }

    #[test]
    fn evens_out_an_uneven_scatter() {
        // A wide piece with points bunched to the left: the nearest-neighbour
        // spread is what should shrink.
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 12.0, 4.0)])[0];
        let (spacing, inset) = (1.0, 0.5);
        let mut points = vec![
            Point::new(1.0, 2.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 2.0),
            Point::new(3.0, 3.0),
            Point::new(2.2, 1.2),
            Point::new(8.0, 2.0),
        ];
        let before = nn_stats(&points);

        relax_scatter(&mut points, piece, &[], spacing, inset);

        let after = nn_stats(&points);
        assert!(
            after.1 < before.1,
            "nearest-neighbour spread should fall: {} -> {}",
            before.1,
            after.1
        );
        assert_invariants(&points, piece, spacing, inset);
    }

    #[test]
    fn is_deterministic() {
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let mut first = clumped(1.0);
        let mut second = clumped(1.0);
        relax_scatter(&mut first, piece, &[], 1.0, 0.5);
        relax_scatter(&mut second, piece, &[], 1.0, 0.5);
        assert_eq!(first, second);
    }

    #[test]
    fn keeps_clear_of_obstacles_it_started_clear_of() {
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let (spacing, inset) = (1.0, 0.5);
        // A wall just past the clump's right edge, in the direction the
        // scatter wants to expand.
        let obstacles = [Rectangle::new(Point::new(5.6, 2.5), 2.0, 6.0)];
        let mut points = clumped(spacing);
        assert!(
            points
                .iter()
                .all(|p| clear_of_obstacles(p.x(), p.y(), &obstacles, inset)),
            "fixture must start clear of the obstacle"
        );

        relax_scatter(&mut points, piece, &obstacles, spacing, inset);

        for (i, p) in points.iter().enumerate() {
            assert!(
                clear_of_obstacles(p.x(), p.y(), &obstacles, inset),
                "point {i} was pushed onto the obstacle"
            );
        }
        assert_invariants(&points, piece, spacing, inset);
    }

    #[test]
    fn a_buried_point_is_free_to_move() {
        // A point that never had obstacle clearance keeps relaxing: holding
        // it to a guarantee it never made would pin it under the label.
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let obstacles = [Rectangle::new(Point::new(5.0, 5.0), 9.0, 9.0)];
        let mut points = vec![Point::new(5.0, 0.6), Point::new(5.0, 1.6)];
        let before = points.clone();

        relax_scatter(&mut points, piece, &obstacles, 1.0, 0.5);

        assert_ne!(points, before, "the buried points should still relax");
        assert_invariants(&points, piece, 1.0, 0.5);
    }

    #[test]
    fn a_lone_point_is_pushed_off_the_boundary() {
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let mut points = vec![Point::new(0.5, 5.0)];
        relax_scatter(&mut points, piece, &[], 1.0, 0.5);
        assert!(
            points[0].x() > 0.5,
            "the boundary force should push inward, got {:?}",
            points[0]
        );
        assert_invariants(&points, piece, 1.0, 0.5);
    }

    #[test]
    fn degenerate_input_is_a_no_op() {
        let piece = &classify_into_pieces(vec![rect_ring(0.0, 0.0, 10.0, 10.0)])[0];
        let mut empty: Vec<Point> = Vec::new();
        relax_scatter(&mut empty, piece, &[], 1.0, 0.5);
        assert!(empty.is_empty());

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut points = clumped(1.0);
            let before = points.clone();
            relax_scatter(&mut points, piece, &[], bad, 0.5);
            assert_eq!(points, before, "spacing = {bad}");
        }
    }
}
