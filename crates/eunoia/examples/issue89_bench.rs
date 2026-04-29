//! Reproduction benchmark for eulerr issue #89.
//!
//! The original issue reports a 17-set ellipse fit that hangs because the
//! ellipse path enumerated all 2^n - 1 = 131,071 region masks per loss
//! evaluation. Eunoia routes the ellipse path through sparse region discovery
//! (`discover_regions`), so only geometrically-non-empty regions are walked.
//!
//! This bench (a) times `compute_exclusive_regions` directly to expose the
//! per-call speed-up, and (b) actually fits the full issue #89 spec to
//! confirm it is now solvable end-to-end.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example issue89_bench
//! ```
use eunoia::geometry::primitives::Point;
use eunoia::geometry::shapes::{Circle, Ellipse};
use eunoia::geometry::traits::DiagramShape;
use eunoia::{spec::DiagramSpec, DiagramSpecBuilder, Fitter, InputType};
use std::time::Instant;

/// The exact disjoint-input spec from
/// <https://github.com/jolars/eulerr/issues/89>.
fn issue89_spec() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 55.0)
        .set("B", 810.0)
        .set("C", 102.0)
        .set("D", 364.0)
        .set("E", 101.0)
        .set("F", 24.0)
        .set("G", 34.0)
        .set("H", 61.0)
        .set("I", 194.0)
        .set("J", 107.0)
        .set("K", 53.0)
        .set("L", 75.0)
        .set("M", 11.0)
        .set("N", 65.0)
        .set("O", 16.0)
        .set("P", 82.0)
        .set("Q", 13.0)
        .intersection(&["F", "O"], 5.0)
        .intersection(&["G", "O"], 5.0)
        .intersection(&["F", "G"], 5.0)
        .intersection(&["D", "I"], 47.0)
        .intersection(&["D", "E"], 33.0)
        .intersection(&["K", "L"], 9.0)
        .intersection(&["A", "K"], 7.0)
        .intersection(&["K", "N"], 7.0)
        .intersection(&["A", "N"], 7.0)
        .intersection(&["A", "L"], 7.0)
        .intersection(&["K", "P"], 7.0)
        .intersection(&["A", "P"], 7.0)
        .intersection(&["L", "N"], 7.0)
        .intersection(&["N", "P"], 7.0)
        .intersection(&["C", "K"], 7.0)
        .intersection(&["A", "C"], 7.0)
        .intersection(&["L", "P"], 7.0)
        .intersection(&["E", "G"], 6.0)
        .intersection(&["J", "K"], 7.0)
        .intersection(&["A", "J"], 7.0)
        .intersection(&["C", "O"], 5.0)
        .intersection(&["G", "H"], 4.0)
        .intersection(&["C", "N"], 7.0)
        .intersection(&["J", "N"], 7.0)
        .intersection(&["E", "F"], 5.0)
        .intersection(&["C", "F"], 5.0)
        .intersection(&["C", "L"], 7.0)
        .intersection(&["J", "L"], 7.0)
        .intersection(&["C", "P"], 7.0)
        .intersection(&["J", "P"], 7.0)
        .intersection(&["C", "G"], 5.0)
        .intersection(&["D", "K"], 15.0)
        .intersection(&["C", "J"], 7.0)
        .intersection(&["A", "D"], 12.0)
        .intersection(&["D", "L"], 12.0)
        .intersection(&["E", "O"], 3.0)
        .intersection(&["E", "H"], 4.0)
        .intersection(&["C", "I"], 6.0)
        .intersection(&["C", "D"], 8.0)
        .intersection(&["D", "H"], 7.0)
        .intersection(&["D", "N"], 7.0)
        .intersection(&["D", "P"], 7.0)
        .intersection(&["D", "J"], 7.0)
        .intersection(&["C", "E"], 3.0)
        .intersection(&["H", "O"], 1.0)
        .intersection(&["B", "D"], 15.0)
        .intersection(&["B", "C"], 11.0)
        .intersection(&["F", "H"], 1.0)
        .intersection(&["E", "M"], 1.0)
        .intersection(&["B", "E"], 8.0)
        .intersection(&["B", "K"], 7.0)
        .intersection(&["I", "K"], 2.0)
        .intersection(&["A", "B"], 7.0)
        .intersection(&["B", "N"], 7.0)
        .intersection(&["B", "L"], 7.0)
        .intersection(&["B", "P"], 7.0)
        .intersection(&["D", "F"], 3.0)
        .intersection(&["B", "J"], 7.0)
        .intersection(&["I", "L"], 2.0)
        .intersection(&["C", "H"], 1.0)
        .intersection(&["D", "O"], 1.0)
        .intersection(&["D", "G"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap()
}

fn region_bench() {
    let n_sets = 17;

    let circles: Vec<Circle> = (0..n_sets)
        .map(|i| {
            let theta = i as f64 * 2.0 * std::f64::consts::PI / n_sets as f64;
            let r = 3.0;
            Circle::new(Point::new(r * theta.cos(), r * theta.sin()), 1.0)
        })
        .collect();

    let ellipses: Vec<Ellipse> = (0..n_sets)
        .map(|i| {
            let theta = i as f64 * 2.0 * std::f64::consts::PI / n_sets as f64;
            let r = 3.0;
            Ellipse::new(
                Point::new(r * theta.cos(), r * theta.sin()),
                1.2,
                0.8,
                theta,
            )
        })
        .collect();

    let n_iters = 20;

    let t0 = Instant::now();
    let mut last_circle = 0;
    for _ in 0..n_iters {
        let regions = Circle::compute_exclusive_regions(&circles);
        last_circle = regions.len();
    }
    let circle_time = t0.elapsed();

    let t0 = Instant::now();
    let mut last_ellipse = 0;
    for _ in 0..n_iters {
        let regions = Ellipse::compute_exclusive_regions(&ellipses);
        last_ellipse = regions.len();
    }
    let ellipse_time = t0.elapsed();

    println!("=== compute_exclusive_regions micro-bench (ring of {n_sets} shapes) ===");
    println!(
        "circles:  {} regions, {:?} per call",
        last_circle,
        circle_time / n_iters
    );
    println!(
        "ellipses: {} regions, {:?} per call",
        last_ellipse,
        ellipse_time / n_iters
    );
    println!(
        "(full enumeration would walk {} regions per call)\n",
        (1usize << n_sets) - 1
    );
}

fn fit_full_spec() {
    let spec = issue89_spec();

    println!("=== full fit on issue #89 spec (17 sets, 67 pair intersections) ===");

    print!("circles  ... ");
    let t0 = Instant::now();
    let layout = Fitter::<Circle>::new(&spec).seed(1).fit().unwrap();
    let dt = t0.elapsed();
    println!(
        "{:>7.2} s, diag_error = {:.4e}, stress = {:.4e}",
        dt.as_secs_f64(),
        layout.diag_error(),
        layout.stress()
    );

    print!("ellipses ... ");
    let t0 = Instant::now();
    let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();
    let dt = t0.elapsed();
    println!(
        "{:>7.2} s, diag_error = {:.4e}, stress = {:.4e}",
        dt.as_secs_f64(),
        layout.diag_error(),
        layout.stress()
    );
}

fn main() {
    region_bench();
    fit_full_spec();
}
