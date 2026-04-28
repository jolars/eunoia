//! Benchmark the candidate MDS solvers used for the initial layout.
//!
//! Per `initial_layout.rs`, MDS loss alone is misleading: MDS-suboptimal
//! inits sometimes downstream-fit perfectly while MDS-optimal ones get stuck
//! (issue #28). So this bench runs the full `Fitter::fit()` pipeline with
//! `n_restarts(1)` to isolate the contribution of the initial solver, and
//! measures wall time per fit.
//!
//! Quality (final `diag_error`) is also reported as throughput-style numbers
//! via a separate quality bench group below — Criterion's wall-time numbers
//! are the primary signal for "is this solver fast enough?", and the quality
//! report tells you "does it actually find the basin?".
//!
//! Run with: `cargo bench -p eunoia --bench initial_layout`
//! Filter:   `cargo bench -p eunoia --bench initial_layout -- 3-circle`

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use eunoia::geometry::shapes::{Circle, Ellipse};
use eunoia::spec::{DiagramSpec, DiagramSpecBuilder, InputType};
use eunoia::{Fitter, MdsSolver};

const SOLVERS: [(MdsSolver, &str); 3] = [
    (MdsSolver::Lbfgs, "lbfgs"),
    (MdsSolver::TrustRegion, "tr_steihaug"),
    (MdsSolver::NewtonCg, "newton_cg"),
];

/// Seed used for the timing benches. A single fixed seed gives Criterion a
/// stable signal — solver-vs-solver wall-time is the headline metric here, and
/// across-seed variance is a separate question handled by the quality report.
const TIMING_SEED: u64 = 42;

/// Seeds swept by the quality report (printed to stdout once per `cargo bench`
/// invocation). Wide enough to catch unlucky basins without dominating runtime.
const QUALITY_SEEDS: [u64; 16] = [1, 2, 3, 7, 13, 17, 23, 29, 31, 37, 41, 42, 47, 53, 59, 61];

fn three_circle_easy() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 10.0)
        .set("C", 10.0)
        .intersection(&["A", "B"], 2.0)
        .intersection(&["B", "C"], 2.0)
        .intersection(&["A", "C"], 2.0)
        .intersection(&["A", "B", "C"], 0.5)
        .build()
        .unwrap()
}

fn three_circle_user_case() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 2.2)
        .set("B", 2.0)
        .set("C", 3.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap()
}

fn issue28_four_set_superset() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 30.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 3.0)
        .intersection(&["A", "D"], 3.0)
        .intersection(&["A", "B", "C"], 2.0)
        .intersection(&["A", "B", "D"], 2.0)
        .intersection(&["A", "C", "D"], 2.0)
        .intersection(&["A", "B", "C", "D"], 1.0)
        .build()
        .unwrap()
}

fn issue28_six_set() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 4.0)
        .set("B", 6.0)
        .set("C", 3.0)
        .set("D", 2.0)
        .set("E", 7.0)
        .set("F", 3.0)
        .intersection(&["A", "B"], 2.0)
        .intersection(&["A", "F"], 2.0)
        .intersection(&["B", "C"], 2.0)
        .intersection(&["B", "D"], 1.0)
        .intersection(&["B", "F"], 2.0)
        .intersection(&["C", "D"], 1.0)
        .intersection(&["D", "E"], 1.0)
        .intersection(&["E", "F"], 1.0)
        .intersection(&["A", "B", "F"], 1.0)
        .intersection(&["B", "C", "D"], 1.0)
        .build()
        .unwrap()
}

fn fit_circle(spec: &DiagramSpec, solver: MdsSolver, seed: u64) -> Option<f64> {
    Fitter::<Circle>::new(spec)
        .seed(seed)
        .n_restarts(1)
        .initial_solver(solver)
        .fit()
        .ok()
        .map(|l| l.diag_error())
}

fn fit_ellipse(spec: &DiagramSpec, solver: MdsSolver, seed: u64) -> Option<f64> {
    Fitter::<Ellipse>::new(spec)
        .seed(seed)
        .n_restarts(1)
        .initial_solver(solver)
        .fit()
        .ok()
        .map(|l| l.diag_error())
}

fn fit_circle_pool(spec: &DiagramSpec, pool: &[MdsSolver], seed: u64) -> Option<f64> {
    Fitter::<Circle>::new(spec)
        .seed(seed)
        .initial_solver_pool(pool.to_vec())
        .fit()
        .ok()
        .map(|l| l.diag_error())
}

fn fit_ellipse_pool(spec: &DiagramSpec, pool: &[MdsSolver], seed: u64) -> Option<f64> {
    Fitter::<Ellipse>::new(spec)
        .seed(seed)
        .initial_solver_pool(pool.to_vec())
        .fit()
        .ok()
        .map(|l| l.diag_error())
}

type Runner = fn(&DiagramSpec, MdsSolver, u64) -> Option<f64>;

struct Case {
    name: &'static str,
    spec: DiagramSpec,
    runner: Runner,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "3circle_easy",
            spec: three_circle_easy(),
            runner: fit_circle,
        },
        Case {
            name: "3circle_user",
            spec: three_circle_user_case(),
            runner: fit_circle,
        },
        Case {
            name: "issue28_4set_superset_ellipse",
            spec: issue28_four_set_superset(),
            runner: fit_ellipse,
        },
        Case {
            name: "issue28_6set_ellipse",
            spec: issue28_six_set(),
            runner: fit_ellipse,
        },
    ]
}

fn bench_solvers(c: &mut Criterion) {
    for case in cases() {
        let mut group = c.benchmark_group(format!("initial_layout/{}", case.name));
        // Each fit is on the order of tens-to-hundreds of ms; cap measurement
        // time so the full sweep runs in minutes, not hours.
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(20);

        for (solver, label) in SOLVERS.iter() {
            group.bench_with_input(BenchmarkId::from_parameter(label), solver, |b, &solver| {
                b.iter(|| {
                    let result = (case.runner)(&case.spec, solver, TIMING_SEED);
                    black_box(result)
                });
            });
        }
        group.finish();
    }
}

/// Quality report — runs once per `cargo bench` invocation and prints a
/// solver × spec table of (good-rate, median diag_error) across seeds. Not a
/// Criterion benchmark; just piggybacks on the bench harness so a single
/// `cargo bench` gives both wall-time and quality signal.
fn quality_report(_c: &mut Criterion) {
    println!(
        "\n=== Quality report (n_restarts=1, {} seeds) ===",
        QUALITY_SEEDS.len()
    );
    println!(
        "{:<32} {:<14} {:>9} {:>13} {:>13} {:>13}",
        "spec", "solver", "good/n", "median diag", "min diag", "max diag"
    );

    let good_threshold = 1e-3;

    for case in cases() {
        for (solver, label) in SOLVERS.iter() {
            let mut diags: Vec<f64> = QUALITY_SEEDS
                .iter()
                .filter_map(|&seed| (case.runner)(&case.spec, *solver, seed))
                .collect();
            diags.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if diags.is_empty() {
                println!(
                    "{:<32} {:<14} {:>9} {:>13} {:>13} {:>13}",
                    case.name, label, "0/0", "n/a", "n/a", "n/a"
                );
                continue;
            }

            let good = diags.iter().filter(|&&d| d < good_threshold).count();
            let median = diags[diags.len() / 2];
            let min = *diags.first().unwrap();
            let max = *diags.last().unwrap();

            println!(
                "{:<32} {:<14} {:>4}/{:<3} {:>13.3e} {:>13.3e} {:>13.3e}",
                case.name,
                label,
                good,
                QUALITY_SEEDS.len(),
                median,
                min,
                max,
            );
        }
    }
}

/// Pool benchmark — compares cycling solver pools across `n_restarts=10`
/// against pinned single-solver pools. Wall-time is best-of-10 with
/// parallel restarts; the quality_report below tracks resulting diag_error.
type PoolRunner = fn(&DiagramSpec, &[MdsSolver], u64) -> Option<f64>;

struct PoolCase {
    name: &'static str,
    spec: DiagramSpec,
    runner: PoolRunner,
}

fn pool_cases() -> Vec<PoolCase> {
    vec![
        PoolCase {
            name: "issue28_4set_superset_ellipse",
            spec: issue28_four_set_superset(),
            runner: fit_ellipse_pool,
        },
        PoolCase {
            name: "issue28_6set_ellipse",
            spec: issue28_six_set(),
            runner: fit_ellipse_pool,
        },
        PoolCase {
            name: "3circle_user",
            spec: three_circle_user_case(),
            runner: fit_circle_pool,
        },
    ]
}

const POOLS: [(&[MdsSolver], &str); 3] = [
    (&[MdsSolver::Lbfgs], "lbfgs_only"),
    (&[MdsSolver::TrustRegion], "tr_only"),
    (&[MdsSolver::Lbfgs, MdsSolver::TrustRegion], "lbfgs+tr"),
];

fn bench_pools(c: &mut Criterion) {
    for case in pool_cases() {
        let mut group = c.benchmark_group(format!("pool_n10/{}", case.name));
        // Best-of-10 fits run in parallel via rayon, so wall time is roughly
        // single-fit / num_cores. Give each sample enough budget.
        group.measurement_time(Duration::from_secs(12));
        group.sample_size(10);

        for (pool, label) in POOLS.iter() {
            group.bench_with_input(BenchmarkId::from_parameter(label), pool, |b, pool| {
                b.iter(|| {
                    let result = (case.runner)(&case.spec, pool, TIMING_SEED);
                    black_box(result)
                });
            });
        }
        group.finish();
    }
}

fn pool_quality_report(_c: &mut Criterion) {
    println!(
        "\n=== Pool quality report (n_restarts=10, {} seeds) ===",
        QUALITY_SEEDS.len()
    );
    println!(
        "{:<32} {:<14} {:>9} {:>13} {:>13} {:>13}",
        "spec", "pool", "good/n", "median diag", "min diag", "max diag"
    );

    let good_threshold = 1e-3;

    for case in pool_cases() {
        for (pool, label) in POOLS.iter() {
            let mut diags: Vec<f64> = QUALITY_SEEDS
                .iter()
                .filter_map(|&seed| (case.runner)(&case.spec, pool, seed))
                .collect();
            diags.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if diags.is_empty() {
                println!(
                    "{:<32} {:<14} {:>9} {:>13} {:>13} {:>13}",
                    case.name, label, "0/0", "n/a", "n/a", "n/a"
                );
                continue;
            }

            let good = diags.iter().filter(|&&d| d < good_threshold).count();
            let median = diags[diags.len() / 2];
            let min = *diags.first().unwrap();
            let max = *diags.last().unwrap();

            println!(
                "{:<32} {:<14} {:>4}/{:<3} {:>13.3e} {:>13.3e} {:>13.3e}",
                case.name,
                label,
                good,
                QUALITY_SEEDS.len(),
                median,
                min,
                max,
            );
        }
    }
}

criterion_group!(
    benches,
    bench_solvers,
    quality_report,
    bench_pools,
    pool_quality_report
);
criterion_main!(benches);
