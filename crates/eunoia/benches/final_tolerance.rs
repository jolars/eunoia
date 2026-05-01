//! Sweep LM stopping tolerance across representative specs.
//!
//! Profiling (`perf record` on `examples/fit_profile`) showed that ~58% of
//! total wall time during a fit is spent inside the `levenberg-marquardt`
//! crate's QR factorization and diagonal-solve, called once per LM
//! trust-region iteration. The exact-conic geometry that does the actual
//! "useful" work is only ~6% of self-time. So per-iteration cost is largely
//! fixed, but iteration *count* is tunable via `Fitter::tolerance` (which
//! feeds `with_ftol` / `with_xtol` / `with_gtol` of LM identically).
//!
//! The `levenberg-marquardt` crate exposes three independent stopping
//! criteria — `ftol` (cost-change), `xtol` (parameter-change), `gtol`
//! (gradient ∞-norm) — plus `patience` (iteration cap). Different specs
//! likely trip different knobs first: a converging fit usually trips
//! `ftol` or `xtol`, a stalled basin trips `patience`. This bench has
//! four groups:
//!
//!   - `joint`: sweeps `Fitter::tolerance(_)` (the existing knob, ties
//!     all three to one value) — the baseline you'd actually ship with.
//!   - `xtol`, `ftol`, `gtol`: per-knob sweeps via the new
//!     `Fitter::xtol/ftol/gtol(_)` overrides, with the other two pinned
//!     at the default `1e-6`. Reveals which knob is actually load-bearing
//!     on each spec.
//!
//! All groups are pinned to `Optimizer::LevenbergMarquardt` (no CmaEsLm
//! wrapper) and `n_restarts=1` to isolate the LM stopping signal — the
//! CmaEsLm threshold gate would confound it.
//!
//! After the criterion timing sweep, two quality-only passes run:
//!   1. **probe quality** — same 4 specs × all 4 knobs × 5 tolerances ×
//!      16 seeds. Fast. Spots which knob (if any) actually changes
//!      quality on the probe specs.
//!   2. **corpus validation** — the *full 27-spec corpus* × candidate
//!      joint tolerances ({1e-3, 1e-4, 1e-6 default}) × 16 seeds at the
//!      production `n_restarts=10`. This is the regression check: a
//!      looser tolerance is only safe to ship if good-rate holds across
//!      the whole corpus, not just the probe set. ~1-2 minutes.
//!
//! Run with:    `cargo bench -p eunoia --bench final_tolerance --features corpus`
//! Filter:      `cargo bench -p eunoia --bench final_tolerance --features corpus -- xtol/6set`

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use eunoia::geometry::shapes::{Circle, Ellipse};
use eunoia::geometry::traits::DiagramShape;
use eunoia::spec::{DiagramSpec, DiagramSpecBuilder, InputType};
use eunoia::test_utils::corpus::{
    all as corpus_all, CorpusEntry, Fittable, QUALITY_SEEDS as CORPUS_QUALITY_SEEDS,
};
use eunoia::{Fitter, Optimizer};

const TIMING_SEED: u64 = 42;

const QUALITY_SEEDS: [u64; 16] = [1, 2, 3, 7, 13, 17, 23, 29, 31, 37, 41, 42, 47, 53, 59, 61];

const TOLERANCES: [(f64, &str); 5] = [
    (1e-3, "1e-3"),
    (1e-4, "1e-4"),
    (1e-6, "1e-6_default"),
    (1e-8, "1e-8"),
    (1e-10, "1e-10"),
];

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
        .input_type(InputType::Exclusive)
        .build()
        .unwrap()
}

/// Which LM knob a sweep value should bind to.
#[derive(Copy, Clone)]
enum Knob {
    /// `Fitter::tolerance(_)` — ties xtol/ftol/gtol to one value (current
    /// public API).
    Joint,
    Xtol,
    Ftol,
    Gtol,
}

impl Knob {
    fn label(self) -> &'static str {
        match self {
            Knob::Joint => "joint",
            Knob::Xtol => "xtol",
            Knob::Ftol => "ftol",
            Knob::Gtol => "gtol",
        }
    }
}

const BASELINE_TOL: f64 = 1e-6;

fn apply_tol_circle(fitter: Fitter<'_, Circle>, knob: Knob, tol: f64) -> Fitter<'_, Circle> {
    match knob {
        Knob::Joint => fitter.tolerance(tol),
        // For per-knob sweeps, the other two knobs are pinned at the default
        // `1e-6` via `tolerance(_)`; the per-knob setter then overrides one.
        Knob::Xtol => fitter.tolerance(BASELINE_TOL).xtol(tol),
        Knob::Ftol => fitter.tolerance(BASELINE_TOL).ftol(tol),
        Knob::Gtol => fitter.tolerance(BASELINE_TOL).gtol(tol),
    }
}

fn apply_tol_ellipse(fitter: Fitter<'_, Ellipse>, knob: Knob, tol: f64) -> Fitter<'_, Ellipse> {
    match knob {
        Knob::Joint => fitter.tolerance(tol),
        Knob::Xtol => fitter.tolerance(BASELINE_TOL).xtol(tol),
        Knob::Ftol => fitter.tolerance(BASELINE_TOL).ftol(tol),
        Knob::Gtol => fitter.tolerance(BASELINE_TOL).gtol(tol),
    }
}

fn fit_circle(spec: &DiagramSpec, knob: Knob, tol: f64, seed: u64) -> Option<f64> {
    apply_tol_circle(
        Fitter::<Circle>::new(spec)
            .seed(seed)
            .n_restarts(1)
            .optimizer(Optimizer::LevenbergMarquardt),
        knob,
        tol,
    )
    .fit()
    .ok()
    .map(|l| l.diag_error())
}

fn fit_ellipse(spec: &DiagramSpec, knob: Knob, tol: f64, seed: u64) -> Option<f64> {
    apply_tol_ellipse(
        Fitter::<Ellipse>::new(spec)
            .seed(seed)
            .n_restarts(1)
            .optimizer(Optimizer::LevenbergMarquardt),
        knob,
        tol,
    )
    .fit()
    .ok()
    .map(|l| l.diag_error())
}

type Runner = fn(&DiagramSpec, Knob, f64, u64) -> Option<f64>;

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
            name: "4set_superset_ellipse",
            spec: issue28_four_set_superset(),
            runner: fit_ellipse,
        },
        Case {
            name: "6set_ellipse",
            spec: issue28_six_set(),
            runner: fit_ellipse,
        },
    ]
}

const KNOBS: [Knob; 4] = [Knob::Joint, Knob::Xtol, Knob::Ftol, Knob::Gtol];

fn bench_tolerance(c: &mut Criterion) {
    for knob in KNOBS {
        for case in cases() {
            let mut group = c.benchmark_group(format!("{}/{}", knob.label(), case.name));
            group.measurement_time(Duration::from_secs(8));
            group.sample_size(20);

            for (tol, label) in TOLERANCES.iter() {
                group.bench_with_input(BenchmarkId::from_parameter(label), tol, |b, &tol| {
                    b.iter(|| {
                        let result = (case.runner)(&case.spec, knob, tol, TIMING_SEED);
                        black_box(result)
                    });
                });
            }
            group.finish();
        }
    }
}

/// Quality report — runs once per `cargo bench` invocation and prints a
/// tolerance × spec table of (good-rate, median diag_error) across seeds.
/// Read alongside the wall-time numbers above to spot the sweet spot:
/// looser tolerance is only a win if `diag_error` doesn't degrade.
fn quality_report(_c: &mut Criterion) {
    println!(
        "\n=== LM tolerance quality report (n_restarts=1, {} seeds; non-swept knobs at {:.0e}) ===",
        QUALITY_SEEDS.len(),
        BASELINE_TOL,
    );
    println!(
        "{:<7} {:<26} {:<14} {:>9} {:>13} {:>13} {:>13}",
        "knob", "spec", "value", "good/n", "median diag", "min diag", "max diag"
    );

    let good_threshold = 1e-3;

    for knob in KNOBS {
        for case in cases() {
            for (tol, label) in TOLERANCES.iter() {
                let mut diags: Vec<f64> = QUALITY_SEEDS
                    .iter()
                    .filter_map(|&seed| (case.runner)(&case.spec, knob, *tol, seed))
                    .collect();
                diags.sort_by(|a, b| a.partial_cmp(b).unwrap());

                if diags.is_empty() {
                    println!(
                        "{:<7} {:<26} {:<14} {:>9} {:>13} {:>13} {:>13}",
                        knob.label(),
                        case.name,
                        label,
                        "0/0",
                        "n/a",
                        "n/a",
                        "n/a"
                    );
                    continue;
                }

                let good = diags.iter().filter(|&&d| d < good_threshold).count();
                let median = diags[diags.len() / 2];
                let min = *diags.first().unwrap();
                let max = *diags.last().unwrap();

                println!(
                    "{:<7} {:<26} {:<14} {:>4}/{:<3} {:>13.3e} {:>13.3e} {:>13.3e}",
                    knob.label(),
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
}

/// Candidate joint-tolerance values to validate across the full corpus.
/// Limited on purpose: the probe pass varies all 5 values × 4 knobs ×
/// 4 specs to spot the candidates worth scaling up. Only the joint
/// `Fitter::tolerance` setter is consulted here because that's the public
/// API a downstream user would tune; per-knob overrides are diagnostic.
const CORPUS_CANDIDATES: [(f64, &str); 3] =
    [(1e-3, "1e-3"), (1e-4, "1e-4"), (1e-6, "1e-6_default")];

/// Median runtime in ms across seeds for one (config, spec, shape) cell.
fn median_ms_circle(spec: &DiagramSpec, tol: f64) -> Option<(f64, f64, usize)> {
    measure_corpus_cell::<Circle>(spec, tol)
}

fn median_ms_ellipse(spec: &DiagramSpec, tol: f64) -> Option<(f64, f64, usize)> {
    measure_corpus_cell::<Ellipse>(spec, tol)
}

fn measure_corpus_cell<S: DiagramShape + Copy + 'static>(
    spec: &DiagramSpec,
    tol: f64,
) -> Option<(f64, f64, usize)> {
    let mut diags: Vec<(f64, f64)> = Vec::with_capacity(CORPUS_QUALITY_SEEDS.len());
    for &seed in CORPUS_QUALITY_SEEDS.iter() {
        let t = std::time::Instant::now();
        let layout = Fitter::<S>::new(spec)
            .seed(seed)
            .optimizer(Optimizer::LevenbergMarquardt)
            .tolerance(tol)
            .fit()
            .ok()?;
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        diags.push((layout.diag_error(), elapsed_ms));
    }
    if diags.is_empty() {
        return None;
    }
    let n = diags.len();
    let mut diag_only: Vec<f64> = diags.iter().map(|x| x.0).collect();
    let mut ms_only: Vec<f64> = diags.iter().map(|x| x.1).collect();
    diag_only.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms_only.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((diag_only[n / 2], ms_only[n / 2], n))
}

/// Quality-only validation across the full corpus.
///
/// Pinned to `Optimizer::LevenbergMarquardt` and the production
/// `n_restarts=10` so the result reflects what a downstream user would
/// see. Each (spec × tolerance × shape) cell reports median diag_error
/// and median wall time over the 16 corpus seeds. Read this table to
/// decide whether a candidate tolerance is safe to ship: the answer is
/// yes only if good-rate doesn't drop on any spec.
fn corpus_validation(_c: &mut Criterion) {
    println!(
        "\n=== Corpus tolerance validation ({} specs × {} seeds, Optimizer::LevenbergMarquardt, n_restarts=10) ===",
        corpus_all().len(),
        CORPUS_QUALITY_SEEDS.len(),
    );
    println!(
        "{:<7} {:<32} {:<14} {:>13} {:>13}  {:<8}",
        "shape", "spec", "tolerance", "median diag", "median ms", "verdict"
    );

    for entry in corpus_all() {
        let spec = (entry.build)();
        let circle_active = matches!(entry.fittable_circle, Fittable::Normal);
        let ellipse_active = matches!(entry.fittable_ellipse, Fittable::Normal);

        let ceiling_circle = entry.ceiling_circle();
        let ceiling_ellipse = entry.ceiling_ellipse();

        for (tol, label) in CORPUS_CANDIDATES.iter() {
            if circle_active {
                report_corpus_row(
                    "circle",
                    entry,
                    label,
                    median_ms_circle(&spec, *tol),
                    ceiling_circle,
                );
            }
            if ellipse_active {
                report_corpus_row(
                    "ellipse",
                    entry,
                    label,
                    median_ms_ellipse(&spec, *tol),
                    ceiling_ellipse,
                );
            }
        }
    }
}

fn report_corpus_row(
    shape: &str,
    entry: &CorpusEntry,
    label: &str,
    cell: Option<(f64, f64, usize)>,
    ceiling: f64,
) {
    match cell {
        None => println!(
            "{:<7} {:<32} {:<14} {:>13} {:>13}  {:<8}",
            shape, entry.name, label, "n/a", "n/a", "fit_err"
        ),
        Some((median_diag, median_ms, _n)) => {
            let verdict = if median_diag <= ceiling {
                "ok"
            } else {
                "REGRESS"
            };
            println!(
                "{:<7} {:<32} {:<14} {:>13.3e} {:>13.2}  {:<8}",
                shape, entry.name, label, median_diag, median_ms, verdict
            );
        }
    }
}

criterion_group!(benches, bench_tolerance, quality_report, corpus_validation);
criterion_main!(benches);
