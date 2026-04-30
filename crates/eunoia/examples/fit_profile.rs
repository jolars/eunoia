//! Deterministic profiling target.
//!
//! Runs N fits of a representative spec — no Criterion warmup, no parallel
//! restarts — so flamegraphs and `perf annotate` attribute time directly
//! to the fitting hot path. A single fit is often <10ms which is too short
//! for useful 100Hz sampling; default iteration count gives ~1s of work.
//!
//! Usage:
//!
//! ```text
//! # Flamegraph (SVG written to ./flamegraph.svg)
//! RAYON_NUM_THREADS=1 cargo flamegraph --profile profiling \
//!     --example fit_profile -- 6set 200
//!
//! # perf record + annotate (line-level)
//! cargo build --profile profiling --example fit_profile
//! RAYON_NUM_THREADS=1 perf record --call-graph fp -F 999 -- \
//!     ./target/profiling/examples/fit_profile 6set 200
//! perf report
//! perf annotate
//!
//! # samply (line-level call tree in Firefox-Profiler UI)
//! cargo build --profile profiling --example fit_profile
//! RAYON_NUM_THREADS=1 samply record \
//!     ./target/profiling/examples/fit_profile 6set 200
//!
//! # cargo-show-asm — inspect codegen for a hot function the profiler points at
//! cargo asm --profile profiling -p eunoia --lib \
//!     eunoia::geometry::shapes::ellipse::region_boundary_arcs_ellipse
//! ```
//!
//! Args: `<case>` (3circle | 4set | 6set, default 6set), `<iters>` (default 200).

use std::env;
use std::time::Instant;

use eunoia::geometry::shapes::Ellipse;
use eunoia::spec::{DiagramSpec, DiagramSpecBuilder, InputType};
use eunoia::Fitter;

fn three_circle() -> DiagramSpec {
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

fn four_set_superset() -> DiagramSpec {
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

fn six_set() -> DiagramSpec {
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

fn main() {
    let case = env::args().nth(1).unwrap_or_else(|| "6set".to_string());
    let iters: u64 = env::args()
        .nth(2)
        .map(|s| s.parse().expect("iters must be a non-negative integer"))
        .unwrap_or(200);

    let (label, spec) = match case.as_str() {
        "3circle" => ("3circle", three_circle()),
        "4set" => ("4set_superset", four_set_superset()),
        "6set" => ("6set", six_set()),
        other => {
            eprintln!("unknown case `{other}`; use 3circle | 4set | 6set");
            std::process::exit(2);
        }
    };

    let mut last_diag = 0.0;
    let t = Instant::now();
    for i in 0..iters {
        let layout = Fitter::<Ellipse>::new(&spec)
            .seed(42 + i)
            .n_restarts(1)
            .fit()
            .expect("fit failed");
        last_diag = layout.diag_error();
    }
    let elapsed = t.elapsed();

    eprintln!(
        "case={label} iters={iters} total={:?} per_fit={:.3?} last_diag={:.3e}",
        elapsed,
        elapsed / iters.max(1) as u32,
        last_diag
    );
}
