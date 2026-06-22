//! MADS speed-lever sweep: poll-size floor (`tolerance`) vs eval budget.
//!
//! `run_mads` wires MADS's poll-size convergence floor to `config.tolerance`
//! (default `1e-6`) and caps cost evals at `max_iterations·(2n+1)`. MADS beats
//! Nelder-Mead on quality but costs ~10-20× the wall time; this probe asks which
//! knob reclaims that time.
//!
//! On Ellipse (the hardest, most non-smooth shape), for one smooth loss
//! (`SumSquared`) and one non-smooth loss (`MaxSquared`), it sweeps two levers
//! against the NM baseline, reporting median loss / `diag_error` / per-fit wall
//! time over the corpus × seeds:
//!   - **Tolerance** (poll-size floor) `∈ {1e-6, 1e-4, 1e-3, 1e-2}` at the
//!     default 200-iter budget. Empirically a *weak* lever — MADS runs into the
//!     eval-budget cap before the mesh reaches even `1e-3`, so raising the floor
//!     barely moves wall time (and `1e-2` costs quality).
//!   - **Eval budget** (`max_iterations ∈ {100, 50, 25}`) at the default floor —
//!     the *real* lever: a clean quality/speed Pareto. At `iter100` MADS still
//!     beats NM on the non-smooth loss at half the cost; by `iter50` its
//!     best-of-restarts converges to NM-parity, i.e. MADS's quality edge is
//!     bought linearly with evals.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example mads_tolerance --features corpus,parallel
//! ```

#[cfg(not(feature = "corpus"))]
fn main() {
    eprintln!("This example requires the `corpus` feature.");
    eprintln!("Run with: cargo run --release --example mads_tolerance --features corpus,parallel");
    std::process::exit(1);
}

#[cfg(feature = "corpus")]
fn main() {
    bench::run();
}

#[cfg(feature = "corpus")]
mod bench {
    use std::time::Instant;

    use eunoia::geometry::shapes::Ellipse;
    use eunoia::loss::LossType;
    use eunoia::test_utils::corpus::{Fittable, all};
    use eunoia::{Fitter, Optimizer};

    const SEEDS: [u64; 3] = [1, 42, 7];

    /// One swept config: display name, optimizer, optional `tolerance`
    /// (poll-size floor), optional `max_iterations` (eval-budget driver).
    /// `None` tolerance = the optimizer's default. The first block holds the
    /// tolerance sweep (fixed default budget); the budget rows at the end probe
    /// the *other* lever. `NM` is the speed/quality baseline.
    struct Cfg {
        name: &'static str,
        opt: Optimizer,
        tol: Option<f64>,
        max_iter: Option<usize>,
    }

    fn configs() -> Vec<Cfg> {
        let c = |name, opt, tol, max_iter| Cfg {
            name,
            opt,
            tol,
            max_iter,
        };
        vec![
            c("NM", Optimizer::NelderMead, None, None),
            // Tolerance (poll-size floor) sweep at the default 200-iter budget.
            c("MADS@tol1e-6", Optimizer::Mads, Some(1e-6), None),
            c("MADS@tol1e-4", Optimizer::Mads, Some(1e-4), None),
            c("MADS@tol1e-3", Optimizer::Mads, Some(1e-3), None),
            c("MADS@tol1e-2", Optimizer::Mads, Some(1e-2), None),
            // Eval-budget sweep at the default 1e-6 floor — the other lever.
            c("MADS@iter100", Optimizer::Mads, None, Some(100)),
            c("MADS@iter50", Optimizer::Mads, None, Some(50)),
            c("MADS@iter25", Optimizer::Mads, None, Some(25)),
        ]
    }

    fn losses() -> Vec<(&'static str, LossType)> {
        vec![
            ("SumSquared (smooth)", LossType::sum_squared()),
            ("MaxSquared (non-smooth)", LossType::MaxSquared),
        ]
    }

    fn median(xs: &mut [f64]) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) / 2.0
        }
    }

    fn fit_one(
        spec: &eunoia::spec::DiagramSpec,
        loss: LossType,
        cfg: &Cfg,
        seed: u64,
    ) -> Option<(f64, f64, f64)> {
        let mut fitter = Fitter::<Ellipse>::new(spec)
            .loss_type(loss)
            .optimizer(cfg.opt)
            .seed(seed);
        if let Some(t) = cfg.tol {
            fitter = fitter.tolerance(t);
        }
        if let Some(mi) = cfg.max_iter {
            fitter = fitter.max_iterations(mi);
        }
        let t = Instant::now();
        let res = fitter.fit();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match res {
            Ok(layout) if layout.loss().is_finite() => {
                Some((layout.loss(), layout.diag_error(), ms))
            }
            _ => None,
        }
    }

    pub fn run() {
        println!("# MADS speed-lever sweep (tolerance floor vs eval budget) — Ellipse\n");
        println!(
            "seeds = {SEEDS:?}, n_restarts = 10 (default); `max_iterations` 200 unless noted."
        );
        println!(
            "Median over (corpus specs × seeds). `loss` = metric minimized (comparable within a \
             loss block); `diag` = common yardstick; `ms` = per-fit wall time (all 10 restarts, \
             `parallel` feature on). Lower is better. `tol*` rows sweep the poll-size floor; \
             `iter*` rows sweep the eval budget. NM is the baseline.\n"
        );

        for (loss_name, loss) in losses() {
            println!("## {loss_name}\n");
            println!("| config | median loss | median diag | median ms | n |");
            println!("|---|---|---|---|---|");
            for cfg in configs() {
                let mut ls = Vec::new();
                let mut ds = Vec::new();
                let mut ms = Vec::new();
                for entry in all() {
                    if matches!(entry.fittable_ellipse, Fittable::Skip(_)) {
                        continue;
                    }
                    let spec = (entry.build)();
                    for &seed in &SEEDS {
                        if let Some((l, d, t)) = fit_one(&spec, loss, &cfg, seed) {
                            ls.push(l);
                            ds.push(d);
                            ms.push(t);
                        }
                    }
                }
                let n = ls.len();
                println!(
                    "| {} | {:.3e} | {:.3e} | {:.1} | {n} |",
                    cfg.name,
                    median(&mut ls),
                    median(&mut ds),
                    median(&mut ms),
                );
            }
            println!();
        }
    }
}
