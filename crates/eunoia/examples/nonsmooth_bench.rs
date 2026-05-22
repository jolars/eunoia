//! Non-smooth-loss fallback benchmark: plain Nelder-Mead vs CmaEsLm.
//!
//! The non-smooth losses (`MaxAbsolute`, `MaxSquared`, `SumAbsolute`,
//! `SumAbsoluteRegionError`, `DiagError`) have no usable gradient, so the
//! gradient optimizers fall back to derivative-free Nelder-Mead. The
//! `Fitter` *default* optimizer is `CmaEsLm`, which for these losses resolves
//! to "NM, plus a conditional bounded-CMA-ES escape with NM polish, keeping
//! the better" — a strictly non-regressing escape layered on top of NM.
//!
//! This benchmark quantifies what that escape buys on the non-smooth losses,
//! over the full corpus × {Circle, Ellipse} × a few seeds. For each
//! (spec, seed) the two optimizers run back-to-back so results pair cleanly.
//! We report the median of the *loss being minimized* and of `diag_error`
//! (a common yardstick across loss types), median wall time, and the share
//! of paired cases where the escape reaches a strictly lower loss than NM.
//!
//! Background: a pure bounded-CMA-ES fallback (escape with no polish) was
//! also measured and lost to NM in every cell — often by 1–2 orders of
//! magnitude — because CMA-ES explores away from the already-good MDS init
//! that NM refines locally. Hence the comparison here is NM vs *NM + escape*,
//! not NM vs CMA-ES. See issue #45.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example nonsmooth_bench --features corpus,parallel
//! ```

#[cfg(not(feature = "corpus"))]
fn main() {
    eprintln!("This example requires the `corpus` feature.");
    eprintln!("Run with: cargo run --release --example nonsmooth_bench --features corpus,parallel");
    std::process::exit(1);
}

#[cfg(feature = "corpus")]
fn main() {
    bench::run();
}

#[cfg(feature = "corpus")]
mod bench {
    use std::time::Instant;

    use eunoia::geometry::shapes::{Circle, Ellipse};
    use eunoia::geometry::traits::DiagramShape;
    use eunoia::loss::LossType;
    use eunoia::test_utils::corpus::{CorpusEntry, Fittable, all};
    use eunoia::{Fitter, Optimizer};

    const SEEDS: [u64; 3] = [1, 42, 7];

    fn losses() -> Vec<(&'static str, LossType)> {
        vec![
            ("MaxAbsolute", LossType::MaxAbsolute),
            ("MaxSquared", LossType::MaxSquared),
            ("SumAbsolute", LossType::SumAbsoute),
            ("SumAbsRegionErr", LossType::SumAbsoluteRegionError),
            ("DiagError", LossType::DiagError),
        ]
    }

    #[derive(Clone, Copy)]
    struct Sample {
        loss: f64,
        diag: f64,
        ms: f64,
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

    fn summarize(s: &[Sample]) -> (f64, f64, f64) {
        let mut l: Vec<f64> = s.iter().map(|x| x.loss).collect();
        let mut d: Vec<f64> = s.iter().map(|x| x.diag).collect();
        let mut m: Vec<f64> = s.iter().map(|x| x.ms).collect();
        (median(&mut l), median(&mut d), median(&mut m))
    }

    fn fit_one<S: DiagramShape + Copy + 'static>(
        spec: &eunoia::spec::DiagramSpec,
        loss: LossType,
        opt: Optimizer,
        seed: u64,
    ) -> Option<Sample> {
        let t = Instant::now();
        let res = Fitter::<S>::new(spec)
            .loss_type(loss)
            .optimizer(opt)
            .seed(seed)
            .fit();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match res {
            Ok(layout) if layout.loss().is_finite() => Some(Sample {
                loss: layout.loss(),
                diag: layout.diag_error(),
                ms,
            }),
            _ => None,
        }
    }

    fn run_shape<S: DiagramShape + Copy + 'static>(
        shape_name: &str,
        fittable: fn(&CorpusEntry) -> Fittable,
    ) {
        // NM = plain Nelder-Mead fallback. POL = CmaEsLm, which for a
        // non-smooth loss is "NM, plus a conditional CMA-ES escape with NM
        // polish, keeping the better" — strictly non-regressing vs NM.
        println!("\n## {shape_name}\n");
        println!(
            "| loss | NM loss | POL loss | NM diag | POL diag | NM ms | POL ms | POL loss-win% | n |"
        );
        println!("|---|---|---|---|---|---|---|---|---|");

        for (loss_name, loss) in losses() {
            let mut nm_s: Vec<Sample> = Vec::new();
            let mut pol_s: Vec<Sample> = Vec::new();
            let mut pol_loss_wins = 0usize;
            let mut paired = 0usize;

            for entry in all() {
                if matches!(fittable(entry), Fittable::Skip(_)) {
                    continue;
                }
                let spec = (entry.build)();
                for &seed in &SEEDS {
                    let nm = fit_one::<S>(&spec, loss, Optimizer::NelderMead, seed);
                    let pol = fit_one::<S>(&spec, loss, Optimizer::CmaEsLm, seed);
                    if let (Some(nm), Some(pol)) = (nm, pol) {
                        // Strictly-lower loss with a small relative slack so
                        // ties (escape didn't fire / matched NM) don't count.
                        if pol.loss < nm.loss * (1.0 - 1e-6) {
                            pol_loss_wins += 1;
                        }
                        nm_s.push(nm);
                        pol_s.push(pol);
                        paired += 1;
                    }
                }
            }

            let (nm_l, nm_d, nm_ms) = summarize(&nm_s);
            let (pol_l, pol_d, pol_ms) = summarize(&pol_s);
            let win = if paired > 0 {
                100.0 * pol_loss_wins as f64 / paired as f64
            } else {
                f64::NAN
            };
            println!(
                "| {loss_name} | {nm_l:.3e} | {pol_l:.3e} | {nm_d:.3e} | {pol_d:.3e} | \
                 {nm_ms:.1} | {pol_ms:.1} | {win:.0}% | {paired} |"
            );
        }
    }

    pub fn run() {
        println!("# Non-smooth fallback benchmark: Nelder-Mead vs CmaEsLm (NM + CMA escape)");
        println!("\nseeds = {SEEDS:?}, n_restarts = 10 (default), max_iterations = 200 (default).");
        println!(
            "Median over (corpus specs × seeds). `loss` = the metric being minimized \
             (comparable within a row); `diag_error` = common yardstick; lower is better. \
             POL is ≥ NM by construction; the win% and the loss/diag deltas show how often, \
             and by how much, the CMA-ES escape improves on plain NM."
        );
        run_shape::<Circle>("Circle", |e| e.fittable_circle);
        run_shape::<Ellipse>("Ellipse", |e| e.fittable_ellipse);
    }
}
