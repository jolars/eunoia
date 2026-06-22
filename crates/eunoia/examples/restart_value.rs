//! Set-size-specific optimization-policy probe.
//!
//! Answers three questions the default pipeline raises for *small* diagrams,
//! bucketed by set count and swept across loss functions:
//!
//!   A. **Do the 10 restarts buy anything, and what's the minimum viable
//!      restart count?** For each spec we run the full pipeline at
//!      `n_restarts ∈ {1, 2, 3, 5, 10}` (same master seed, so restart prefixes
//!      are shared and best-loss is monotone in the count). `min-k` is the
//!      smallest restart count at which the *worst* seed already reaches its
//!      own loss@10 plateau (within 1%) — i.e. the restart count you could
//!      safely drop to. `spread@1` is the across-seed relative spread of the
//!      single-restart loss: ~0 means restart 0 is effectively deterministic
//!      (the canonical Venn warm-start governs) so one restart is reliable;
//!      large means a single restart is a gamble.
//!
//!   B. **Does the CMA-ES escape ever fire / help?** We compare plain
//!      Levenberg-Marquardt at `n_restarts=10` (`lm`) against the default
//!      `CmaEsTrf` ladder's `loss@10` (which *is* the escape path). `fire rate`
//!      is the fraction of single-restart LM fits landing above the 1e-3
//!      fallback threshold (only then can the escape stage run). `seeds helped`
//!      counts seeds where the escape actually lowered the loss.
//!
//!   C. **Does the loss function change the answer?** Everything is swept over
//!      `{SumSquared (default), SumAbsolute, MaxAbsolute}`. Loss *values* are
//!      not comparable across loss types, so cross-loss reads use `diag_error`
//!      (scale-invariant) and the relative within-loss metrics.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example restart_value --features corpus,parallel
//! ```
//!
//! Optional: `RESTART_SEEDS=8` caps the seed count (default = all 16
//! `QUALITY_SEEDS`) for a faster pass.

#[cfg(not(feature = "corpus"))]
fn main() {
    eprintln!("This example requires the `corpus` feature.");
    eprintln!("Run with: cargo run --release --example restart_value --features corpus,parallel");
    std::process::exit(1);
}

#[cfg(feature = "corpus")]
fn main() {
    restart_value::run();
}

#[cfg(feature = "corpus")]
mod restart_value {
    use std::env;
    use std::time::Instant;

    use eunoia::geometry::shapes::{Circle, Ellipse};
    use eunoia::geometry::traits::DiagramShape;
    use eunoia::loss::LossType;
    use eunoia::test_utils::corpus::{CorpusEntry, Fittable, QUALITY_SEEDS, all};
    use eunoia::{Fitter, Optimizer};

    /// The fallback threshold the default `CmaEsTrf` uses: plain LM at or below
    /// this skips the escape stage. Mirrors `FinalLayoutConfig`'s default. Note
    /// this is compared against the *raw* loss value, so it is calibrated for
    /// the scale-invariant `SumSquared` and means something different under
    /// `SumAbsolute`/`MaxAbsolute` (whose values track raw areas).
    const FALLBACK_THRESHOLD: f64 = 1e-3;

    /// Restart counts swept in experiment A.
    const RESTART_LADDER: [usize; 5] = [1, 2, 3, 5, 10];

    /// `diag_error` below which a fit is "already good": this is the standard,
    /// scale- and loss-invariant quality metric the corpus ceilings use, so it
    /// gates "is there anything left to improve?" uniformly across loss types.
    const DIAG_FLOOR: f64 = 1e-3;

    /// Minimum relative loss drop to count a seed as genuinely "helped" (1%).
    const REL_GAIN_MIN: f64 = 1e-2;

    /// Tolerance for "reached the loss@10 plateau" when computing `min-k`.
    const PLATEAU_REL: f64 = 1e-2;

    fn losses() -> [(&'static str, LossType); 3] {
        [
            ("sumsq", LossType::SumSquared),
            ("sumabs", LossType::SumAbsolute),
            ("maxabs", LossType::MaxAbsolute),
        ]
    }

    fn seeds() -> Vec<u64> {
        match env::var("RESTART_SEEDS").ok().and_then(|s| s.parse().ok()) {
            Some(n) => QUALITY_SEEDS.iter().copied().take(n).collect(),
            None => QUALITY_SEEDS.to_vec(),
        }
    }

    pub fn run() {
        let seeds = seeds();
        eprintln!(
            "restart_value: {} specs × {{circle, ellipse}} × {} losses × {} seeds",
            all().len(),
            losses().len(),
            seeds.len()
        );

        println!("# Set-size-specific optimization policy probe\n");
        println!("Seeds (n={}): {:?}\n", seeds.len(), seeds);
        println!(
            "`min-k` = worst-seed minimum restart count reaching the loss@10 plateau (≤1% above). \
             `spread@1` = across-seed relative spread of single-restart loss \
             (≈0 ⇒ restart 0 deterministic ⇒ 1 restart reliable). \
             `diag@1`/`diag@10` = median diag_error (scale-invariant) at 1 vs 10 restarts.\n"
        );

        let mut all_rows: Vec<Row> = Vec::new();
        run_shape::<Circle>("circle", &seeds, &mut all_rows);
        run_shape::<Ellipse>("ellipse", &seeds, &mut all_rows);

        render_cross_loss_summary(&all_rows);
    }

    fn run_shape<S>(shape: &'static str, seeds: &[u64], all_rows: &mut Vec<Row>)
    where
        S: DiagramShape + Copy + 'static,
    {
        println!("## Shape: {shape}\n");
        for (lname, loss) in losses() {
            eprintln!("== shape: {shape} | loss: {lname} ==");
            println!("### Loss: `{lname}`\n");

            let mut rows: Vec<SpecRow> = Vec::new();
            for entry in all() {
                if !matches!(fittable_for::<S>(entry), Fittable::Normal) {
                    continue;
                }
                let spec = (entry.build)();
                let n_sets = spec.set_names().len();
                let started = Instant::now();
                if let Some(row) = measure_spec::<S>(entry.name, n_sets, &spec, loss, seeds) {
                    rows.push(row);
                }
                eprintln!(
                    "  [{shape}/{lname}] {:32} n={n_sets} in {:.2}s",
                    entry.name,
                    started.elapsed().as_secs_f64()
                );
            }
            rows.sort_by_key(|r| (r.n_sets, r.spec));
            render_restart_table(&rows);
            render_escape_table(&rows);
            for r in &rows {
                all_rows.push(Row {
                    shape,
                    loss: lname,
                    inner: r.clone(),
                });
            }
        }
    }

    #[derive(Clone)]
    struct SpecRow {
        spec: &'static str,
        n_sets: usize,
        n_seeds: usize,
        // Experiment A.
        loss_at_restart: [f64; RESTART_LADDER.len()],
        diag_at_1: f64,
        diag_at_10: f64,
        spread1_rel: f64,
        min_k_worst: usize,
        seeds_helped_by_restarts: usize,
        median_rel_gain_restarts: f64,
        // Experiment B.
        lm_loss_med: f64,
        cmaes_loss_med: f64,
        fire_rate: f64,
        seeds_helped_by_escape: usize,
        median_rel_gain_escape: f64,
        lm_ms_med: f64,
    }

    struct Row {
        shape: &'static str,
        loss: &'static str,
        inner: SpecRow,
    }

    /// One fit → `(loss, diag_error, elapsed_ms)`. Panics are swallowed.
    fn fit_one<S>(
        spec: &eunoia::spec::DiagramSpec,
        loss: LossType,
        seed: u64,
        n_restarts: usize,
        cfg: impl Fn(Fitter<'_, S>) -> Fitter<'_, S>,
    ) -> Option<(f64, f64, f64)>
    where
        S: DiagramShape + Copy + 'static,
    {
        let started = Instant::now();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cfg(Fitter::<S>::new(spec)
                .seed(seed)
                .n_restarts(n_restarts)
                .loss_type(loss))
            .fit()
        }));
        let ms = started.elapsed().as_secs_f64() * 1e3;
        match res {
            Ok(Ok(layout)) if layout.loss().is_finite() => {
                Some((layout.loss(), layout.diag_error(), ms))
            }
            _ => None,
        }
    }

    fn measure_spec<S>(
        spec_name: &'static str,
        n_sets: usize,
        spec: &eunoia::spec::DiagramSpec,
        loss: LossType,
        seeds: &[u64],
    ) -> Option<SpecRow>
    where
        S: DiagramShape + Copy + 'static,
    {
        let mut loss_by_k: Vec<Vec<f64>> = vec![Vec::new(); RESTART_LADDER.len()];
        let mut loss1_samples: Vec<f64> = Vec::new();
        let mut diag1: Vec<f64> = Vec::new();
        let mut diag10: Vec<f64> = Vec::new();
        let mut rel_gain_restarts: Vec<f64> = Vec::new();
        let mut seeds_helped_restarts = 0usize;
        let mut min_k_worst = 1usize;

        // Experiment A: default-optimizer (CmaEsTrf) restart ladder.
        for &seed in seeds {
            let mut per_k = [f64::NAN; RESTART_LADDER.len()];
            let mut diag_seed1 = f64::NAN;
            for (i, &k) in RESTART_LADDER.iter().enumerate() {
                if let Some((l, d, _)) = fit_one::<S>(spec, loss, seed, k, |f| f) {
                    per_k[i] = l;
                    loss_by_k[i].push(l);
                    if i == 0 {
                        diag1.push(d);
                        diag_seed1 = d;
                    }
                    if k == 10 {
                        diag10.push(d);
                    }
                }
            }
            let l1 = per_k[0];
            let l10 = per_k[RESTART_LADDER.len() - 1];
            // Only an imperfect single-restart fit can need more restarts; a
            // perfect one's sub-floor loss jitter must not inflate min-k/spread.
            let imperfect = diag_seed1.is_finite() && diag_seed1 > DIAG_FLOOR;
            if imperfect && l1.is_finite() {
                loss1_samples.push(l1);
            }
            // min-k for this seed: smallest ladder k within PLATEAU_REL of l10.
            if imperfect && l1.is_finite() && l10.is_finite() {
                let target = l10 * (1.0 + PLATEAU_REL) + f64::EPSILON;
                let mut k_seed = RESTART_LADDER[RESTART_LADDER.len() - 1];
                for (i, &k) in RESTART_LADDER.iter().enumerate() {
                    if per_k[i].is_finite() && per_k[i] <= target {
                        k_seed = k;
                        break;
                    }
                }
                min_k_worst = min_k_worst.max(k_seed);
            }
            if imperfect && l1.is_finite() && l10.is_finite() && l1 > 0.0 {
                let gain = ((l1 - l10) / l1).max(0.0);
                rel_gain_restarts.push(gain);
                if gain >= REL_GAIN_MIN {
                    seeds_helped_restarts += 1;
                }
            }
        }

        // Experiment B: plain LM @10 (escape off) vs default ladder's @10
        // (escape on), plus LM@1 for the fire-rate proxy.
        let mut lm_loss = Vec::new();
        let mut lm_diag = Vec::new();
        let mut lm_ms = Vec::new();
        let mut rel_gain_escape = Vec::new();
        let mut seeds_helped_escape = 0usize;
        let mut fire_count = 0usize;
        let mut fire_total = 0usize;

        for (si, &seed) in seeds.iter().enumerate() {
            let lm = fit_one::<S>(spec, loss, seed, 10, |f| {
                f.optimizer(Optimizer::LevenbergMarquardt)
            });
            if let Some((l, d, ms)) = lm {
                lm_loss.push(l);
                lm_diag.push(d);
                lm_ms.push(ms);
                // escape help vs the default ladder's @10 for the same seed.
                let cmaes_l = loss_by_k[RESTART_LADDER.len() - 1].get(si).copied();
                if let Some(cl) = cmaes_l
                    && d > DIAG_FLOOR
                    && l > 0.0
                {
                    let gain = ((l - cl) / l).max(0.0);
                    rel_gain_escape.push(gain);
                    if gain >= REL_GAIN_MIN {
                        seeds_helped_escape += 1;
                    }
                }
            }
            if let Some((l1, _, _)) = fit_one::<S>(spec, loss, seed, 1, |f| {
                f.optimizer(Optimizer::LevenbergMarquardt)
            }) {
                fire_total += 1;
                if l1 > FALLBACK_THRESHOLD {
                    fire_count += 1;
                }
            }
        }

        let loss_at_restart = std::array::from_fn(|i| median(&loss_by_k[i]));
        let cmaes_loss_med = loss_at_restart[RESTART_LADDER.len() - 1];

        Some(SpecRow {
            spec: spec_name,
            n_sets,
            n_seeds: seeds.len(),
            loss_at_restart,
            diag_at_1: median(&diag1),
            diag_at_10: median(&diag10),
            spread1_rel: rel_spread(&loss1_samples),
            min_k_worst,
            seeds_helped_by_restarts: seeds_helped_restarts,
            median_rel_gain_restarts: median(&rel_gain_restarts),
            lm_loss_med: median(&lm_loss),
            cmaes_loss_med,
            fire_rate: if fire_total == 0 {
                f64::NAN
            } else {
                fire_count as f64 / fire_total as f64
            },
            seeds_helped_by_escape: seeds_helped_escape,
            median_rel_gain_escape: median(&rel_gain_escape),
            lm_ms_med: median(&lm_ms),
        })
    }

    fn fittable_for<S: 'static>(entry: &CorpusEntry) -> Fittable {
        use std::any::TypeId;
        let t = TypeId::of::<S>();
        if t == TypeId::of::<Circle>() {
            entry.fittable_circle
        } else if t == TypeId::of::<Ellipse>() {
            entry.fittable_ellipse
        } else {
            Fittable::Skip("unsupported shape")
        }
    }

    fn render_restart_table(rows: &[SpecRow]) {
        println!("#### A. Restart value (default `CmaEsTrf`)\n");
        println!(
            "| spec | n | loss@1 | loss@2 | loss@3 | loss@10 | spread@1 | diag@1 | diag@10 | min-k | helped | gain |"
        );
        println!(
            "| ---- | -:| ------:| ------:| ------:| -------:| -------:| -----:| ------:| ----:| -----:| ---:|"
        );
        for r in rows {
            println!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} |",
                r.spec,
                r.n_sets,
                e(r.loss_at_restart[0]),
                e(r.loss_at_restart[1]),
                e(r.loss_at_restart[2]),
                e(r.loss_at_restart[4]),
                pct(r.spread1_rel),
                e(r.diag_at_1),
                e(r.diag_at_10),
                r.min_k_worst,
                r.seeds_helped_by_restarts,
                r.n_seeds,
                pct(r.median_rel_gain_restarts),
            );
        }
        println!();
    }

    fn render_escape_table(rows: &[SpecRow]) {
        println!("#### B. CMA-ES escape value (lm vs default `CmaEsTrf`, both `n_restarts=10`)\n");
        println!("| spec | n | lm loss | cmaes loss | fire rate | helped | gain | lm ms |");
        println!("| ---- | -:| -------:| ---------:| --------:| -----:| ---:| ----:|");
        for r in rows {
            println!(
                "| {} | {} | {} | {} | {} | {}/{} | {} | {} |",
                r.spec,
                r.n_sets,
                e(r.lm_loss_med),
                e(r.cmaes_loss_med),
                pct(r.fire_rate),
                r.seeds_helped_by_escape,
                r.n_seeds,
                pct(r.median_rel_gain_escape),
                ms(r.lm_ms_med),
            );
        }
        println!();
    }

    fn render_cross_loss_summary(rows: &[Row]) {
        println!("## Cross-loss summary (bucketed by set count)\n");
        println!(
            "For each (shape, loss, n_sets) bucket: how many specs ever need >1 restart, the \
             max `min-k` in the bucket, how many specs the escape ever helps, and the median \
             `diag@10` (achieved quality). The 3-set rows are the headline.\n"
        );
        println!(
            "| shape | loss | n | specs | restarts help | max min-k | escape helps | median diag@10 |"
        );
        println!(
            "| ----- | ---- | -:| ----: | ------------: | --------: | -----------: | -------------: |"
        );
        let shapes = ["circle", "ellipse"];
        let lnames = ["sumsq", "sumabs", "maxabs"];
        for shape in shapes {
            for lname in lnames {
                let max_n = rows
                    .iter()
                    .filter(|r| r.shape == shape && r.loss == lname)
                    .map(|r| r.inner.n_sets)
                    .max()
                    .unwrap_or(0);
                for n in 2..=max_n {
                    let bucket: Vec<&SpecRow> = rows
                        .iter()
                        .filter(|r| r.shape == shape && r.loss == lname && r.inner.n_sets == n)
                        .map(|r| &r.inner)
                        .collect();
                    if bucket.is_empty() {
                        continue;
                    }
                    let restart_help = bucket
                        .iter()
                        .filter(|r| r.seeds_helped_by_restarts > 0)
                        .count();
                    let escape_help = bucket
                        .iter()
                        .filter(|r| r.seeds_helped_by_escape > 0)
                        .count();
                    let max_min_k = bucket.iter().map(|r| r.min_k_worst).max().unwrap_or(1);
                    let diags: Vec<f64> = bucket.iter().map(|r| r.diag_at_10).collect();
                    println!(
                        "| {} | {} | {} | {} | {}/{} | {} | {}/{} | {} |",
                        shape,
                        lname,
                        n,
                        bucket.len(),
                        restart_help,
                        bucket.len(),
                        max_min_k,
                        escape_help,
                        bucket.len(),
                        e(median(&diags)),
                    );
                }
            }
        }
        println!();
    }

    fn median(xs: &[f64]) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    /// Relative spread (max−min)/max, in [0,1). 0 ⇒ identical across seeds.
    fn rel_spread(xs: &[f64]) -> f64 {
        let finite: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.len() < 2 {
            return 0.0;
        }
        let mn = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if mx <= 0.0 { 0.0 } else { (mx - mn) / mx }
    }

    fn e(x: f64) -> String {
        if x.is_finite() {
            format!("{x:.2e}")
        } else {
            "—".to_string()
        }
    }
    fn ms(x: f64) -> String {
        if x.is_finite() {
            format!("{x:.0}")
        } else {
            "—".to_string()
        }
    }
    fn pct(x: f64) -> String {
        if x.is_finite() {
            format!("{:.1}%", x * 100.0)
        } else {
            "—".to_string()
        }
    }
}
