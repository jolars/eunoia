//! Aggregate quality-report binary.
//!
//! Sweeps a list of named [`Fitter`] configurations × the 27-spec corpus ×
//! `QUALITY_SEEDS` (16 seeds) × {Circle, Ellipse} and reports the *loss being
//! minimized* (not `diag_error`) as the primary metric, with `diag_error`
//! and runtime as secondary diagnostics.
//!
//! All configs share the same loss type (the `Fitter` default,
//! `LossType::SumSquared = Σ(f-t)² / Σt²`), so loss values are directly
//! comparable across configs and across specs (the loss is bounded
//! roughly in `[0, 1]` regardless of input area scale). Cross-spec
//! aggregates still use the median to keep one bad outlier from
//! dominating.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example quality_report --features corpus
//! ```
//!
//! Markdown is printed to stdout; JSON is written to
//! `${CARGO_TARGET_DIR:-target}/quality_report.json` for PR-vs-PR diffing.
//!
//! To compare a new optimizer (CMA-ES, DE+L-BFGS, etc.), add an entry to
//! [`shape_configs`] with a closure that applies the relevant builder calls
//! and re-run.

#[cfg(not(feature = "corpus"))]
fn main() {
    eprintln!("This example requires the `corpus` feature.");
    eprintln!("Run with: cargo run --release --example quality_report --features corpus");
    std::process::exit(1);
}

#[cfg(feature = "corpus")]
fn main() {
    quality_report::run();
}

#[cfg(feature = "corpus")]
mod quality_report {
    use std::collections::BTreeMap;
    use std::env;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use eunoia::geometry::shapes::{Circle, Ellipse};
    use eunoia::geometry::traits::DiagramShape;
    use eunoia::loss::LossType;
    use eunoia::test_utils::corpus::{all, CorpusEntry, Fittable, QUALITY_SEEDS};
    use eunoia::{Fitter, InitialSampler, MdsSolver, Optimizer};

    /// Common-across-configs description. Things that vary across configs
    /// (final-stage optimizer, MDS solver pool) are listed per config.
    const COMMON_CONFIG: &str =
        "n_restarts=10, max_iterations=200, tolerance=1e-6, loss=SumSquared \
         (Fitter default after dropping NM from the optimizer pool)";

    /// Type of a config-builder closure: takes a fresh `Fitter` and applies
    /// the config's builder calls. Non-capturing so it coerces to `fn`.
    type ConfigFn<S> = fn(Fitter<'_, S>) -> Fitter<'_, S>;

    /// The list of configs swept per shape. Keep names short and stable —
    /// they appear as column headers in cross-config tables and as keys in
    /// the JSON snapshot. To add a config, append a `(name, builder)` pair;
    /// builders must be non-capturing closures (so they coerce to `fn`).
    fn shape_configs<S: DiagramShape + Copy + 'static>() -> Vec<(&'static str, ConfigFn<S>)> {
        vec![
            // Current default: L-BFGS-only final-stage pool, L-BFGS-only MDS.
            ("default", |f| f),
            // Mixed pool with NM cycled in (the previous default). Kept as a
            // sentinel so we can verify NM stays harmful and don't quietly
            // regress the change.
            ("nm_lbfgs_mix", |f| {
                f.optimizer_pool(vec![Optimizer::NelderMead, Optimizer::Lbfgs])
            }),
            // Pure Nelder-Mead final-stage. Tracks how badly NM-only does
            // (huge ellipse loss) — useful as the lower bound on quality.
            ("neldermead_only", |f| f.optimizer(Optimizer::NelderMead)),
            // Mixed MDS pool. Default keeps initial-layout on L-BFGS-only
            // because the mix has been observed to hang on real eulerr-style
            // specs; this config flips that on so we can quantify the cost.
            ("mds_mixed", |f| {
                f.initial_solver_pool(vec![MdsSolver::Lbfgs, MdsSolver::TrustRegion])
            }),
            // Levenberg-Marquardt at the final stage only. LM approximates
            // the Hessian as JᵀJ from the analytical region-area Jacobian
            // we already compute for L-BFGS — strictly stronger curvature
            // info for least-squares losses.
            ("lm_final", |f| f.optimizer(Optimizer::LevenbergMarquardt)),
            // LM at the MDS stage only.
            ("lm_initial", |f| {
                f.initial_solver(MdsSolver::LevenbergMarquardt)
            }),
            // LM in both stages.
            ("lm_full", |f| {
                f.optimizer(Optimizer::LevenbergMarquardt)
                    .initial_solver(MdsSolver::LevenbergMarquardt)
            }),
            // Bounded CMA-ES global step + LM polish on the final stage.
            // Targets the three specs LM-on-LM can't escape (issue91_6_set,
            // issue44_4_set_inclusive, issue92_3_set_dropped_pair). Expensive
            // — ~1k–2k extra region-area evals per restart on hard 6-set
            // ellipse fits.
            ("cmaes_lm", |f| f.optimizer(Optimizer::CmaEsLm)),
            // Same plus LM at the MDS stage. Tests whether the global step
            // benefits from a tighter MDS init.
            ("cmaes_lm_full", |f| {
                f.optimizer(Optimizer::CmaEsLm)
                    .initial_solver(MdsSolver::LevenbergMarquardt)
            }),
            // Latin-hypercube initial draws across the n_restarts batch
            // instead of independent uniform draws. Probe: does stratified
            // coverage of [0, scale]^(2·n_sets) lift best-of-N quality on
            // specs where multiple uniform attempts collide in the same
            // basin? Same final-stage as `default` (CmaEsLm) so the diff
            // is exclusively the initial-position distribution.
            ("lhs", |f| f.initial_sampler(InitialSampler::LatinHypercube)),
        ]
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Status {
        Ok,
        Err,
        Skip,
        Panic,
        SanityOk,
        SanityErr,
    }

    impl Status {
        fn as_str(self) -> &'static str {
            match self {
                Status::Ok => "ok",
                Status::Err => "err",
                Status::Skip => "skip",
                Status::Panic => "panic",
                Status::SanityOk => "sanity_ok",
                Status::SanityErr => "sanity_err",
            }
        }
    }

    struct Row {
        config: &'static str,
        shape: &'static str,
        spec: &'static str,
        seed: u64,
        status: Status,
        /// Final value of the loss the optimizer minimized.
        loss: Option<f64>,
        /// `Layout::loss_type()` rendered as a short string. None if the fit
        /// did not produce a layout (skip/err/panic).
        loss_type: Option<&'static str>,
        /// `Layout::diag_error()` — secondary diagnostic.
        diag_error: Option<f64>,
        iterations: Option<usize>,
        elapsed_ms: Option<f64>,
        note: String,
    }

    pub fn run() {
        let circle_configs = shape_configs::<Circle>();
        let ellipse_configs = shape_configs::<Ellipse>();
        eprintln!(
            "Running quality sweep: {} configs × 2 shapes × 27 specs × {} seeds.",
            circle_configs.len(),
            QUALITY_SEEDS.len()
        );
        eprintln!("Common config: {COMMON_CONFIG}");
        eprintln!();

        let mut all_rows: Vec<Row> = Vec::new();

        for (cfg_name, cfg_fn) in &circle_configs {
            eprintln!("== config: {cfg_name} | shape: circle ==");
            run_config::<Circle>(
                cfg_name,
                *cfg_fn,
                "circle",
                |e| e.fittable_circle,
                &mut all_rows,
            );
            eprintln!();
        }
        for (cfg_name, cfg_fn) in &ellipse_configs {
            eprintln!("== config: {cfg_name} | shape: ellipse ==");
            run_config::<Ellipse>(
                cfg_name,
                *cfg_fn,
                "ellipse",
                |e| e.fittable_ellipse,
                &mut all_rows,
            );
            eprintln!();
        }

        let active_loss = active_loss_or_die(&all_rows);

        let config_names: Vec<&'static str> = circle_configs.iter().map(|(n, _)| *n).collect();

        let md = render_markdown(&all_rows, &config_names, active_loss);
        print!("{md}");

        let json = render_json(&all_rows, &config_names, active_loss);
        let target_dir = env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target"));
        if let Err(e) = fs::create_dir_all(&target_dir) {
            eprintln!("warning: could not create {}: {e}", target_dir.display());
        }
        let json_path = target_dir.join("quality_report.json");
        match fs::write(&json_path, json) {
            Ok(()) => eprintln!("\nWrote JSON snapshot: {}", json_path.display()),
            Err(e) => eprintln!("\nERROR writing {}: {e}", json_path.display()),
        }
    }

    fn run_config<S>(
        config_name: &'static str,
        config_fn: ConfigFn<S>,
        shape_name: &'static str,
        fittable_fn: fn(&CorpusEntry) -> Fittable,
        out: &mut Vec<Row>,
    ) where
        S: DiagramShape + Copy + 'static,
    {
        for entry in all() {
            let mut spec_ok = 0usize;
            let mut spec_runs = 0usize;
            let spec_start = Instant::now();

            for &seed in &QUALITY_SEEDS {
                let fittable = fittable_fn(entry);

                if let Fittable::Skip(reason) = fittable {
                    out.push(Row {
                        config: config_name,
                        shape: shape_name,
                        spec: entry.name,
                        seed,
                        status: Status::Skip,
                        loss: None,
                        loss_type: None,
                        diag_error: None,
                        iterations: None,
                        elapsed_ms: None,
                        note: reason.to_string(),
                    });
                    continue;
                }

                let spec = (entry.build)();
                let started = Instant::now();
                let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    config_fn(Fitter::<S>::new(&spec).seed(seed)).fit()
                }));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                spec_runs += 1;

                let row = match (fittable, fit_result) {
                    (Fittable::Skip(_), _) => unreachable!(),
                    (_, Err(_panic)) => Row {
                        config: config_name,
                        shape: shape_name,
                        spec: entry.name,
                        seed,
                        status: Status::Panic,
                        loss: None,
                        loss_type: None,
                        diag_error: None,
                        iterations: None,
                        elapsed_ms: Some(elapsed_ms),
                        note: "panic during fit".to_string(),
                    },
                    (Fittable::Normal, Ok(Ok(layout))) => {
                        spec_ok += 1;
                        Row {
                            config: config_name,
                            shape: shape_name,
                            spec: entry.name,
                            seed,
                            status: Status::Ok,
                            loss: Some(layout.loss()),
                            loss_type: Some(loss_type_name(layout.loss_type())),
                            diag_error: Some(layout.diag_error()),
                            iterations: Some(layout.iterations()),
                            elapsed_ms: Some(elapsed_ms),
                            note: String::new(),
                        }
                    }
                    (Fittable::Normal, Ok(Err(e))) => Row {
                        config: config_name,
                        shape: shape_name,
                        spec: entry.name,
                        seed,
                        status: Status::Err,
                        loss: None,
                        loss_type: None,
                        diag_error: None,
                        iterations: None,
                        elapsed_ms: Some(elapsed_ms),
                        note: format!("{e}"),
                    },
                    (Fittable::SanityOnly, Ok(Ok(layout))) => {
                        spec_ok += 1;
                        Row {
                            config: config_name,
                            shape: shape_name,
                            spec: entry.name,
                            seed,
                            status: Status::SanityOk,
                            loss: Some(layout.loss()),
                            loss_type: Some(loss_type_name(layout.loss_type())),
                            diag_error: None,
                            iterations: Some(layout.iterations()),
                            elapsed_ms: Some(elapsed_ms),
                            note: String::new(),
                        }
                    }
                    (Fittable::SanityOnly, Ok(Err(_))) => {
                        spec_ok += 1;
                        Row {
                            config: config_name,
                            shape: shape_name,
                            spec: entry.name,
                            seed,
                            status: Status::SanityErr,
                            loss: None,
                            loss_type: None,
                            diag_error: None,
                            iterations: None,
                            elapsed_ms: Some(elapsed_ms),
                            note: "sanity-only spec returned Err (expected)".to_string(),
                        }
                    }
                };

                out.push(row);
            }

            eprintln!(
                "  [{config_name}/{shape_name}] {:32} {}/{} ok in {:.2}s",
                entry.name,
                spec_ok,
                spec_runs,
                spec_start.elapsed().as_secs_f64()
            );
        }
    }

    fn loss_type_name(t: LossType) -> &'static str {
        match t {
            LossType::SumSquared => "SumSquared",
            LossType::SumAbsoute => "SumAbsolute",
            LossType::SumAbsoluteRegionError => "SumAbsoluteRegionError",
            LossType::SumSquaredRegionError => "SumSquaredRegionError",
            LossType::MaxAbsolute => "MaxAbsolute",
            LossType::MaxSquared => "MaxSquared",
            LossType::RootMeanSquared => "RootMeanSquared",
            LossType::Stress => "Stress",
            LossType::DiagError => "DiagError",
        }
    }

    /// Returns the single loss-type string used across all rows. Every
    /// successful fit must report the same loss; otherwise cross-config
    /// loss numbers can't be compared. Aborts with a clear message if not.
    fn active_loss_or_die(rows: &[Row]) -> &'static str {
        let losses: std::collections::BTreeSet<&'static str> =
            rows.iter().filter_map(|r| r.loss_type).collect();
        match losses.len() {
            0 => "<unknown>",
            1 => losses.into_iter().next().unwrap(),
            _ => {
                eprintln!(
                    "ERROR: configs reported mixed loss types ({:?}); cross-config loss numbers \
                     would be incomparable. Pin loss_type to a single value across configs and re-run.",
                    losses.into_iter().collect::<Vec<_>>()
                );
                std::process::exit(2);
            }
        }
    }

    /// Best/median/mean/worst plus run counts for one (config, shape, spec) cell.
    struct CellSummary {
        ok_count: usize,
        skip_count: usize,
        err_count: usize,
        panic_count: usize,
        loss_best: Option<f64>,
        loss_median: Option<f64>,
        loss_mean: Option<f64>,
        loss_worst: Option<f64>,
        diag_best: Option<f64>,
        diag_median: Option<f64>,
        diag_worst: Option<f64>,
        elapsed_best_ms: Option<f64>,
        elapsed_median_ms: Option<f64>,
        elapsed_mean_ms: Option<f64>,
        elapsed_worst_ms: Option<f64>,
    }

    fn summarize_cell(rows: &[&Row]) -> CellSummary {
        let mut ok_count = 0usize;
        let mut skip_count = 0usize;
        let mut err_count = 0usize;
        let mut panic_count = 0usize;
        for r in rows {
            match r.status {
                Status::Ok | Status::SanityOk | Status::SanityErr => ok_count += 1,
                Status::Skip => skip_count += 1,
                Status::Err => err_count += 1,
                Status::Panic => panic_count += 1,
            }
        }
        let losses: Vec<f64> = rows.iter().filter_map(|r| r.loss).collect();
        let diags: Vec<f64> = rows.iter().filter_map(|r| r.diag_error).collect();
        let elapsed: Vec<f64> = rows.iter().filter_map(|r| r.elapsed_ms).collect();

        CellSummary {
            ok_count,
            skip_count,
            err_count,
            panic_count,
            loss_best: stat_min(&losses),
            loss_median: stat_median(&losses),
            loss_mean: stat_mean(&losses),
            loss_worst: stat_max(&losses),
            diag_best: stat_min(&diags),
            diag_median: stat_median(&diags),
            diag_worst: stat_max(&diags),
            elapsed_best_ms: stat_min(&elapsed),
            elapsed_median_ms: stat_median(&elapsed),
            elapsed_mean_ms: stat_mean(&elapsed),
            elapsed_worst_ms: stat_max(&elapsed),
        }
    }

    fn stat_min(xs: &[f64]) -> Option<f64> {
        xs.iter()
            .copied()
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.min(x))))
    }
    fn stat_max(xs: &[f64]) -> Option<f64> {
        xs.iter()
            .copied()
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.max(x))))
    }
    fn stat_mean(xs: &[f64]) -> Option<f64> {
        if xs.is_empty() {
            return None;
        }
        Some(xs.iter().copied().sum::<f64>() / xs.len() as f64)
    }
    fn stat_median(xs: &[f64]) -> Option<f64> {
        if xs.is_empty() {
            return None;
        }
        let mut sorted = xs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 1 {
            Some(sorted[n / 2])
        } else {
            Some(0.5 * (sorted[n / 2 - 1] + sorted[n / 2]))
        }
    }

    fn render_markdown(
        rows: &[Row],
        config_names: &[&'static str],
        active_loss: &'static str,
    ) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Eunoia quality report");
        let _ = writeln!(s);
        let _ = writeln!(s, "- Common config: {COMMON_CONFIG}");
        let _ = writeln!(
            s,
            "- Active loss (primary metric): **{active_loss}** (same across all configs)"
        );
        let _ = writeln!(
            s,
            "- Seeds (`QUALITY_SEEDS`, n={}): {:?}",
            QUALITY_SEEDS.len(),
            QUALITY_SEEDS
        );
        let _ = writeln!(s, "- Configs swept: `{}`", config_names.join("`, `"));
        let _ = writeln!(s);

        for &shape in &["circle", "ellipse"] {
            render_shape_block(&mut s, rows, config_names, shape, active_loss);
        }

        s
    }

    fn render_shape_block(
        s: &mut String,
        rows: &[Row],
        config_names: &[&'static str],
        shape: &str,
        active_loss: &str,
    ) {
        let _ = writeln!(s, "## Shape: {shape}");
        let _ = writeln!(s);

        // Per-config detail table.
        for &cfg in config_names {
            render_per_config_table(s, rows, cfg, shape, active_loss);
        }

        // Head-to-head: median loss per (spec, config).
        render_h2h_table(s, rows, config_names, shape, "median loss", |c| {
            c.loss_median
        });
        render_h2h_table(s, rows, config_names, shape, "best loss", |c| c.loss_best);
        render_h2h_table(s, rows, config_names, shape, "mean ms", |c| {
            c.elapsed_mean_ms
        });

        render_aggregate_ranking(s, rows, config_names, shape);
    }

    fn render_per_config_table(
        s: &mut String,
        rows: &[Row],
        config: &str,
        shape: &str,
        active_loss: &str,
    ) {
        let cell_rows: Vec<&Row> = rows
            .iter()
            .filter(|r| r.config == config && r.shape == shape)
            .collect();
        let total_ms: f64 = cell_rows.iter().filter_map(|r| r.elapsed_ms).sum();
        let n_total = cell_rows.len();
        let n_ok = cell_rows
            .iter()
            .filter(|r| matches!(r.status, Status::Ok | Status::SanityOk | Status::SanityErr))
            .count();
        let n_skip = cell_rows
            .iter()
            .filter(|r| matches!(r.status, Status::Skip))
            .count();
        let n_err = cell_rows
            .iter()
            .filter(|r| matches!(r.status, Status::Err))
            .count();
        let n_panic = cell_rows
            .iter()
            .filter(|r| matches!(r.status, Status::Panic))
            .count();

        let _ = writeln!(s, "### {shape} / `{config}`");
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "Total wall time: **{:.2} s** across {} seed-rows ({} ok, {} skip, {} err, {} panic). \
             Loss values are {active_loss} (lower is better).",
            total_ms / 1e3,
            n_total,
            n_ok,
            n_skip,
            n_err,
            n_panic,
        );
        let _ = writeln!(s);

        let _ = writeln!(
            s,
            "| spec | ok | skip | err+panic | best loss | median loss | worst loss | median diag | mean ms |"
        );
        let _ = writeln!(
            s,
            "| ---- | --:| ---:| ---------:| ---------:| -----------:| ----------:| -----------:| -------:|"
        );

        for entry in all() {
            let cells: Vec<&Row> = cell_rows
                .iter()
                .copied()
                .filter(|r| r.spec == entry.name)
                .collect();
            if cells.is_empty() {
                continue;
            }
            let summary = summarize_cell(&cells);
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                entry.name,
                summary.ok_count,
                summary.skip_count,
                summary.err_count + summary.panic_count,
                fmt_opt_e(summary.loss_best),
                fmt_opt_e(summary.loss_median),
                fmt_opt_e(summary.loss_worst),
                fmt_opt_e(summary.diag_median),
                fmt_opt_ms(summary.elapsed_mean_ms),
            );
        }
        let _ = writeln!(s);
    }

    fn render_h2h_table(
        s: &mut String,
        rows: &[Row],
        config_names: &[&'static str],
        shape: &str,
        metric_label: &str,
        pick: fn(&CellSummary) -> Option<f64>,
    ) {
        let _ = writeln!(s, "### {shape}: {metric_label} per (spec × config)");
        let _ = writeln!(s);

        let mut header = String::from("| spec |");
        let mut sep = String::from("| ---- |");
        for cfg in config_names {
            let _ = write!(header, " {cfg} |");
            sep.push_str(" ---: |");
        }
        let _ = writeln!(s, "{header}");
        let _ = writeln!(s, "{sep}");

        for entry in all() {
            // Find best (lowest) value across configs to mark the winner.
            let mut values_by_cfg: Vec<Option<f64>> = Vec::with_capacity(config_names.len());
            for &cfg in config_names {
                let cells: Vec<&Row> = rows
                    .iter()
                    .filter(|r| r.config == cfg && r.shape == shape && r.spec == entry.name)
                    .collect();
                if cells.is_empty() {
                    values_by_cfg.push(None);
                } else {
                    values_by_cfg.push(pick(&summarize_cell(&cells)));
                }
            }

            // If every value is None (e.g. ellipse-skip spec), still show a
            // row with em-dashes so the table stays rectangular.
            let any_some = values_by_cfg.iter().any(|v| v.is_some());
            let best = if any_some {
                values_by_cfg
                    .iter()
                    .filter_map(|v| *v)
                    .filter(|x| x.is_finite())
                    .fold(None, |acc, x| Some(acc.map_or(x, |a: f64| a.min(x))))
            } else {
                None
            };

            let mut row = format!("| {} |", entry.name);
            for v in &values_by_cfg {
                let cell = fmt_opt_e(*v);
                let mark_winner =
                    matches!((v, best), (Some(x), Some(b)) if (x - b).abs() <= b * 1e-9 + 1e-12);
                if mark_winner && config_names.len() > 1 {
                    let _ = write!(row, " **{cell}** |");
                } else {
                    let _ = write!(row, " {cell} |");
                }
            }
            let _ = writeln!(s, "{row}");
        }
        let _ = writeln!(s);
    }

    fn render_aggregate_ranking(
        s: &mut String,
        rows: &[Row],
        config_names: &[&'static str],
        shape: &str,
    ) {
        let _ = writeln!(s, "### {shape}: aggregate ranking");
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| config | median loss | mean diag | total wall time | mean ms / fit | spec wins (median loss) |"
        );
        let _ = writeln!(
            s,
            "| ------ | ----------: | --------: | --------------: | ------------: | ----------------------: |"
        );

        // Pre-compute per-spec medians for win-counting.
        let mut wins: BTreeMap<&str, usize> = config_names.iter().map(|&c| (c, 0usize)).collect();
        for entry in all() {
            let mut best_val: Option<f64> = None;
            let mut best_cfgs: Vec<&str> = Vec::new();
            for &cfg in config_names {
                let cells: Vec<&Row> = rows
                    .iter()
                    .filter(|r| r.config == cfg && r.shape == shape && r.spec == entry.name)
                    .collect();
                if cells.is_empty() {
                    continue;
                }
                if let Some(median) = summarize_cell(&cells).loss_median {
                    if !median.is_finite() {
                        continue;
                    }
                    match best_val {
                        None => {
                            best_val = Some(median);
                            best_cfgs.clear();
                            best_cfgs.push(cfg);
                        }
                        Some(b) if median < b - (b.abs() * 1e-9 + 1e-12) => {
                            best_val = Some(median);
                            best_cfgs.clear();
                            best_cfgs.push(cfg);
                        }
                        Some(b) if (median - b).abs() <= b.abs() * 1e-9 + 1e-12 => {
                            best_cfgs.push(cfg);
                        }
                        _ => {}
                    }
                }
            }
            for cfg in best_cfgs {
                *wins.entry(cfg).or_insert(0) += 1;
            }
        }

        for &cfg in config_names {
            let cells: Vec<&Row> = rows
                .iter()
                .filter(|r| r.config == cfg && r.shape == shape)
                .collect();
            let losses: Vec<f64> = cells.iter().filter_map(|r| r.loss).collect();
            let diags: Vec<f64> = cells.iter().filter_map(|r| r.diag_error).collect();
            let elapsed: Vec<f64> = cells.iter().filter_map(|r| r.elapsed_ms).collect();
            let total_ms: f64 = elapsed.iter().sum();
            let _ = writeln!(
                s,
                "| {} | {} | {} | {:.2} s | {} | {} |",
                cfg,
                fmt_opt_e(stat_median(&losses)),
                fmt_opt_e(stat_mean(&diags)),
                total_ms / 1e3,
                fmt_opt_ms(stat_mean(&elapsed)),
                wins.get(cfg).copied().unwrap_or(0),
            );
        }
        let _ = writeln!(s);
    }

    fn fmt_opt_e(v: Option<f64>) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:.3e}"),
            Some(_) => "nan".to_string(),
            None => "—".to_string(),
        }
    }

    fn fmt_opt_ms(v: Option<f64>) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:.1}"),
            Some(_) => "nan".to_string(),
            None => "—".to_string(),
        }
    }

    fn render_json(rows: &[Row], config_names: &[&'static str], active_loss: &str) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str("  \"version\": 2,\n");
        let _ = writeln!(s, "  \"common_config\": {},", json_str(COMMON_CONFIG));
        let _ = writeln!(s, "  \"active_loss\": {},", json_str(active_loss));
        s.push_str("  \"seeds\": [");
        for (i, &seed) in QUALITY_SEEDS.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "{seed}");
        }
        s.push_str("],\n");
        s.push_str("  \"configs\": [\n");
        for (ci, &cfg) in config_names.iter().enumerate() {
            let last_cfg = ci + 1 == config_names.len();
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"name\": {},", json_str(cfg));
            s.push_str("      \"shapes\": [\n");
            for (si, &shape) in ["circle", "ellipse"].iter().enumerate() {
                let last_shape = si + 1 == 2;
                write_shape_json(&mut s, rows, cfg, shape, last_shape);
            }
            s.push_str("      ]\n");
            let trailing = if last_cfg { "" } else { "," };
            let _ = writeln!(s, "    }}{trailing}");
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }

    fn write_shape_json(s: &mut String, rows: &[Row], cfg: &str, shape: &str, last: bool) {
        let cell_rows: Vec<&Row> = rows
            .iter()
            .filter(|r| r.config == cfg && r.shape == shape)
            .collect();
        let total_ms: f64 = cell_rows.iter().filter_map(|r| r.elapsed_ms).sum();

        s.push_str("        {\n");
        let _ = writeln!(s, "          \"name\": {},", json_str(shape));
        let _ = writeln!(s, "          \"total_elapsed_ms\": {},", fmt_num(total_ms));
        s.push_str("          \"rows\": [\n");
        for (i, r) in cell_rows.iter().enumerate() {
            let comma = if i + 1 == cell_rows.len() { "" } else { "," };
            s.push_str("            {");
            let _ = write!(s, "\"spec\": {}, ", json_str(r.spec));
            let _ = write!(s, "\"seed\": {}, ", r.seed);
            let _ = write!(s, "\"status\": {}, ", json_str(r.status.as_str()));
            let _ = write!(s, "\"loss\": {}, ", fmt_opt_num(r.loss));
            let _ = write!(
                s,
                "\"loss_type\": {}, ",
                r.loss_type
                    .map(json_str)
                    .unwrap_or_else(|| "null".to_string())
            );
            let _ = write!(s, "\"diag_error\": {}, ", fmt_opt_num(r.diag_error));
            let _ = write!(
                s,
                "\"iterations\": {}, ",
                r.iterations
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "null".to_string())
            );
            let _ = write!(s, "\"elapsed_ms\": {}, ", fmt_opt_num(r.elapsed_ms));
            let _ = write!(s, "\"note\": {}", json_str(&r.note));
            let _ = writeln!(s, "}}{comma}");
        }
        s.push_str("          ],\n");
        s.push_str("          \"summary_per_spec\": [\n");
        let mut entries_iter = all().iter().peekable();
        while let Some(entry) = entries_iter.next() {
            let cells: Vec<&Row> = cell_rows
                .iter()
                .copied()
                .filter(|r| r.spec == entry.name)
                .collect();
            if cells.is_empty() {
                continue;
            }
            let sm = summarize_cell(&cells);
            let comma = if entries_iter.peek().is_none() {
                ""
            } else {
                ","
            };
            s.push_str("            {");
            let _ = write!(s, "\"spec\": {}, ", json_str(entry.name));
            let _ = write!(s, "\"ok_count\": {}, ", sm.ok_count);
            let _ = write!(s, "\"skip_count\": {}, ", sm.skip_count);
            let _ = write!(s, "\"err_count\": {}, ", sm.err_count);
            let _ = write!(s, "\"panic_count\": {}, ", sm.panic_count);
            let _ = write!(s, "\"loss_best\": {}, ", fmt_opt_num(sm.loss_best));
            let _ = write!(s, "\"loss_median\": {}, ", fmt_opt_num(sm.loss_median));
            let _ = write!(s, "\"loss_mean\": {}, ", fmt_opt_num(sm.loss_mean));
            let _ = write!(s, "\"loss_worst\": {}, ", fmt_opt_num(sm.loss_worst));
            let _ = write!(s, "\"diag_best\": {}, ", fmt_opt_num(sm.diag_best));
            let _ = write!(s, "\"diag_median\": {}, ", fmt_opt_num(sm.diag_median));
            let _ = write!(s, "\"diag_worst\": {}, ", fmt_opt_num(sm.diag_worst));
            let _ = write!(
                s,
                "\"elapsed_best_ms\": {}, ",
                fmt_opt_num(sm.elapsed_best_ms)
            );
            let _ = write!(
                s,
                "\"elapsed_median_ms\": {}, ",
                fmt_opt_num(sm.elapsed_median_ms)
            );
            let _ = write!(
                s,
                "\"elapsed_mean_ms\": {}, ",
                fmt_opt_num(sm.elapsed_mean_ms)
            );
            let _ = write!(
                s,
                "\"elapsed_worst_ms\": {}",
                fmt_opt_num(sm.elapsed_worst_ms)
            );
            let _ = writeln!(s, "}}{comma}");
        }
        s.push_str("          ]\n");
        let trailing = if last { "" } else { "," };
        let _ = writeln!(s, "        }}{trailing}");
    }

    fn json_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn fmt_num(x: f64) -> String {
        if x.is_finite() {
            format!("{x}")
        } else {
            "null".to_string()
        }
    }

    fn fmt_opt_num(x: Option<f64>) -> String {
        match x {
            Some(v) if v.is_finite() => format!("{v}"),
            _ => "null".to_string(),
        }
    }
}
