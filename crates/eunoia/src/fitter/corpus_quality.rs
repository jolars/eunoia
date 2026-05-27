//! Static-corpus fit-quality regression test.
//!
//! Iterates the 17-spec corpus in [`crate::test_utils::corpus`] across
//! [`TEST_SEEDS`] and asserts each fit produces a `diag_error` within a
//! permissive per-spec ceiling. Three test functions (one per shape) so a
//! contributor can run circles only during local iteration.
//!
//! All failures are collected into a single panic at the end of each test
//! so a single run surfaces the full picture rather than failing fast.

#[cfg(test)]
mod tests {
    use crate::Fitter;
    use crate::geometry::shapes::{Circle, Ellipse, Square};
    use crate::geometry::traits::DiagramShape;
    use crate::test_utils::corpus::{CorpusEntry, Fittable, TEST_SEEDS, all};

    /// One row of the per-shape report.
    struct Row {
        name: &'static str,
        seed: u64,
        diag: Option<f64>,
        ceiling: f64,
        ok: bool,
        note: &'static str,
    }

    fn run<S>(
        shape_name: &str,
        ceiling_fn: fn(&CorpusEntry) -> f64,
        fittable_fn: fn(&CorpusEntry) -> Fittable,
    ) where
        S: DiagramShape + Copy + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // (spec, seed) work items in corpus order.
        let work: Vec<(&'static CorpusEntry, u64)> = all()
            .iter()
            .flat_map(|entry| TEST_SEEDS.iter().map(move |&seed| (entry, seed)))
            .collect();

        // Each (spec, seed) fit is independent and deterministic, so fan the
        // loop out across cores. cargo runs the three per-shape tests
        // concurrently, but that only saturates three cores while each test
        // does ~80 sequential fits — the inner loop is where the wall time
        // lives. A shared work-stealing counter keeps the very uneven per-spec
        // costs balanced; results are re-sorted into corpus order afterwards so
        // the printed table stays byte-for-byte stable regardless of
        // interleaving.
        let next = AtomicUsize::new(0);
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(work.len().max(1));

        let mut indexed: Vec<(usize, Row)> = std::thread::scope(|scope| {
            let work = &work;
            let next = &next;
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(entry, seed)) = work.get(i) else {
                                break;
                            };
                            local.push((i, fit_one::<S>(entry, seed, ceiling_fn, fittable_fn)));
                        }
                        local
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("corpus worker thread panicked"))
                .collect()
        });

        indexed.sort_by_key(|(i, _)| *i);
        let rows: Vec<Row> = indexed.into_iter().map(|(_, row)| row).collect();

        print_report(shape_name, &rows);

        let failures: Vec<&Row> = rows.iter().filter(|r| !r.ok).collect();
        if !failures.is_empty() {
            panic!(
                "{} corpus quality: {} of {} (spec, seed) rows exceeded ceiling — see table above",
                shape_name,
                failures.len(),
                rows.len()
            );
        }
    }

    /// Fit a single (spec, seed) under shape `S` and classify the outcome.
    /// Pure per-item work with no shared state, so [`run`] can call it from
    /// many threads at once.
    fn fit_one<S>(
        entry: &CorpusEntry,
        seed: u64,
        ceiling_fn: fn(&CorpusEntry) -> f64,
        fittable_fn: fn(&CorpusEntry) -> Fittable,
    ) -> Row
    where
        S: DiagramShape + Copy + 'static,
    {
        let ceiling = ceiling_fn(entry);
        let fittable = fittable_fn(entry);

        if let Fittable::Skip(reason) = fittable {
            return Row {
                name: entry.name,
                seed,
                diag: None,
                ceiling,
                ok: true,
                note: reason,
            };
        }

        let spec = (entry.build)();
        // Catch panics (notably the `normalize_layout changed fitted exclusive
        // regions` debug_assert that can trip on some ellipse fits) so the row
        // is recorded against the offending spec rather than aborting the run.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Fitter::<S>::new(&spec).seed(seed).fit()
        })) {
            Ok(r) => r,
            Err(_) => {
                return Row {
                    name: entry.name,
                    seed,
                    diag: None,
                    ceiling,
                    ok: false,
                    note: "PANIC during fit",
                };
            }
        };

        match (fittable, result) {
            (Fittable::Normal, Ok(layout)) => {
                let diag = layout.diag_error();
                let ok = diag.is_finite() && diag <= ceiling;
                Row {
                    name: entry.name,
                    seed,
                    diag: Some(diag),
                    ceiling,
                    ok,
                    note: if ok { "" } else { "OVER CEILING" },
                }
            }
            (Fittable::Normal, Err(e)) => Row {
                name: entry.name,
                seed,
                diag: None,
                ceiling,
                ok: false,
                note: leak_msg(format!("ERR: {e}")),
            },
            (Fittable::SanityOnly, Ok(layout)) => {
                let ok = layout.loss().is_finite();
                Row {
                    name: entry.name,
                    seed,
                    diag: None,
                    ceiling,
                    ok,
                    note: if ok { "sanity-ok" } else { "sanity-loss-nan" },
                }
            }
            // Expected for specs like single_set that fail preprocessing — no
            // panic, no diag_error to check.
            (Fittable::SanityOnly, Err(_)) => Row {
                name: entry.name,
                seed,
                diag: None,
                ceiling,
                ok: true,
                note: "sanity-err",
            },
            (Fittable::Skip(_), _) => unreachable!("Skip handled above the fit"),
        }
    }

    fn print_report(shape_name: &str, rows: &[Row]) {
        eprintln!();
        eprintln!("# Corpus quality report ({shape_name})");
        eprintln!("| spec | seed | diag_error | ceiling | status |");
        eprintln!("|------|------|------------|---------|--------|");
        for r in rows {
            let diag = r
                .diag
                .map(|d| format!("{d:.3e}"))
                .unwrap_or_else(|| "—".to_string());
            let status = if r.ok { "OK" } else { "FAIL" };
            let note = if r.note.is_empty() {
                String::new()
            } else {
                format!(" ({})", r.note)
            };
            eprintln!(
                "| {} | {} | {} | {:.1e} | {}{} |",
                r.name, r.seed, diag, r.ceiling, status, note
            );
        }
        eprintln!();
    }

    /// Leak a `String` into a `&'static str` so it can sit in the `Row`
    /// alongside the static notes. Cheap — at most a handful of failures
    /// per run, only on the panic path.
    fn leak_msg(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    #[test]
    fn corpus_circles_diag_error() {
        run::<Circle>("circle", |e| e.ceiling_circle(), |e| e.fittable_circle);
    }

    #[test]
    fn corpus_ellipses_diag_error() {
        run::<Ellipse>("ellipse", |e| e.ceiling_ellipse(), |e| e.fittable_ellipse);
    }

    #[test]
    fn corpus_squares_diag_error() {
        run::<Square>("square", |e| e.ceiling_square(), |e| e.fittable_square);
    }
}
