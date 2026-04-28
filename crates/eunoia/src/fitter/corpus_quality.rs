//! Static-corpus fit-quality regression test.
//!
//! Iterates the 17-spec corpus in [`crate::test_utils::corpus`] across
//! [`TEST_SEEDS`] and asserts each fit produces a `diag_error` within a
//! permissive per-spec ceiling. Two test functions (one per shape) so a
//! contributor can run circles only during local iteration.
//!
//! All failures are collected into a single panic at the end of each test
//! so a single run surfaces the full picture rather than failing fast.

#[cfg(test)]
mod tests {
    use crate::geometry::shapes::{Circle, Ellipse};
    use crate::geometry::traits::DiagramShape;
    use crate::test_utils::corpus::{all, CorpusEntry, Fittable, TEST_SEEDS};
    use crate::Fitter;

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
        let mut rows = Vec::new();

        for entry in all() {
            for &seed in &TEST_SEEDS {
                let ceiling = ceiling_fn(entry);
                let fittable = fittable_fn(entry);

                if let Fittable::Skip(reason) = fittable {
                    rows.push(Row {
                        name: entry.name,
                        seed,
                        diag: None,
                        ceiling,
                        ok: true,
                        note: reason,
                    });
                    continue;
                }

                let spec = (entry.build)();
                // Catch panics (notably the `normalize_layout changed
                // fitted exclusive regions` debug_assert that can trip on
                // some ellipse fits) so the row is recorded against the
                // offending spec rather than aborting the whole run.
                let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Fitter::<S>::new(&spec).seed(seed).fit()
                })) {
                    Ok(r) => r,
                    Err(_) => {
                        rows.push(Row {
                            name: entry.name,
                            seed,
                            diag: None,
                            ceiling,
                            ok: false,
                            note: "PANIC during fit",
                        });
                        continue;
                    }
                };

                match (fittable, result) {
                    (Fittable::Normal, Ok(layout)) => {
                        let diag = layout.diag_error();
                        let ok = diag.is_finite() && diag <= ceiling;
                        rows.push(Row {
                            name: entry.name,
                            seed,
                            diag: Some(diag),
                            ceiling,
                            ok,
                            note: if ok { "" } else { "OVER CEILING" },
                        });
                    }
                    (Fittable::Normal, Err(e)) => {
                        rows.push(Row {
                            name: entry.name,
                            seed,
                            diag: None,
                            ceiling,
                            ok: false,
                            note: leak_msg(format!("ERR: {e}")),
                        });
                    }
                    (Fittable::SanityOnly, Ok(layout)) => {
                        let ok = layout.loss().is_finite();
                        rows.push(Row {
                            name: entry.name,
                            seed,
                            diag: None,
                            ceiling,
                            ok,
                            note: if ok { "sanity-ok" } else { "sanity-loss-nan" },
                        });
                    }
                    (Fittable::SanityOnly, Err(_)) => {
                        // Expected for specs like single_set that fail
                        // preprocessing — no panic, no diag_error to check.
                        rows.push(Row {
                            name: entry.name,
                            seed,
                            diag: None,
                            ceiling,
                            ok: true,
                            note: "sanity-err",
                        });
                    }
                    (Fittable::Skip(_), _) => unreachable!("Skip handled above the fit"),
                }
            }
        }

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
}
