//! Shared spec fixtures for fit-quality tests and benchmarks.
//!
//! Ports the 17 reproducibility specs from eulerr's
//! `tests/testthat/test-reproducibility.R`. Each entry exposes a
//! [`CorpusEntry::build`] function returning a [`DiagramSpec`] plus a
//! permissive `diag_error` ceiling per shape type. The integration tests in
//! `fitter::corpus_quality` consume this list to characterise the *default*
//! `Fitter::<S>` user experience across a representative spec mix.

use crate::spec::{DiagramSpec, DiagramSpecBuilder, InputType};

/// Difficulty class for a corpus spec, used to pick a default `diag_error`
/// ceiling when an entry doesn't specify one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Trivially representable layouts (uniform overlap, fully disjoint).
    Easy,
    /// Realistic mixed overlaps; expected to fit well by default.
    Medium,
    /// Known-hard for the relevant shape (containment, near-degenerate).
    Hard,
}

/// How the static-corpus test should treat a spec for one shape type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fittable {
    /// Run `fit()` and assert `diag_error` is within the configured ceiling.
    Normal,
    /// `fit()` is expected to either fail (e.g. single-set specs that fail
    /// preprocessing) or succeed with no meaningful `diag_error` to assert
    /// against. The harness only checks the call doesn't panic and, if it
    /// returns `Ok`, that `loss().is_finite()`.
    SanityOnly,
    /// Skip this spec entirely for the relevant shape. Used for specs that
    /// hit a known-broken code path upstream — record the spec, don't
    /// attempt the fit. Each `Skip` carries a short note used in the
    /// report.
    Skip(&'static str),
}

/// One spec in the test corpus.
pub struct CorpusEntry {
    /// Stable identifier used in test output and lookup-by-name.
    pub name: &'static str,
    /// Builder for the spec — kept as a fn pointer so the corpus list can
    /// live in a `static` without runtime allocation.
    pub build: fn() -> DiagramSpec,
    /// Difficulty classification (drives the default ceiling).
    pub category: Category,
    /// Explicit per-shape `diag_error` ceiling. `None` falls back to the
    /// category default ([`DEFAULT_MAX_DIAG_ERROR_*`]).
    pub max_diag_error_circle: Option<f64>,
    /// Explicit per-shape `diag_error` ceiling for ellipses.
    pub max_diag_error_ellipse: Option<f64>,
    /// How the harness should treat the spec under [`Circle`].
    ///
    /// [`Circle`]: crate::geometry::shapes::Circle
    pub fittable_circle: Fittable,
    /// How the harness should treat the spec under [`Ellipse`].
    ///
    /// [`Ellipse`]: crate::geometry::shapes::Ellipse
    pub fittable_ellipse: Fittable,
}

impl CorpusEntry {
    /// Resolve the active `diag_error` ceiling for circles, falling back
    /// to the category default when the per-spec override is `None`.
    pub fn ceiling_circle(&self) -> f64 {
        self.max_diag_error_circle
            .unwrap_or_else(|| default_ceiling(self.category))
    }

    /// Resolve the active `diag_error` ceiling for ellipses.
    pub fn ceiling_ellipse(&self) -> f64 {
        self.max_diag_error_ellipse
            .unwrap_or_else(|| default_ceiling(self.category))
    }
}

/// Default `diag_error` ceiling for [`Category::Easy`] specs.
pub const DEFAULT_MAX_DIAG_ERROR_EASY: f64 = 5e-3;
/// Default `diag_error` ceiling for [`Category::Medium`] specs.
pub const DEFAULT_MAX_DIAG_ERROR_MEDIUM: f64 = 5e-3;
/// Default `diag_error` ceiling for [`Category::Hard`] specs.
pub const DEFAULT_MAX_DIAG_ERROR_HARD: f64 = 5e-2;

fn default_ceiling(c: Category) -> f64 {
    match c {
        Category::Easy => DEFAULT_MAX_DIAG_ERROR_EASY,
        Category::Medium => DEFAULT_MAX_DIAG_ERROR_MEDIUM,
        Category::Hard => DEFAULT_MAX_DIAG_ERROR_HARD,
    }
}

/// Seeds used by the bench-style quality sweep. Wide enough to catch
/// unlucky basins without dominating runtime. Mirrors the legacy
/// `QUALITY_SEEDS` array in `benches/initial_layout.rs`.
pub const QUALITY_SEEDS: [u64; 16] = [1, 2, 3, 7, 13, 17, 23, 29, 31, 37, 41, 42, 47, 53, 59, 61];

/// Small fixed seed list for the default-suite quality test. Three seeds is
/// enough to surface obvious regressions without blowing the wall-time
/// budget; the full 16-seed sweep lives in the bench harness / future
/// quality-report binary.
pub const TEST_SEEDS: [u64; 3] = [1, 42, 7];

/// All 17 corpus specs in a deterministic order matching
/// `eulerr/tests/testthat/test-reproducibility.R:1-188`.
pub fn all() -> &'static [CorpusEntry] {
    &CORPUS
}

/// Look up a corpus entry by its `name`.
pub fn get(name: &str) -> Option<&'static CorpusEntry> {
    CORPUS.iter().find(|e| e.name == name)
}

// 17 reproducibility specs from eulerr.
// The categorisation reflects what circles and ellipses can faithfully
// represent at default `Fitter` settings. Russian-doll containment and
// 6-set Wilkinson are the obvious "Hard" cases for both shapes; specs
// that can't be fit at all (single-set #16) are marked `SanityOnly`.

#[allow(clippy::too_many_lines)]
static CORPUS: [CorpusEntry; 17] = [
    CorpusEntry {
        name: "uniform_3_set",
        build: spec_uniform_3_set,
        category: Category::Easy,
        // Symmetric 3-Venn with uniform overlaps cannot be fit exactly by
        // circles (eulerr reports ~2-3% here); ellipses get it to ~0.
        max_diag_error_circle: Some(3e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "three_disjoint",
        build: spec_three_disjoint,
        category: Category::Easy,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "one_contained",
        build: spec_one_contained,
        category: Category::Medium,
        // Containment is hard for circles; eulerr accepts ~2-3% here.
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "two_inside_third",
        build: spec_two_inside_third,
        category: Category::Medium,
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "one_contained_others_interacting",
        build: spec_one_contained_others_interacting,
        category: Category::Medium,
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "russian_doll",
        build: spec_russian_doll,
        category: Category::Hard,
        // Circles cannot represent strict containment exactly.
        max_diag_error_circle: Some(1e-1),
        max_diag_error_ellipse: Some(5e-2),
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "unequal_overlaps",
        build: spec_unequal_overlaps,
        category: Category::Medium,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "two_disjoint",
        build: spec_two_disjoint,
        category: Category::Easy,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "wilkinson_6_set",
        build: spec_wilkinson_6_set,
        category: Category::Hard,
        // Circles can't represent the 6-set Wilkinson layout faithfully;
        // ellipses can (eulerr reaches ~1e-9 with tighter settings, but
        // default Fitter at TEST_SEEDS varies).
        max_diag_error_circle: Some(1e-1),
        max_diag_error_ellipse: Some(5e-2),
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "gene_sets",
        build: spec_gene_sets,
        category: Category::Medium,
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "three_inside_fourth",
        build: spec_three_inside_fourth,
        category: Category::Hard,
        max_diag_error_circle: Some(1e-1),
        max_diag_error_ellipse: Some(5e-2),
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "eulerape_3_set",
        build: spec_eulerape_3_set,
        category: Category::Easy,
        // Asymmetric 3-set from the eulerAPE article: circles cannot
        // achieve the exact triple intersection. Ellipses *can* fit this
        // exactly in principle, but the default Fitter currently lands at
        // ~1.2e-2 across our TEST_SEEDS — a known suboptimum worth
        // investigating but not blocking this harness on.
        max_diag_error_circle: Some(2e-2),
        max_diag_error_ellipse: Some(2e-2),
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "one_disjoint_two_intersecting",
        build: spec_one_disjoint_two_intersecting,
        category: Category::Medium,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "four_uniform_interactions",
        build: spec_four_uniform_interactions,
        category: Category::Medium,
        // 4-set uniform overlaps stretch what circles can fit.
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
    CorpusEntry {
        name: "two_overlapping_completely",
        build: spec_two_overlapping_completely,
        category: Category::Easy,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        fittable_circle: Fittable::Normal,
        // The ellipse path trips a `normalize_layout`-vs-exclusive-areas
        // debug_assert on coincident ellipses (a separate upstream bug).
        // Skip ellipses for this spec until that's resolved; circles fit
        // it perfectly.
        fittable_ellipse: Fittable::Skip("normalize_layout debug_assert on coincident ellipses"),
    },
    CorpusEntry {
        name: "single_set",
        build: spec_single_set,
        category: Category::Hard,
        max_diag_error_circle: None,
        max_diag_error_ellipse: None,
        // Preprocessing rejects n_sets <= 1, so fit() returns Err.
        fittable_circle: Fittable::SanityOnly,
        fittable_ellipse: Fittable::SanityOnly,
    },
    CorpusEntry {
        name: "random_4_set",
        build: spec_random_4_set,
        category: Category::Medium,
        // Hardcoded seeded random values — no expected analytic answer,
        // but a well-formed 4-set with all pairwise + triple + quad
        // intersections. Random area inputs aren't guaranteed to be
        // representable by any shape, so ceilings are loose for both.
        max_diag_error_circle: Some(5e-2),
        max_diag_error_ellipse: Some(5e-2),
        fittable_circle: Fittable::Normal,
        fittable_ellipse: Fittable::Normal,
    },
];

// Builders. Names mirror the comments in `test-reproducibility.R`.
// All values use exclusive (`disjoint` in eulerr terminology) input.

fn spec_uniform_3_set() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 10.0)
        .set("C", 10.0)
        .intersection(&["A", "B"], 4.0)
        .intersection(&["A", "C"], 4.0)
        .intersection(&["B", "C"], 4.0)
        .intersection(&["A", "B", "C"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("uniform_3_set")
}

fn spec_three_disjoint() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 1.0)
        .set("B", 1.0)
        .set("C", 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("three_disjoint")
}

fn spec_one_contained() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 10.0)
        .set("C", 0.0)
        .intersection(&["A", "B"], 4.0)
        .intersection(&["A", "C"], 0.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["A", "B", "C"], 3.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("one_contained")
}

fn spec_two_inside_third() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 15.0)
        .set("B", 0.0)
        .set("C", 0.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 3.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["A", "B", "C"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("two_inside_third")
}

fn spec_one_contained_others_interacting() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 15.0)
        .set("B", 15.0)
        .set("C", 0.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 0.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["A", "B", "C"], 3.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("one_contained_others_interacting")
}

fn spec_russian_doll() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 15.0)
        .set("B", 0.0)
        .set("C", 0.0)
        .intersection(&["A", "B"], 10.0)
        .intersection(&["A", "C"], 0.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["A", "B", "C"], 5.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("russian_doll")
}

fn spec_unequal_overlaps() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 7.0)
        .set("B", 6.0)
        .set("C", 0.0)
        .intersection(&["A", "B"], 0.0)
        .intersection(&["A", "C"], 1.0)
        .intersection(&["B", "C"], 1.0)
        .intersection(&["A", "B", "C"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("unequal_overlaps")
}

fn spec_two_disjoint() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 9.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("two_disjoint")
}

fn spec_wilkinson_6_set() -> DiagramSpec {
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
        .expect("wilkinson_6_set")
}

fn spec_gene_sets() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("SE", 13.0)
        .set("Treat", 28.0)
        .set("Anti-CCP", 101.0)
        .set("DAS28", 91.0)
        .intersection(&["SE", "Treat"], 1.0)
        .intersection(&["SE", "DAS28"], 14.0)
        .intersection(&["Treat", "Anti-CCP"], 6.0)
        .intersection(&["SE", "Anti-CCP", "DAS28"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("gene_sets")
}

fn spec_three_inside_fourth() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 30.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 3.0)
        .intersection(&["A", "D"], 3.0)
        .intersection(&["A", "B", "C"], 2.0)
        .intersection(&["A", "B", "D"], 2.0)
        .intersection(&["A", "C", "D"], 2.0)
        .intersection(&["A", "B", "C", "D"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("three_inside_fourth")
}

fn spec_eulerape_3_set() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("a", 3491.0)
        .set("b", 3409.0)
        .set("c", 3503.0)
        .intersection(&["a", "b"], 120.0)
        .intersection(&["a", "c"], 114.0)
        .intersection(&["b", "c"], 132.0)
        .intersection(&["a", "b", "c"], 126.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("eulerape_3_set")
}

fn spec_one_disjoint_two_intersecting() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 1.0)
        .set("B", 0.4)
        .set("C", 3.0)
        .intersection(&["A", "B"], 0.2)
        .intersection(&["A", "C"], 0.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["A", "B", "C"], 0.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("one_disjoint_two_intersecting")
}

fn spec_four_uniform_interactions() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 10.0)
        .set("B", 10.0)
        .set("C", 10.0)
        .set("D", 10.0)
        .intersection(&["A", "B"], 3.0)
        .intersection(&["A", "C"], 3.0)
        .intersection(&["A", "D"], 0.0)
        .intersection(&["B", "C"], 0.0)
        .intersection(&["B", "D"], 3.0)
        .intersection(&["C", "D"], 3.0)
        .intersection(&["A", "B", "C"], 1.0)
        .intersection(&["A", "B", "D"], 1.0)
        .intersection(&["A", "C", "D"], 1.0)
        .intersection(&["B", "C", "D"], 1.0)
        .intersection(&["A", "B", "C", "D"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("four_uniform_interactions")
}

fn spec_two_overlapping_completely() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 0.0)
        .set("B", 0.0)
        .intersection(&["A", "B"], 10.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("two_overlapping_completely")
}

fn spec_single_set() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .expect("single_set")
}

// 4-set spec from `set.seed(1); runif(15)` in R, ordered:
// A, B, C, E, A&B, A&C, A&E, B&C, B&E, C&E, A&B&C, A&B&E, A&C&E, B&C&E, A&B&C&E.
// Reproduces eulerr's `s[[17]]` exactly.
fn spec_random_4_set() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 0.265_508_663_142_1)
        .set("B", 0.372_123_899_636_79)
        .set("C", 0.572_853_363_351_896_4)
        .set("E", 0.908_207_789_994_776_2)
        .intersection(&["A", "B"], 0.201_681_931_037_455_8)
        .intersection(&["A", "C"], 0.898_389_684_967_696_7)
        .intersection(&["A", "E"], 0.944_675_268_605_351_4)
        .intersection(&["B", "C"], 0.660_797_792_486_846_4)
        .intersection(&["B", "E"], 0.629_114_043_898_880_5)
        .intersection(&["C", "E"], 0.061_786_270_467_564_46)
        .intersection(&["A", "B", "C"], 0.205_974_574_899_301)
        .intersection(&["A", "B", "E"], 0.176_556_752_528_995_3)
        .intersection(&["A", "C", "E"], 0.687_022_846_657_782_8)
        .intersection(&["B", "C", "E"], 0.384_103_718_213_737)
        .intersection(&["A", "B", "C", "E"], 0.769_841_419_998_556_4)
        .input_type(InputType::Exclusive)
        .build()
        .expect("random_4_set")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_has_17_unique_named_entries() {
        let entries = all();
        assert_eq!(entries.len(), 17);
        let mut names: Vec<&str> = entries.iter().map(|e| e.name).collect();
        names.sort();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped, "corpus names must be unique");
    }

    #[test]
    fn every_corpus_spec_builds() {
        for entry in all() {
            let _ = (entry.build)();
        }
    }

    #[test]
    fn lookup_by_name_works() {
        assert!(get("eulerape_3_set").is_some());
        assert!(get("nonexistent_spec").is_none());
    }
}
