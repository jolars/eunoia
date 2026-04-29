//! Regression test for eulerr issue #89.
//!
//! The original issue (<https://github.com/jolars/eulerr/issues/89>) reports a
//! 17-set ellipse fit that hangs because the ellipse path enumerated all
//! `2^n - 1 = 131,071` region masks per loss evaluation. Eunoia routes the
//! ellipse path through sparse region discovery (`discover_regions`), so only
//! geometrically-non-empty regions are walked.
//!
//! These tests are `#[ignore]` because a full 17-set fit takes seconds — too
//! slow for the default suite. Run with:
//!
//! ```text
//! cargo test --test issue89_regression -- --ignored --nocapture
//! ```
use eunoia::geometry::shapes::{Circle, Ellipse};
use eunoia::spec::DiagramSpec;
use eunoia::{DiagramSpecBuilder, Fitter, InputType};
use std::time::{Duration, Instant};

/// The exact disjoint-input spec from eulerr issue #89.
fn issue89_spec() -> DiagramSpec {
    DiagramSpecBuilder::new()
        .set("A", 55.0)
        .set("B", 810.0)
        .set("C", 102.0)
        .set("D", 364.0)
        .set("E", 101.0)
        .set("F", 24.0)
        .set("G", 34.0)
        .set("H", 61.0)
        .set("I", 194.0)
        .set("J", 107.0)
        .set("K", 53.0)
        .set("L", 75.0)
        .set("M", 11.0)
        .set("N", 65.0)
        .set("O", 16.0)
        .set("P", 82.0)
        .set("Q", 13.0)
        .intersection(&["F", "O"], 5.0)
        .intersection(&["G", "O"], 5.0)
        .intersection(&["F", "G"], 5.0)
        .intersection(&["D", "I"], 47.0)
        .intersection(&["D", "E"], 33.0)
        .intersection(&["K", "L"], 9.0)
        .intersection(&["A", "K"], 7.0)
        .intersection(&["K", "N"], 7.0)
        .intersection(&["A", "N"], 7.0)
        .intersection(&["A", "L"], 7.0)
        .intersection(&["K", "P"], 7.0)
        .intersection(&["A", "P"], 7.0)
        .intersection(&["L", "N"], 7.0)
        .intersection(&["N", "P"], 7.0)
        .intersection(&["C", "K"], 7.0)
        .intersection(&["A", "C"], 7.0)
        .intersection(&["L", "P"], 7.0)
        .intersection(&["E", "G"], 6.0)
        .intersection(&["J", "K"], 7.0)
        .intersection(&["A", "J"], 7.0)
        .intersection(&["C", "O"], 5.0)
        .intersection(&["G", "H"], 4.0)
        .intersection(&["C", "N"], 7.0)
        .intersection(&["J", "N"], 7.0)
        .intersection(&["E", "F"], 5.0)
        .intersection(&["C", "F"], 5.0)
        .intersection(&["C", "L"], 7.0)
        .intersection(&["J", "L"], 7.0)
        .intersection(&["C", "P"], 7.0)
        .intersection(&["J", "P"], 7.0)
        .intersection(&["C", "G"], 5.0)
        .intersection(&["D", "K"], 15.0)
        .intersection(&["C", "J"], 7.0)
        .intersection(&["A", "D"], 12.0)
        .intersection(&["D", "L"], 12.0)
        .intersection(&["E", "O"], 3.0)
        .intersection(&["E", "H"], 4.0)
        .intersection(&["C", "I"], 6.0)
        .intersection(&["C", "D"], 8.0)
        .intersection(&["D", "H"], 7.0)
        .intersection(&["D", "N"], 7.0)
        .intersection(&["D", "P"], 7.0)
        .intersection(&["D", "J"], 7.0)
        .intersection(&["C", "E"], 3.0)
        .intersection(&["H", "O"], 1.0)
        .intersection(&["B", "D"], 15.0)
        .intersection(&["B", "C"], 11.0)
        .intersection(&["F", "H"], 1.0)
        .intersection(&["E", "M"], 1.0)
        .intersection(&["B", "E"], 8.0)
        .intersection(&["B", "K"], 7.0)
        .intersection(&["I", "K"], 2.0)
        .intersection(&["A", "B"], 7.0)
        .intersection(&["B", "N"], 7.0)
        .intersection(&["B", "L"], 7.0)
        .intersection(&["B", "P"], 7.0)
        .intersection(&["D", "F"], 3.0)
        .intersection(&["B", "J"], 7.0)
        .intersection(&["I", "L"], 2.0)
        .intersection(&["C", "H"], 1.0)
        .intersection(&["D", "O"], 1.0)
        .intersection(&["D", "G"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap()
}

/// Generous wall-time budget. The point of this test is "doesn't hang" — not
/// "fits in X seconds". A pre-fix ellipse fit would have to enumerate
/// 131,071 region masks per loss eval and never complete in any reasonable
/// time; a few minutes here is a safe ceiling for slow CI machines.
const TIME_BUDGET: Duration = Duration::from_secs(180);

#[test]
#[ignore = "slow: full 17-set fit"]
fn issue89_circles_terminate() {
    let spec = issue89_spec();
    let t0 = Instant::now();
    let layout = Fitter::<Circle>::new(&spec).seed(1).fit().unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < TIME_BUDGET,
        "circle fit took {:?}, exceeds budget {:?}",
        elapsed,
        TIME_BUDGET
    );
    assert!(
        layout.diag_error().is_finite(),
        "diag_error must be finite, got {}",
        layout.diag_error()
    );
}

#[test]
#[ignore = "slow: full 17-set fit"]
fn issue89_ellipses_terminate() {
    // The actual regression: pre-fix this would not terminate because the
    // ellipse path walked all 2^17 - 1 = 131,071 region masks per loss eval.
    let spec = issue89_spec();
    let t0 = Instant::now();
    let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < TIME_BUDGET,
        "ellipse fit took {:?}, exceeds budget {:?}",
        elapsed,
        TIME_BUDGET
    );
    assert!(
        layout.diag_error().is_finite(),
        "diag_error must be finite, got {}",
        layout.diag_error()
    );
}
