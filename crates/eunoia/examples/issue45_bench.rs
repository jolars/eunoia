//! Reproduction benchmark for issue #45.
//!
//! Times the 6-set ellipse fit from the issue across smooth and non-smooth
//! losses, including the `LossType::Smooth*` surrogate variants
//! (logsumexp / Huber smoothing) that route through the L-BFGS path.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example issue45_bench
//! ```

use eunoia::geometry::shapes::Ellipse;
use eunoia::loss::LossType;
use eunoia::{DiagramSpecBuilder, Fitter, InputType};
use std::io::Write;
use std::time::Instant;

fn six_set_spec() -> eunoia::spec::DiagramSpec {
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

fn bench(label: &str, n: usize, loss: LossType) {
    let spec = six_set_spec();
    print!("  {:<46}", label);
    std::io::stdout().flush().unwrap();
    let _ = Fitter::<Ellipse>::new(&spec)
        .loss_type(loss)
        .seed(1)
        .fit()
        .unwrap();
    let t = Instant::now();
    let mut last_diag = 0.0;
    for i in 0..n {
        let layout = Fitter::<Ellipse>::new(&spec)
            .loss_type(loss)
            .seed(i as u64 + 2)
            .fit()
            .unwrap();
        last_diag = layout.diag_error();
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!(
        ": {:>2} fits, {:>9.1} ms/fit, last diag_error = {:.4e}",
        n, per, last_diag
    );
}

fn main() {
    println!("Smooth losses (control):");
    bench("loss=SumSquared", 3, LossType::SumSquared);

    println!("\nNon-smooth losses (NM dispatch):");
    bench("loss=SumAbsolute", 3, LossType::sum_absolute());
    bench("loss=MaxSquared", 3, LossType::MaxSquared);
    bench("loss=MaxAbsolute", 3, LossType::MaxAbsolute);

    println!("\nSmooth surrogates (eps=1e-3, L-BFGS dispatch):");
    bench(
        "loss=SmoothSumAbsolute{eps=1e-3}",
        3,
        LossType::smooth_sum_absolute(1e-3),
    );
    bench(
        "loss=SmoothMaxSquared{eps=1e-3}",
        3,
        LossType::smooth_max_squared(1e-3),
    );
    bench(
        "loss=SmoothMaxAbsolute{eps=1e-3}",
        3,
        LossType::smooth_max_absolute(1e-3),
    );
}
