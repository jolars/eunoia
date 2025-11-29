//! Integration tests for the NLM optimization library
//!
//! These tests verify the full optimization workflow on various test problems.

use eunoia_nlm::{optimize, NlmConfig};
use nalgebra::{dvector, DVector};

/// Simple quadratic: f(x) = (x-2)^2 + (y-3)^2
/// Minimum at (2, 3) with f* = 0
fn quadratic(x: &DVector<f64>) -> f64 {
    (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
}

fn quadratic_grad(x: &DVector<f64>) -> DVector<f64> {
    dvector![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)]
}

/// Rosenbrock function: f(x,y) = 100*(y - x^2)^2 + (1 - x)^2
/// Minimum at (1, 1) with f* = 0
fn rosenbrock(x: &DVector<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    100.0 * (x1 - x0 * x0).powi(2) + (1.0 - x0).powi(2)
}

fn rosenbrock_grad(x: &DVector<f64>) -> DVector<f64> {
    let x0 = x[0];
    let x1 = x[1];
    dvector![
        -400.0 * x0 * (x1 - x0 * x0) - 2.0 * (1.0 - x0),
        200.0 * (x1 - x0 * x0)
    ]
}

/// Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
/// Has 4 local minima, all with f* = 0
fn himmelblau(x: &DVector<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
}

fn himmelblau_grad(x: &DVector<f64>) -> DVector<f64> {
    let x0 = x[0];
    let x1 = x[1];
    dvector![
        4.0 * x0 * (x0 * x0 + x1 - 11.0) + 2.0 * (x0 + x1 * x1 - 7.0),
        2.0 * (x0 * x0 + x1 - 11.0) + 4.0 * x1 * (x0 + x1 * x1 - 7.0)
    ]
}

/// Booth function: f(x,y) = (x + 2*y - 7)^2 + (2*x + y - 5)^2
/// Minimum at (1, 3) with f* = 0
fn booth(x: &DVector<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (x0 + 2.0 * x1 - 7.0).powi(2) + (2.0 * x0 + x1 - 5.0).powi(2)
}

fn booth_grad(x: &DVector<f64>) -> DVector<f64> {
    let x0 = x[0];
    let x1 = x[1];
    dvector![
        2.0 * (x0 + 2.0 * x1 - 7.0) + 4.0 * (2.0 * x0 + x1 - 5.0),
        4.0 * (x0 + 2.0 * x1 - 7.0) + 2.0 * (2.0 * x0 + x1 - 5.0)
    ]
}

/// Beale function: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
/// Minimum at (3, 0.5) with f* = 0
fn beale(x: &DVector<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (1.5 - x0 + x0 * x1).powi(2)
        + (2.25 - x0 + x0 * x1 * x1).powi(2)
        + (2.625 - x0 + x0 * x1.powi(3)).powi(2)
}

fn beale_grad(x: &DVector<f64>) -> DVector<f64> {
    let x0 = x[0];
    let x1 = x[1];
    let t1 = 1.5 - x0 + x0 * x1;
    let t2 = 2.25 - x0 + x0 * x1 * x1;
    let t3 = 2.625 - x0 + x0 * x1.powi(3);
    dvector![
        2.0 * t1 * (-1.0 + x1) + 2.0 * t2 * (-1.0 + x1 * x1) + 2.0 * t3 * (-1.0 + x1.powi(3)),
        2.0 * t1 * x0 + 2.0 * t2 * 2.0 * x0 * x1 + 2.0 * t3 * 3.0 * x0 * x1 * x1
    ]
}

#[test]
fn test_gradient_check() {
    // Verify gradient computation is working
    let x = dvector![1.0, 2.0];
    let g_analytic = quadratic_grad(&x);

    // Compute numeric gradient
    let eps = 1e-8;
    let mut g_numeric = DVector::zeros(2);
    for i in 0..2 {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        g_numeric[i] = (quadratic(&x_plus) - quadratic(&x_minus)) / (2.0 * eps);
    }

    // Check they match
    for i in 0..2 {
        assert!(
            (g_analytic[i] - g_numeric[i]).abs() < 1e-5,
            "Gradient mismatch at component {}: analytic={}, numeric={}",
            i,
            g_analytic[i],
            g_numeric[i]
        );
    }
}

#[test]
fn test_function_evaluation() {
    // Test that our test functions are correct
    assert_eq!(quadratic(&dvector![2.0, 3.0]), 0.0);
    assert_eq!(quadratic(&dvector![0.0, 0.0]), 13.0);

    let g = quadratic_grad(&dvector![0.0, 0.0]);
    assert_eq!(g[0], -4.0);
    assert_eq!(g[1], -6.0);
}

#[test]
fn test_quadratic_optimization() {
    let x0 = dvector![0.0, 0.0];
    let config = NlmConfig::default();

    let result = optimize(x0, quadratic, quadratic_grad, config);

    assert!(
        result.termination.is_converged(),
        "Optimization should converge, got {:?}",
        result.termination
    );
    assert!(
        (result.xpls[0] - 2.0).abs() < 1e-4,
        "x should be close to 2.0, got {}",
        result.xpls[0]
    );
    assert!(
        (result.xpls[1] - 3.0).abs() < 1e-4,
        "y should be close to 3.0, got {}",
        result.xpls[1]
    );
    assert!(
        result.fpls < 1e-6,
        "Function value should be close to 0, got {}",
        result.fpls
    );
}

#[test]
fn test_rosenbrock_optimization() {
    let x0 = dvector![-1.2, 1.0]; // Standard starting point
    let mut config = NlmConfig::default();
    config.max_iter = 50; // Just run a few iterations to see progress
    config.expensive = true; // Back to default

    let result = optimize(x0, rosenbrock, rosenbrock_grad, config);

    // Debug output
    eprintln!("Rosenbrock result after {} iters:", result.iterations);
    eprintln!("  x = [{}, {}]", result.xpls[0], result.xpls[1]);
    eprintln!("  f = {}", result.fpls);
    eprintln!("  ||g|| = {}", result.gpls.norm());
    eprintln!("  termination = {:?}", result.termination);

    // For now, just check it doesn't crash
    assert!(result.iterations > 0);
    assert!(
        (result.xpls[0] - 1.0).abs() < 1e-3,
        "x should be close to 1.0, got {}",
        result.xpls[0]
    );
    assert!(
        (result.xpls[1] - 1.0).abs() < 1e-3,
        "y should be close to 1.0, got {}",
        result.xpls[1]
    );
    assert!(
        result.fpls < 1e-4,
        "Function value should be close to 0, got {}",
        result.fpls
    );
}

#[test]
fn test_himmelblau_optimization() {
    // Test one of the four minima: (3, 2)
    let x0 = dvector![2.5, 1.5];
    let config = NlmConfig::default();

    let result = optimize(x0, himmelblau, himmelblau_grad, config);

    assert!(
        result.termination.is_converged(),
        "Optimization should converge, got {:?}",
        result.termination
    );

    // Check if we found one of the four minima
    let minima = vec![
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ];

    let found_minimum = minima
        .iter()
        .any(|(mx, my)| (result.xpls[0] - mx).abs() < 1e-2 && (result.xpls[1] - my).abs() < 1e-2);

    assert!(
        found_minimum,
        "Should find one of the four minima, got ({}, {})",
        result.xpls[0], result.xpls[1]
    );
    assert!(
        result.fpls < 1e-4,
        "Function value should be close to 0, got {}",
        result.fpls
    );
}

#[test]
fn test_booth_optimization() {
    let x0 = dvector![0.0, 0.0];
    let config = NlmConfig::default();

    let result = optimize(x0, booth, booth_grad, config);

    assert!(
        result.termination.is_converged(),
        "Optimization should converge, got {:?}",
        result.termination
    );
    assert!(
        (result.xpls[0] - 1.0).abs() < 1e-4,
        "x should be close to 1.0, got {}",
        result.xpls[0]
    );
    assert!(
        (result.xpls[1] - 3.0).abs() < 1e-4,
        "y should be close to 3.0, got {}",
        result.xpls[1]
    );
    assert!(
        result.fpls < 1e-6,
        "Function value should be close to 0, got {}",
        result.fpls
    );
}

#[test]
fn test_beale_optimization() {
    let x0 = dvector![1.0, 1.0];
    let mut config = NlmConfig::default();
    config.max_iter = 500;

    let result = optimize(x0, beale, beale_grad, config);

    assert!(
        result.termination.is_converged(),
        "Optimization should converge, got {:?}",
        result.termination
    );
    assert!(
        (result.xpls[0] - 3.0).abs() < 1e-3,
        "x should be close to 3.0, got {}",
        result.xpls[0]
    );
    assert!(
        (result.xpls[1] - 0.5).abs() < 1e-3,
        "y should be close to 0.5, got {}",
        result.xpls[1]
    );
    assert!(
        result.fpls < 1e-4,
        "Function value should be close to 0, got {}",
        result.fpls
    );
}

#[test]
fn test_max_iterations() {
    let x0 = dvector![-1.2, 1.0];
    let mut config = NlmConfig::default();
    config.max_iter = 5; // Very few iterations

    let result = optimize(x0, rosenbrock, rosenbrock_grad, config);

    // Should stop due to max iterations
    assert!(result.iterations <= 5, "Should not exceed max iterations");
}

#[test]
fn test_tolerance_settings() {
    let x0 = dvector![0.0, 0.0];
    let mut config = NlmConfig::default();
    config.grad_tol = 1e-2; // Loose tolerance

    let result = optimize(x0, quadratic, quadratic_grad, config);

    // Should converge faster with loose tolerance
    assert!(
        result.termination.is_converged(),
        "Should converge with loose tolerance, got {:?}",
        result.termination
    );
    assert!(
        result.iterations < 50,
        "Should converge quickly with loose tolerance"
    );
}
