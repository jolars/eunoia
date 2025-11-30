use eunoia_nlm::{optdrv, optimize, NlmConfig, OptimizationConfig};
use nalgebra::dvector;

fn main() {
    fn build_cfg(n: usize) -> OptimizationConfig {
        let nlm = NlmConfig::default();
        OptimizationConfig {
            typsiz: nlm
                .typsize
                .unwrap_or_else(|| nalgebra::DVector::from_element(n, 1.0)),
            fscale: nlm.fscale,
            method: nlm.method,
            expensive: nlm.expensive,
            ndigit: nlm.ndigit.unwrap_or(-1),
            itnlim: nlm.max_iter,
            has_gradient: false,
            has_hessian: false,
            dlt: nlm.delta.unwrap_or(-1.0),
            gradtl: nlm.grad_tol,
            stepmx: nlm.max_step,
            steptl: nlm.step_tol,
        }
    }

    let rosen =
        |x: &nalgebra::DVector<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
    let rosen_grad = |x: &nalgebra::DVector<f64>| {
        dvector![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2))
        ]
    };
    let x0 = dvector![-1.2, 1.0];
    let res_num = optdrv(&x0, &rosen, None, None, &build_cfg(2));
    let res_an = optimize(
        x0.clone(),
        rosen,
        rosen_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Rosenbrock: numeric={} analytic={} (R=22)",
        res_num.iterations, res_an.iterations
    );

    let quad = |x: &nalgebra::DVector<f64>| (x[0] - 3.0).powi(2) + 10.0 * (x[1] + 1.0).powi(2);
    let quad_grad = |x: &nalgebra::DVector<f64>| dvector![2.0 * (x[0] - 3.0), 20.0 * (x[1] + 1.0)];
    let xq = dvector![0.0, 0.0];
    let res_q_num = optdrv(&xq, &quad, None, None, &build_cfg(2));
    let res_q_an = optimize(
        xq.clone(),
        quad,
        quad_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Quadratic: numeric={} analytic={} (R=8)",
        res_q_num.iterations, res_q_an.iterations
    );

    let powell = |x: &nalgebra::DVector<f64>| {
        (x[0] + 10.0 * x[1]).powi(2)
            + 5.0 * (x[2] - x[3]).powi(2)
            + (x[1] - 2.0 * x[2]).powi(4)
            + 10.0 * (x[0] - x[3]).powi(4)
    };
    let powell_grad = |x: &nalgebra::DVector<f64>| {
        dvector![
            2.0 * (x[0] + 10.0 * x[1]) + 40.0 * (x[0] - x[3]).powi(3),
            20.0 * (x[0] + 10.0 * x[1]) + 4.0 * (x[1] - 2.0 * x[2]).powi(3),
            10.0 * (x[2] - x[3]) - 8.0 * (x[1] - 2.0 * x[2]).powi(3),
            -10.0 * (x[2] - x[3]) - 40.0 * (x[0] - x[3]).powi(3)
        ]
    };
    let xp = dvector![3.0, -1.0, 0.0, 1.0];
    let res_p_num = optdrv(&xp, &powell, None, None, &build_cfg(4));
    let res_p_an = optimize(
        xp.clone(),
        powell,
        powell_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Powell: numeric={} analytic={} (R=31)",
        res_p_num.iterations, res_p_an.iterations
    );

    let wood = |x: &nalgebra::DVector<f64>| {
        100.0 * (x[1] - x[0].powi(2)).powi(2)
            + (1.0 - x[0]).powi(2)
            + 90.0 * (x[3] - x[2].powi(2)).powi(2)
            + (1.0 - x[2]).powi(2)
            + 10.1 * ((x[1] - 1.0).powi(2) + (x[3] - 1.0).powi(2))
            + 19.8 * (x[1] - 1.0) * (x[3] - 1.0)
    };
    let wood_grad = |x: &nalgebra::DVector<f64>| {
        dvector![
            -400.0 * x[0] * (x[1] - x[0].powi(2)) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0].powi(2)) + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0),
            -360.0 * x[2] * (x[3] - x[2].powi(2)) - 2.0 * (1.0 - x[2]),
            180.0 * (x[3] - x[2].powi(2)) + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0)
        ]
    };
    let xw = dvector![-3.0, -1.0, -3.0, -1.0];
    let res_w_num = optdrv(&xw, &wood, None, None, &build_cfg(4));
    let res_w_an = optimize(
        xw.clone(),
        wood,
        wood_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Wood: numeric={} analytic={} (R=30)",
        res_w_num.iterations, res_w_an.iterations
    );

    let helical = |x: &nalgebra::DVector<f64>| {
        let theta = x[1].atan2(x[0]) / (2.0 * std::f64::consts::PI);
        let r = (x[0].powi(2) + x[1].powi(2)).sqrt();
        100.0 * ((x[2] - 10.0 * theta).powi(2) + (r - 1.0).powi(2)) + x[2].powi(2)
    };
    let helical_grad = |x: &nalgebra::DVector<f64>| {
        let r2 = x[0].powi(2) + x[1].powi(2);
        let r = r2.sqrt().max(1e-12);
        let theta = x[1].atan2(x[0]) / (2.0 * std::f64::consts::PI);
        let dtheta_dx0 = -x[1] / (2.0 * std::f64::consts::PI * r2);
        let dtheta_dx1 = x[0] / (2.0 * std::f64::consts::PI * r2);
        dvector![
            -2000.0 * (x[2] - 10.0 * theta) * dtheta_dx0 + 200.0 * (r - 1.0) * x[0] / r,
            -2000.0 * (x[2] - 10.0 * theta) * dtheta_dx1 + 200.0 * (r - 1.0) * x[1] / r,
            200.0 * (x[2] - 10.0 * theta) + 2.0 * x[2]
        ]
    };
    let xh = dvector![-1.0, 0.0, 0.0];
    let res_h_num = optdrv(&xh, &helical, None, None, &build_cfg(3));
    let res_h_an = optimize(
        xh.clone(),
        helical,
        helical_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Helical: numeric={} analytic={} (R=24)",
        res_h_num.iterations, res_h_an.iterations
    );

    let beale = |x: &nalgebra::DVector<f64>| {
        (1.5 - x[0] + x[0] * x[1]).powi(2)
            + (2.25 - x[0] + x[0] * x[1].powi(2)).powi(2)
            + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
    };
    let beale_grad = |x: &nalgebra::DVector<f64>| {
        let t1 = 1.5 - x[0] + x[0] * x[1];
        let t2 = 2.25 - x[0] + x[0] * x[1].powi(2);
        let t3 = 2.625 - x[0] + x[0] * x[1].powi(3);
        dvector![
            2.0 * t1 * (x[1] - 1.0)
                + 2.0 * t2 * (x[1].powi(2) - 1.0)
                + 2.0 * t3 * (x[1].powi(3) - 1.0),
            2.0 * t1 * x[0] + 4.0 * t2 * x[0] * x[1] + 6.0 * t3 * x[0] * x[1].powi(2)
        ]
    };
    let xb = dvector![1.0, 1.0];
    let res_b_num = optdrv(&xb, &beale, None, None, &build_cfg(2));
    let res_b_an = optimize(
        xb.clone(),
        beale,
        beale_grad,
        NlmConfig {
            has_gradient: true,
            ..Default::default()
        },
    );
    println!(
        "Beale: numeric={} analytic={} (R=15)",
        res_b_num.iterations, res_b_an.iterations
    );
}
