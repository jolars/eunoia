use std::borrow::Borrow;

#[cfg(test)]
pub fn approx_eq<I, J>(a: I, b: J, epsilon: f64) -> bool
where
    I: IntoIterator,
    J: IntoIterator,
    I::Item: Borrow<f64>,
    J::Item: Borrow<f64>,
{
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(x), Some(y)) => {
                if (x.borrow() - y.borrow()).abs() > epsilon {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false, // different lengths
        }
    }
}

#[cfg(test)]
pub fn approx_eq_scalar(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() <= epsilon
}

#[cfg(test)]
#[macro_export]
macro_rules! assert_approx_eq {
    // Special case for f64 literals or expressions that are clearly f64
    ($left:expr, $right:expr, $epsilon:expr) => {{
        $crate::_assert_approx_eq_impl!($left, $right, $epsilon)
    }};
    ($left:expr, $right:expr, $epsilon:expr, $($arg:tt)+) => {{
        $crate::_assert_approx_eq_impl!($left, $right, $epsilon, $($arg)+)
    }};
}

#[cfg(test)]
#[macro_export]
#[doc(hidden)]
macro_rules! _assert_approx_eq_impl {
    ($left:expr, $right:expr, $epsilon:expr) => {{
        // Helper trait to dispatch
        trait ApproxEqDispatch {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool;
        }

        impl ApproxEqDispatch for f64 {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool {
                $crate::test_utils::approx_eq_scalar(*self, *other, epsilon)
            }
        }

        impl<T> ApproxEqDispatch for &T
        where
            for<'a> &'a T: IntoIterator,
            for<'a> <&'a T as IntoIterator>::Item: std::borrow::Borrow<f64>,
        {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool {
                $crate::test_utils::approx_eq(*self, *other, epsilon)
            }
        }

        let eps = $epsilon;
        let left_val = &$left;
        let right_val = &$right;

        if !left_val.approx_eq_dispatch(right_val, eps) {
            panic!(
                "assertion failed: `approx_eq(left, right, epsilon)`\n  left: `{:?}`\n right: `{:?}`\nepsilon: `{:?}`",
                left_val, right_val, eps
            );
        }
    }};
    ($left:expr, $right:expr, $epsilon:expr, $($arg:tt)+) => {{
        trait ApproxEqDispatch {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool;
        }

        impl ApproxEqDispatch for f64 {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool {
                $crate::test_utils::approx_eq_scalar(*self, *other, epsilon)
            }
        }

        impl<T> ApproxEqDispatch for &T
        where
            for<'a> &'a T: IntoIterator,
            for<'a> <&'a T as IntoIterator>::Item: std::borrow::Borrow<f64>,
        {
            fn approx_eq_dispatch(&self, other: &Self, epsilon: f64) -> bool {
                $crate::test_utils::approx_eq(*self, *other, epsilon)
            }
        }

        let eps = $epsilon;
        let left_val = &$left;
        let right_val = &$right;

        if !left_val.approx_eq_dispatch(right_val, eps) {
            panic!(
                "assertion failed: `approx_eq(left, right, epsilon)`\n  left: `{:?}`\n right: `{:?}`\nepsilon: `{:?}`: {}",
                left_val, right_val, eps, format_args!($($arg)+)
            );
        }
    }};
}
