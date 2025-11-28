use std::borrow::Borrow;

#[cfg(test)]
pub fn approx_eq<I, J>(a: I, b: J, epsilon: f64) -> bool
where
    I: IntoIterator,
    J: IntoIterator,
    I::Item: std::borrow::Borrow<f64>,
    J::Item: std::borrow::Borrow<f64>,
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
