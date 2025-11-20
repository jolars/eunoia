pub mod geometry;

pub trait Intersects<S> {
    fn intersection_area(&self, other: &S) -> f64;
}

pub trait Parameters {
    fn n_params(&self) -> usize;
    fn update(&mut self, params: &[f64]);
}

pub struct Diagram<S> {
    shapes: Vec<S>,
}

#[cfg(test)]
mod tests {
    use super::*;
}
