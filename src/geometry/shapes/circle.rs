use crate::geometry::area::Area;

pub struct Circle {
    center: (f64, f64),
    radius: f64,
}

impl Area for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}
