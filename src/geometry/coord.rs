use crate::geometry::operations::Distance;

pub struct Coord {
    x: f64,
    y: f64,
}

impl Coord {
    pub fn new(x: f64, y: f64) -> Self {
        Coord { x, y }
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}

impl Distance for Coord {
    fn distance(&self, other: &Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
