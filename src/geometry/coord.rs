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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_new() {
        let coord = Coord::new(3.0, 4.0);
        assert_eq!(coord.x(), 3.0);
        assert_eq!(coord.y(), 4.0);
    }

    #[test]
    fn test_coord_getters() {
        let coord = Coord::new(-2.5, 7.8);
        assert_eq!(coord.x(), -2.5);
        assert_eq!(coord.y(), 7.8);
    }

    #[test]
    fn test_distance_same_point() {
        let coord1 = Coord::new(1.0, 1.0);
        let coord2 = Coord::new(1.0, 1.0);
        assert_eq!(coord1.distance(&coord2), 0.0);
    }

    #[test]
    fn test_distance_horizontal() {
        let coord1 = Coord::new(0.0, 0.0);
        let coord2 = Coord::new(3.0, 0.0);
        assert_eq!(coord1.distance(&coord2), 3.0);
    }

    #[test]
    fn test_distance_vertical() {
        let coord1 = Coord::new(0.0, 0.0);
        let coord2 = Coord::new(0.0, 4.0);
        assert_eq!(coord1.distance(&coord2), 4.0);
    }

    #[test]
    fn test_distance_pythagorean() {
        let coord1 = Coord::new(0.0, 0.0);
        let coord2 = Coord::new(3.0, 4.0);
        assert_eq!(coord1.distance(&coord2), 5.0);
    }

    #[test]
    fn test_distance_negative_coords() {
        let coord1 = Coord::new(-1.0, -1.0);
        let coord2 = Coord::new(2.0, 3.0);
        assert_eq!(coord1.distance(&coord2), 5.0);
    }

    #[test]
    fn test_distance_symmetric() {
        let coord1 = Coord::new(1.5, 2.5);
        let coord2 = Coord::new(4.5, 6.5);
        assert_eq!(coord1.distance(&coord2), coord2.distance(&coord1));
    }
}
