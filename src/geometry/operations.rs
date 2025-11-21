use crate::geometry::coord::Coord;

pub trait Area {
    fn area(&self) -> f64;
}

pub trait Centroid {
    fn centroid(&self) -> Coord;
}

pub trait Intersects {
    fn intersects(&self, other: &Self) -> bool;
}

pub trait Contains {
    fn contains(&self, other: &Self) -> bool;
}

pub trait Distance {
    fn distance(&self, other: &Self) -> f64;
}

pub trait Perimeter {
    fn perimeter(&self) -> f64;
}

pub trait IntersectionArea {
    fn intersection_area(&self, other: &Self) -> f64;
}
