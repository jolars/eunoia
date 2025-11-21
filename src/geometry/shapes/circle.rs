use crate::geometry::coord::Coord;
use crate::geometry::operations::Area;
use crate::geometry::operations::Contains;
use crate::geometry::operations::Distance;
use crate::geometry::operations::IntersectionArea;
use crate::geometry::operations::Intersects;

pub struct Circle {
    center: Coord,
    radius: f64,
}

impl Area for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

impl Distance for Circle {
    fn distance(&self, other: &Self) -> f64 {
        let center_distance = self.center.distance(&other.center);
        let radius_sum = self.radius + other.radius;

        if center_distance > radius_sum {
            center_distance - radius_sum
        } else {
            0.0
        }
    }
}

impl Contains for Circle {
    fn contains(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance + other.radius <= self.radius
    }
}

impl Intersects for Circle {
    fn intersects(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance >= self.radius + other.radius
    }
}

impl IntersectionArea for Circle {
    fn intersection_area(&self, other: &Self) -> f64 {
        let d = self.center.distance(&other.center);

        if d >= self.radius + other.radius {
            return 0.0; // No intersection
        }

        if d <= (self.radius - other.radius).abs() {
            // One circle is completely inside the other
            let smaller_radius = self.radius.min(other.radius);
            return std::f64::consts::PI * smaller_radius * smaller_radius;
        }

        let r1 = self.radius;
        let r2 = other.radius;

        let part1 = r1 * r1 * (((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos());
        let part2 = r2 * r2 * (((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos());
        let part3 = 0.5 * ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();

        part1 + part2 - part3
    }
}
