use std::collections::HashMap;

use super::*;
use crate::geometry::shapes::Polygon;
use crate::plotting::placement::{PlacementKind, TetherSource, place_labels, union_bbox};
use crate::plotting::regions::classify_into_pieces;

fn rectangle(x: f64, y: f64, w: f64, h: f64) -> RegionPiece {
    RegionPiece {
        outer: Polygon::new(vec![
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        holes: Vec::new(),
    }
}

fn obstacle<'a>(combo: &'a Combination, piece: &'a RegionPiece) -> Obstacle<'a> {
    let points = piece.outer.vertices();
    Obstacle {
        combo,
        piece,
        bounds: InteriorAabb {
            xmin: points.iter().map(Point::x).fold(f64::INFINITY, f64::min),
            ymin: points.iter().map(Point::y).fold(f64::INFINITY, f64::min),
            xmax: points
                .iter()
                .map(Point::x)
                .fold(f64::NEG_INFINITY, f64::max),
            ymax: points
                .iter()
                .map(Point::y)
                .fold(f64::NEG_INFINITY, f64::max),
        },
    }
}

#[test]
fn traversal_matches_analytic_rectangle_lengths() {
    let piece = rectangle(0.0, 0.0, 2.0, 2.0);
    let combo = Combination::new(&["B"]);
    let o = obstacle(&combo, &piece);
    for (start, end, expected) in [
        ((-1.0, 1.0), (3.0, 1.0), 2.0),
        ((-1.0, -1.0), (3.0, 3.0), 2.0 * 2.0_f64.sqrt()),
        ((0.5, 1.0), (1.5, 1.0), 1.0),
        ((-1.0, 0.0), (3.0, 0.0), 0.0),
        ((-1.0, 1.0), (1.0, 3.0), 0.0),
        ((3.0, 3.0), (3.0, 3.0), 0.0),
        ((1.0, 1.0), (1.0, 1.0), 0.0),
        ((3.0, -1.0), (3.0, 3.0), 0.0),
    ] {
        let start = Point::new(start.0, start.1);
        let end = Point::new(end.0, end.1);
        for (a, b) in [(start, end), (end, start)] {
            let actual = segment_length_in_piece(a, b, &o, 1e-8);
            assert!(
                (actual - expected).abs() < 1e-7,
                "{a:?} -> {b:?}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn traversal_respects_holes_and_concavity() {
    let mut piece = rectangle(0.0, 0.0, 10.0, 10.0);
    piece.holes.push(rectangle(4.0, 4.0, 2.0, 2.0).outer);
    let combo = Combination::new(&["B"]);
    let a = Point::new(-1.0, 5.0);
    let b = Point::new(11.0, 5.0);
    assert!((segment_length_in_piece(a, b, &obstacle(&combo, &piece), 1e-8) - 8.0).abs() < 1e-7);

    piece.outer = Polygon::new(
        [
            (0., 0.),
            (3., 0.),
            (3., 3.),
            (2., 3.),
            (2., 1.),
            (1., 1.),
            (1., 3.),
            (0., 3.),
        ]
        .into_iter()
        .map(|(x, y)| Point::new(x, y))
        .collect(),
    );
    piece.holes.clear();
    let a = Point::new(-1.0, 2.0);
    let b = Point::new(4.0, 2.0);
    assert!((segment_length_in_piece(a, b, &obstacle(&combo, &piece), 1e-8) - 2.0).abs() < 1e-7);
}

#[test]
fn traversal_counts_disconnected_foreign_pieces_but_excludes_the_source() {
    let own = Combination::new(&["A", "B"]);
    let foreign = Combination::new(&["A"]);
    let pieces = [
        rectangle(0., 0., 1., 1.),
        rectangle(2., 0., 1., 1.),
        rectangle(4., 0., 1., 1.),
    ];
    let obstacles = [
        obstacle(&foreign, &pieces[0]),
        obstacle(&own, &pieces[1]),
        obstacle(&foreign, &pieces[2]),
    ];
    assert!(
        (foreign_length(
            (Point::new(-1., 0.5), Point::new(6., 0.5)),
            &own,
            &obstacles,
            1e-8
        ) - 2.0)
            .abs()
            < 1e-7
    );
}

fn gallery() -> (
    RegionPolygons,
    HashMap<String, (f64, f64)>,
    serde_json::Value,
) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/leader_routing_gallery.json"
    )))
    .unwrap();
    let polygon = |v: &serde_json::Value| {
        Polygon::new(
            serde_json::from_value::<Vec<[f64; 2]>>(v.clone())
                .unwrap()
                .into_iter()
                .map(|[x, y]| Point::new(x, y))
                .collect(),
        )
    };
    let map = fixture["regions"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, pieces)| {
            let pieces = pieces
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|piece| {
                    let mut rings = vec![polygon(&piece["outer"])];
                    rings.extend(piece["holes"].as_array().unwrap().iter().map(polygon));
                    classify_into_pieces(rings)
                })
                .collect();
            (key.parse().unwrap(), pieces)
        })
        .collect();
    let sizes = serde_json::from_value(fixture["sizes"].clone()).unwrap();
    (RegionPolygons::from_map(map), sizes, fixture)
}

fn policies() -> [ExteriorPolicy; 3] {
    [
        ExteriorPolicy::Raycast { margin: None },
        ExteriorPolicy::ForceDirected {
            margin: None,
            iterations: None,
        },
        ExteriorPolicy::Matched { margin: None },
    ]
}

#[test]
fn gallery_straight_leaders_clear_foreign_regions_and_labels() {
    let (regions, sizes, fixture) = gallery();
    let bbox = union_bbox(&regions).unwrap();
    let epsilon = 1e-6 * bbox.width().hypot(bbox.height());
    let own = Combination::new(&["Anti-CCP", "Treat"]);
    let obstacles: Vec<_> = regions
        .iter()
        .flat_map(|(key, ps)| ps.iter().map(move |p| obstacle(key, p)))
        .collect();
    let baseline = &fixture["baseline"]["raycast/poi"]["Anti-CCP&Treat"]["foreign_length"]["Treat"];
    assert!(baseline.as_f64().unwrap() > 2.0);
    assert!(
        fixture["baseline"]["elbow/poi"]["Anti-CCP&Treat"]["foreign_length"]["Anti-CCP"]
            .as_f64()
            .unwrap()
            > 10.0
    );
    for policy in policies() {
        for tether in [TetherSource::Poi, TetherSource::Boundary] {
            let strategy = PlacementStrategy::default()
                .leader(LeaderStrategy::Straight(policy))
                .tether(tether);
            let result = place_labels(&regions, &sizes, None, &strategy);
            assert_eq!(result.len(), sizes.len());
            let p = &result["Anti-CCP&Treat"];
            let segment = (p.tether.unwrap(), p.leader_end.unwrap());
            let length = foreign_length(segment, &own, &obstacles, epsilon * 0.01);
            assert!(
                length <= epsilon,
                "{policy:?}/{tether:?}: foreign length {length}, placement {p:?}"
            );
            assert!(p.leader_waypoints.is_empty());
            let poi = regions.label_points(strategy.precision)[&own];
            if tether == TetherSource::Poi {
                assert_eq!(segment.0, poi);
            } else {
                assert!(
                    signed_clearance(segment.0.x(), segment.0.y(), &regions.get(&own).unwrap()[0])
                        .abs()
                        <= epsilon
                );
            }
            for (key, placement) in &result {
                let (w, h) = sizes[key];
                let b = label_box(&placement.anchor, w, h);
                for (other, q) in &result {
                    if key == other {
                        continue;
                    }
                    let (ow, oh) = sizes[other];
                    assert!(
                        !boxes_overlap(&b, &label_box(&q.anchor, ow, oh), epsilon),
                        "{key} overlaps {other}"
                    );
                    if let (Some(start), Some(end)) = (placement.tether, placement.leader_end) {
                        assert!(
                            !leader_hits_box((start, end), &label_box(&q.anchor, ow, oh)),
                            "{key} leader crosses {other} label"
                        );
                        if let (Some(q_start), Some(q_end)) = (q.tether, q.leader_end) {
                            assert!(!segments_properly_cross(&start, &end, &q_start, &q_end));
                        }
                    }
                }
            }
            if matches!(policy, ExteriorPolicy::Matched { .. }) {
                let angle = |p: Point| (p.y() - bbox.center().y()).atan2(p.x() - bbox.center().x());
                let mut before: Vec<_> = fixture["baseline"]["matched/poi"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(key, p)| {
                        let [x, y]: [f64; 2] =
                            serde_json::from_value(p["placement"]["anchor"].clone()).unwrap();
                        (key, angle(Point::new(x, y)))
                    })
                    .collect();
                before.sort_by(|a, b| a.1.total_cmp(&b.1));
                let after: Vec<_> = before
                    .iter()
                    .map(|(key, _)| angle(result[*key].anchor))
                    .collect();
                let descents = (0..after.len())
                    .filter(|&i| after[(i + 1) % after.len()] < after[i])
                    .count();
                assert!(descents <= 1, "matched angular order changed: {after:?}");
            }
        }
    }
}

#[test]
fn enclosed_routing_uses_the_same_tolerance_across_coordinate_scales() {
    let own = Combination::new(&["A"]);
    let foreign = Combination::new(&["B"]);
    for scale in [1e-3, 1.0, 1e9] {
        let source = rectangle(-0.5 * scale, -0.5 * scale, scale, scale);
        let mut surround = rectangle(-5.0 * scale, -10.0 * scale, 10.0 * scale, 20.0 * scale);
        surround.holes.push(source.outer.clone());
        let regions = RegionPolygons::from_map(HashMap::from([
            (own.clone(), vec![source]),
            (foreign.clone(), vec![surround.clone()]),
        ]));
        let sizes = HashMap::from([("A".into(), (2.0 * scale, scale))]);
        let strategy = PlacementStrategy::default().precision(0.01 * scale);
        let result = place_labels(&regions, &sizes, None, &strategy);
        let p = &result["A"];
        let cost = foreign_length(
            (p.tether.unwrap(), p.leader_end.unwrap()),
            &own,
            &[obstacle(&foreign, &surround)],
            1e-8 * scale,
        );
        assert!((cost / scale - 4.5).abs() < 1e-6, "scale {scale}: {cost}");
    }
}

#[test]
fn enclosed_sources_reduce_traversal_without_moving_the_tether_to_the_container() {
    let source = rectangle(-0.5, -0.5, 1.0, 1.0);
    let mut surround = rectangle(-5.0, -10.0, 10.0, 20.0);
    surround.holes.push(source.outer.clone());
    let own = Combination::new(&["A"]);
    let foreign = Combination::new(&["B"]);
    let obstacles = [obstacle(&foreign, &surround)];
    let regions = RegionPolygons::from_map(HashMap::from([
        (own.clone(), vec![source.clone()]),
        (foreign.clone(), vec![surround.clone()]),
    ]));
    // The surrounding region remains an obstacle even though it has no label.
    let sizes = HashMap::from([("A".into(), (2.0, 1.0))]);
    let container = Rectangle::new(Point::new(0.0, 0.0), 20.0, 40.0);
    for policy in policies() {
        for tether in [TetherSource::Poi, TetherSource::Boundary] {
            let strategy = PlacementStrategy::default()
                .leader(LeaderStrategy::Straight(policy))
                .tether(tether)
                .leader_gap(0.4);
            let result = place_labels(&regions, &sizes, Some(&container), &strategy);
            let p = &result["A"];
            let segment = (p.tether.unwrap(), p.leader_end.unwrap());
            let length = foreign_length(segment, &own, &obstacles, 1e-8);
            assert!(
                (length - 4.5).abs() < 1e-6,
                "{policy:?}/{tether:?}: {length}"
            );
            assert!(segment.0.x().abs() <= 0.5 + 1e-8 && segment.0.y().abs() <= 0.5 + 1e-8);
            assert!(p.anchor.x().abs() - 1.0 >= 11.0 || p.anchor.y().abs() - 0.5 >= 21.0);
            assert!(
                (segment.1.x() - p.anchor.x()).abs() >= 1.4 - 1e-8
                    || (segment.1.y() - p.anchor.y()).abs() >= 0.9 - 1e-8
            );
        }
    }
}

#[test]
fn gallery_placement_is_independent_of_map_insertion_order() {
    let (regions, sizes, _) = gallery();
    for policy in policies() {
        let strategy = PlacementStrategy::default().leader(LeaderStrategy::Straight(policy));
        let expected = place_labels(&regions, &sizes, None, &strategy);
        for _ in 0..3 {
            let reversed = RegionPolygons::from_map(
                regions
                    .iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .map(|(k, ps)| (k.clone(), ps.clone()))
                    .collect(),
            );
            let reversed_sizes: HashMap<_, _> =
                sizes.iter().map(|(k, s)| (k.clone(), *s)).collect();
            assert_eq!(
                place_labels(&reversed, &reversed_sizes, None, &strategy),
                expected
            );
        }
    }
}

#[test]
fn an_already_clear_leader_keeps_its_original_anchor() {
    let regions = RegionPolygons::from_map(HashMap::from([(
        Combination::new(&["A"]),
        vec![rectangle(0., 0., 2., 2.)],
    )]));
    let sizes = HashMap::from([("A".into(), (4.0, 2.0))]);
    let result = place_labels(&regions, &sizes, None, &PlacementStrategy::default());
    assert_eq!(result["A"].kind, PlacementKind::ExteriorRaycast);
    assert_eq!(result["A"].anchor, Point::new(1.0, 5.0));
}
