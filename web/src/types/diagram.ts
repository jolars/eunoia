export interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  vertices: Point[];
}

export interface RegionPolygons {
  [combination: string]: Polygon[];
}
