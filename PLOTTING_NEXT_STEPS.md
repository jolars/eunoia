# Plotting Feature - Next Steps

**Status**: All phases complete ✅ (Core Rust + WASM bindings + Web app)  
**Date**: 2024-12-02

## Phase 1 Summary (COMPLETED)

### What Was Built

- ✅ Feature flag system (`plotting` feature in Cargo.toml)
- ✅ Polygon clipping operations (intersection, union, difference, xor)
- ✅ Region decomposition algorithm
- ✅ Integration with Layout API
- ✅ FloatPointCompatible implementation for Point
- ✅ Comprehensive tests (8 unit tests + 3 integration tests)
- ✅ Example application (`plotting_demo.rs`)

### Key Files Created

```
crates/eunoia/src/
├── plotting.rs                    # Module definition
└── plotting/
    ├── clip.rs                    # Polygon clipping wrapper
    └── regions.rs                 # Region decomposition

crates/eunoia/examples/
└── plotting_demo.rs               # Usage example

crates/eunoia/tests/
└── plotting_integration.rs        # Integration tests
```

### API Usage

```rust
use eunoia::{DiagramSpecBuilder, Fitter, InputType};
use eunoia::geometry::shapes::Circle;

let spec = DiagramSpecBuilder::new()
    .set("A", 5.0)
    .set("B", 3.0)
    .intersection(&["A", "B"], 1.0)
    .input_type(InputType::Exclusive)
    .build()?;

let layout = Fitter::<Circle>::new(&spec).fit()?;

// Get polygons for each exclusive region
let regions = layout.region_polygons(&spec, 64);

for (combination, polygons) in regions.iter() {
    println!("{}: {} polygon(s)", combination, polygons.len());
    for poly in polygons {
        println!("  Area: {:.3}", poly.area());
        println!("  Vertices: {}", poly.vertices().len());
    }
}
```

---

## Phase 2: WASM Bindings (COMPLETED ✅)

### What Was Built

- ✅ Updated `eunoia-wasm/Cargo.toml` to enable `plotting` feature
- ✅ Created WASM types for region polygons:
  - `WasmRegion`: A single region with combination name and polygons
  - `WasmRegionPolygons`: Collection of all regions
- ✅ Added `area()` method to `WasmPolygon` (shoelace formula)
- ✅ Implemented `generate_region_polygons_circles()` function
- ✅ Implemented `generate_region_polygons_ellipses()` function
- ✅ Built WASM package successfully
- ✅ TypeScript definitions generated correctly

### API Usage (JavaScript/TypeScript)

```typescript
import {
  generate_region_polygons_circles,
  DiagramSpec,
} from "./pkg/eunoia_wasm.js";

// Create diagram specs
const specs = [
  new DiagramSpec("A", 5.0),
  new DiagramSpec("B", 3.0),
  new DiagramSpec("A&B", 1.0),
];

// Generate region polygons
const regionPolygons = generate_region_polygons_circles(
  specs,
  "exclusive",
  64, // number of vertices per shape
  null, // seed (optional)
  null, // optimizer (optional)
);

// Access regions
for (const region of regionPolygons.regions) {
  console.log(`Region: ${region.combination}`);
  console.log(`Total area: ${region.total_area}`);

  for (const polygon of region.polygons) {
    console.log(`  Polygon with ${polygon.vertices.length} vertices`);
    console.log(`  Area: ${polygon.area}`);
  }
}
```

---

## Phase 3: Web Application Integration (COMPLETED ✅)

### What Was Built

- ✅ Added `RegionPolygon` interface and state variables
- ✅ Added `showRegions` toggle state
- ✅ Updated `generateFromSpec()` to call region polygon functions
- ✅ Added `polygonToPath()` helper function
- ✅ Added `calculateCentroid()` helper function
- ✅ Updated viewBox calculation to support region polygons
- ✅ Added SVG rendering for filled regions
- ✅ Added UI checkbox toggle ("Show filled regions")
- ✅ Updated reactive effect to track visualization mode

### Features

**Visualization Modes**:
1. **Shape outlines** (default): Shows circles/ellipses as polygon borders with transparency
2. **Filled regions** (toggle on): Shows exclusive regions as filled, colored polygons

**How to Use**:
1. Start the web app: `cd web && npm run dev`
2. Visit http://localhost:5174
3. Add sets and intersections in the table
4. Check "Show filled regions" to see the filled visualization
5. Toggle back to see shape outlines

### Files Modified

1. **`web/src/lib/DiagramViewer.svelte`**
   - Lines ~25-34: Added `RegionPolygon` interface and state
   - Lines ~95-131: Added helper functions
   - Lines ~229-341: Updated `generateFromSpec()` with region logic
   - Lines ~382-400: Updated reactive effect tracking
   - Lines ~408-470: Updated viewBox calculation
   - Lines ~640-649: Added UI toggle
   - Lines ~900-1108: Updated SVG rendering with conditional regions

---

## Phase 4: Label Placement (FUTURE)

### Goal

Add intelligent label placement for regions (poles of inaccessibility).

### Tasks

#### 4.1: Implement Pole of Inaccessibility

Use the polylabel-mini rust crate, which features a polylabel() function, that
finds the pole of inaccessibility for a polygon.

Add it to our Polygon struct:

```rust
impl Polygon {
    /// Find the pole of inaccessibility (visual center)
    pub fn pole_of_inaccessibility(&self, precision: f64) -> Point {
        // Implementation
    }
}
```

It might require converting our Polygon struct into the format expected by
polylabel-mini.

#### 4.3: Implement for both set labels and region labels

Set labels are the labels for each individual set (A, B, C, etc). Region labels
are mapped to an exclusive region (A&B, A only, etc), and a size (area)
associated with that region.

Each shape needs a label. And we need to handle many edge cases where for
instance that set is inside another set.

And if two shapes are completely overlapping, we need to label both of them at
the same point. For our library, we only need to provide the location of that
label and the labels for the sets that should be placed there.

We might also need to handle cases where the polygon region is very small. When
that's the case for a set label, then we can pick another region that it belongs
to to place the label. But for the region label, we might eventually need to
place the label outside the shape with a line pointing to it. But that can be a
future enhancement.

#### 4.4: Use in Web App

Replace simple centroid with pole of inaccessibility for better label placement.

---

## Testing Checklist

### Phase 2 (WASM) ✅

- ✅ WASM builds without errors
- ✅ TypeScript definitions are correct
- ✅ Can call `generate_region_polygons_*()` from JavaScript
- ✅ Returns properly formatted data

### Phase 3 (Web) ✅

- ✅ Regions render correctly in browser
- ✅ Colors are distinguishable
- ✅ Labels are readable and centered
- ✅ Toggle between modes works
- ✅ ViewBox adjusts correctly for both modes
- ✅ Works with both circles and ellipses
- ✅ Works with different optimizers
- ✅ Performance is acceptable

### To Test Manually

1. Start dev server: `cd web && npm run dev`
2. Visit http://localhost:5174
3. Try the default 3-set example
4. Toggle "Show filled regions" on/off
5. Change shape type (circles ↔ ellipses)
6. Add more sets and intersections
7. Verify colors are distinct and regions don't have gaps

---

## Useful Commands

```bash
# Build with plotting feature
cargo build --features plotting

# Run tests
cargo test --features plotting

# Run example
cargo run -p eunoia --example plotting_demo --features plotting

# Build WASM
cd crates/eunoia-wasm
wasm-pack build --target web --out-dir ../../web/pkg

# Run web app
cd web
npm run dev
```

---

## Known Issues / Considerations

1. **Polygonization Accuracy**:
   - Current implementation uses fixed vertex count
   - Higher vertex counts = more accuracy but more data
   - Consider adaptive refinement for large/small shapes

2. **Performance**:
   - i_overlay is fast, but many operations can add up
   - Consider caching region polygons if spec doesn't change
   - May need web worker for large diagrams (10+ sets)

3. **Visual Quality**:
   - Anti-aliasing in SVG
   - Smooth curves vs polygonal edges
   - Consider adding curve fitting for smoother appearance

4. **Color Scheme**:
   - Need accessible color palette
   - Consider colorblind-friendly options
   - Allow user customization

---

## Questions to Consider

1. **Should we cache the polygon decomposition?**
   - Pro: Faster re-renders
   - Con: More memory usage
   - Decision: Start without caching, add if needed

2. **What's the default vertex count?**
   - 32 = fast but jagged
   - 64 = balanced (current default)
   - 128 = smooth but slower
   - Decision: Expose as parameter, default to 64

3. **Should regions be clickable?**
   - Could show details on click/hover
   - Needs interaction layer
   - Future enhancement

---

## Reference Files

- **Rust Implementation**: `crates/eunoia/src/plotting/`
- **Example**: `crates/eunoia/examples/plotting_demo.rs`
- **Tests**: `crates/eunoia/tests/plotting_integration.rs`
- **eulerr Reference**: `eulerr/R/setup_geometry.R` (lines 63-98)

---

## Contact / Notes

Phase 1 completed: 2024-12-02  
All tests passing ✅  
Ready for WASM integration when needed.

**Pro tip**: Test the Rust example first to understand the API:

```bash
cargo run -p eunoia --example plotting_demo --features plotting
```

This will show you exactly what data the API returns, which helps when
implementing the WASM layer.
