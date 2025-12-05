<script lang="ts">
  import { onMount } from "svelte";

  interface Circle {
    x: number;
    y: number;
    radius: number;
    label: string;
  }

  interface Ellipse {
    x: number;
    y: number;
    semi_major: number;
    semi_minor: number;
    rotation: number;
    label: string;
  }

  interface Point {
    x: number;
    y: number;
  }

  interface Polygon {
    vertices: Point[];
    label: string;
    labelPosition?: Point; // Pre-computed optimal label position
  }

  interface RegionPolygon {
    combination: string;
    polygons: Polygon[];
    totalArea: number;
  }

  interface DiagramRow {
    input: string;
    size: number;
  }

  let circles = $state<Circle[]>([]);
  let ellipses = $state<Ellipse[]>([]);
  let polygons = $state<Polygon[]>([]);
  let regionPolygons = $state<RegionPolygon[]>([]);
  let wasmModule = $state<any>(null);
  let loading = $state(true);
  let error = $state("");
  let loss = $state<number>(0);
  let targetAreas = $state<Record<string, number>>({});
  let fittedAreas = $state<Record<string, number>>({});
  let showShapeParams = $state(false);
  let showRegions = $state(false);

  // Diagram specification
  let diagramRows = $state<DiagramRow[]>([
    { input: "A", size: 3 },
    { input: "B", size: 5 },
    { input: "A&B", size: 1 },
  ]);

  let inputType = $state<"exclusive" | "inclusive">("exclusive");
  let shapeType = $state<"circle" | "ellipse">("circle");
  // Fixed polygon vertex count - always render as polygons
  const POLYGON_VERTICES = 256;
  let optimizer = $state<
    "NelderMead" | "Lbfgs" | "ConjugateGradient" | "TrustRegion"
  >("Lbfgs");
  let seed = $state<number | undefined>(undefined);
  let useSeed = $state(false);

  const colors = [
    "rgba(59, 130, 246, 0.3)", // blue
    "rgba(239, 68, 68, 0.3)", // red
    "rgba(34, 197, 94, 0.3)", // green
    "rgba(234, 179, 8, 0.3)", // yellow
    "rgba(168, 85, 247, 0.3)", // purple
  ];

  onMount(async () => {
    try {
      // Import and initialize the WASM module
      const wasm = await import("../../pkg/eunoia_wasm.js");
      await wasm.default(); // Initialize WASM
      wasmModule = wasm;

      loading = false;
      // The reactive statement will generate the diagram from spec
    } catch (e) {
      error = `Failed to load WASM: ${e}`;
      loading = false;
      console.error(e);
    }
  });

  function addRow() {
    diagramRows = [...diagramRows, { input: "", size: 0 }];
  }

  function removeRow(index: number) {
    diagramRows = diagramRows.filter((_, i) => i !== index);
  }

  function polygonToPath(polygon: Polygon): string {
    if (polygon.vertices.length === 0) return "";

    const first = polygon.vertices[0];
    let path = `M ${first.x},${first.y}`;

    for (let i = 1; i < polygon.vertices.length; i++) {
      const v = polygon.vertices[i];
      path += ` L ${v.x},${v.y}`;
    }

    path += " Z"; // Close path
    return path;
  }

  function calculateLabelPosition(
    polygon: Polygon,
    usePole: boolean = true,
  ): Point {
    // Use pre-computed label position if available
    if (polygon.labelPosition) {
      return polygon.labelPosition;
    }

    // Fallback to simple centroid (average of vertices)
    const n = polygon.vertices.length;
    let cx = 0,
      cy = 0;
    for (const v of polygon.vertices) {
      cx += v.x;
      cy += v.y;
    }
    return { x: cx / n, y: cy / n };
  }

  function calculateCentroid(polygon: Polygon): Point {
    // Keep for backward compatibility, but prefer calculateLabelPosition
    const n = polygon.vertices.length;
    let cx = 0,
      cy = 0;

    for (const v of polygon.vertices) {
      cx += v.x;
      cy += v.y;
    }

    return { x: cx / n, y: cy / n };
  }

  function normalizeRegionPolygons(regions: RegionPolygon[]): RegionPolygon[] {
    // Find bounding box of all region polygons
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    for (const region of regions) {
      for (const polygon of region.polygons) {
        for (const vertex of polygon.vertices) {
          minX = Math.min(minX, vertex.x);
          minY = Math.min(minY, vertex.y);
          maxX = Math.max(maxX, vertex.x);
          maxY = Math.max(maxY, vertex.y);
        }
      }
    }

    const width = maxX - minX;
    const height = maxY - minY;
    const maxDim = Math.max(width, height);

    // Target size: scale so largest dimension is 100 units
    const targetSize = 100;
    const scale = targetSize / maxDim;

    // Compute label positions BEFORE normalization (using WASM pole_of_inaccessibility)
    // Then normalize both vertices and label positions
    return regions.map((region) => ({
      combination: region.combination,
      totalArea: region.totalArea,
      polygons: region.polygons.map((polygon: any) => {
        // Calculate precision relative to polygon size
        // Use ~0.1% of the maximum dimension for good accuracy
        const precision = maxDim * 0.001;

        // Compute pole of inaccessibility on original coordinates
        const pole = polygon.pole_of_inaccessibility(precision);

        // Normalize vertices and label position
        return {
          label: polygon.label,
          vertices: polygon.vertices.map((v: Point) => ({
            x: (v.x - minX) * scale,
            y: (v.y - minY) * scale,
          })),
          labelPosition: {
            x: (pole.x - minX) * scale,
            y: (pole.y - minY) * scale,
          },
        };
      }),
    }));
  }

  function normalizeCoordinates(shapes: {
    polygons: any[];
    circles: any[];
    ellipses: any[];
  }) {
    const { polygons, circles, ellipses } = shapes;

    // Find bounding box of all shapes
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    if (polygons.length > 0) {
      for (const polygon of polygons) {
        for (const vertex of polygon.vertices) {
          minX = Math.min(minX, vertex.x);
          minY = Math.min(minY, vertex.y);
          maxX = Math.max(maxX, vertex.x);
          maxY = Math.max(maxY, vertex.y);
        }
      }
    } else if (circles.length > 0) {
      for (const circle of circles) {
        minX = Math.min(minX, circle.x - circle.radius);
        minY = Math.min(minY, circle.y - circle.radius);
        maxX = Math.max(maxX, circle.x + circle.radius);
        maxY = Math.max(maxY, circle.y + circle.radius);
      }
    } else if (ellipses.length > 0) {
      for (const ellipse of ellipses) {
        const a = ellipse.semi_major;
        const b = ellipse.semi_minor;
        minX = Math.min(minX, ellipse.x - a);
        minY = Math.min(minY, ellipse.y - b);
        maxX = Math.max(maxX, ellipse.x + a);
        maxY = Math.max(maxY, ellipse.y + b);
      }
    }

    const width = maxX - minX;
    const height = maxY - minY;
    const maxDim = Math.max(width, height);

    // Target size: scale so largest dimension is 100 units
    const targetSize = 100;
    const scale = targetSize / maxDim;

    // Return normalized shapes (don't modify in place)
    const result = {
      polygons: polygons.map((polygon) => ({
        label: polygon.label, // Explicitly copy label from WASM object
        vertices: polygon.vertices.map((v) => ({
          x: (v.x - minX) * scale,
          y: (v.y - minY) * scale,
        })),
      })),
      circles: circles.map((circle) => ({
        label: circle.label, // Explicitly copy label from WASM object
        x: (circle.x - minX) * scale,
        y: (circle.y - minY) * scale,
        radius: circle.radius * scale,
      })),
      ellipses: ellipses.map((ellipse) => ({
        label: ellipse.label, // Explicitly copy label from WASM object
        x: (ellipse.x - minX) * scale,
        y: (ellipse.y - minY) * scale,
        semi_major: ellipse.semi_major * scale,
        semi_minor: ellipse.semi_minor * scale,
        rotation: ellipse.rotation,
      })),
    };

    console.log("Normalized output:", {
      firstPolygon: result.polygons[0],
      firstCircle: result.circles[0],
      firstEllipse: result.ellipses[0],
    });

    return result;
  }

  function generateFromSpec() {
    if (!wasmModule || diagramRows.length === 0) return;

    try {
      // Convert diagramRows to DiagramSpec objects, filtering out invalid/incomplete entries
      const specs = diagramRows
        .filter((row) => {
          const input = row.input.trim();
          const size = row.size;
          // Skip empty inputs, zero/negative sizes, or incomplete combinations (e.g., "A&")
          if (input === "" || size <= 0) return false;
          // Skip incomplete combinations that end with & or |
          if (input.endsWith("&") || input.endsWith("|")) return false;
          return true;
        })
        .map((row) => new wasmModule.DiagramSpec(row.input, row.size));

      if (specs.length === 0) {
        circles = [];
        ellipses = [];
        polygons = [];
        error = "";
        loss = 0;
        targetAreas = {};
        fittedAreas = {};
        return;
      }

      // Determine seed value - convert to BigInt for WASM
      const seedValue =
        useSeed && seed !== undefined ? BigInt(seed) : undefined;

      // Map optimizer string to WasmOptimizer enum value
      const optimizerValue =
        optimizer === "NelderMead"
          ? wasmModule.WasmOptimizer.NelderMead
          : optimizer === "Lbfgs"
            ? wasmModule.WasmOptimizer.Lbfgs
            : optimizer === "ConjugateGradient"
              ? wasmModule.WasmOptimizer.ConjugateGradient
              : optimizer === "TrustRegion"
                ? wasmModule.WasmOptimizer.TrustRegion
                : wasmModule.WasmOptimizer.NelderMead; // default fallback

      // Generate diagram - either as regions or shape outlines
      if (showRegions) {
        // Generate region polygons for filled visualization
        if (shapeType === "circle") {
          const result = wasmModule.generate_region_polygons_circles(
            specs,
            inputType,
            POLYGON_VERTICES,
            seedValue,
            optimizerValue,
          );

          // Extract loss
          loss = result.loss;

          // Extract target and fitted areas
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);

          // Keep WASM polygon objects to retain their methods (pole_of_inaccessibility, etc.)
          const rawRegions = Array.from(result.regions).map((region: any) => ({
            combination: region.combination,
            polygons: Array.from(region.polygons), // Keep WASM polygons
            totalArea: region.total_area,
          }));

          // Normalize the region polygons
          regionPolygons = normalizeRegionPolygons(rawRegions);

          // Clear shape outlines
          polygons = [];
          circles = [];
          ellipses = [];
        } else {
          const result = wasmModule.generate_region_polygons_ellipses(
            specs,
            inputType,
            POLYGON_VERTICES,
            seedValue,
            optimizerValue,
          );

          // Extract loss
          loss = result.loss;

          // Extract target and fitted areas
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);

          // Keep WASM polygon objects to retain their methods (pole_of_inaccessibility, etc.)
          const rawRegions = Array.from(result.regions).map((region: any) => ({
            combination: region.combination,
            polygons: Array.from(region.polygons), // Keep WASM polygons
            totalArea: region.total_area,
          }));

          // Normalize the region polygons
          regionPolygons = normalizeRegionPolygons(rawRegions);

          // Clear shape outlines
          polygons = [];
          circles = [];
          ellipses = [];
        }

        // For regions, we don't have loss/areas in the same format
        // We could compute them but for now just clear them
        loss = 0;
        targetAreas = {};
        fittedAreas = {};
      } else {
        // Generate shape outlines (existing code)
        if (shapeType === "circle") {
          const result = wasmModule.generate_circles_as_polygons(
            specs,
            inputType,
            POLYGON_VERTICES,
            seedValue,
            optimizerValue,
          );

          // Normalize coordinates before assigning to state
          const normalized = normalizeCoordinates({
            polygons: Array.from(result.polygons),
            circles: Array.from(result.circles),
            ellipses: [],
          });

          polygons = normalized.polygons;
          circles = normalized.circles;
          ellipses = normalized.ellipses;

          // Extract areas from the result
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        } else {
          const result = wasmModule.generate_ellipses_as_polygons(
            specs,
            inputType,
            POLYGON_VERTICES,
            seedValue,
            optimizerValue,
          );

          // Normalize coordinates before assigning to state
          const normalized = normalizeCoordinates({
            polygons: Array.from(result.polygons),
            circles: [],
            ellipses: Array.from(result.ellipses),
          });

          polygons = normalized.polygons;
          circles = normalized.circles;
          ellipses = normalized.ellipses;

          // Extract areas from the result
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        }

        // Clear region data
        regionPolygons = [];
      }
      error = "";

      // No need for separate debug info call - we already have the areas!
    } catch (e) {
      error = `Failed to generate diagram: ${e}`;
      circles = [];
      ellipses = [];
      polygons = [];
      loss = 0;
      targetAreas = {};
      fittedAreas = {};
      console.error(e);
    }
  }

  // Auto-generate diagram when specification changes
  $effect(() => {
    if (wasmModule && diagramRows.length > 0) {
      // Track all relevant parameters - explicitly reference optimizer to ensure tracking
      const sizeSignature = diagramRows.map((row) => row.size).join(",");
      const effectiveSeed = useSeed && seed !== undefined ? seed : "none";
      const currentOptimizer = optimizer; // Explicitly track optimizer
      const visualizationMode = showRegions ? "regions" : "shapes"; // Track visualization mode
      console.log(
        "Generating diagram:",
        sizeSignature,
        inputType,
        shapeType,
        "seed:",
        effectiveSeed,
        "optimizer:",
        currentOptimizer,
        "mode:",
        visualizationMode,
      );
      generateFromSpec();
    }
  });

  // Calculate SVG viewBox and sizes
  let viewBox = $state("0 0 400 400");
  let strokeWidth = $state(1.5);
  let fontSize = $state(8);
  let svgAspectRatio = $state(1); // width / height

  // Compute viewBox for SVG - fits all shapes with padding
  $effect(() => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    if (showRegions && regionPolygons.length > 0) {
      // Calculate bounds from region polygons
      for (const region of regionPolygons) {
        for (const polygon of region.polygons) {
          for (const vertex of polygon.vertices) {
            minX = Math.min(minX, vertex.x);
            minY = Math.min(minY, vertex.y);
            maxX = Math.max(maxX, vertex.x);
            maxY = Math.max(maxY, vertex.y);
          }
        }
      }
    } else if (polygons.length > 0) {
      for (const polygon of polygons) {
        for (const vertex of polygon.vertices) {
          minX = Math.min(minX, vertex.x);
          minY = Math.min(minY, vertex.y);
          maxX = Math.max(maxX, vertex.x);
          maxY = Math.max(maxY, vertex.y);
        }
      }
    } else if (circles.length > 0) {
      for (const circle of circles) {
        minX = Math.min(minX, circle.x - circle.radius);
        minY = Math.min(minY, circle.y - circle.radius);
        maxX = Math.max(maxX, circle.x + circle.radius);
        maxY = Math.max(maxY, circle.y + circle.radius);
      }
    } else if (ellipses.length > 0) {
      for (const ellipse of ellipses) {
        // Compute bounding box for rotated ellipse
        const a = ellipse.semi_major;
        const b = ellipse.semi_minor;
        const angle = ellipse.rotation;

        // Calculate the extents of the rotated ellipse
        const cos_angle = Math.cos(angle);
        const sin_angle = Math.sin(angle);

        const dx = Math.sqrt(
          a * a * cos_angle * cos_angle + b * b * sin_angle * sin_angle,
        );
        const dy = Math.sqrt(
          a * a * sin_angle * sin_angle + b * b * cos_angle * cos_angle,
        );

        minX = Math.min(minX, ellipse.x - dx);
        minY = Math.min(minY, ellipse.y - dy);
        maxX = Math.max(maxX, ellipse.x + dx);
        maxY = Math.max(maxY, ellipse.y + dy);
      }
    } else {
      viewBox = "0 0 200 200";
      return;
    }

    // Calculate padding as percentage of diagram size (10%)
    const rawWidth = maxX - minX;
    const rawHeight = maxY - minY;
    const padding = Math.max(rawWidth, rawHeight) * 0.1;

    const width = rawWidth + 2 * padding;
    const height = rawHeight + 2 * padding;

    viewBox = `${minX - padding} ${minY - padding} ${width} ${height}`;
    svgAspectRatio = width / height;

    // Use fixed sizes since coordinates are normalized to ~100 units
    strokeWidth = 1; // Fixed stroke width
    fontSize = 6; // Fixed font size (8 units in a 100-unit space)

    console.log("Sizes:", { strokeWidth, fontSize });
  });

  function renderRegions(
    layout: WasmLayout,
    spec: WasmDiagramSpec,
    svg: d3.Selection<SVGGElement, unknown, null, undefined>,
  ) {
    // Get region polygons
    const regions = layout.regionPolygons(spec, 64);
    const regionData = regions.getRegions();

    // Define color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Render each region
    Object.entries(regionData).forEach(([combination, polygons], i) => {
      polygons.forEach((polygon: Polygon) => {
        // Create SVG path from polygon vertices
        const pathData = createPathFromPolygon(polygon);

        svg
          .append("path")
          .attr("d", pathData)
          .attr("fill", colorScale(i))
          .attr("fill-opacity", 0.5)
          .attr("stroke", colorScale(i))
          .attr("stroke-width", 1.5);
      });

      // Add label at pole of inaccessibility (better than centroid)
      const firstPolygon = polygons[0];
      const labelPos = calculateLabelPosition(firstPolygon);

      svg
        .append("text")
        .attr("x", labelPos.x)
        .attr("y", labelPos.y)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .text(combination);
    });
  }
</script>

<div class="bg-gray-50 p-8">
  <div class="max-w-6xl mx-auto">
    <header class="mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">Eunoia</h1>
      <p class="text-gray-600">Area-proportional Euler and Venn diagrams</p>
    </header>

    {#if loading}
      <div class="bg-white rounded-lg shadow p-8 text-center">
        <div
          class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"
        ></div>
        <p class="mt-4 text-gray-600">Loading WASM module...</p>
      </div>
    {:else}
      {#if error}
        <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p class="text-red-800">{error}</p>
        </div>
      {/if}

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Controls -->
        <div class="lg:col-span-1 space-y-6">
          <!-- Diagram Specification -->
          <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-xl font-semibold mb-4">Diagram Specification</h2>

            <!-- Input Type Selection -->
            <div class="mb-4">
              <div class="block text-sm font-medium text-gray-700 mb-2">
                Input Type
              </div>
              <div class="flex gap-4">
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={inputType}
                    value="exclusive"
                    class="mr-2"
                  />
                  <span class="text-sm">Exclusive</span>
                </label>
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={inputType}
                    value="inclusive"
                    class="mr-2"
                  />
                  <span class="text-sm">Inclusive</span>
                </label>
              </div>
            </div>

            <!-- Shape Selection -->
            <div class="mb-4">
              <div class="block text-sm font-medium text-gray-700 mb-2">
                Shapes
              </div>
              <div class="flex gap-4">
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={shapeType}
                    value="circle"
                    class="mr-2"
                  />
                  <span class="text-sm">Circles</span>
                </label>
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={shapeType}
                    value="ellipse"
                    class="mr-2"
                  />
                  <span class="text-sm">Ellipses</span>
                </label>
              </div>
            </div>

            <!-- Visualization Mode -->
            <div class="mb-4">
              <label class="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={showRegions}
                  class="mr-2"
                />
                <span class="text-sm font-medium text-gray-700"
                  >Show filled regions</span
                >
              </label>
              <p class="mt-1 text-xs text-gray-500">
                Display exclusive regions with colors instead of shape outlines
              </p>
            </div>

            <!-- Optimizer Selection -->
            <div class="mb-4">
              <label
                for="optimizer"
                class="block text-sm font-medium text-gray-700 mb-2"
              >
                Optimizer
              </label>
              <select
                bind:value={optimizer}
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Lbfgs">L-BFGS</option>
                <option value="NelderMead">Nelder-Mead</option>
                <option value="ConjugateGradient">Conjugate Gradient</option>
                <option value="TrustRegion">Trust Region (Cauchy Point)</option>
              </select>
              <p class="mt-1 text-xs text-gray-500">
                Optimization method for fitting shapes
              </p>
            </div>

            <!-- Random Seed Option -->
            <div class="mb-4">
              <label class="flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={useSeed} class="mr-2" />
                <span class="text-sm font-medium text-gray-700"
                  >Use random seed</span
                >
              </label>
              <p class="mt-1 text-xs text-gray-500">
                Set seed for reproducible layouts
              </p>
              {#if useSeed}
                <div class="mt-2 ml-6">
                  <label class="flex items-center gap-2">
                    <span class="text-xs text-gray-600">Seed:</span>
                    <input
                      type="number"
                      bind:value={seed}
                      min="0"
                      step="1"
                      placeholder="e.g., 42"
                      class="w-32 px-2 py-1 text-sm border border-gray-300 rounded"
                    />
                  </label>
                </div>
              {/if}
            </div>

            <div class="space-y-3">
              <div
                class="grid grid-cols-12 gap-2 text-sm font-medium text-gray-700"
              >
                <div class="col-span-6">Input</div>
                <div class="col-span-4">Size</div>
                <div class="col-span-2"></div>
              </div>

              {#each diagramRows as row, i}
                <div class="grid grid-cols-12 gap-2">
                  <input
                    type="text"
                    bind:value={row.input}
                    placeholder="e.g., A or A&B"
                    class="col-span-6 px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <input
                    type="number"
                    bind:value={row.size}
                    min="0"
                    step="0.1"
                    class="col-span-4 px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    onclick={() => removeRow(i)}
                    class="col-span-2 px-2 py-2 bg-red-100 text-red-600 rounded hover:bg-red-200"
                    title="Remove row"
                  >
                    ×
                  </button>
                </div>
              {/each}

              <button
                onclick={() => addRow()}
                class="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                + Add Row
              </button>
            </div>
          </div>

          <!-- Fit -->
          {#if circles.length > 0 || ellipses.length > 0 || polygons.length > 0 || regionPolygons.length > 0}
            <div class="bg-white rounded-lg shadow p-6 mt-6">
              <h2 class="text-xl font-semibold mb-4">Goodness of Fit</h2>

              <div class="mb-4">
                <div class="text-sm font-medium text-gray-700 mb-1">
                  Loss: <span class="font-mono">{loss.toFixed(4)}</span>
                </div>
                <div class="text-xs text-gray-500 mt-2">
                  Target keys: {Object.keys(targetAreas).length} | Fitted keys: {Object.keys(
                    fittedAreas,
                  ).length}
                </div>
              </div>

              <div class="grid grid-cols-2 gap-8 text-sm">
                <div>
                  <h3 class="font-semibold text-gray-700 mb-2">Target</h3>
                  <div class="space-y-1 font-mono text-xs">
                    {#if Object.keys(targetAreas).length === 0}
                      <div class="text-gray-400">No data</div>
                    {:else}
                      {#each Object.entries(targetAreas).sort() as [combo, area]}
                        <div class="flex justify-between">
                          <span>{combo}:</span>
                          <span>{area.toFixed(3)}</span>
                        </div>
                      {/each}
                    {/if}
                  </div>
                </div>

                <div>
                  <h3 class="font-semibold text-gray-700 mb-2">Fitted</h3>
                  <div class="space-y-1 font-mono text-xs">
                    {#if Object.keys(fittedAreas).length === 0}
                      <div class="text-gray-400">No data</div>
                    {:else}
                      {#each Object.entries(fittedAreas).sort() as [combo, area]}
                        <div class="flex justify-between">
                          <span>{combo}:</span>
                          <span
                            class:text-red-600={Math.abs(
                              area - (targetAreas[combo] || 0),
                            ) > 0.1}
                          >
                            {area.toFixed(3)}
                          </span>
                        </div>
                      {/each}
                    {/if}
                  </div>
                </div>
              </div>
            </div>
          {/if}

          <!-- Shape Parameters Debug Panel -->
          <div class="mt-6">
            <button
              onclick={() => (showShapeParams = !showShapeParams)}
              class="text-sm font-medium text-blue-600 hover:text-blue-800 mb-2"
            >
              {showShapeParams ? "▼" : "▶"} Shape Parameters (Debug)
            </button>
            {#if showShapeParams}
              <div class="bg-gray-50 rounded-lg p-4 border border-gray-200">
                {#if circles.length === 0 && ellipses.length === 0}
                  <div class="text-sm text-gray-600 italic">
                    No shapes generated yet. Add diagram specifications above.
                  </div>
                {:else}
                  <div class="space-y-2 text-sm font-mono">
                    {#if circles.length > 0}
                      {#each circles as circle}
                        <div class="border-b border-gray-300 pb-2">
                          <div class="font-semibold text-gray-700">
                            {circle.label}:
                          </div>
                          <div class="pl-4 text-xs">
                            <div>
                              center: ({circle.x.toFixed(6)}, {circle.y.toFixed(
                                6,
                              )})
                            </div>
                            <div>radius: {circle.radius.toFixed(6)}</div>
                            <div>
                              area: {(Math.PI * circle.radius ** 2).toFixed(6)}
                            </div>
                          </div>
                        </div>
                      {/each}
                    {/if}
                    {#if ellipses.length > 0}
                      {#each ellipses as ellipse}
                        <div class="border-b border-gray-300 pb-2">
                          <div class="font-semibold text-gray-700">
                            {ellipse.label}:
                          </div>
                          <div class="pl-4 text-xs">
                            <div>
                              center: ({ellipse.x.toFixed(6)}, {ellipse.y.toFixed(
                                6,
                              )})
                            </div>
                            <div>
                              semi_major: {ellipse.semi_major.toFixed(6)}
                            </div>
                            <div>
                              semi_minor: {ellipse.semi_minor.toFixed(6)}
                            </div>
                            <div>
                              rotation: {ellipse.rotation.toFixed(6)} rad ({(
                                (ellipse.rotation * 180) /
                                Math.PI
                              ).toFixed(2)}°)
                            </div>
                            <div>
                              aspect: {(
                                ellipse.semi_minor / ellipse.semi_major
                              ).toFixed(6)}
                            </div>
                            <div>
                              area: {(
                                Math.PI *
                                ellipse.semi_major *
                                ellipse.semi_minor
                              ).toFixed(6)}
                            </div>
                          </div>
                        </div>
                      {/each}
                    {/if}
                  </div>
                  <div class="mt-4 text-xs text-gray-600">
                    <div class="font-semibold mb-1">Copy for unit test:</div>
                    <textarea
                      readonly
                      class="w-full h-32 p-2 bg-white border border-gray-300 rounded font-mono text-xs"
                      value={circles.length > 0
                        ? circles
                            .map(
                              (c) =>
                                `Circle::new(Point::new(${c.x.toFixed(6)}, ${c.y.toFixed(6)}), ${c.radius.toFixed(6)}) // ${c.label}`,
                            )
                            .join("\n")
                        : ellipses
                            .map(
                              (e) =>
                                `Ellipse::new(Point::new(${e.x.toFixed(6)}, ${e.y.toFixed(6)}), ${e.semi_major.toFixed(6)}, ${e.semi_minor.toFixed(6)}, ${e.rotation.toFixed(6)}) // ${e.label}`,
                            )
                            .join("\n")}
                    ></textarea>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <!-- Visualization -->
        <div class="lg:col-span-2">
          <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-xl font-semibold mb-4">Diagram</h2>

            <svg
              {viewBox}
              class="w-full border border-gray-200 rounded"
              style="aspect-ratio: {svgAspectRatio}; max-height: 70vh;"
              preserveAspectRatio="xMidYMid meet"
            >
              {#if showRegions && regionPolygons.length > 0}
                <!-- Render filled regions -->
                {#each regionPolygons as region, idx}
                  {#each region.polygons as polygon}
                    <path
                      d={polygonToPath(polygon)}
                      fill={colors[idx % colors.length]}
                      stroke="black"
                      stroke-width={strokeWidth}
                      opacity="0.7"
                    />
                  {/each}
                {/each}

                <!-- Labels at region poles of inaccessibility -->
                {#each regionPolygons as region}
                  {#each region.polygons as polygon}
                    {@const labelPos = calculateLabelPosition(polygon)}
                    <text
                      x={labelPos.x}
                      y={labelPos.y}
                      text-anchor="middle"
                      dominant-baseline="middle"
                      fill="black"
                      font-size={fontSize}
                      font-weight="bold"
                    >
                      {region.combination}
                    </text>
                  {/each}
                {/each}
              {:else}
                <!-- Polygons -->
                {#if polygons.length > 0}
                  {#each polygons as polygon, i}
                    <polygon
                      points={polygon.vertices
                        .map((v) => `${v.x},${v.y}`)
                        .join(" ")}
                      fill={colors[i % colors.length]}
                      stroke={colors[i % colors.length].replace("0.3", "1")}
                      stroke-width={strokeWidth}
                    />
                    {@const labelPos = calculateLabelPosition(polygon)}
                    <text
                      x={labelPos.x}
                      y={labelPos.y}
                      text-anchor="middle"
                      dominant-baseline="middle"
                      font-size={fontSize}
                      class="font-semibold"
                    >
                      {polygon.label}
                    </text>
                  {/each}
                {/if}

                <!-- Circles -->
                {#if circles.length > 0}
                  {#each circles as circle, i}
                    <circle
                      cx={circle.x}
                      cy={circle.y}
                      r={circle.radius}
                      fill={colors[i % colors.length]}
                      stroke={colors[i % colors.length].replace("0.3", "1")}
                      stroke-width={strokeWidth}
                    />
                    <text
                      x={circle.x}
                      y={circle.y}
                      text-anchor="middle"
                      dominant-baseline="middle"
                      font-size={fontSize}
                      class="font-semibold"
                    >
                      {circle.label}
                    </text>
                  {/each}
                {/if}

                <!-- Ellipses -->
                {#if ellipses.length > 0}
                  {#each ellipses as ellipse, i}
                    <ellipse
                      cx={ellipse.x}
                      cy={ellipse.y}
                      rx={ellipse.semi_major}
                      ry={ellipse.semi_minor}
                      transform="rotate({(ellipse.rotation * 180) /
                        Math.PI} {ellipse.x} {ellipse.y})"
                      fill={colors[i % colors.length]}
                      stroke={colors[i % colors.length].replace("0.3", "1")}
                      stroke-width={strokeWidth}
                    />
                    <text
                      x={ellipse.x}
                      y={ellipse.y}
                      text-anchor="middle"
                      dominant-baseline="middle"
                      font-size={fontSize}
                      class="font-semibold"
                    >
                      {ellipse.label}
                    </text>
                  {/each}
                {/if}
              {/if}
            </svg>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
