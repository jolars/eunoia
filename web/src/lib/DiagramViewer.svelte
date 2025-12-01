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
  }

  interface DiagramRow {
    input: string;
    size: number;
  }

  let circles = $state<Circle[]>([]);
  let ellipses = $state<Ellipse[]>([]);
  let polygons = $state<Polygon[]>([]);
  let wasmModule = $state<any>(null);
  let loading = $state(true);
  let error = $state("");
  let loss = $state<number>(0);
  let targetAreas = $state<Record<string, number>>({});
  let fittedAreas = $state<Record<string, number>>({});
  let showShapeParams = $state(false);

  // Diagram specification
  let diagramRows = $state<DiagramRow[]>([
    { input: "A", size: 3 },
    { input: "B", size: 5 },
    { input: "A&B", size: 1 },
  ]);

  let inputType = $state<"exclusive" | "inclusive">("exclusive");
  let useInitialOnly = $state(false);
  let shapeType = $state<"circle" | "ellipse">("circle");
  let usePolygons = $state(true);
  let polygonVertices = $state(64);
  let optimizer = $state<
    "NelderMead" | "Lbfgs" | "ConjugateGradient" | "TrustRegion" | "Nlm"
  >("NelderMead");
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
                : optimizer === "Nlm"
                  ? wasmModule.WasmOptimizer.Nlm
                  : wasmModule.WasmOptimizer.NelderMead; // default fallback

      // Generate diagram based on shape type and polygon preference
      if (usePolygons) {
        // Generate as polygons
        if (shapeType === "circle") {
          const result = wasmModule.generate_circles_as_polygons(
            specs,
            inputType,
            polygonVertices,
            seedValue,
            optimizerValue,
          );
          polygons = Array.from(result.polygons);
          circles = [];
          ellipses = [];

          // Extract areas from the result (no need to refit!)
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        } else {
          const result = wasmModule.generate_ellipses_as_polygons(
            specs,
            inputType,
            polygonVertices,
            seedValue,
            optimizerValue,
          );
          polygons = Array.from(result.polygons);
          circles = [];
          ellipses = [];

          // Extract areas from the result (no need to refit!)
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        }
      } else {
        // Generate as analytical shapes
        polygons = [];

        console.log(
          "Using optimizer:",
          optimizer,
          "enum value:",
          optimizerValue,
        );

        if (shapeType === "circle") {
          const generateFn = useInitialOnly
            ? wasmModule.generate_from_spec_initial
            : wasmModule.generate_from_spec;
          const result = generateFn(
            specs,
            inputType,
            seedValue,
            optimizerValue,
          );
          circles = Array.from(result.circles);
          ellipses = [];

          // Extract areas from the result (no need to refit!)
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        } else {
          const result = wasmModule.generate_ellipses_from_spec(
            specs,
            inputType,
            seedValue,
            optimizerValue,
          );
          ellipses = Array.from(result.ellipses);
          circles = [];

          // Extract areas from the result (no need to refit!)
          loss = result.loss;
          targetAreas = JSON.parse(result.target_areas_json);
          fittedAreas = JSON.parse(result.fitted_areas_json);
        }
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
      console.log(
        "Generating diagram:",
        sizeSignature,
        inputType,
        shapeType,
        usePolygons,
        polygonVertices,
        "initial-only:",
        useInitialOnly,
        "seed:",
        effectiveSeed,
        "optimizer:",
        currentOptimizer,
      );
      generateFromSpec();
    }
  });

  // Calculate SVG viewBox and sizes
  let viewBox = $state("0 0 400 400");
  let strokeWidth = $state(2);
  let fontSize = $state(16);

  // Compute viewBox for SVG - fits all shapes with padding
  $effect(() => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

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
      strokeWidth = 2;
      fontSize = 16;
      return;
    }

    // Calculate padding as percentage of diagram size (10%)
    const rawWidth = maxX - minX;
    const rawHeight = maxY - minY;
    const padding = Math.max(rawWidth, rawHeight) * 0.1;

    const width = rawWidth + 2 * padding;
    const height = rawHeight + 2 * padding;

    viewBox = `${minX - padding} ${minY - padding} ${width} ${height}`;

    console.log("ViewBox Debug:", {
      minX,
      maxX,
      minY,
      maxY,
      width,
      height,
      viewBox,
      numShapes: circles.length || ellipses.length || polygons.length,
    });

    // Scale stroke and font based on the coordinate space
    const dimension = Math.max(width, height);
    strokeWidth = dimension * 0.01; // 1% of diagram
    fontSize = dimension * 0.06; // 8% of diagram

    console.log("Sizes:", { strokeWidth, fontSize, dimension });
  });
</script>

<div class="bg-gray-50 p-8">
  <div class="max-w-6xl mx-auto">
    <header class="mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">Eunoia Debug Viewer</h1>
      <p class="text-gray-600">Visualize Euler diagram layouts</p>
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

            <!-- Initial Layout Only Option -->
            <div class="mb-4">
              <label class="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={useInitialOnly}
                  disabled={usePolygons}
                  class="mr-2"
                />
                <span
                  class="text-sm font-medium"
                  class:text-gray-700={!usePolygons}
                  class:text-gray-400={usePolygons}
                  >Use initial layout only (skip optimization)</span
                >
              </label>
              <p class="mt-1 text-xs text-gray-500">
                Shows MDS-based initial positions without final optimization
                {#if usePolygons}(Disabled with polygon rendering){/if}
              </p>
            </div>

            <!-- Shape Type Selection -->
            <div class="mb-4">
              <div class="block text-sm font-medium text-gray-700 mb-2">
                Shape Type
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

            <!-- Optimizer Selection -->
            <div class="mb-4">
              <label class="block text-sm font-medium text-gray-700 mb-2">
                Optimizer
              </label>
              <select
                bind:value={optimizer}
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="NelderMead">Nelder-Mead (default, robust)</option
                >
                <option value="Lbfgs">L-BFGS (gradient-based)</option>
                <option value="ConjugateGradient">Conjugate Gradient</option>
                <option value="TrustRegion">Trust Region (Cauchy Point)</option>
                <option value="Nlm">NLM (Dennis-Schnabel)</option>
              </select>
              <p class="mt-1 text-xs text-gray-500">
                Optimization method for fitting shapes
              </p>
            </div>

            <!-- Polygon Rendering Option -->
            <div class="mb-4">
              <label class="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={usePolygons}
                  class="mr-2"
                />
                <span class="text-sm font-medium text-gray-700"
                  >Render as polygons</span
                >
              </label>
              <p class="mt-1 text-xs text-gray-500">
                Convert shapes to polygons for rendering
              </p>
              {#if usePolygons}
                <div class="mt-2 ml-6">
                  <label class="flex items-center gap-2">
                    <span class="text-xs text-gray-600">Vertices:</span>
                    <input
                      type="number"
                      bind:value={polygonVertices}
                      min="8"
                      max="256"
                      step="8"
                      class="w-20 px-2 py-1 text-sm border border-gray-300 rounded"
                    />
                  </label>
                </div>
              {/if}
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

          <!-- Debug Information -->
          {#if circles.length > 0 || ellipses.length > 0 || polygons.length > 0}
            <div class="bg-white rounded-lg shadow p-6 mt-6">
              <h2 class="text-xl font-semibold mb-4">Debug Information</h2>

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

              <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <h3 class="font-semibold text-gray-700 mb-2">
                    Target (Disjoint)
                  </h3>
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
                  <h3 class="font-semibold text-gray-700 mb-2">
                    Fitted (Disjoint)
                  </h3>
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
                {#if usePolygons}
                  <div class="text-sm text-gray-600 italic">
                    Shape parameters not available when using polygon rendering.
                    <button
                      onclick={() => (usePolygons = false)}
                      class="text-blue-600 hover:text-blue-800 underline ml-1"
                    >
                      Disable polygon rendering
                    </button>
                    to view shape parameters.
                  </div>
                {:else if circles.length === 0 && ellipses.length === 0}
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
              style="height: 600px; max-height: 80vh;"
              preserveAspectRatio="xMidYMid meet"
            >
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
                  {@const centroidX =
                    polygon.vertices.reduce((sum, v) => sum + v.x, 0) /
                    polygon.vertices.length}
                  {@const centroidY =
                    polygon.vertices.reduce((sum, v) => sum + v.y, 0) /
                    polygon.vertices.length}
                  <text
                    x={centroidX}
                    y={centroidY}
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
            </svg>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
