<script lang="ts">
  import { onMount } from 'svelte';
  
  interface Circle {
    x: number;
    y: number;
    radius: number;
  }
  
  interface DiagramRow {
    input: string;
    size: number;
  }
  
  let circles: Circle[] = [];
  let wasmModule: any = null;
  let loading = true;
  let error = '';
  
  // Diagram specification
  let diagramRows: DiagramRow[] = [
    { input: 'A', size: 3 },
    { input: 'B', size: 5 },
    { input: 'A&B', size: 1 }
  ];
  
  const colors = [
    'rgba(59, 130, 246, 0.3)',   // blue
    'rgba(239, 68, 68, 0.3)',    // red
    'rgba(34, 197, 94, 0.3)',    // green
    'rgba(234, 179, 8, 0.3)',    // yellow
    'rgba(168, 85, 247, 0.3)',   // purple
  ];

  onMount(async () => {
    try {
      // Import and initialize the WASM module
      const wasm = await import('../../pkg/eunoia.js');
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
    diagramRows = [...diagramRows, { input: '', size: 0 }];
  }
  
  function removeRow(index: number) {
    diagramRows = diagramRows.filter((_, i) => i !== index);
  }
  
  function generateFromSpec() {
    if (!wasmModule || diagramRows.length === 0) return;
    
    try {
      // Convert diagramRows to DiagramSpec objects
      const specs = diagramRows
        .filter(row => row.input.trim() !== '' && row.size > 0)
        .map(row => new wasmModule.DiagramSpec(row.input, row.size));
      
      if (specs.length === 0) {
        circles = [];
        error = '';
        return;
      }
      
      // Generate diagram from specification (returns Result)
      const result = wasmModule.generate_from_spec(specs);
      circles = Array.from(result);
      error = '';
    } catch (e) {
      error = `Failed to generate diagram: ${e}`;
      console.error(e);
    }
  }
  
  // Auto-generate diagram when specification changes
  // Serialize diagramRows to detect deep changes
  $: diagramSpec = JSON.stringify(diagramRows);
  $: if (wasmModule && diagramRows.length > 0 && diagramSpec) {
    console.log('Generating diagram from spec:', diagramSpec);
    generateFromSpec();
  }
  
  function generateLayout(n: number) {
    if (!wasmModule) return;
    
    const newCircles = wasmModule.compute_layout(n);
    circles = Array.from(newCircles);
  }
  
  // Calculate SVG viewBox to fit all circles
  $: viewBox = (() => {
    if (circles.length === 0) return '0 0 400 400';
    
    const xs = circles.map(c => c.x);
    const ys = circles.map(c => c.y);
    const rs = circles.map(c => c.radius);
    
    const minX = Math.min(...xs.map((x, i) => x - rs[i]));
    const maxX = Math.max(...xs.map((x, i) => x + rs[i]));
    const minY = Math.min(...ys.map((y, i) => y - rs[i]));
    const maxY = Math.max(...ys.map((y, i) => y + rs[i]));
    
    const padding = 20;
    const width = maxX - minX + padding * 2;
    const height = maxY - minY + padding * 2;
    
    return `${minX - padding} ${minY - padding} ${width} ${height}`;
  })();
</script>

<div class="min-h-screen bg-gray-50 p-8">
  <div class="max-w-6xl mx-auto">
    <header class="mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">Eunoia Debug Viewer</h1>
      <p class="text-gray-600">Visualize Euler diagram layouts</p>
    </header>
    
    {#if loading}
      <div class="bg-white rounded-lg shadow p-8 text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
        <p class="mt-4 text-gray-600">Loading WASM module...</p>
      </div>
    {:else if error}
      <div class="bg-red-50 border border-red-200 rounded-lg p-4">
        <p class="text-red-800">{error}</p>
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
            
            <div class="space-y-3">
              <div class="grid grid-cols-12 gap-2 text-sm font-medium text-gray-700">
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
                onclick={addRow}
                class="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                + Add Row
              </button>
            </div>
          </div>
        </div>
        
        <!-- Visualization -->
        <div class="lg:col-span-2">
          <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-xl font-semibold mb-4">Diagram</h2>
            
            <svg
              {viewBox}
              class="w-full h-auto border border-gray-200 rounded"
              preserveAspectRatio="xMidYMid meet"
            >
              <!-- Grid lines for reference -->
              <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                  <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
              
              <!-- Circles -->
              {#each circles as circle, i}
                <circle
                  cx={circle.x}
                  cy={circle.y}
                  r={circle.radius}
                  fill={colors[i % colors.length]}
                  stroke={colors[i % colors.length].replace('0.3', '1')}
                  stroke-width="2"
                />
                <text
                  x={circle.x}
                  y={circle.y}
                  text-anchor="middle"
                  dominant-baseline="middle"
                  class="text-sm font-semibold"
                >
                  {String.fromCharCode(65 + i)}
                </text>
              {/each}
            </svg>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
