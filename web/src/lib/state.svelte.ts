import { DEFAULT_FONT_FAMILY, isKnownFontFamily } from "./fonts";
import type {
  AdvancedOptions,
  DiagramStyle,
  DiagramType,
  ExportSettings,
  FitResult,
  InputType,
  PersistedState,
  Row,
  ShapeType,
  VennRegion,
  VennSetCount,
} from "./types/diagram";
import {
  emptyVennRegions,
  isVennSetCount,
  vennCombinations,
  vennSetNames,
} from "./venn";

const STORAGE_KEY = "eunoia.app.v1";

// Member names ship with the defaults so the "Member names" glyph mode has
// something to draw the moment it's selected, without the user typing a roster.
const DEFAULT_ROWS: Row[] = [
  { input: "A", size: 5, members: "Ada, Grace, Barbara, Karen, Ida" },
  { input: "B", size: 3, members: "Alan, Edsger, Tony" },
  { input: "A&B", size: 2, members: "Katherine, Hedy" },
];

// Venn defaults follow the same reasoning as the rows above: the quantity and
// member columns should do something the moment they are switched on. Every
// region of the default 3-set diagram is filled in, and each quantity equals
// its roster length so "Dots" and "Member names" agree.
const DEFAULT_VENN_REGIONS: Record<string, VennRegion> = {
  ...emptyVennRegions(),
  A: { size: 3, members: "Ada, Grace, Barbara" },
  B: { size: 3, members: "Alan, Edsger, Tony" },
  C: { size: 2, members: "Katherine, Dorothy" },
  "A&B": { size: 2, members: "Karen, Ida" },
  "A&C": { size: 1, members: "Hedy" },
  "B&C": { size: 1, members: "Donald" },
  "A&B&C": { size: 1, members: "Margaret" },
};

const DEFAULT_STYLE: DiagramStyle = {
  palette: "default",
  colors: {},
  alpha: 1,
  showLegend: false,
  legendPosition: "right",
  fontBold: true,
  fontItalic: false,
  fontFamily: DEFAULT_FONT_FAMILY,
  strokeWidth: 0.5,
  labelSize: 6,
  showCounts: false,
  labelPlacement: "raycast",
  setLabelMode: "inside",
  labelTether: "boundary",
  glyphMode: "none",
  glyphArrangement: "uniform",
  glyphGap: 0.25,
  glyphSeed: 0,
  memberLabelSize: 4,
};

const DEFAULT_ADVANCED: AdvancedOptions = {
  optimizer: "CmaEsLm",
  lossType: "SumSquared",
  showRegions: true,
  seed: 1,
  useSeed: true,
  tolerance: 1e-3,
  useComplement: false,
  complement: null,
};

const DEFAULT_EXPORT: ExportSettings = {
  format: "svg",
  raster: { width: 1200, height: 900 },
  vector: { width: 6, height: 4.5 },
};

class AppState {
  // Specification
  rows: Row[] = $state([...DEFAULT_ROWS]);
  inputType: InputType = $state("exclusive");
  shapeType: ShapeType = $state("circle");
  diagramType: DiagramType = $state("euler");
  vennN: VennSetCount = $state(3);
  // Keyed by canonical combination, and pre-seeded for *every* region of the
  // largest supported n so the editor can bind straight into it — no
  // fill-the-gaps pass when `vennN` changes, and switching n back and forth
  // keeps what was typed.
  vennRegions: Record<string, VennRegion> = $state({ ...DEFAULT_VENN_REGIONS });

  // Style
  style: DiagramStyle = $state({ ...DEFAULT_STYLE });

  // Advanced
  advanced: AdvancedOptions = $state({ ...DEFAULT_ADVANCED });

  // Export
  exportSettings: ExportSettings = $state({ ...DEFAULT_EXPORT });

  // Runtime
  result: FitResult | null = $state(null);
  error = $state("");
  loading = $state(true);
  fitting = $state(false);

  // Stable, input-driven set ordering used for palette indices and the legend.
  // Derived from the spec (rows / vennN) rather than the fit output, so colors
  // don't shuffle when the seed changes or the fit re-runs.
  setNames: string[] = $derived.by(() => {
    if (this.diagramType === "venn") return vennSetNames(this.vennN);
    const seen = new Set<string>();
    const out: string[] = [];
    for (const row of this.rows) {
      const t = row.input.trim();
      if (!t) continue;
      for (const part of t.split(/[&|]/)) {
        const p = part.trim();
        if (!p || seen.has(p)) continue;
        seen.add(p);
        out.push(p);
      }
    }
    return out;
  });

  /** Region keys of the current Venn, in editor order. */
  vennCombos: string[] = $derived(vennCombinations(this.vennN));

  /** Clear every quantity and roster of the current Venn's regions. */
  clearVennRegions() {
    for (const combo of this.vennCombos) {
      this.vennRegions[combo] = { size: null, members: "" };
    }
  }

  addRow() {
    this.rows = [...this.rows, { input: "", size: 0 }];
  }

  removeRow(index: number) {
    this.rows = this.rows.filter((_, i) => i !== index);
  }

  reset() {
    this.rows = [...DEFAULT_ROWS];
    this.inputType = "exclusive";
    this.shapeType = "circle";
    this.diagramType = "euler";
    this.vennN = 3;
    this.vennRegions = { ...DEFAULT_VENN_REGIONS };
    this.style = { ...DEFAULT_STYLE };
    this.advanced = { ...DEFAULT_ADVANCED };
    this.exportSettings = { ...DEFAULT_EXPORT };
  }

  toPersisted(): PersistedState {
    return {
      rows: $state.snapshot(this.rows),
      inputType: this.inputType,
      shapeType: this.shapeType,
      diagramType: this.diagramType,
      vennN: this.vennN,
      vennRegions: $state.snapshot(this.vennRegions),
      style: $state.snapshot(this.style),
      advanced: $state.snapshot(this.advanced),
      exportSettings: $state.snapshot(this.exportSettings),
    };
  }

  loadPersisted(p: PersistedState) {
    if (p.rows && Array.isArray(p.rows)) this.rows = p.rows;
    if (p.inputType) this.inputType = p.inputType;
    if (p.shapeType) this.shapeType = p.shapeType;
    if (p.diagramType) this.diagramType = p.diagramType;
    if (isVennSetCount(p.vennN)) this.vennN = p.vennN;
    // Over an *empty* key space, not the seeded defaults: a blob written after
    // the user cleared a region must not resurrect the example data. Older
    // blobs have no `vennRegions` at all and keep the defaults.
    if (p.vennRegions) {
      this.vennRegions = { ...emptyVennRegions(), ...p.vennRegions };
    }
    if (p.style) {
      const { showGlyphs, ...style } = p.style;
      this.style = { ...DEFAULT_STYLE, ...style };
      // Pre-`glyphMode` blobs carried a `showGlyphs` boolean. Honour it unless
      // the blob already has the newer field.
      if (typeof showGlyphs === "boolean" && p.style.glyphMode === undefined) {
        this.style.glyphMode = showGlyphs ? "dots" : "none";
      }
      // Drop any persisted font that's no longer one of the bundled stacks,
      // so the picker never lands on a blank/unknown value.
      if (!isKnownFontFamily(this.style.fontFamily)) {
        this.style.fontFamily = DEFAULT_FONT_FAMILY;
      }
    }
    if (p.advanced) this.advanced = { ...DEFAULT_ADVANCED, ...p.advanced };
    if (p.exportSettings) {
      this.exportSettings = { ...DEFAULT_EXPORT, ...p.exportSettings };
    }
  }
}

export const appState = new AppState();

export function hydrateFromStorage() {
  if (typeof localStorage === "undefined") return;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as PersistedState;
    appState.loadPersisted(parsed);
  } catch {
    // ignore corrupt storage
  }
}

/**
 * Persist a state blob. Takes the payload rather than reading `appState`
 * itself: the caller is an `$effect`, and only a read there registers the
 * dependencies that should re-trigger a save. See `routes/app/+page.svelte`.
 */
export function saveToStorage(state: PersistedState = appState.toPersisted()) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore quota / serialization errors
  }
}
