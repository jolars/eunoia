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
  VennSetCount,
} from "./types/diagram";

const STORAGE_KEY = "eunoia.app.v1";

const DEFAULT_ROWS: Row[] = [
  { input: "A", size: 5 },
  { input: "B", size: 3 },
  { input: "A&B", size: 2 },
];

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
  labelTether: "boundary",
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
    if (this.diagramType === "venn") {
      const out: string[] = [];
      for (let i = 0; i < this.vennN; i++) {
        out.push(String.fromCharCode(65 + i));
      }
      return out;
    }
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
    if (p.vennN && p.vennN >= 1 && p.vennN <= 5) this.vennN = p.vennN;
    if (p.style) {
      this.style = { ...DEFAULT_STYLE, ...p.style };
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

export function saveToStorage() {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(appState.toPersisted()));
  } catch {
    // ignore quota / serialization errors
  }
}
