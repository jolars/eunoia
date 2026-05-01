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
  TabKey,
  VennSetCount,
} from "../types/diagram";

const STORAGE_KEY = "eunoia.app.v1";

const DEFAULT_ROWS: Row[] = [
  { input: "A", size: 5 },
  { input: "B", size: 3 },
  { input: "A&B", size: 2 },
];

const DEFAULT_STYLE: DiagramStyle = {
  colors: {},
  alpha: 0.5,
  showLegend: false,
  legendPosition: "right",
  fontBold: true,
  fontItalic: false,
  strokeWidth: 1,
  labelSize: 10,
  showCounts: false,
};

const DEFAULT_ADVANCED: AdvancedOptions = {
  optimizer: "CmaEsLm",
  lossType: "SumSquared",
  showRegions: true,
  seed: 1,
  useSeed: false,
  tolerance: 1e-3,
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
  tab: TabKey = $state("app");
  result: FitResult | null = $state(null);
  error = $state("");
  loading = $state(true);
  fitting = $state(false);

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
    if (p.style) this.style = { ...DEFAULT_STYLE, ...p.style };
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

export function syncTabToHash() {
  if (typeof window === "undefined") return;
  const hash = window.location.hash.replace(/^#/, "").toLowerCase();
  if (hash === "about" || hash === "cite") {
    appState.tab = hash;
  } else {
    appState.tab = "app";
  }
}
