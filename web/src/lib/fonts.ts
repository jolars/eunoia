/**
 * Diagram label fonts.
 *
 * Two families are bundled (vendored woff2, Latin subset) so labels render
 * with the *same glyphs* in the browser and in every export path, instead of
 * resolving the generic `sans-serif` / `serif` keywords to whatever font each
 * environment happens to provide:
 *
 * - **Arimo** — metric-compatible with Arial / Helvetica.
 * - **Tinos** — metric-compatible with Times New Roman / Times.
 *
 * The metric compatibility is the trick that keeps exports consistent without
 * shipping megabytes of TTFs:
 *
 * - **App** — registered via the `FontFace` API ({@link registerDiagramFonts}).
 * - **SVG / PNG** — the used faces are inlined into the exported SVG as
 *   base64 `@font-face` rules ({@link embeddableFontFaceCss}); where a viewer
 *   ignores them (e.g. some `<img>`-rasterization paths) the CSS stack falls
 *   back to the metric-identical system font (Arial / Times), so the layout is
 *   unchanged either way.
 * - **PDF** — jsPDF can't embed woff2, so PDF export maps each family onto its
 *   metric-compatible base-14 font ({@link pdfFontFor}); the layout matches.
 *
 * Both fonts are Apache-2.0 licensed — see `assets/fonts/NOTICE.md`.
 */
import { browser } from "$app/environment";

import arimo400 from "./assets/fonts/arimo-400.woff2";
import arimo400i from "./assets/fonts/arimo-400-italic.woff2";
import arimo700 from "./assets/fonts/arimo-700.woff2";
import arimo700i from "./assets/fonts/arimo-700-italic.woff2";
import tinos400 from "./assets/fonts/tinos-400.woff2";
import tinos400i from "./assets/fonts/tinos-400-italic.woff2";
import tinos700 from "./assets/fonts/tinos-700.woff2";
import tinos700i from "./assets/fonts/tinos-700-italic.woff2";

type FontWeight = 400 | 700;
type FontStyleName = "normal" | "italic";

interface BundledFace {
  weight: FontWeight;
  style: FontStyleName;
  /** Vite asset URL for the woff2 (hashed in production, same-origin). */
  url: string;
}

export interface FontOption {
  /** Dropdown label. */
  label: string;
  /**
   * CSS `font-family` stack written onto the SVG root. The bundled family
   * comes first, then the metric-compatible system font, then a generic
   * fallback — so the layout is identical whether or not the bundled woff2
   * is available.
   */
  value: string;
  /** The bundled `@font-face` family name used at the head of `value`. */
  family: string;
  /**
   * jsPDF base-14 font used for PDF export, where the bundled woff2 can't be
   * embedded. Chosen to be metric-compatible with `family`.
   */
  pdfFont: "helvetica" | "times";
  /** Vendored faces, for `FontFace` registration and export embedding. */
  faces: BundledFace[];
}

const ARIMO: FontOption = {
  label: "Sans-serif",
  value: "Arimo, Arial, Helvetica, sans-serif",
  family: "Arimo",
  pdfFont: "helvetica",
  faces: [
    { weight: 400, style: "normal", url: arimo400 },
    { weight: 400, style: "italic", url: arimo400i },
    { weight: 700, style: "normal", url: arimo700 },
    { weight: 700, style: "italic", url: arimo700i },
  ],
};

const TINOS: FontOption = {
  label: "Serif",
  value: "Tinos, 'Times New Roman', Times, serif",
  family: "Tinos",
  pdfFont: "times",
  faces: [
    { weight: 400, style: "normal", url: tinos400 },
    { weight: 400, style: "italic", url: tinos400i },
    { weight: 700, style: "normal", url: tinos700 },
    { weight: 700, style: "italic", url: tinos700i },
  ],
};

export const FONT_FAMILIES: FontOption[] = [ARIMO, TINOS];

/** Default diagram font — bundled sans-serif. */
export const DEFAULT_FONT_FAMILY = ARIMO.value;

/** Resolve a stored `fontFamily` value to its option, defaulting to sans. */
export function fontOptionFor(value: string): FontOption {
  return FONT_FAMILIES.find((f) => f.value === value) ?? ARIMO;
}

/** Whether `value` is one of the known bundled stacks. */
export function isKnownFontFamily(value: string): boolean {
  return FONT_FAMILIES.some((f) => f.value === value);
}

/** Metric-compatible jsPDF base-14 font for a diagram font value. */
export function pdfFontFor(value: string): "helvetica" | "times" {
  return fontOptionFor(value).pdfFont;
}

let registered = false;

/**
 * Register the bundled faces with the browser's font set. Idempotent and a
 * no-op during SSR. Kicked off at module load (below) so the faces are
 * already loading by the time the first diagram measures its labels.
 */
export function registerDiagramFonts(): void {
  if (!browser || registered || typeof FontFace === "undefined") return;
  registered = true;
  for (const opt of FONT_FAMILIES) {
    for (const f of opt.faces) {
      const face = new FontFace(opt.family, `url(${f.url}) format('woff2')`, {
        weight: String(f.weight),
        style: f.style,
        display: "swap",
      });
      // Add while loading so `document.fonts.ready` waits for it. On failure
      // we simply keep the metric-compatible system fallback.
      document.fonts.add(face);
      face.load().catch(() => {});
    }
  }
}

if (browser) registerDiagramFonts();

/**
 * The faces a diagram actually uses for a given bold/italic combination:
 * the quantity/complement labels are always regular (400 normal); the set
 * names and legend use the toggled weight/style.
 */
function facesUsed(
  opt: FontOption,
  bold: boolean,
  italic: boolean,
): BundledFace[] {
  const titleWeight: FontWeight = bold ? 700 : 400;
  const titleStyle: FontStyleName = italic ? "italic" : "normal";
  const wanted: [FontWeight, FontStyleName][] = [
    [400, "normal"],
    [titleWeight, titleStyle],
  ];
  return opt.faces.filter((f) =>
    wanted.some(([w, s]) => w === f.weight && s === f.style),
  );
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/**
 * Build base64 `@font-face` rules for the faces a diagram uses, to inline into
 * an exported SVG so it carries its own font. Returns `""` during SSR, when
 * the font has no bundled faces, or if any fetch fails (the export then relies
 * on the CSS stack's metric-compatible system fallback).
 */
export async function embeddableFontFaceCss(
  value: string,
  opts: { bold: boolean; italic: boolean },
): Promise<string> {
  if (!browser) return "";
  const opt = fontOptionFor(value);
  const faces = facesUsed(opt, opts.bold, opts.italic);
  try {
    const rules = await Promise.all(
      faces.map(async (f) => {
        const res = await fetch(f.url);
        const buf = new Uint8Array(await res.arrayBuffer());
        const b64 = bytesToBase64(buf);
        return (
          `@font-face{font-family:'${opt.family}';` +
          `font-style:${f.style};font-weight:${f.weight};` +
          `src:url(data:font/woff2;base64,${b64}) format('woff2');}`
        );
      }),
    );
    return rules.join("");
  } catch {
    return "";
  }
}
