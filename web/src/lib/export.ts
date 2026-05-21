import type { ExportFormat, PersistedState } from "./types/diagram";

function todayStamp(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function defaultFilename(format: ExportFormat): string {
  return `eunoia-${todayStamp()}.${format}`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Clone the live SVG for export: drop the hidden label-measurement group
 * (renderer scratch space, not part of the picture) and, when `fontFaceCss`
 * is supplied, inline it as a `<style>` so the file carries its own font.
 */
function cloneForExport(
  svg: SVGSVGElement,
  fontFaceCss?: string,
): SVGSVGElement {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) {
    clone.setAttribute("xmlns", SVG_NS);
  }
  if (!clone.getAttribute("xmlns:xlink")) {
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  }
  for (const m of Array.from(clone.querySelectorAll("[data-fit-measure]"))) {
    m.remove();
  }
  if (fontFaceCss) {
    const style = document.createElementNS(SVG_NS, "style");
    style.textContent = fontFaceCss;
    clone.insertBefore(style, clone.firstChild);
  }
  return clone;
}

function serializeSvg(svg: SVGSVGElement, fontFaceCss?: string): string {
  return new XMLSerializer().serializeToString(
    cloneForExport(svg, fontFaceCss),
  );
}

export function exportSvg(
  svg: SVGSVGElement,
  fontFaceCss = "",
  filename = defaultFilename("svg"),
) {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n${serializeSvg(svg, fontFaceCss)}`;
  downloadBlob(
    new Blob([xml], { type: "image/svg+xml;charset=utf-8" }),
    filename,
  );
}

export async function exportPng(
  svg: SVGSVGElement,
  width: number,
  height: number,
  fontFaceCss = "",
  filename = defaultFilename("png"),
) {
  const xml = serializeSvg(svg, fontFaceCss);
  const svgUrl = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to rasterize SVG"));
    img.src = svgUrl;
  });
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  await new Promise<void>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to encode PNG"));
        return;
      }
      downloadBlob(blob, filename);
      resolve();
    }, "image/png");
  });
}

export async function exportPdf(
  svg: SVGSVGElement,
  widthInches: number,
  heightInches: number,
  pdfFont = "helvetica",
  filename = defaultFilename("pdf"),
) {
  const [{ jsPDF }, svg2pdfModule] = await Promise.all([
    import("jspdf"),
    import("svg2pdf.js"),
  ]);
  const svg2pdf = (
    svg2pdfModule as {
      svg2pdf: (
        svg: SVGSVGElement,
        doc: unknown,
        opts?: unknown,
      ) => Promise<unknown>;
    }
  ).svg2pdf;
  const orientation = widthInches >= heightInches ? "landscape" : "portrait";
  const doc = new jsPDF({
    orientation,
    unit: "in",
    format: [widthInches, heightInches],
  });

  // jsPDF can't embed woff2, so render with the family rewritten to the
  // metric-compatible base-14 font (Helvetica/Times). The bundled Arimo/Tinos
  // share metrics with those, so the layout is unchanged. svg2pdf reads the
  // computed font-family per text node, so we rewrite it on an off-screen
  // clone (inline style + attribute) rather than mutating the live SVG.
  const clone = cloneForExport(svg);
  clone.setAttribute("font-family", pdfFont);
  clone.style.fontFamily = pdfFont;
  clone.style.position = "fixed";
  clone.style.left = "-100000px";
  clone.style.top = "0";
  document.body.appendChild(clone);
  try {
    await svg2pdf(clone, doc, {
      x: 0,
      y: 0,
      width: widthInches,
      height: heightInches,
    });
  } finally {
    clone.remove();
  }
  doc.save(filename);
}

export function exportJson(
  state: PersistedState,
  filename = defaultFilename("json"),
) {
  const json = JSON.stringify(state, null, 2);
  downloadBlob(new Blob([json], { type: "application/json" }), filename);
}
