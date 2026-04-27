import type { ExportFormat, PersistedState } from "../types/diagram";

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

function serializeSvg(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) {
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  }
  if (!clone.getAttribute("xmlns:xlink")) {
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  }
  return new XMLSerializer().serializeToString(clone);
}

export function exportSvg(svg: SVGSVGElement, filename = defaultFilename("svg")) {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n${serializeSvg(svg)}`;
  downloadBlob(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }), filename);
}

export async function exportPng(
  svg: SVGSVGElement,
  width: number,
  height: number,
  filename = defaultFilename("png"),
) {
  const xml = serializeSvg(svg);
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
  filename = defaultFilename("pdf"),
) {
  const [{ jsPDF }, svg2pdfModule] = await Promise.all([
    import("jspdf"),
    import("svg2pdf.js"),
  ]);
  const svg2pdf = (svg2pdfModule as { svg2pdf: (svg: SVGSVGElement, doc: unknown, opts?: unknown) => Promise<unknown> }).svg2pdf;
  const orientation = widthInches >= heightInches ? "landscape" : "portrait";
  const doc = new jsPDF({
    orientation,
    unit: "in",
    format: [widthInches, heightInches],
  });
  await svg2pdf(svg, doc, { x: 0, y: 0, width: widthInches, height: heightInches });
  doc.save(filename);
}

export function exportJson(state: PersistedState, filename = defaultFilename("json")) {
  const json = JSON.stringify(state, null, 2);
  downloadBlob(new Blob([json], { type: "application/json" }), filename);
}
