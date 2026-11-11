import { cp, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { canonicalPath, legacyRoutes, pageTitles } from "../site.config.mjs";
import { VERSION, VERSION_MINOR } from "../version.config.js";
import { escapeHtml, files, json, webRoot } from "./docs-common.mjs";

export function plainText(value) {
  return value
    .replace(/<[^>]*>/g, "")
    .replace(/&#(?:x([\da-f]+)|(\d+));/gi, (_, hex, decimal) =>
      String.fromCodePoint(parseInt(hex ?? decimal, hex ? 16 : 10)),
    )
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&amp;", "&")
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

export function rewriteUrl(value, current) {
  const decoded = value.replaceAll("&amp;", "&");
  const url = new URL(decoded, `https://eunoia.bz${current}`);
  if (url.origin !== "https://eunoia.bz") return value;
  const previous =
    legacyRoutes[url.pathname] ?? legacyRoutes[`${url.pathname}/`];
  if (!previous) return value;
  return `${canonicalPath(previous.artifact)}${url.search}${url.hash}`;
}

export function assetUrl(value, current) {
  const url = new URL(
    value.replaceAll("&amp;", "&"),
    `https://eunoia.bz${current}`,
  );
  return url.origin === "https://eunoia.bz"
    ? `${url.pathname}${url.search}${url.hash}`
    : value;
}

export function preserveAnchors(html, markdown, entry) {
  const known = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  for (const [id, label] of Object.entries({
    ...entry.headings,
    ...entry.aliases,
  })) {
    if (known.has(id)) continue;
    const marker = `<a id="${escapeHtml(id)}"></a>`;
    let found = false;
    html = html.replace(/<main\b[^>]*>[\s\S]*?<\/main>/, (main) =>
      main.replace(
        /<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/g,
        (heading, _level, content) => {
          if (found || plainText(content) !== plainText(label)) return heading;
          found = true;
          return marker + heading;
        },
      ),
    );
    if (!found)
      throw new Error(
        `Cannot preserve anchor ${id}: heading ${label} is missing.`,
      );
    const lines = markdown.split("\n");
    const position = lines.findIndex(
      (line) =>
        /^#{1,6} /.test(line) &&
        plainText(
          line
            .replace(/^#+ /, "")
            .replace(/\[([^\]]+)\]\((?:<[^>]+>|[^)]*)\)/g, "$1")
            .replace(/\\+([^\w\s]|_)/g, "$1")
            .replace(/[`*]/g, ""),
        ) === plainText(label),
    );
    if (position < 0)
      throw new Error(
        `Cannot preserve Markdown anchor ${id}: ${label} is missing.`,
      );
    lines.splice(position, 0, marker, "");
    markdown = lines.join("\n");
    known.add(id);
  }
  return { html, markdown };
}

export async function finishSite(root) {
  for (const entry of await readdir(join(webRoot, "static"), {
    withFileTypes: true,
  })) {
    if (entry.name === "_diplodocus" || entry.name === "robots.txt") continue;
    await cp(join(webRoot, "static", entry.name), join(root, entry.name), {
      recursive: true,
    });
  }
  const descriptions = await json(join(webRoot, "page-descriptions.json"));
  const pins = await json(join(webRoot, "docs-sources.lock.json"));
  const sitemap = await readFile(join(root, "sitemap.xml"), "utf8");
  const aliases = new Map(
    Object.entries(legacyRoutes).map(([route, entry]) => [
      entry.artifact,
      { ...entry, route },
    ]),
  );
  for (const path of (await files(root)).filter((path) =>
    path.endsWith(".html"),
  )) {
    const artifact = relative(root, path).replaceAll("\\", "/");
    const markdownPath = path.replace(/\.html$/, ".md");
    let html = await readFile(path, "utf8");
    let markdown = await readFile(markdownPath, "utf8");
    // Diplodocus adds an icon for every source repository, including bindings.
    html = html.replace(
      /<a class="repository-link" href="([^"]*)"[^>]*>[\s\S]*?<\/a>/g,
      (link, url) => (url === "https://github.com/jolars/eunoia" ? link : ""),
    );
    const previous = aliases.get(artifact);
    if (previous)
      ({ html, markdown } = preserveAnchors(html, markdown, previous));
    const current = canonicalPath(artifact);
    if (
      current.startsWith("/packages/") &&
      current.includes("/reference/") &&
      !sitemap.includes(`<loc>https://eunoia.bz${current}</loc>`)
    ) {
      html = html.replace(
        "</head>",
        '<meta name="robots" content="noindex"></head>',
      );
    }
    html = html.replace(
      /(<a\b[^>]*?\bhref=")([^"]*)(")/g,
      (_, start, value, end) =>
        start +
        escapeHtml(rewriteUrl(value, current).replaceAll("&amp;", "&")) +
        end,
    );
    // The navigation shell remains mounted between pages. Root-relative
    // candidates keep responsive images independent of the current page URL.
    html = html.replace(
      /\b(src|poster)="([^"]*)"/g,
      (_, attribute, value) =>
        `${attribute}="${escapeHtml(assetUrl(value, current))}"`,
    );
    html = html.replace(
      /\bsrcset="([^"]*)"/g,
      (_, value) =>
        `srcset="${value
          .split(",")
          .map((candidate) => {
            const [url, descriptor] = candidate.trim().split(/\s+/);
            return `${escapeHtml(assetUrl(url, current))}${descriptor ? ` ${descriptor}` : ""}`;
          })
          .join(", ")}"`,
    );
    markdown = markdown.replace(
      /\]\((<[^>]+>|[^\s)]+)\)/g,
      (original, value) => {
        const raw = value.startsWith("<") ? value.slice(1, -1) : value;
        const target = rewriteUrl(raw, current);
        return target === raw
          ? original
          : `](${target.includes(" ") ? `<${target}>` : target})`;
      },
    );
    const title = pageTitles[current];
    if (title) {
      html = html.replace(
        /<title>[^<]*<\/title>/,
        `<title>${escapeHtml(title)}</title>`,
      );
      html = html.replace(
        /(<meta (?:property="og:title"|name="twitter:title") content=")[^"]*(")/g,
        (_, start, end) => start + escapeHtml(title) + end,
      );
    }
    const language = current.match(/^\/packages\/(python|julia)\//)?.[1];
    const pageName = plainText(
      html.match(/<title>([^<]*)<\/title>/)[1],
    ).replace(/ · Eunoia$/, "");
    const description =
      descriptions[current] ??
      descriptions[previous?.route] ??
      (language
        ? `${pageName}: eunoia ${pins[language].version} ${language === "python" ? "Python" : "Julia"} documentation.`
        : undefined);
    if (description)
      html = html.replace(
        /(<meta (?:name="description"|property="og:description"|name="twitter:description") content=")[^"]*(")/g,
        (_, start, end) => start + escapeHtml(description) + end,
      );
    html = html.replace(
      "</head>",
      '<link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="icon" href="/favicon.ico"><link rel="apple-touch-icon" href="/apple-touch-icon.png"><link rel="manifest" href="/site.webmanifest"></head>',
    );
    const tokens = (text) =>
      text
        .replaceAll("%EUNOIA_VERSION_MINOR%", VERSION_MINOR)
        .replaceAll("%EUNOIA_VERSION%", VERSION);
    await writeFile(path, tokens(html));
    await writeFile(markdownPath, tokens(markdown));
  }
  const legacy = await json(join(webRoot, "julia-docs-legacy.json"));
  const redirectManifest = join(root, "_diplodocus/julia/redirects.json");
  await mkdir(dirname(redirectManifest), { recursive: true });
  await writeFile(
    redirectManifest,
    `${JSON.stringify({
      format_version: 1,
      generator: "eunoia-julia-docs",
      base_url: "https://eunoia.bz/docs/reference/julia/",
      package_version: pins.julia.version,
      source_revision: pins.julia.revision,
      pages: {
        "": "",
        api: "api",
        gallery: "gallery",
        CHANGELOG: "changelog",
        changelog: "changelog",
      },
      api: legacy.api,
    })}\n`,
  );
}
