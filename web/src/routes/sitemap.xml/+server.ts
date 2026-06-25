// Auto-discovered sitemap. Every `+page.svelte`/`+page.svx` under `src/routes`
// becomes a `<url>` entry, so new pages are picked up without touching this
// file. The site is fully static (no dynamic route params), so a plain glob
// over the route files is exhaustive.
const hostname = "https://eunoia.bz";

const pageFiles = import.meta.glob("/src/routes/**/+page.{svelte,svx}");

/** Turn a route file path into a site-absolute URL path with a trailing slash. */
function toPath(file: string): string {
  const route = file
    .replace("/src/routes", "")
    .replace(/\/\+page\.(svelte|svx)$/, "")
    // Drop SvelteKit layout groups like `(marketing)` from the URL.
    .replace(/\/\([^/]+\)/g, "");
  return route === "" ? "/" : `${route}/`;
}

const routes = [...new Set(Object.keys(pageFiles).map(toPath))].sort();

export const prerender = true;

export function GET() {
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${routes.map((path) => `  <url><loc>${hostname}${path}</loc></url>`).join("\n")}
</urlset>
`;
  return new Response(body, {
    headers: { "content-type": "application/xml" },
  });
}
