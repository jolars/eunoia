// Static sitemap. Add new routes here when they land.
const routes = ["/", "/app/", "/about/", "/cite/"];
const hostname = "https://eunoia.bz";

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
