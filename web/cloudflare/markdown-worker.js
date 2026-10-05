const markdownRoot = "/_agent_markdown";

export function wantsMarkdown(accept) {
  return (
    accept?.split(",").some((range) => {
      const [type, ...parameters] = range.trim().split(";");
      if (type.trim().toLowerCase() !== "text/markdown") return false;
      const quality = parameters.find((parameter) =>
        /^\s*q\s*=/i.test(parameter),
      );
      return !quality || Number(quality.split("=")[1]) > 0;
    }) ?? false
  );
}

function pagePath(pathname) {
  if (pathname === "/") return "";
  const normalized = pathname.replace(/\/$/, "");
  if (
    normalized === "/app" ||
    normalized === "/cite" ||
    normalized === "/docs" ||
    normalized.startsWith("/docs/")
  ) {
    return normalized;
  }
  return null;
}

function varyAccept(headers) {
  const vary = headers.get("Vary");
  if (
    !vary?.split(",").some((part) => part.trim().toLowerCase() === "accept")
  ) {
    headers.set("Vary", vary ? `${vary}, Accept` : "Accept");
  }
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const route = pagePath(url.pathname);
    if (route === null || !["GET", "HEAD"].includes(request.method))
      return fetch(request);

    if (wantsMarkdown(request.headers.get("Accept"))) {
      const markdownUrl = new URL(`${markdownRoot}${route}/index.txt`, url);
      const markdown = await fetch(
        new Request(markdownUrl, { method: request.method }),
      );
      if (markdown.ok) {
        const headers = new Headers(markdown.headers);
        headers.set("Content-Type", "text/markdown; charset=utf-8");
        varyAccept(headers);
        return new Response(request.method === "HEAD" ? null : markdown.body, {
          status: markdown.status,
          headers,
        });
      }
    }

    const html = await fetch(request);
    const headers = new Headers(html.headers);
    varyAccept(headers);
    return new Response(request.method === "HEAD" ? null : html.body, {
      status: html.status,
      statusText: html.statusText,
      headers,
    });
  },
};
