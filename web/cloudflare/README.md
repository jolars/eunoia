# Markdown responses on eunoia.bz

The site is prerendered for GitHub Pages. The web build writes Markdown source
files to `dist/_agent_markdown/`, and `markdown-worker.js` serves them when a
page request includes `Accept: text/markdown`. The Worker forwards other
requests to GitHub Pages and adds `Vary: Accept` to page responses.

Deploy the Worker to the `eunoia.bz/*` route after the matching GitHub Pages
build is live:

```sh
cd web/cloudflare
pnpm dlx wrangler deploy
```

The Worker needs no bindings or secrets. Keep the source file and the deployed
Worker in sync when changing negotiation behavior. Cloudflare's built-in
Markdown for Agents setting is unavailable on the zone's Free plan.

Verify with:

```sh
curl -i -H 'Accept: text/markdown' https://eunoia.bz/docs/
curl -i https://eunoia.bz/docs/
```

The first response should be Markdown with `Content-Type: text/markdown`; the
second should remain HTML. Both should include `Vary: Accept`.
