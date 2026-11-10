# Cloudflare Workers hosting for eunoia.bz

Diplodocus generates the site in `web/dist/`, including adjacent Markdown
representations and `__diplodocus/worker.js`. The checked-in Worker entry
re-exports that generated handler. Wrangler uploads the matching site assets
and Worker in one deployment.

The Worker negotiates `Accept: text/markdown`, respects media quality and
representation validators, and adds `Vary: Accept` to page responses. HTML and
Markdown use the same canonical paths. Diplodocus generates `_redirects` for the
former Python and Julia reference URLs; the build preserves their heading and
method anchors. Unknown paths remain 404s.

The existing release workflow deploys to `eunoia.bz/*` on crate release tags or
a manual trigger. It requires `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` as GitHub Actions secrets. No DNS or deployment policy
change is needed for the generator migration.

From `web/`, validate the built site through the actual local Worker:

```sh
pnpm build
pnpm preview --port 8787
curl -i -H 'Accept: text/markdown' http://127.0.0.1:8787/docs/
curl -i http://127.0.0.1:8787/docs/
```

The Markdown response must have `Content-Type: text/markdown`. Both page
responses must include `Vary: Accept`. Also check an old package reference URL,
its destination fragment, and a nonexistent path.

To package a deployment without publishing it:

```sh
pnpm exec wrangler deploy --config cloudflare/wrangler.jsonc --dry-run
```

To deploy an approved local build with the two account environment variables:

```sh
pnpm exec wrangler deploy --config cloudflare/wrangler.jsonc
```
