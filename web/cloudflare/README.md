# Cloudflare Workers hosting for eunoia.bz

The SvelteKit site is prerendered into `web/dist/`. Wrangler uploads that
directory as Worker assets, including the Markdown files in
`dist/_agent_markdown/`. `markdown-worker.js` serves those files when a page
request includes `Accept: text/markdown`; other requests use the asset binding.
Page responses include `Vary: Accept`.

The release workflow builds the site and deploys the Worker to the existing
`eunoia.bz/*` route. It needs `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` as GitHub Actions secrets. The token needs the
**Edit Cloudflare Workers** permission for the account and zone.

To deploy a locally built site, set those two environment variables and run:

```sh
cd web
pnpm run build
pnpm exec wrangler deploy --config cloudflare/wrangler.jsonc
```

The Worker needs only its `ASSETS` binding and no runtime secrets. Keep the
Worker and the site build in the same deployment so Markdown and HTML remain
in sync. The existing proxied DNS record is required for the Worker route; the
Worker serves the site without fetching the former GitHub Pages origin.

Verify with:

```sh
curl -i -H 'Accept: text/markdown' https://eunoia.bz/docs/
curl -i https://eunoia.bz/docs/
```

The first response should be Markdown with `Content-Type: text/markdown`; the
second should be HTML. Both should include `Vary: Accept`.
