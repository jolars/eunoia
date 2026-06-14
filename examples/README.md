# Examples

Browser usage of Eunoia with **no bundler**, via the self-contained
`@jolars/eunoia/web` entry (a single ESM file with the WebAssembly module
inlined; call `await init()` once before fitting).

| File                  | Run it with                                              |
| --------------------- | -------------------------------------------------------- |
| `quarto/euler.qmd`    | `quarto preview quarto/euler.qmd` (Observable `ojs` cells) |
| `html/index.html`     | Any static server, e.g. `python -m http.server` then open it |

Both import from a CDN:

```js
import { euler, init } from "https://cdn.jsdelivr.net/npm/@jolars/eunoia/web.js";
```

## Before the `/web` entry is published

The `@jolars/eunoia/web` subpath ships from the first npm release that includes
it. Until then the CDN URL 404s. To try the examples against a local build:

```sh
task build-web                      # writes npm/web.js
python -m http.server -d npm 8000   # serve it
```

then point the import at `http://localhost:8000/web.js`.

Once published, pin a version for a stable CDN cache, e.g.
`https://cdn.jsdelivr.net/npm/@jolars/eunoia@1.2.0/web.js`.
