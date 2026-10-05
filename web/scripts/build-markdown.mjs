import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { VERSION, VERSION_MINOR } from "../version.config.js";

const webRoot = fileURLToPath(new URL("../", import.meta.url));
const routesRoot = join(webRoot, "src/routes/docs");
const outputRoot = join(webRoot, "dist/_agent_markdown");

export function cleanDocsSource(source) {
  const lines = source.split("\n");
  const result = [];
  let fence = false;
  let script = false;
  let component = false;
  let caption = "";

  for (const line of lines) {
    if (/^```/.test(line.trimStart())) fence = !fence;
    if (fence || /^```/.test(line.trimStart())) {
      result.push(line);
      continue;
    }
    if (script) {
      if (/<\/script>/.test(line)) script = false;
      continue;
    }
    if (component) {
      caption += ` ${line}`;
      if (line.includes("/>")) {
        const description = caption.match(/caption="([^"]+)"/)?.[1];
        if (description) result.push(`*${description}*`);
        component = false;
        caption = "";
      }
      continue;
    }
    if (/^<script(?:\s|>)/.test(line)) {
      script = !line.includes("</script>");
      continue;
    }
    if (/^<[A-Z][\w]*\b/.test(line)) {
      component = !line.includes("/>");
      caption = line;
      if (!component) {
        const description = caption.match(/caption="([^"]+)"/)?.[1];
        if (description) result.push(`*${description}*`);
        caption = "";
      }
      continue;
    }
    result.push(line);
  }

  return `${result
    .join("\n")
    .replaceAll("%EUNOIA_VERSION_MINOR%", VERSION_MINOR)
    .replaceAll("%EUNOIA_VERSION%", VERSION)
    .replace(/^\s+/, "")
    .trimEnd()}\n`;
}

async function writePage(route, markdown) {
  const target = join(outputRoot, route, "index.txt");
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, markdown);
}

async function walkDocs(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) await walkDocs(path);
    else if (entry.name === "+page.svx") {
      const route = join("docs", relative(routesRoot, dir));
      await writePage(route, cleanDocsSource(await readFile(path, "utf8")));
    }
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await walkDocs(routesRoot);
  await writePage("", await readFile(join(webRoot, "static/llms.txt"), "utf8"));
  for (const route of ["app", "cite"]) {
    await writePage(
      route,
      await readFile(join(webRoot, `markdown/${route}.md`), "utf8"),
    );
  }
}
