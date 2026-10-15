import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";

const [baseURL, output, toolSource = process.env.DIPLODOCUS_SOURCE] =
  process.argv.slice(2);
if (!baseURL || !output || !toolSource) {
  throw new Error(
    "Usage: node scripts/python-docs.browser.mjs URL SCREENSHOTS DIPLODOCUS_SOURCE",
  );
}
const require = createRequire(resolve(toolSource, "package.json"));
const { chromium } = require("@playwright/test");
const browser = await chromium.launch({
  executablePath: chromium.executablePath(),
  headless: true,
});
try {
  for (const [size, viewport] of [
    ["desktop", { width: 1440, height: 1000 }],
    ["mobile", { width: 390, height: 844 }],
  ]) {
    for (const theme of ["light", "dark"]) {
      const context = await browser.newContext({
        baseURL,
        viewport,
        colorScheme: theme,
      });
      await context.addInitScript(
        (choice) => localStorage.setItem("eunoia.theme", choice),
        theme,
      );
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
      });
      const directory = join(output, `${size}-${theme}`);
      await mkdir(directory, { recursive: true });
      for (const [path, name] of [
        ["", "index"],
        ["euler/", "euler"],
        ["EulerFit/#plot", "plot"],
        ["VennFit/", "venn-result"],
      ]) {
        const response = await page.goto(`/docs/reference/python/${path}`);
        assert.equal(response.status(), 200);
        await page.locator(".diplodocus-content h1").waitFor();
        if (size === "mobile") {
          const target = path.includes("#")
            ? "#plot"
            : ".diplodocus-content h1";
          await page
            .locator(target)
            .evaluate((element) => element.scrollIntoView({ block: "start" }));
        }
        assert.equal(await page.locator("h1").count(), 1);
        assert.equal(
          await page
            .locator("html")
            .evaluate((element) => element.classList.contains("dark")),
          theme === "dark",
        );
        assert.ok(
          await page.evaluate(
            () =>
              document.documentElement.scrollWidth <=
              document.documentElement.clientWidth,
          ),
          `${size}-${theme}/${name} overflows horizontally`,
        );
        await page.screenshot({ path: join(directory, `${name}.png`) });
      }
      await page
        .locator(
          '.diplodocus-content a[href="/docs/reference/python/EulerFit/#plot"]',
        )
        .click();
      await page.waitForURL("**/EulerFit/#plot");
      await page.locator("#plot").waitFor();
      assert.ok(
        await page
          .locator(
            '.source-links a[href^="https://github.com/jolars/eunoia-py/blob/"]',
          )
          .count(),
      );
      await page.goto("/docs/concepts/fitter-pipeline/");
      await page.getByRole("button", { name: /Initial layout/ }).click();
      await page.locator("main svg").first().waitFor();
      await page.screenshot({ path: join(directory, "existing-pipeline.png") });
      assert.deepEqual(errors, []);
      await context.close();
    }
  }
  const context = await browser.newContext({
    baseURL,
    javaScriptEnabled: false,
  });
  const page = await context.newPage();
  assert.equal(
    (await page.goto("/docs/reference/python/EulerFit/#plot")).status(),
    200,
  );
  assert.ok(await page.locator(".diplodocus-content pre.syntax-code").count());
  assert.ok(await page.locator("#plot").count());
  await page.screenshot({ path: join(output, "no-javascript.png") });
  await context.close();
  console.log(
    `Desktop/mobile light/dark, inherited links, existing interactive component, static API content, console errors, and horizontal overflow passed. Screenshots: ${output}`,
  );
} finally {
  await browser.close();
}
