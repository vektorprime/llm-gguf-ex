const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.env.GGUF_EXPLORER_URL || "http://127.0.0.1:8765";
const openPath = process.env.GGUF_EXPLORER_OPEN_PATH || "";
const referencePath = process.env.GGUF_EXPLORER_REFERENCE_PATH || "";
const drillPath = (process.env.GGUF_EXPLORER_DRILL_PATH || "blk.0.ffn_down")
  .split(".")
  .filter(Boolean);
const tensorLabel = process.env.GGUF_EXPLORER_TENSOR_LABEL || "weight";
const expectFinal = process.env.GGUF_EXPLORER_EXPECT_FINAL || "-4";
const expectStatic = process.env.GGUF_EXPLORER_EXPECT_STATIC || "-16";
const screenshotName = process.env.GGUF_EXPLORER_SCREENSHOT || "gguf-explorer-sample.png";
const screenshotPath = path.resolve("screenshots", screenshotName);

async function drill(page, label) {
  const row = page.locator(".node-row", { hasText: label });
  await row.waitFor({ state: "visible", timeout: 5000 });
  await row.dblclick();
}

async function headerOrder(page) {
  return await page.locator("#values-table thead th").evaluateAll((headers) =>
    headers.map((header) => header.dataset.columnId),
  );
}

async function columnWidth(page, columnId) {
  return await page.locator(`th[data-column-id="${columnId}"]`).evaluate((header) =>
    Math.round(header.getBoundingClientRect().width),
  );
}

(async () => {
  const launchOptions = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    launchOptions.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
    const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 860 } });
    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (key.startsWith("ggufExplorer.valueColumns.")
          || key.startsWith("ggufExplorer.visibleColumns.")
          || key.startsWith("ggufExplorer.columnWidths.")
          || key.startsWith("ggufExplorer.visibleColumnsVersion.")
          || key === "ggufExplorer.referencePath") {
          localStorage.removeItem(key);
        }
      }
    });
    await page.reload({ waitUntil: "domcontentloaded" });
    const shell = page.locator(".app-shell");
    const splitter = page.locator("#split-resizer");
    await splitter.waitFor({ state: "visible", timeout: 5000 });
    const splitBox = await splitter.boundingBox();
    if (!splitBox) throw new Error("Expected visible split resizer");
    const beforeWidth = await shell.evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).getPropertyValue("--sidebar-width")),
    );
    await page.mouse.move(splitBox.x + splitBox.width / 2, splitBox.y + splitBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(splitBox.x + splitBox.width / 2 + 80, splitBox.y + splitBox.height / 2, {
      steps: 4,
    });
    await page.mouse.up();
    const afterWidth = await shell.evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).getPropertyValue("--sidebar-width")),
    );
    if (Math.abs(afterWidth - beforeWidth) < 20) {
      throw new Error("Expected split resizer to change sidebar width");
    }

    if (openPath) {
      await page.fill("#path-input", openPath);
      await page.click('#open-form button[type="submit"]');
    } else {
      const firstModel = page.locator("[data-model-load]");
      await firstModel.first().waitFor({ state: "visible", timeout: 20000 });
      const firstModelPath = await firstModel.first().getAttribute("data-model-load");
      if (!firstModelPath) throw new Error("Expected a detected model path");
      await firstModel.first().click();
      process.env.GGUF_EXPLORER_OPEN_PATH = firstModelPath;
    }
    const expectedName = path.basename(openPath || process.env.GGUF_EXPLORER_OPEN_PATH);
    await page.locator(".summary-title", { hasText: expectedName }).waitFor({
      state: "visible",
      timeout: 15000,
    });
    if (referencePath) {
      await page.fill("#reference-path-input", referencePath);
      await page.click('#reference-form button[type="submit"]');
      await page.locator("#reference-summary", { hasText: path.basename(referencePath) }).waitFor({
        state: "visible",
        timeout: 15000,
      });
    }

    for (const part of drillPath) {
      await drill(page, part);
    }
    await page.locator(".node-row", { hasText: tensorLabel }).click();

    const valuesPanel = page.locator(".panel", { hasText: "Values" });
    await valuesPanel.locator("table").waitFor({ state: "visible", timeout: 10000 });
    const coordsTooltip = await valuesPanel
      .locator('th[data-column-id="coords"]')
      .getAttribute("data-tooltip");
    if (!coordsTooltip || !coordsTooltip.includes("Tensor coordinates")) {
      throw new Error("Expected tooltip text on the Coords column");
    }
    await valuesPanel.locator('th[data-column-id="coords"]').dispatchEvent("pointerenter", {
      clientX: 540,
      clientY: 450,
    });
    await page.waitForTimeout(50);
    const renderedTooltip = await page.locator(".app-tooltip").evaluate((tooltip) => ({
      text: tooltip.textContent || "",
      hidden: tooltip.classList.contains("hidden"),
      background: getComputedStyle(tooltip).backgroundColor,
    }));
    if (renderedTooltip.hidden || !renderedTooltip.text.includes("Tensor coordinates")) {
      throw new Error(`Expected rendered black tooltip, got ${JSON.stringify(renderedTooltip)}`);
    }
    await valuesPanel.locator('th[data-column-id="coords"]').dispatchEvent("pointerleave");

    const finalText = await page.locator("#values-table").textContent();
    if (!finalText.includes(expectFinal)) {
      throw new Error(`Expected dequantized value ${expectFinal} in final mode`);
    }

    await page.click('button[data-mode="static"]');
    await page.locator("#values-table").waitFor({ state: "visible", timeout: 10000 });
    const staticText = await page.locator("#values-table").textContent();
    if (!staticText.includes(expectStatic)) {
      throw new Error(`Expected raw value ${expectStatic} in static mode`);
    }

    const initialOrder = await headerOrder(page);
    const rawHeader = page.locator('th[data-column-id="raw"]');
    const blockHeader = page.locator('th[data-column-id="block"]');
    if ((await rawHeader.count()) && (await blockHeader.count())) {
      const diffState = await page.locator("#values-table").evaluate((table) => {
        const headers = [...table.querySelectorAll("thead th")].map((header) => header.dataset.columnId);
        const referenceIndex = headers.indexOf("reference");
        const diffIndex = headers.indexOf("diff");
        const diffHeader = table.querySelector('th[data-column-id="diff"]');
        const firstDiffCell = diffIndex >= 0
          ? table.querySelector(`tbody tr:first-child td:nth-child(${diffIndex + 1})`)
          : null;
        const diffStyle = firstDiffCell ? getComputedStyle(firstDiffCell) : null;
        return {
          referenceIndex,
          diffIndex,
          tooltip: diffHeader?.getAttribute("data-tooltip") || "",
          text: firstDiffCell?.textContent?.trim() || "",
          background: diffStyle?.backgroundColor || "",
        };
      });
      if (diffState.diffIndex < 0) {
        throw new Error("Expected quant table to include a Diff column");
      }
      if (diffState.referenceIndex < 0) {
        throw new Error("Expected quant table to include a Reference column");
      }
      if (!diffState.tooltip.includes("matching reference")) {
        throw new Error("Expected tooltip text on the Diff column");
      }
      if (referencePath && (!diffState.text || diffState.text === "-")) {
        throw new Error(`Expected reference diff value in first quant row, got "${diffState.text}"`);
      }
      if (referencePath && !diffState.background.includes("rgba") && !diffState.background.includes("rgb")) {
        throw new Error("Expected diff cell to have a visible heat tint");
      }
      const headerVisibility = await page.locator("#values-table").evaluate((table) =>
        [...table.querySelectorAll("thead th")].map((header) => {
          const label = header.querySelector(".column-head span:first-child");
          const grip = header.querySelector(".column-grip");
          return {
            id: header.dataset.columnId,
            headerWidth: header.getBoundingClientRect().width,
            needed: (label?.getBoundingClientRect().width || 0) + (grip?.getBoundingClientRect().width || 0) + 24,
          };
        }),
      );
      const clippedHeader = headerVisibility.find((header) => header.needed > header.headerWidth + 1);
      if (clippedHeader) {
        throw new Error(`Expected header label to fit: ${JSON.stringify(clippedHeader)}`);
      }

      const ratiosBeforeResize = await page.locator("#values-table").evaluate((table) => {
        const tableWidth = table.getBoundingClientRect().width;
        return [...table.querySelectorAll("thead th")].map((header) => ({
          id: header.dataset.columnId,
          ratio: header.getBoundingClientRect().width / tableWidth,
        }));
      });
      await page.setViewportSize({ width: 1120, height: 860 });
      await page.waitForTimeout(100);
      const ratiosAfterResize = await page.locator("#values-table").evaluate((table) => {
        const tableWidth = table.getBoundingClientRect().width;
        return [...table.querySelectorAll("thead th")].map((header) => ({
          id: header.dataset.columnId,
          ratio: header.getBoundingClientRect().width / tableWidth,
        }));
      });
      for (const before of ratiosBeforeResize) {
        const after = ratiosAfterResize.find((item) => item.id === before.id);
        if (after && Math.abs(before.ratio - after.ratio) > 0.012) {
          throw new Error(`Expected ${before.id} ratio to hold on viewport resize. Before=${before.ratio} After=${after.ratio}`);
        }
      }
      await page.setViewportSize({ width: 1366, height: 860 });
      await page.waitForTimeout(100);

      await page.click("#column-menu-button");
      await page.locator('input[data-column-toggle="diff"]').click();
      await page.locator('th[data-column-id="diff"]').waitFor({ state: "detached", timeout: 5000 });
      await page.click("#column-reset");
      await page.locator('th[data-column-id="diff"]').waitFor({ state: "visible", timeout: 5000 });

      const beforeIndexWidth = await columnWidth(page, "index");
      const indexResizer = page.locator('[data-column-resizer="index"]');
      const resizerBox = await indexResizer.boundingBox();
      if (!resizerBox) throw new Error("Expected index column resizer");
      await page.mouse.move(resizerBox.x + resizerBox.width / 2, resizerBox.y + resizerBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(resizerBox.x + resizerBox.width / 2 + 42, resizerBox.y + resizerBox.height / 2);
      await page.mouse.up();
      const afterIndexWidth = await columnWidth(page, "index");
      if (afterIndexWidth <= beforeIndexWidth + 20) {
        throw new Error(`Expected index column resize. Before=${beforeIndexWidth} After=${afterIndexWidth}`);
      }

      const shrinkBox = await indexResizer.boundingBox();
      if (!shrinkBox) throw new Error("Expected index column resizer before shrink");
      await page.mouse.move(shrinkBox.x + shrinkBox.width / 2, shrinkBox.y + shrinkBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(shrinkBox.x + shrinkBox.width / 2 - 30, shrinkBox.y + shrinkBox.height / 2);
      await page.mouse.up();
      const shrunkIndexWidth = await columnWidth(page, "index");
      if (shrunkIndexWidth >= afterIndexWidth - 10) {
        throw new Error(`Expected index column to shrink. Resized=${afterIndexWidth} Shrunk=${shrunkIndexWidth}`);
      }

      await page.evaluate(() => document.querySelector("#column-autosize")?.click());
      await page.locator("#values-table").waitFor({ state: "visible", timeout: 10000 });
      const autoIndexWidth = await columnWidth(page, "index");
      if (autoIndexWidth > afterIndexWidth) {
        throw new Error(`Expected auto-size to keep index compact. Resized=${afterIndexWidth} Auto=${autoIndexWidth}`);
      }

      await rawHeader.focus();
      await page.keyboard.press("Alt+ArrowLeft");
      await page.keyboard.press("Alt+ArrowLeft");
      const reordered = await headerOrder(page);
      if (reordered.indexOf("raw") > reordered.indexOf("block")) {
        throw new Error(`Expected Raw column to move before Block. Before=${initialOrder} After=${reordered}`);
      }
    }

    await page.evaluate(() => document.querySelector('th[data-column-id="diff"]')?.scrollIntoView());

    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`UI verified: ${screenshotPath}`);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
