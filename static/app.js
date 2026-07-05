const SIDEBAR_WIDTH_DEFAULT = 390;
const SIDEBAR_WIDTH_MIN = 260;
const SIDEBAR_WORKSPACE_MIN = 360;
const SIDEBAR_WIDTH_MAX = 760;
const COLUMN_WIDTH_MIN = 58;
const COLUMN_WIDTH_MAX = 520;
const MODEL_SCAN_DEBOUNCE_MS = 350;

const VALUE_COLUMN_DEFS = {
  q8_0: [
    {
      id: "index",
      label: "Index",
      tooltip: "Zero-based flat index within the tensor.",
      cell: (row) => row.index,
    },
    {
      id: "coords",
      label: "Coords",
      tooltip: "Tensor coordinates decoded from the flat index using GGUF dimension order.",
      className: "mono",
      cell: (row) => escapeHtml(formatDimensions(row.coords)),
    },
    {
      id: "block",
      label: "Block",
      tooltip: "Q8_0 block number. Each block stores one scale and 32 signed int8 values.",
      cell: (row) => row.block,
    },
    {
      id: "in_block",
      label: "In block",
      tooltip: "Position inside the current 32-value Q8_0 block.",
      cell: (row) => row.in_block,
    },
    {
      id: "raw",
      label: "Raw q",
      tooltip: "The signed int8 value stored on disk before the block scale is applied.",
      cell: (row) => row.raw,
    },
    {
      id: "scale",
      label: "Scale",
      tooltip: "Half-precision scale stored at the start of the Q8_0 block.",
      cell: (row) => formatNumber(row.scale),
    },
    {
      id: "value",
      label: "Value",
      tooltip: "Displayed value for the active mode. Static shows raw q; Final shows dequantized.",
      cell: (row) => formatNumber(row.value),
    },
    {
      id: "decoded",
      label: "Final",
      tooltip: "Dequantized value computed as scale multiplied by the raw q value.",
      cell: (row) => formatNumber(row.decoded),
    },
    {
      id: "reference",
      label: "Reference",
      tooltip: "Decoded value from the matching tensor in the loaded reference GGUF, usually BF16.",
      cell: (row) => formatOptionalNumber(row.reference_value),
    },
    {
      id: "diff",
      label: "Diff",
      tooltip: "Final quantized value minus the matching reference value. The cell tint grows with absolute difference.",
      className: "diff-cell",
      cell: (row) => formatOptionalNumber(referenceDiff(row)),
      style: (row) => `--diff-alpha: ${diffHeatAlpha(referenceDiff(row))};`,
    },
  ],
  q4_0: [
    {
      id: "index",
      label: "Index",
      tooltip: "Zero-based flat index within the tensor.",
      cell: (row) => row.index,
    },
    {
      id: "coords",
      label: "Coords",
      tooltip: "Tensor coordinates decoded from the flat index using GGUF dimension order.",
      className: "mono",
      cell: (row) => escapeHtml(formatDimensions(row.coords)),
    },
    {
      id: "block",
      label: "Block",
      tooltip: "Q4_0 block number. Each block stores one scale and 32 values packed as 4-bit nibbles.",
      cell: (row) => row.block,
    },
    {
      id: "in_block",
      label: "In block",
      tooltip: "Position inside the current 32-value Q4_0 block.",
      cell: (row) => row.in_block,
    },
    {
      id: "raw",
      label: "Raw q",
      tooltip: "The signed 4-bit value (-8 to 7) stored on disk before the block scale is applied.",
      cell: (row) => row.raw,
    },
    {
      id: "scale",
      label: "Scale",
      tooltip: "Half-precision scale stored at the start of the Q4_0 block.",
      cell: (row) => formatNumber(row.scale),
    },
    {
      id: "value",
      label: "Value",
      tooltip: "Displayed value for the active mode. Static shows raw q; Final shows dequantized.",
      cell: (row) => formatNumber(row.value),
    },
    {
      id: "decoded",
      label: "Final",
      tooltip: "Dequantized value computed as scale multiplied by the raw q value.",
      cell: (row) => formatNumber(row.decoded),
    },
    {
      id: "reference",
      label: "Reference",
      tooltip: "Decoded value from the matching tensor in the loaded reference GGUF, usually BF16.",
      cell: (row) => formatOptionalNumber(row.reference_value),
    },
    {
      id: "diff",
      label: "Diff",
      tooltip: "Final quantized value minus the matching reference value. The cell tint grows with absolute difference.",
      className: "diff-cell",
      cell: (row) => formatOptionalNumber(referenceDiff(row)),
      style: (row) => `--diff-alpha: ${diffHeatAlpha(referenceDiff(row))};`,
    },
  ],
  scalar: [
    {
      id: "index",
      label: "Index",
      tooltip: "Zero-based flat index within the tensor.",
      cell: (row) => row.index,
    },
    {
      id: "coords",
      label: "Coords",
      tooltip: "Tensor coordinates decoded from the flat index using GGUF dimension order.",
      className: "mono",
      cell: (row) => escapeHtml(formatDimensions(row.coords)),
    },
    {
      id: "raw",
      label: "Raw",
      tooltip: "Raw on-disk bytes for this sampled scalar value.",
      className: "mono",
      cell: (row) => escapeHtml(row.raw),
    },
    {
      id: "value",
      label: "Value",
      tooltip: "Displayed value for the active mode. Static shows raw bytes; Final shows decoded numeric data.",
      cell: (row) => formatNumber(row.value),
    },
    {
      id: "decoded",
      label: "Decoded",
      tooltip: "Decoded numeric value after interpreting the raw scalar bytes.",
      cell: (row) => formatNumber(row.decoded),
    },
  ],
};

const VALUE_COLUMN_DEFAULT_WIDTHS = {
  q8_0: {
    index: 72,
    coords: 112,
    block: 88,
    in_block: 96,
    raw: 88,
    scale: 138,
    value: 104,
    decoded: 116,
    reference: 126,
    diff: 112,
  },
  scalar: {
    index: 72,
    coords: 112,
    raw: 142,
    value: 118,
    decoded: 118,
  },
};

const VALUE_COLUMN_LABEL_MIN_WIDTHS = {
  q8_0: {
    index: 84,
    coords: 90,
    block: 84,
    in_block: 104,
    raw: 86,
    scale: 82,
    value: 82,
    decoded: 82,
    reference: 124,
    diff: 76,
  },
  scalar: {
    index: 84,
    coords: 90,
    raw: 72,
    value: 82,
    decoded: 98,
  },
};

const VALUE_COLUMN_VISIBILITY_VERSION = {
  q8_0: 2,
  q4_0: 2,
  scalar: 1,
};

const state = {
  open: false,
  file: null,
  referenceFile: null,
  metadata: [],
  tree: null,
  currentPath: [],
  selectedTensor: null,
  tensorDetail: null,
  valuesMode: "dequantized",
  valuePayload: null,
  valueError: null,
  valuesLoading: false,
  sampleStart: 0,
  sampleCount: 64,
  modelDirectory: "",
  discoveredModels: [],
  modelsLoading: false,
  modelsError: "",
  modelBrowserOpen: true,
  valueColumnOrder: {
    q8_0: loadValueColumnOrder("q8_0"),
    scalar: loadValueColumnOrder("scalar"),
  },
  valueColumnVisibility: {
    q8_0: loadValueColumnVisibility("q8_0"),
    scalar: loadValueColumnVisibility("scalar"),
  },
  valueColumnWidths: {
    q8_0: loadValueColumnWidths("q8_0"),
    scalar: loadValueColumnWidths("scalar"),
  },
};

const els = {
  appShell: document.querySelector(".app-shell"),
  openForm: document.querySelector("#open-form"),
  pathInput: document.querySelector("#path-input"),
  scanButton: document.querySelector("#scan-button"),
  modelBrowser: document.querySelector("#model-browser"),
  referenceForm: document.querySelector("#reference-form"),
  referencePathInput: document.querySelector("#reference-path-input"),
  referenceClearButton: document.querySelector("#reference-clear-button"),
  referenceSummary: document.querySelector("#reference-summary"),
  splitResizer: document.querySelector("#split-resizer"),
  filterInput: document.querySelector("#filter-input"),
  breadcrumbs: document.querySelector("#breadcrumbs"),
  nodeList: document.querySelector("#node-list"),
  summaryStrip: document.querySelector("#summary-strip"),
  emptyState: document.querySelector("#empty-state"),
  detailPanel: document.querySelector("#detail-panel"),
  toast: document.querySelector("#toast"),
  tooltip: null,
};

const numberFormatter = new Intl.NumberFormat();
let draggedValueColumnId = null;
let resizingValueColumn = null;
let modelScanTimer = 0;

init();

function init() {
  const storedPath = localStorage.getItem("ggufExplorer.path");
  if (storedPath) els.pathInput.value = storedPath;
  const storedReferencePath = localStorage.getItem("ggufExplorer.referencePath");
  if (storedReferencePath) els.referencePathInput.value = storedReferencePath;
  initSplitter();
  initTooltip();

  els.openForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await openPath(els.pathInput.value.trim());
  });
  els.pathInput.addEventListener("input", () => scheduleModelScan());
  els.pathInput.addEventListener("focus", () => {
    state.modelBrowserOpen = true;
    renderModelBrowser();
  });
  els.scanButton.addEventListener("click", () => {
    state.modelBrowserOpen = true;
    scanModels(els.pathInput.value.trim());
  });
  els.referenceForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await openReferencePath(els.referencePathInput.value.trim());
  });
  els.referenceClearButton.addEventListener("click", clearReference);
  els.filterInput.addEventListener("input", renderNodeList);

  scanModels(storedPath || "");
  loadExistingState();
}

function initSplitter() {
  if (!els.appShell || !els.splitResizer) return;
  const storedWidth = Number(localStorage.getItem("ggufExplorer.sidebarWidth"));
  setSidebarWidth(Number.isFinite(storedWidth) && storedWidth > 0 ? storedWidth : SIDEBAR_WIDTH_DEFAULT);
  els.splitResizer.addEventListener("pointerdown", startSidebarResize);
  els.splitResizer.addEventListener("keydown", handleSplitterKeydown);
}

function startSidebarResize(event) {
  event.preventDefault();
  els.appShell.classList.add("resizing");
  els.splitResizer.setPointerCapture(event.pointerId);

  const onPointerMove = (moveEvent) => {
    const rect = els.appShell.getBoundingClientRect();
    const nextWidth = moveEvent.clientX - rect.left;
    setSidebarWidth(nextWidth, true);
  };
  const stopResize = () => {
    els.appShell.classList.remove("resizing");
    els.splitResizer.removeEventListener("pointermove", onPointerMove);
    els.splitResizer.removeEventListener("pointerup", stopResize);
    els.splitResizer.removeEventListener("pointercancel", stopResize);
  };

  els.splitResizer.addEventListener("pointermove", onPointerMove);
  els.splitResizer.addEventListener("pointerup", stopResize);
  els.splitResizer.addEventListener("pointercancel", stopResize);
}

function handleSplitterKeydown(event) {
  const currentWidth = sidebarWidthValue();
  let nextWidth = currentWidth;
  if (event.key === "ArrowLeft") nextWidth -= event.shiftKey ? 60 : 24;
  if (event.key === "ArrowRight") nextWidth += event.shiftKey ? 60 : 24;
  if (event.key === "Home") nextWidth = SIDEBAR_WIDTH_MIN;
  if (event.key === "End") nextWidth = SIDEBAR_WIDTH_MAX;
  if (event.key.toLowerCase() === "r") nextWidth = SIDEBAR_WIDTH_DEFAULT;
  if (nextWidth === currentWidth) return;
  event.preventDefault();
  setSidebarWidth(nextWidth, true);
}

function setSidebarWidth(width, persist = false) {
  if (!els.appShell || !els.splitResizer) return;
  const rect = els.appShell.getBoundingClientRect();
  const availableMax = rect.width > 0 ? rect.width - SIDEBAR_WORKSPACE_MIN : SIDEBAR_WIDTH_MAX;
  const maxWidth = Math.max(SIDEBAR_WIDTH_MIN, Math.min(SIDEBAR_WIDTH_MAX, availableMax));
  const nextWidth = clamp(Math.round(width), SIDEBAR_WIDTH_MIN, maxWidth);
  els.appShell.style.setProperty("--sidebar-width", `${nextWidth}px`);
  els.splitResizer.setAttribute("aria-valuemin", String(SIDEBAR_WIDTH_MIN));
  els.splitResizer.setAttribute("aria-valuemax", String(maxWidth));
  els.splitResizer.setAttribute("aria-valuenow", String(nextWidth));
  if (persist) localStorage.setItem("ggufExplorer.sidebarWidth", String(nextWidth));
}

function sidebarWidthValue() {
  const value = getComputedStyle(els.appShell).getPropertyValue("--sidebar-width").trim();
  return Number.parseFloat(value) || SIDEBAR_WIDTH_DEFAULT;
}

function resetValueState() {
  state.valuePayload = null;
  state.valueError = null;
  state.valuesLoading = false;
  state.sampleStart = 0;
  state.sampleCount = 64;
}

async function loadExistingState() {
  try {
    const payload = await api("/api/state");
    applyStatePayload(payload);
  } catch {
    render();
  }
}

async function openPath(path) {
  if (!path) {
    showToast("Path is required", true);
    return;
  }
  if (!path.toLowerCase().endsWith(".gguf")) {
    await scanModels(path);
    return;
  }
  localStorage.setItem("ggufExplorer.path", path);
  const payload = await api("/api/open", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
  state.modelBrowserOpen = false;
  applyStatePayload(payload);
}

async function openReferencePath(path) {
  if (!path) {
    showToast("Reference path is required", true);
    return;
  }
  localStorage.setItem("ggufExplorer.referencePath", path);
  const payload = await api("/api/reference/open", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
  applyReferencePayload(payload.reference);
  showToast("Reference loaded");
  await reloadCurrentValues();
}

function scheduleModelScan() {
  window.clearTimeout(modelScanTimer);
  state.modelBrowserOpen = true;
  modelScanTimer = window.setTimeout(() => scanModels(els.pathInput.value.trim()), MODEL_SCAN_DEBOUNCE_MS);
}

async function scanModels(pathHint = "") {
  state.modelBrowserOpen = true;
  state.modelsLoading = true;
  state.modelsError = "";
  renderModelBrowser();
  renderWelcomePage();
  try {
    const payload = await api(`/api/models?dir=${encodeURIComponent(pathHint)}`);
    state.modelDirectory = payload.directory || "";
    state.discoveredModels = payload.models || [];
    state.modelsError = "";
  } catch (error) {
    state.discoveredModels = [];
    state.modelsError = error.message;
  } finally {
    state.modelsLoading = false;
    renderModelBrowser();
    renderWelcomePage();
  }
}

async function loadDiscoveredModel(path) {
  els.pathInput.value = path;
  await openPath(path);
}

async function useDiscoveredModelAsReference(path) {
  els.referencePathInput.value = path;
  await openReferencePath(path);
}

async function clearReference() {
  const payload = await api("/api/reference/clear", { method: "POST" });
  localStorage.removeItem("ggufExplorer.referencePath");
  els.referencePathInput.value = "";
  applyReferencePayload(payload.reference);
  showToast("Reference cleared");
  await reloadCurrentValues();
}

function applyStatePayload(payload) {
  applyReferencePayload(payload.reference, false);
  if (!payload.open) {
    state.open = false;
    state.file = null;
    state.metadata = [];
    state.tree = null;
    state.currentPath = [];
    state.selectedTensor = null;
    state.tensorDetail = null;
    resetValueState();
    render();
    return;
  }
  state.open = true;
  state.file = payload.file;
  state.metadata = payload.metadata || [];
  state.tree = payload.tree;
  state.currentPath = [];
  state.selectedTensor = null;
  state.tensorDetail = null;
  resetValueState();
  render();
}

function applyReferencePayload(reference, shouldRender = true) {
  state.referenceFile = reference?.open ? reference.file : null;
  if (state.referenceFile?.path) {
    els.referencePathInput.value = state.referenceFile.path;
  }
  if (shouldRender) render();
}

async function reloadCurrentValues() {
  if (state.tensorDetail?.supports_values) {
    const name = state.tensorDetail.name;
    try {
      const detail = await api(`/api/tensor?name=${encodeURIComponent(name)}&_=${Date.now()}`);
      state.tensorDetail = detail;
    } catch { /* keep current detail */ }
    renderTensorDetail();
    await loadValues();
  } else {
    render();
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error || `Request failed: ${response.status}`;
    showToast(message, true);
    throw new Error(message);
  }
  return payload;
}

function render() {
  renderSummary();
  renderModelBrowser();
  renderReferenceSummary();
  renderBreadcrumbs();
  renderNodeList();
  renderDetails();
}

function renderSummary() {
  if (!state.file) {
    els.summaryStrip.innerHTML = "<span>No file</span>";
    return;
  }
  const modelName = displayModelName(state.file);
  els.summaryStrip.innerHTML = `
    <div class="summary-title" title="${escapeHtml(state.file.path)}">${escapeHtml(state.file.name)}</div>
    ${modelName ? `<div class="summary-model" title="${escapeHtml(modelName)}">${escapeHtml(modelName)}</div>` : ""}
    <div class="summary-meta">
      <span>${numberFormatter.format(state.file.tensor_count)} tensors</span>
      <span>${formatBytes(state.file.size_bytes)}</span>
      <span>v${state.file.version}</span>
    </div>
  `;
}

function renderModelBrowser() {
  if (!els.modelBrowser) return;
  if (!state.modelBrowserOpen) {
    els.modelBrowser.replaceChildren();
    return;
  }
  const status = state.modelsLoading
    ? `<div class="model-browser-status">Scanning folder</div>`
    : state.modelsError
      ? `<div class="model-browser-status error-text">${escapeHtml(state.modelsError)}</div>`
      : state.discoveredModels.length
        ? `<div class="model-browser-status">${numberFormatter.format(state.discoveredModels.length)} GGUF files in ${escapeHtml(state.modelDirectory)}</div>`
        : `<div class="model-browser-status">No GGUF files found in ${escapeHtml(state.modelDirectory || "this folder")}</div>`;
  const rows = state.discoveredModels.map(modelRowHtml).join("");
  els.modelBrowser.innerHTML = `
    ${status}
    ${rows ? `<div class="model-list">${rows}</div>` : ""}
  `;
  els.modelBrowser.querySelectorAll("[data-model-load]").forEach((button) => {
    button.addEventListener("click", () => loadDiscoveredModel(button.dataset.modelLoad));
  });
  els.modelBrowser.querySelectorAll("[data-model-reference]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      useDiscoveredModelAsReference(button.dataset.modelReference);
    });
  });
}

function modelRowHtml(model) {
  const modelName = displayModelName(model);
  const meta = [
    model.primary_type,
    model.tensor_count ? `${numberFormatter.format(model.tensor_count)} tensors` : "",
    formatBytes(model.size_bytes),
    model.error ? "Unreadable" : "",
  ].filter(Boolean).join(" - ");
  return `
    <div class="model-row">
      <button type="button" class="model-load" data-model-load="${escapeHtml(model.path)}">
        <span class="model-name">${escapeHtml(model.name)}</span>
        ${modelName ? `<span class="model-sub">${escapeHtml(modelName)}</span>` : ""}
        <span class="model-meta">${escapeHtml(meta)}</span>
      </button>
      <button type="button" class="model-reference" data-model-reference="${escapeHtml(model.path)}">Ref</button>
    </div>
  `;
}

function renderReferenceSummary() {
  if (!els.referenceSummary) return;
  if (!state.referenceFile) {
    els.referenceSummary.textContent = "No reference";
    return;
  }
  els.referenceSummary.innerHTML = `
    <span title="${escapeHtml(state.referenceFile.path)}">
      <strong>${escapeHtml(state.referenceFile.name)}</strong>
      ${displayModelName(state.referenceFile) ? `${escapeHtml(displayModelName(state.referenceFile))}, ` : ""}
      ${numberFormatter.format(state.referenceFile.tensor_count)} tensors, ${formatBytes(state.referenceFile.size_bytes)}
    </span>
  `;
}

function renderBreadcrumbs() {
  els.breadcrumbs.replaceChildren();
  if (!state.tree) return;

  const rootButton = crumbButton("root", []);
  els.breadcrumbs.append(rootButton);
  state.currentPath.forEach((part, index) => {
    els.breadcrumbs.append(crumbButton(part, state.currentPath.slice(0, index + 1)));
  });
}

function crumbButton(label, path) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", () => {
    state.currentPath = path;
    state.selectedTensor = null;
    state.tensorDetail = null;
    render();
  });
  return button;
}

function renderNodeList() {
  els.nodeList.replaceChildren();
  if (!state.tree) {
    const row = document.createElement("div");
    row.className = "node-row";
    row.innerHTML = `<span class="node-icon">-</span><span class="node-main"><span class="node-name">No file</span></span>`;
    els.nodeList.append(row);
    return;
  }

  const node = getCurrentNode();
  const filter = els.filterInput.value.trim().toLowerCase();
  const children = (node.children || []).filter((child) => matchesFilter(child, filter));
  if (children.length === 0) {
    const row = document.createElement("div");
    row.className = "node-row";
    row.innerHTML = `<span class="node-icon">-</span><span class="node-main"><span class="node-name">No matches</span></span>`;
    els.nodeList.append(row);
    return;
  }

  for (const child of children) {
    els.nodeList.append(nodeRow(child));
  }
}

function nodeRow(node) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "node-row";
  if (state.selectedTensor === node.tensor_name) button.classList.add("selected");
  const isGroup = node.kind === "group";
  const sub = isGroup
    ? `${numberFormatter.format(node.tensor_count)} tensors`
    : `${node.tensor_type} ${formatDimensions(node.dimensions)}`;
  const pill = isGroup ? "group" : formatBytes(node.byte_size);
  button.innerHTML = `
    <span class="node-icon">${isGroup ? ">" : "T"}</span>
    <span class="node-main">
      <span class="node-name" title="${escapeHtml(node.path)}">${escapeHtml(node.name)}</span>
      <span class="node-sub">${escapeHtml(sub)}</span>
    </span>
    <span class="node-pill">${escapeHtml(pill)}</span>
  `;

  button.addEventListener("click", () => {
    if (isGroup) {
      renderGroupDetail(node);
    } else {
      selectTensor(node.tensor_name);
    }
  });
  button.addEventListener("dblclick", () => {
    if (isGroup) {
      state.currentPath = node.path ? node.path.split(".") : [];
      state.selectedTensor = null;
      state.tensorDetail = null;
      render();
    } else {
      selectTensor(node.tensor_name);
    }
  });
  return button;
}

function matchesFilter(node, filter) {
  if (!filter) return true;
  const text = `${node.name} ${node.path} ${node.tensor_type || ""}`.toLowerCase();
  return text.includes(filter);
}

function getCurrentNode() {
  let node = state.tree;
  for (const part of state.currentPath) {
    node = (node.children || []).find((child) => child.kind === "group" && child.name === part);
    if (!node) return state.tree;
  }
  return node;
}

function renderDetails() {
  els.emptyState.classList.toggle("hidden", state.open);
  els.detailPanel.classList.toggle("hidden", !state.open);
  if (!state.open) {
    renderWelcomePage();
    return;
  }
  if (state.tensorDetail) {
    renderTensorDetail();
    return;
  }
  renderFileDetail();
}

function renderWelcomePage() {
  if (!els.emptyState || state.open) return;
  const modelRows = state.discoveredModels
    .slice(0, 8)
    .map(
      (model) => `
        <div class="welcome-model">
          <div>
            <strong>${escapeHtml(model.name)}</strong>
            ${displayModelName(model) ? `<span>${escapeHtml(displayModelName(model))}</span>` : ""}
          </div>
          <button type="button" data-welcome-model-load="${escapeHtml(model.path)}">Load</button>
        </div>
      `,
    )
    .join("");
  const modelStatus = state.modelsLoading
    ? "Scanning this folder for GGUF files."
    : state.discoveredModels.length
      ? `Found ${numberFormatter.format(state.discoveredModels.length)} GGUF files in ${state.modelDirectory}.`
      : "No GGUF files found in the current folder yet.";
  els.emptyState.innerHTML = `
    <div class="welcome">
      <h1>GGUF Explorer</h1>
      <p class="welcome-lede">Open a GGUF, drill into tensors, and compare quantized values against a native reference.</p>
      <ol class="welcome-steps">
        <li>Choose a model from the detected list or paste a GGUF path in the top bar.</li>
        <li>Load a BF16/native file as Reference when comparing quantized files.</li>
        <li>Double-click groups in the left browser, then select a tensor to inspect values.</li>
        <li>Use the table controls to resize, hide, reorder, and reset value columns.</li>
      </ol>
      <div class="welcome-detected">
        <div class="welcome-detected-title">${escapeHtml(modelStatus)}</div>
        ${modelRows || `<div class="notice">Put GGUF files in ${escapeHtml(state.modelDirectory || "this folder")} or enter a folder path above, then scan.</div>`}
      </div>
    </div>
  `;
  els.emptyState.querySelectorAll("[data-welcome-model-load]").forEach((button) => {
    button.addEventListener("click", () => loadDiscoveredModel(button.dataset.welcomeModelLoad));
  });
}

function renderFileDetail() {
  const typeRows = Object.entries(state.file.type_counts || {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([type, count]) => {
      const bytes = state.file.type_bytes?.[type];
      return `<tr><td>${escapeHtml(type)}</td><td>${numberFormatter.format(count)}</td><td>${formatBytes(bytes)}</td></tr>`;
    })
    .join("");
  const metadataRows = state.metadata
    .slice(0, 80)
    .map(
      (entry) => `
      <div class="metadata-row">
        <div class="metadata-key">${escapeHtml(entry.key)} <span class="mono">${escapeHtml(entry.value_type)}</span></div>
        <div class="metadata-value">${metadataValueHtml(entry.value)}</div>
      </div>
    `,
    )
    .join("");

  els.detailPanel.innerHTML = `
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <h2>${escapeHtml(state.file.name)}</h2>
          ${displayModelName(state.file) ? `<p>${escapeHtml(displayModelName(state.file))}</p>` : ""}
          <p class="mono">${escapeHtml(state.file.path)}</p>
        </div>
      </div>
      <div class="panel-body">
        <div class="stat-grid">
          ${stat("File size", formatBytes(state.file.size_bytes))}
          ${stat("GGUF version", state.file.version)}
          ${stat("Tensors", numberFormatter.format(state.file.tensor_count))}
          ${stat("Metadata", numberFormatter.format(state.file.metadata_count))}
          ${stat("Alignment", `${state.file.alignment} bytes`)}
          ${stat("Data start", state.file.data_start)}
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title"><h3>Tensor Types</h3></div>
      </div>
      <div class="panel-body">
        <div class="table-wrap">
          <table>
            <thead><tr><th>Type</th><th>Count</th><th>Bytes</th></tr></thead>
            <tbody>${typeRows}</tbody>
          </table>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title"><h3>Metadata</h3></div>
      </div>
      <div class="panel-body metadata-list">${metadataRows}</div>
    </section>
  `;
  renderNodeList();
}

function renderGroupDetail(node) {
  state.selectedTensor = null;
  state.tensorDetail = null;
  els.emptyState.classList.add("hidden");
  els.detailPanel.classList.remove("hidden");
  els.detailPanel.innerHTML = `
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <h2>${escapeHtml(node.path || "root")}</h2>
          <p>${numberFormatter.format(node.tensor_count)} tensors</p>
        </div>
      </div>
      <div class="panel-body">
        <div class="stat-grid">
          ${stat("Groups", numberFormatter.format((node.children || []).filter((child) => child.kind === "group").length))}
          ${stat("Tensor leaves", numberFormatter.format((node.children || []).filter((child) => child.kind === "tensor").length))}
        </div>
      </div>
    </section>
  `;
}

async function selectTensor(name) {
  state.selectedTensor = name;
  try {
    state.tensorDetail = await api(`/api/tensor?name=${encodeURIComponent(name)}`);
    resetValueState();
    renderNodeList();
    renderTensorDetail();
    if (state.tensorDetail.supports_values) await loadValues();
  } catch (error) {
    console.error(error);
  }
}

function renderTensorDetail() {
  const tensor = state.tensorDetail;
  if (!tensor) return;
  const type = tensor.type || {};
  const typeBits = type.block_size > 1
    ? `block ${type.block_size}, ${type.type_size} bytes`
    : `${type.type_size || "?"} bytes/value`;
  const valuesHtml = valuePanelHtml(tensor);

  els.emptyState.classList.add("hidden");
  els.detailPanel.classList.remove("hidden");
  els.detailPanel.innerHTML = `
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <h2>${escapeHtml(tensor.name)}</h2>
          <p>${escapeHtml(tensor.type_name)} ${escapeHtml(formatDimensions(tensor.dimensions))}</p>
        </div>
      </div>
      <div class="panel-body">
        <div class="stat-grid">
          ${stat("Elements", numberFormatter.format(tensor.element_count))}
          ${stat("Bytes", formatBytes(tensor.byte_size))}
          ${stat("GGUF offset", tensor.offset)}
          ${stat("Absolute offset", tensor.absolute_offset)}
          ${stat("Type id", tensor.type_id)}
          ${stat("Type layout", typeBits)}
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <h3>Values</h3>
          <p>${state.valuesMode === "static" ? "On-disk/static" : "Dequantized"}</p>
        </div>
        ${valueToolbar(tensor)}
      </div>
      <div class="panel-body" id="values-body">${valuesHtml}</div>
    </section>
  `;
  wireValueControls();
}

function valuePanelHtml(tensor) {
  if (state.valueError) {
    return `<div class="notice error">${escapeHtml(state.valueError)}</div>`;
  }
  if (state.valuePayload) {
    const kind = valueColumnKind(tensor.type_name);
    return `${comparisonStatusHtml(tensor, state.valuePayload.reference)}${valueStatsHtml(kind)}${valuesTable(state.valuePayload.rows, tensor.type_name)}`;
  }
  if (state.valuesLoading) {
    return `<div class="notice">Loading values</div>`;
  }
  if (tensor.supports_values) {
    return `<div class="notice">Choose a sample range and refresh values.</div>`;
  }
  return `<div class="notice">Value sampling is not implemented for ${escapeHtml(tensor.type_name)}.</div>`;
}

function valueStatsHtml(kind) {
  if (kind !== "q8_0" && kind !== "q4_0") return "";
  const stats = state.tensorDetail?.stats;
  if (!stats) return "";
  const deq = state.valuesMode === "dequantized";
  const valueMin = deq ? stats.decoded_min : stats.raw_min;
  const valueMax = deq ? stats.decoded_max : stats.raw_max;
  let html = `<div class="value-stats">`;
  html += stat("Scales", numberFormatter.format(stats.unique_scales));
  html += stat("Scale min", formatNumber(stats.scale_min));
  html += stat("Scale max", formatNumber(stats.scale_max));
  html += stat("Value min", formatNumber(valueMin));
  html += stat("Value max", formatNumber(valueMax));
  html += stat("Final min", formatNumber(stats.decoded_min));
  html += stat("Final max", formatNumber(stats.decoded_max));
  if (Number.isFinite(stats.reference_min)) {
    html += stat("Ref min", formatNumber(stats.reference_min));
    html += stat("Ref max", formatNumber(stats.reference_max));
    html += stat("Diff min", formatOptionalNumber(stats.diff_min));
    html += stat("Diff max", formatOptionalNumber(stats.diff_max));
  }
  html += `</div>`;
  return html;
}

function comparisonStatusHtml(tensor, reference) {
  const kind = valueColumnKind(tensor.type_name);
  if (kind !== "q8_0" && kind !== "q4_0") return "";
  if (!reference?.open) {
    return `<div class="compare-status">Load a BF16 reference GGUF to populate Reference and Diff for this ${tensor.type_name} tensor.</div>`;
  }
  if (!reference.compatible) {
    return `<div class="compare-status warning">${escapeHtml(reference.message || "Reference tensor cannot be compared with this tensor.")}</div>`;
  }
  const typeName = reference.type_name ? `${reference.type_name} ` : "";
  const matched = Number.isFinite(Number(reference.matched))
    ? `${numberFormatter.format(reference.matched)} values`
    : "sampled values";
  return `<div class="compare-status ready">Comparing against ${escapeHtml(typeName)}reference: ${escapeHtml(reference.file?.name || "loaded GGUF")} (${matched}).</div>`;
}

function valueToolbar(tensor) {
  const disabled = tensor.supports_values ? "" : "disabled";
  const kind = valueColumnKind(tensor.type_name);
  const isQuantized = kind === "q8_0" || kind === "q4_0";
  return `
    <div class="toolbar">
      <div class="segmented" role="group">
        <button type="button" data-mode="static" class="${state.valuesMode === "static" ? "active" : ""}" ${disabled}>Static</button>
        <button type="button" data-mode="dequantized" class="${state.valuesMode === "dequantized" ? "active" : ""}" ${disabled}>Final</button>
      </div>
      <label class="field">
        <span class="field-label">Start</span>
        <input id="sample-start" type="number" min="0" step="1" value="${state.sampleStart}" ${disabled} />
      </label>
      <label class="field">
        <span class="field-label">Count</span>
        <input id="sample-count" type="number" min="1" max="1024" step="1" value="${state.sampleCount}" ${disabled} />
      </label>
      <button type="button" id="sample-refresh" ${disabled}>Refresh</button>
      ${isQuantized ? `<button type="button" id="analyze-duplicates" ${disabled} title="Count consecutive duplicate raw Q values across the entire tensor">Dup count</button>` : ""}
      <div class="column-tools">
        <button type="button" id="column-menu-button" aria-expanded="false" ${disabled}>Columns</button>
        <button type="button" id="column-reset" title="Reset column order, visibility, and widths" ${disabled}>Reset</button>
        <button type="button" id="column-autosize" title="Smart auto-size visible columns" ${disabled}>Auto size</button>
        <div id="column-menu" class="column-menu hidden" role="menu">
          ${columnMenuHtml(kind)}
        </div>
      </div>
    </div>
  `;
}

function columnMenuHtml(kind) {
  return getAllValueColumns(kind)
    .map((column) => {
      const checked = isColumnVisible(kind, column.id) ? "checked" : "";
      return `
        <label class="column-menu-row">
          <input type="checkbox" data-column-toggle="${escapeHtml(column.id)}" ${checked} />
          <span>${escapeHtml(column.label)}</span>
          <span class="column-menu-width">${columnWidth(kind, column.id)} px</span>
        </label>
      `;
    })
    .join("");
}

function wireValueControls() {
  const panel = els.detailPanel;
  panel.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.valuesMode = button.dataset.mode;
      await loadValues();
    });
  });
  const refresh = panel.querySelector("#sample-refresh");
  if (refresh) {
    refresh.addEventListener("click", loadValues);
  }
  const dupButton = panel.querySelector("#analyze-duplicates");
  if (dupButton) {
    dupButton.addEventListener("click", analyzeConsecutiveDuplicates);
  }
  wireColumnToolbarControls();
  wireValueColumnControls();
}

function wireColumnToolbarControls() {
  const menuButton = document.querySelector("#column-menu-button");
  const menu = document.querySelector("#column-menu");
  if (menuButton && menu) {
    menuButton.addEventListener("click", () => {
      const isOpen = !menu.classList.contains("hidden");
      menu.classList.toggle("hidden", isOpen);
      menuButton.setAttribute("aria-expanded", String(!isOpen));
    });
  }

  document.querySelector("#column-reset")?.addEventListener("click", () => {
    resetCurrentColumnLayout();
  });
  document.querySelector("#column-autosize")?.addEventListener("click", () => {
    autoSizeVisibleColumns();
  });

  menu?.querySelectorAll("[data-column-toggle]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      setColumnVisible(currentValueColumnKind(), checkbox.dataset.columnToggle, checkbox.checked);
    });
  });
}

async function analyzeConsecutiveDuplicates() {
  const tensor = state.tensorDetail;
  if (!tensor || !tensor.supports_values) return;
  const name = tensor.name;
  try {
    const payload = await api(`/api/tensor/consecutive_duplicates?name=${encodeURIComponent(name)}`);
    const count = payload.consecutive_duplicates;
    const total = payload.element_count;
    const pct = total > 0 ? ((count / (total - 1)) * 100).toFixed(1) : "0.0";
    showToast(`Consecutive duplicates: ${numberFormatter.format(count)} of ${numberFormatter.format(total)} values (${pct}%)`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadValues() {
  const tensor = state.tensorDetail;
  if (!tensor || !tensor.supports_values) return;
  const startInput = document.querySelector("#sample-start");
  const countInput = document.querySelector("#sample-count");
  const start = startInput ? Number(startInput.value || 0) : 0;
  const count = countInput ? Number(countInput.value || 64) : 64;
  state.sampleStart = start;
  state.sampleCount = count;
  state.valuePayload = null;
  state.valueError = null;
  state.valuesLoading = true;
  renderTensorDetail();
  try {
    const payload = await api(
      `/api/values?name=${encodeURIComponent(tensor.name)}&start=${encodeURIComponent(start)}&count=${encodeURIComponent(count)}&mode=${state.valuesMode}`,
    );
    state.valuePayload = payload;
    state.valueError = null;
  } catch (error) {
    state.valuePayload = null;
    state.valueError = error.message;
  } finally {
    state.valuesLoading = false;
    renderTensorDetail();
  }
}

function valuesTable(rows, typeName) {
  if (!rows.length) return `<div class="notice">No sampled values</div>`;
  const kind = valueColumnKind(typeName);
  const columns = getOrderedValueColumns(kind);
  if (!columns.length) return `<div class="notice">No visible columns</div>`;
  const totalWidth = totalColumnWidth(kind, columns);
  const colgroup = columns.map((column) => valueColHtml(kind, column, totalWidth)).join("");
  const head = columns.map(valueHeaderHtml).join("");
  const body = rows
    .map((row) => `<tr>${columns.map((column) => valueCellHtml(column, row)).join("")}</tr>`)
    .join("");
  return `
    <div class="table-wrap">
      <table id="values-table" data-column-kind="${kind}" style="width: 100%; min-width: ${totalWidth}px">
        <colgroup>${colgroup}</colgroup>
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function valueColHtml(kind, column, totalWidth = null) {
  const total = totalWidth || totalColumnWidth(kind);
  const percent = total > 0 ? (columnWidth(kind, column.id) / total) * 100 : 0;
  return `<col data-column-id="${escapeHtml(column.id)}" style="width: ${percent.toFixed(4)}%" />`;
}

function valueHeaderHtml(column) {
  const tooltip = `${column.tooltip} Drag this header to rearrange columns.`;
  return `
    <th
      class="value-th"
      draggable="true"
      tabindex="0"
      data-column-id="${escapeHtml(column.id)}"
      data-tooltip="${escapeHtml(tooltip)}"
      aria-label="${escapeHtml(`${column.label}. ${tooltip}`)}"
    >
      <span class="column-head">
        <span>${escapeHtml(column.label)}</span>
        <span class="column-grip" aria-hidden="true">::</span>
      </span>
      <span class="column-resizer" data-column-resizer="${escapeHtml(column.id)}" aria-label="Resize ${escapeHtml(column.label)} column"></span>
    </th>
  `;
}

function valueCellHtml(column, row) {
  const className = column.className ? ` class="${escapeHtml(column.className)}"` : "";
  const style = column.style ? ` style="${escapeHtml(column.style(row))}"` : "";
  const title = column.title ? ` title="${escapeHtml(column.title(row))}"` : "";
  return `<td${className}${style}${title}><span class="value-cell-content">${column.cell(row)}</span></td>`;
}

function valueColumnKind(typeName) {
  if (typeName === "Q8_0") return "q8_0";
  if (typeName === "Q4_0") return "q4_0";
  return "scalar";
}

function getOrderedValueColumns(kind) {
  const defById = new Map(getAllValueColumns(kind).map((column) => [column.id, column]));
  const visible = new Set(state.valueColumnVisibility[kind] || defaultColumnOrder(kind));
  return normalizeColumnOrder(kind, state.valueColumnOrder[kind])
    .filter((id) => visible.has(id))
    .map((id) => defById.get(id))
    .filter(Boolean);
}

function getAllValueColumns(kind) {
  return VALUE_COLUMN_DEFS[kind] || VALUE_COLUMN_DEFS.scalar;
}

function loadValueColumnOrder(kind) {
  const stored = localStorage.getItem(`ggufExplorer.valueColumns.${kind}`);
  if (!stored) return defaultColumnOrder(kind);
  try {
    return normalizeColumnOrder(kind, JSON.parse(stored));
  } catch {
    return defaultColumnOrder(kind);
  }
}

function saveValueColumnOrder(kind, order) {
  const normalized = normalizeColumnOrder(kind, order);
  state.valueColumnOrder[kind] = normalized;
  localStorage.setItem(`ggufExplorer.valueColumns.${kind}`, JSON.stringify(normalized));
}

function normalizeColumnOrder(kind, order) {
  const defaults = defaultColumnOrder(kind);
  if (!Array.isArray(order)) return defaults;
  const allowed = new Set(defaults);
  const normalized = order.filter((id, index) => allowed.has(id) && order.indexOf(id) === index);
  for (const id of defaults) {
    if (normalized.includes(id)) continue;
    const defaultIndex = defaults.indexOf(id);
    const nextDefaultId = defaults.slice(defaultIndex + 1).find((candidate) => normalized.includes(candidate));
    if (nextDefaultId) {
      normalized.splice(normalized.indexOf(nextDefaultId), 0, id);
    } else {
      normalized.push(id);
    }
  }
  return normalized;
}

function defaultColumnOrder(kind) {
  return getAllValueColumns(kind).map((column) => column.id);
}

function loadValueColumnVisibility(kind) {
  const stored = localStorage.getItem(`ggufExplorer.visibleColumns.${kind}`);
  if (!stored) return defaultColumnOrder(kind);
  try {
    const visible = normalizeVisibleColumns(kind, JSON.parse(stored));
    return migrateVisibleColumns(kind, visible);
  } catch {
    return defaultColumnOrder(kind);
  }
}

function saveValueColumnVisibility(kind, visibleColumns) {
  const normalized = normalizeVisibleColumns(kind, visibleColumns);
  state.valueColumnVisibility[kind] = normalized;
  localStorage.setItem(`ggufExplorer.visibleColumns.${kind}`, JSON.stringify(normalized));
  localStorage.setItem(`ggufExplorer.visibleColumnsVersion.${kind}`, String(VALUE_COLUMN_VISIBILITY_VERSION[kind] || 1));
}

function normalizeVisibleColumns(kind, visibleColumns) {
  const defaults = defaultColumnOrder(kind);
  if (!Array.isArray(visibleColumns)) return defaults;
  const allowed = new Set(defaults);
  const normalized = visibleColumns.filter((id, index) => allowed.has(id) && visibleColumns.indexOf(id) === index);
  return normalized.length ? normalized : defaults;
}

function migrateVisibleColumns(kind, visibleColumns) {
  const currentVersion = VALUE_COLUMN_VISIBILITY_VERSION[kind] || 1;
  const storedVersion = Number(localStorage.getItem(`ggufExplorer.visibleColumnsVersion.${kind}`) || "1");
  const normalized = normalizeVisibleColumns(kind, visibleColumns);
  if ((kind === "q8_0" || kind === "q4_0") && storedVersion < 2 && !normalized.includes("reference")) {
    const diffIndex = normalized.indexOf("diff");
    if (diffIndex >= 0) {
      normalized.splice(diffIndex, 0, "reference");
    } else {
      normalized.push("reference");
    }
  }
  localStorage.setItem(`ggufExplorer.visibleColumnsVersion.${kind}`, String(currentVersion));
  return normalized;
}

function isColumnVisible(kind, columnId) {
  return normalizeVisibleColumns(kind, state.valueColumnVisibility[kind]).includes(columnId);
}

function setColumnVisible(kind, columnId, isVisible) {
  if (!columnId) return;
  const visible = normalizeVisibleColumns(kind, state.valueColumnVisibility[kind]);
  const nextVisible = isVisible
    ? [...visible, columnId]
    : visible.filter((id) => id !== columnId);
  if (!isVisible && nextVisible.length === 0) {
    showToast("At least one column must stay visible", true);
    renderTensorDetail();
    return;
  }
  saveValueColumnVisibility(kind, nextVisible);
  renderTensorDetail();
}

function loadValueColumnWidths(kind) {
  const defaults = defaultColumnWidths(kind);
  const stored = localStorage.getItem(`ggufExplorer.columnWidths.${kind}`);
  if (!stored) return defaults;
  try {
    return normalizeColumnWidths(kind, JSON.parse(stored));
  } catch {
    return defaults;
  }
}

function saveValueColumnWidths(kind, widths) {
  const normalized = normalizeColumnWidths(kind, widths);
  state.valueColumnWidths[kind] = normalized;
  localStorage.setItem(`ggufExplorer.columnWidths.${kind}`, JSON.stringify(normalized));
}

function normalizeColumnWidths(kind, widths) {
  const defaults = defaultColumnWidths(kind);
  const normalized = { ...defaults };
  if (widths && typeof widths === "object") {
    for (const id of defaultColumnOrder(kind)) {
      const width = Number(widths[id]);
      if (Number.isFinite(width)) normalized[id] = clampColumnWidth(kind, id, width);
    }
  }
  return normalized;
}

function defaultColumnWidths(kind) {
  const defaults = VALUE_COLUMN_DEFAULT_WIDTHS[kind] || {};
  const widths = {};
  for (const id of defaultColumnOrder(kind)) {
    widths[id] = clampColumnWidth(kind, id, defaults[id] || 112);
  }
  return widths;
}

function columnWidth(kind, columnId) {
  return normalizeColumnWidths(kind, state.valueColumnWidths[kind])[columnId] || 112;
}

function minColumnWidth(kind, columnId) {
  const labelMins = VALUE_COLUMN_LABEL_MIN_WIDTHS[kind] || {};
  return Math.max(COLUMN_WIDTH_MIN, labelMins[columnId] || COLUMN_WIDTH_MIN);
}

function clampColumnWidth(kind, columnId, width) {
  return clamp(Math.round(width), minColumnWidth(kind, columnId), COLUMN_WIDTH_MAX);
}

function totalColumnWidth(kind, columns = getOrderedValueColumns(kind)) {
  return columns.reduce((sum, column) => sum + columnWidth(kind, column.id), 0);
}

function wireValueColumnControls() {
  const table = document.querySelector("#values-table");
  if (!table) return;
  table.querySelectorAll("th[data-column-id]").forEach((header) => {
    header.addEventListener("pointerenter", showHeaderTooltip);
    header.addEventListener("pointermove", positionHeaderTooltip);
    header.addEventListener("pointerleave", hideHeaderTooltip);
    header.addEventListener("focus", showHeaderTooltip);
    header.addEventListener("blur", hideHeaderTooltip);
    header.addEventListener("dragstart", handleColumnDragStart);
    header.addEventListener("dragover", handleColumnDragOver);
    header.addEventListener("dragleave", (event) => clearColumnDropClasses(event.currentTarget));
    header.addEventListener("drop", handleColumnDrop);
    header.addEventListener("dragend", clearAllColumnDragState);
    header.addEventListener("keydown", handleColumnKeydown);
  });
  table.querySelectorAll("[data-column-resizer]").forEach((resizer) => {
    resizer.addEventListener("pointerdown", startColumnResize);
    resizer.addEventListener("dblclick", (event) => {
      event.preventDefault();
      event.stopPropagation();
      autoSizeVisibleColumns([event.currentTarget.dataset.columnResizer]);
    });
  });
}

function handleColumnDragStart(event) {
  hideHeaderTooltip();
  draggedValueColumnId = event.currentTarget.dataset.columnId;
  event.currentTarget.classList.add("dragging");
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", draggedValueColumnId);
}

function initTooltip() {
  const tooltip = document.createElement("div");
  tooltip.className = "app-tooltip hidden";
  tooltip.setAttribute("role", "tooltip");
  document.body.append(tooltip);
  els.tooltip = tooltip;
}

function showHeaderTooltip(event) {
  const text = event.currentTarget?.dataset?.tooltip;
  if (!text || !els.tooltip) return;
  els.tooltip.textContent = text;
  els.tooltip.classList.remove("hidden");
  positionHeaderTooltip(event);
}

function positionHeaderTooltip(event) {
  if (!els.tooltip || els.tooltip.classList.contains("hidden")) return;
  const pointerX = Number(event.clientX);
  const pointerY = Number(event.clientY);
  const rect = event.currentTarget?.getBoundingClientRect();
  const x = Number.isFinite(pointerX) ? pointerX : (rect?.left ?? 0) + 16;
  const y = Number.isFinite(pointerY) ? pointerY : (rect?.bottom ?? 0);
  const tooltipRect = els.tooltip.getBoundingClientRect();
  const margin = 10;
  const nextLeft = clamp(x + 12, margin, window.innerWidth - tooltipRect.width - margin);
  const nextTop = clamp(y + 14, margin, window.innerHeight - tooltipRect.height - margin);
  els.tooltip.style.left = `${nextLeft}px`;
  els.tooltip.style.top = `${nextTop}px`;
}

function hideHeaderTooltip() {
  if (!els.tooltip) return;
  els.tooltip.classList.add("hidden");
}

function handleColumnDragOver(event) {
  event.preventDefault();
  const placement = columnDropPlacement(event, event.currentTarget);
  clearColumnDropClasses(event.currentTarget);
  event.currentTarget.classList.add(placement === "after" ? "drop-after" : "drop-before");
  event.dataTransfer.dropEffect = "move";
}

function handleColumnDrop(event) {
  event.preventDefault();
  const targetId = event.currentTarget.dataset.columnId;
  const sourceId = event.dataTransfer.getData("text/plain") || draggedValueColumnId;
  const placement = columnDropPlacement(event, event.currentTarget);
  clearAllColumnDragState();
  moveValueColumn(sourceId, targetId, placement);
}

function handleColumnKeydown(event) {
  if (!event.altKey || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const kind = currentValueColumnKind();
  const order = normalizeColumnOrder(kind, state.valueColumnOrder[kind]);
  const sourceId = event.currentTarget.dataset.columnId;
  const sourceIndex = order.indexOf(sourceId);
  if (sourceIndex < 0) return;
  const targetIndex = event.key === "ArrowLeft" ? sourceIndex - 1 : sourceIndex + 1;
  if (targetIndex < 0 || targetIndex >= order.length) return;
  const targetId = order[targetIndex];
  moveValueColumn(sourceId, targetId, event.key === "ArrowLeft" ? "before" : "after", sourceId);
}

function moveValueColumn(sourceId, targetId, placement = "before", focusId = "") {
  if (!sourceId || !targetId || sourceId === targetId) return;
  const kind = currentValueColumnKind();
  const currentOrder = normalizeColumnOrder(kind, state.valueColumnOrder[kind]);
  if (!currentOrder.includes(sourceId) || !currentOrder.includes(targetId)) return;
  const withoutSource = currentOrder.filter((id) => id !== sourceId);
  let targetIndex = withoutSource.indexOf(targetId);
  if (placement === "after") targetIndex += 1;
  withoutSource.splice(targetIndex, 0, sourceId);
  saveValueColumnOrder(kind, withoutSource);
  renderTensorDetail();
  const nextFocus = focusId || sourceId;
  document.querySelector(`th[data-column-id="${cssEscape(nextFocus)}"]`)?.focus();
}

function columnDropPlacement(event, header) {
  const rect = header.getBoundingClientRect();
  return event.clientX > rect.left + rect.width / 2 ? "after" : "before";
}

function currentValueColumnKind() {
  return valueColumnKind(state.tensorDetail?.type_name);
}

function clearColumnDropClasses(header) {
  header.classList.remove("drop-before", "drop-after");
}

function clearAllColumnDragState() {
  document.querySelectorAll(".value-th").forEach((header) => {
    header.classList.remove("dragging", "drop-before", "drop-after");
  });
  draggedValueColumnId = null;
}

function startColumnResize(event) {
  event.preventDefault();
  event.stopPropagation();
  const columnId = event.currentTarget.dataset.columnResizer;
  const kind = currentValueColumnKind();
  const table = document.querySelector("#values-table");
  const renderedTableWidth = table?.getBoundingClientRect().width || totalColumnWidth(kind);
  const scale = Math.max(0.1, renderedTableWidth / totalColumnWidth(kind));
  resizingValueColumn = {
    kind,
    columnId,
    startX: event.clientX,
    startWidth: columnWidth(kind, columnId),
    scale,
  };
  document.body.classList.add("column-resizing");
  event.currentTarget.setPointerCapture(event.pointerId);
  window.addEventListener("pointermove", handleColumnResizeMove);
  window.addEventListener("pointerup", stopColumnResize, { once: true });
  window.addEventListener("pointercancel", stopColumnResize, { once: true });
}

function handleColumnResizeMove(event) {
  if (!resizingValueColumn) return;
  const delta = (event.clientX - resizingValueColumn.startX) / resizingValueColumn.scale;
  const nextWidth = clampColumnWidth(
    resizingValueColumn.kind,
    resizingValueColumn.columnId,
    resizingValueColumn.startWidth + delta,
  );
  setColumnWidth(resizingValueColumn.kind, resizingValueColumn.columnId, nextWidth, false);
}

function stopColumnResize() {
  if (resizingValueColumn) {
    const kind = resizingValueColumn.kind;
    saveValueColumnWidths(kind, state.valueColumnWidths[kind]);
  }
  resizingValueColumn = null;
  document.body.classList.remove("column-resizing");
  window.removeEventListener("pointermove", handleColumnResizeMove);
}

function setColumnWidth(kind, columnId, width, persist = true) {
  if (!columnId) return;
  const widths = normalizeColumnWidths(kind, state.valueColumnWidths[kind]);
  widths[columnId] = clampColumnWidth(kind, columnId, width);
  state.valueColumnWidths[kind] = widths;
  applyValueTableSizing(kind);
  if (persist) saveValueColumnWidths(kind, widths);
}

function applyValueTableSizing(kind) {
  const table = document.querySelector("#values-table");
  if (!table) return;
  const columns = getOrderedValueColumns(kind);
  const total = totalColumnWidth(kind, columns);
  table.style.setProperty("width", "100%");
  table.style.setProperty("min-width", `${total}px`);
  for (const column of columns) {
    const selector = `[data-column-id="${cssEscape(column.id)}"]`;
    const percent = total > 0 ? (columnWidth(kind, column.id) / total) * 100 : 0;
    table.querySelector(`col${selector}`)?.style.setProperty("width", `${percent.toFixed(4)}%`);
  }
}

function autoSizeVisibleColumns(columnIds = null) {
  const table = document.querySelector("#values-table");
  if (!table) return;
  const kind = currentValueColumnKind();
  const visibleIds = getOrderedValueColumns(kind).map((column) => column.id);
  const ids = (columnIds || visibleIds).filter((id) => visibleIds.includes(id));
  if (!ids.length) return;
  const widths = normalizeColumnWidths(kind, state.valueColumnWidths[kind]);
  for (const id of ids) {
    widths[id] = clampColumnWidth(kind, id, smartColumnWidth(table, id));
  }
  saveValueColumnWidths(kind, widths);
  renderTensorDetail();
}

function smartColumnWidth(table, columnId) {
  const headers = [...table.querySelectorAll("thead th")];
  const index = headers.findIndex((header) => header.dataset.columnId === columnId);
  if (index < 0) return columnWidth(currentValueColumnKind(), columnId);
  const rows = [...table.querySelectorAll("tbody tr")].slice(0, 96);
  const samples = [
    headers[index].querySelector(".column-head span:first-child")?.textContent || "",
    ...rows.map((row) => row.children[index]?.textContent || ""),
  ];
  const context = smartColumnWidth.context || (smartColumnWidth.context = document.createElement("canvas").getContext("2d"));
  const sampleElement = rows[0]?.children[index] || headers[index];
  context.font = getComputedStyle(sampleElement).font;
  const textWidth = samples.reduce((max, text) => Math.max(max, context.measureText(text.trim()).width), 0);
  return clampColumnWidth(currentValueColumnKind(), columnId, Math.ceil(textWidth + 44));
}

function resetCurrentColumnLayout() {
  const kind = currentValueColumnKind();
  saveValueColumnOrder(kind, defaultColumnOrder(kind));
  saveValueColumnVisibility(kind, defaultColumnOrder(kind));
  saveValueColumnWidths(kind, defaultColumnWidths(kind));
  renderTensorDetail();
  showToast("Columns reset");
}

function stat(label, value) {
  return `
    <div class="stat">
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value">${escapeHtml(String(value ?? "-"))}</div>
    </div>
  `;
}

function metadataValueHtml(value) {
  if (value && typeof value === "object" && value.kind === "array") {
    const preview = value.preview.map((item) => stringifyMetadataValue(item)).join(", ");
    const suffix = value.truncated ? ", ..." : "";
    return `<span>${escapeHtml(value.element_type)}[${numberFormatter.format(value.length)}]</span> <code>${escapeHtml(preview + suffix)}</code>`;
  }
  if (value && typeof value === "object" && Object.hasOwn(value, "text")) {
    return `<code>${escapeHtml(value.text)}${value.truncated ? "..." : ""}</code>`;
  }
  return `<code>${escapeHtml(stringifyMetadataValue(value))}</code>`;
}

function stringifyMetadataValue(value) {
  if (value && typeof value === "object" && Object.hasOwn(value, "text")) return value.text;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function displayModelName(file) {
  if (!file?.model_name || file.model_name === file.name) return "";
  return file.model_name;
}

function formatDimensions(dimensions) {
  if (!dimensions || dimensions.length === 0) return "[]";
  return `[${dimensions.join(", ")}]`;
}

function formatBytes(value) {
  if (value == null || Number.isNaN(Number(value))) return "-";
  const bytes = Number(value);
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const digits = unitIndex === 0 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
}

function formatNumber(value) {
  if (typeof value === "string") return escapeHtml(value);
  if (!Number.isFinite(value)) return String(value);
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.0001 || abs >= 100000)) return value.toExponential(6);
  return String(Number(value.toPrecision(8)));
}

function formatOptionalNumber(value) {
  return value == null || !Number.isFinite(Number(value)) ? "-" : formatNumber(Number(value));
}

function referenceDiff(row) {
  const diff = Number(row.diff);
  if (!Number.isFinite(diff)) return null;
  return diff;
}

function diffHeatAlpha(diff) {
  if (diff == null) return "0";
  const magnitude = Math.abs(Number(diff));
  if (!Number.isFinite(magnitude)) return "0";
  if (magnitude <= 0) return "0.06";
  const scaled = Math.log1p(magnitude * 128) / Math.log1p(128);
  return String(Math.min(0.52, 0.06 + scaled * 0.46).toFixed(3));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function cssEscape(value) {
  if (window.CSS?.escape) return window.CSS.escape(String(value));
  return String(value).replace(/["\\]/g, "\\$&");
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    els.toast.classList.add("hidden");
  }, 4500);
}
