const API_BASE_URL = "https://lung-nodules-api-5qbieyivxa-de.a.run.app";

const imageInput = document.querySelector("#imageInput");
const confidenceInput = document.querySelector("#confidenceInput");
const iouInput = document.querySelector("#iouInput");
const detectButton = document.querySelector("#detectButton");
const healthButton = document.querySelector("#healthButton");
const copyButton = document.querySelector("#copyButton");
const previewImage = document.querySelector("#previewImage");
const overlayCanvas = document.querySelector("#overlayCanvas");
const emptyState = document.querySelector("#emptyState");
const imageMeta = document.querySelector("#imageMeta");
const serviceStatus = document.querySelector("#serviceStatus");
const resultCount = document.querySelector("#resultCount");
const resultList = document.querySelector("#resultList");
const rawJson = document.querySelector("#rawJson");
const imageStage = document.querySelector("#imageStage");

const state = {
  file: null,
  objectUrl: null,
  lastResponse: null,
  healthOk: false,
};

function setStatus(text, mode = "") {
  serviceStatus.textContent = text;
  serviceStatus.className = `status-pill ${mode}`.trim();
}

function formatNumber(value) {
  return Number(value).toFixed(2);
}

function resetResults() {
  state.lastResponse = null;
  resultCount.textContent = "0 detections";
  resultList.innerHTML = "";
  rawJson.textContent = "{}";
  copyButton.disabled = true;
  clearCanvas();
}

function clearCanvas() {
  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function syncOverlaySize() {
  const imageRect = previewImage.getBoundingClientRect();
  const stageRect = imageStage.getBoundingClientRect();
  overlayCanvas.style.width = `${imageRect.width}px`;
  overlayCanvas.style.height = `${imageRect.height}px`;
  overlayCanvas.style.left = `${imageRect.left - stageRect.left}px`;
  overlayCanvas.style.top = `${imageRect.top - stageRect.top}px`;
}

function drawDetections(detections = []) {
  if (!previewImage.naturalWidth || !previewImage.naturalHeight) return;

  overlayCanvas.width = previewImage.naturalWidth;
  overlayCanvas.height = previewImage.naturalHeight;
  syncOverlaySize();

  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  context.lineWidth = Math.max(3, Math.round(previewImage.naturalWidth / 160));
  context.font = `${Math.max(14, Math.round(previewImage.naturalWidth / 28))}px sans-serif`;

  detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.bbox_xyxy;
    const width = x2 - x1;
    const height = y2 - y1;
    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
    const textMetrics = context.measureText(label);
    const labelHeight = 24;

    context.strokeStyle = "#dc2626";
    context.fillStyle = "rgba(220, 38, 38, 0.12)";
    context.fillRect(x1, y1, width, height);
    context.strokeRect(x1, y1, width, height);

    context.fillStyle = "#dc2626";
    context.fillRect(x1, Math.max(0, y1 - labelHeight), textMetrics.width + 14, labelHeight);
    context.fillStyle = "#ffffff";
    context.fillText(label, x1 + 7, Math.max(17, y1 - 7));
  });

  overlayCanvas.style.display = detections.length ? "block" : "none";
}

function renderResults(response) {
  const detections = response.detections || [];
  resultCount.textContent = `${detections.length} detection${detections.length === 1 ? "" : "s"}`;
  rawJson.textContent = JSON.stringify(response, null, 2);
  copyButton.disabled = false;

  if (!detections.length) {
    resultList.innerHTML = '<div class="empty-state">No detections</div>';
    overlayCanvas.style.display = "none";
    return;
  }

  resultList.innerHTML = detections
    .map((detection, index) => {
      const [x, y, width, height] = detection.bbox_xywh;
      return `
        <article class="result-card">
          <strong>${index + 1}. ${detection.class_name}</strong>
          <div class="metric-row"><span>Confidence</span><span>${(detection.confidence * 100).toFixed(2)}%</span></div>
          <div class="metric-row"><span>x / y</span><span>${formatNumber(x)} / ${formatNumber(y)}</span></div>
          <div class="metric-row"><span>w / h</span><span>${formatNumber(width)} / ${formatNumber(height)}</span></div>
        </article>
      `;
    })
    .join("");

  drawDetections(detections);
}

function renderError(error) {
  const message = typeof error === "string" ? error : error.message || "Prediction failed";
  resultList.innerHTML = `
    <article class="result-card">
      <strong>Request failed</strong>
      <div class="metric-row"><span>Status</span><span>${message}</span></div>
    </article>
  `;
}

async function checkApiHealth() {
  setStatus("Checking", "busy");
  healthButton.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      cache: "no-store",
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(JSON.stringify(payload));
    }

    state.healthOk = true;
    setStatus("API OK", "ok");
    rawJson.textContent = JSON.stringify(payload, null, 2);
    copyButton.disabled = false;
    return true;
  } catch (error) {
    state.healthOk = false;
    setStatus("API Error", "error");
    rawJson.textContent = JSON.stringify(
      {
        error: error.message,
        note:
          "The browser could not reach the Cloud Run API. Open /health directly, try another browser/network, and check the DevTools Network tab.",
        api: `${API_BASE_URL}/health`,
      },
      null,
      2,
    );
    return false;
  } finally {
    healthButton.disabled = false;
  }
}

async function predict() {
  if (!state.file) return;

  setStatus("Running", "busy");
  detectButton.disabled = true;
  resetResults();

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("confidence", confidenceInput.value || "0.25");
  formData.append("iou", iouInput.value || "0.45");
  formData.append("return_annotated_image", "false");
  formData.append("return_annotated_image_base64", "false");

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || payload);
      throw new Error(detail || `HTTP ${response.status}`);
    }

    state.lastResponse = payload;
    setStatus("Complete", "ok");
    renderResults(payload);
  } catch (error) {
    setStatus("Error", "error");
    renderError(error);
    rawJson.textContent = JSON.stringify(
      {
        error: error.message,
        note:
          "If this says Failed to fetch, the browser connection to Cloud Run was reset or blocked. If the API returns a Vertex error JSON, check whether Start Vertex Model has run.",
      },
      null,
      2,
    );
  } finally {
    detectButton.disabled = !state.file;
  }
}

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (!file) return;

  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  previewImage.src = state.objectUrl;
  previewImage.style.display = "block";
  emptyState.style.display = "none";
  detectButton.disabled = false;
  setStatus("Ready");
  resetResults();
});

previewImage.addEventListener("load", () => {
  imageMeta.textContent = `${previewImage.naturalWidth} x ${previewImage.naturalHeight}`;
  overlayCanvas.width = previewImage.naturalWidth;
  overlayCanvas.height = previewImage.naturalHeight;
  syncOverlaySize();
});

detectButton.addEventListener("click", predict);
healthButton.addEventListener("click", checkApiHealth);

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(rawJson.textContent);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy JSON";
  }, 1200);
});

window.addEventListener("resize", () => {
  if (state.lastResponse) {
    drawDetections(state.lastResponse.detections || []);
  } else {
    syncOverlaySize();
  }
});

checkApiHealth();
