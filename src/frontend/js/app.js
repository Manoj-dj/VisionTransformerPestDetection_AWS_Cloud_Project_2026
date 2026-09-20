// AgriVision PestGuard - frontend application logic.
// Plain HTML/CSS/JS by design so this can later be served from S3/CloudFront
// without a build step.

// Configure this to point at your running FastAPI backend.
const API_BASE_URL = "http://127.0.0.1:8000";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
const ACCEPTED_MIME_TYPES = ["image/jpeg", "image/jpg", "image/png"];
const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png"];

// --- Element references -----------------------------------------------------
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const dropZoneContent = document.getElementById("dropZoneContent");

const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const fileMeta = document.getElementById("fileMeta");
const resetBtn = document.getElementById("resetBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingState = document.getElementById("loadingState");
const errorAlert = document.getElementById("errorAlert");

const resultsSection = document.getElementById("resultsSection");
const predictedName = document.getElementById("predictedName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceFill = document.getElementById("confidenceFill");
const latencyValue = document.getElementById("latencyValue");
const modelVersionValue = document.getElementById("modelVersionValue");
const topKList = document.getElementById("topKList");
const originalImage = document.getElementById("originalImage");
const gradcamImage = document.getElementById("gradcamImage");
const gradcamNote = document.getElementById("gradcamNote");

// --- State -------------------------------------------------------------------
let selectedFile = null;
let previewObjectUrl = null;

// --- Health check --------------------------------------------------------------
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    statusDot.className = "status-dot status-ok";
    statusText.textContent = `API online (${data.device})`;
  } catch (err) {
    statusDot.className = "status-dot status-error";
    statusText.textContent = "API unreachable";
  }
}

// --- File selection / validation -----------------------------------------------
function hasAcceptedExtension(filename) {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function validateFile(file) {
  if (!file) return "No file selected.";
  if (!hasAcceptedExtension(file.name)) {
    return "Unsupported file type. Please upload a JPG, JPEG, or PNG image.";
  }
  if (file.type && !ACCEPTED_MIME_TYPES.includes(file.type)) {
    return "Unsupported file type. Please upload a JPG, JPEG, or PNG image.";
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return "File is too large. Maximum allowed size is 10 MB.";
  }
  if (file.size === 0) {
    return "File is empty.";
  }
  return null;
}

function formatBytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function handleFileSelected(file) {
  hideError();
  const validationError = validateFile(file);
  if (validationError) {
    showError(validationError);
    return;
  }

  selectedFile = file;

  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
  }
  previewObjectUrl = URL.createObjectURL(file);
  previewImg.src = previewObjectUrl;

  fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;

  dropZoneContent.hidden = true;
  previewWrap.hidden = false;
  resultsSection.hidden = true;
}

function resetSelection() {
  selectedFile = null;
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  previewImg.src = "";
  fileInput.value = "";
  previewWrap.hidden = true;
  dropZoneContent.hidden = false;
  resultsSection.hidden = true;
  hideError();
}

// --- Errors --------------------------------------------------------------------
function showError(message) {
  errorAlert.textContent = message;
  errorAlert.hidden = false;
}

function hideError() {
  errorAlert.hidden = true;
  errorAlert.textContent = "";
}

// --- Drag and drop ---------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    handleFileSelected(fileInput.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer && event.dataTransfer.files;
  if (files && files[0]) {
    handleFileSelected(files[0]);
  }
});

resetBtn.addEventListener("click", resetSelection);

// --- Analyze -----------------------------------------------------------------------
analyzeBtn.addEventListener("click", analyzeImage);

async function analyzeImage() {
  if (!selectedFile) {
    showError("Please select an image first.");
    return;
  }

  hideError();
  resultsSection.hidden = true;
  loadingState.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    // Let the browser set the multipart boundary automatically.
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `Request failed with status ${response.status}.`;
      try {
        const errorBody = await response.json();
        if (errorBody && errorBody.detail) {
          message = String(errorBody.detail);
        } else if (errorBody && errorBody.error) {
          message = String(errorBody.error);
        }
      } catch (_) {
        // Ignore JSON parse failures; use the generic message above.
      }
      throw new Error(message);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError(
      err instanceof TypeError
        ? "Could not reach the AgriVision PestGuard API. Is the backend running?"
        : err.message || "An unexpected error occurred."
    );
  } finally {
    loadingState.hidden = true;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

// --- Rendering -----------------------------------------------------------------------
function base64ToDataUrl(base64String) {
  return `data:image/png;base64,${base64String}`;
}

function renderResults(data) {
  const predicted = data.predicted_class;

  predictedName.textContent = predicted.class_name;

  const confidencePercent = Math.round(predicted.confidence * 1000) / 10;
  confidenceValue.textContent = `${confidencePercent}%`;
  confidenceFill.style.width = `${confidencePercent}%`;
  confidenceBar.setAttribute("aria-valuenow", String(confidencePercent));

  latencyValue.textContent = data.latency_ms.toFixed(1);
  modelVersionValue.textContent = data.model_version;

  topKList.innerHTML = "";
  (data.top_k_predictions || []).forEach((entry, index) => {
    const li = document.createElement("li");

    const nameSpan = document.createElement("span");
    const rankSpan = document.createElement("span");
    rankSpan.className = "top-k-rank";
    rankSpan.textContent = `#${index + 1}`;
    nameSpan.appendChild(rankSpan);
    nameSpan.appendChild(document.createTextNode(entry.class_name));

    const confSpan = document.createElement("span");
    confSpan.textContent = `${(entry.confidence * 100).toFixed(2)}%`;

    li.appendChild(nameSpan);
    li.appendChild(confSpan);
    topKList.appendChild(li);
  });

  originalImage.src = previewObjectUrl || "";

  const explanation = data.explanation;
  if (explanation && explanation.enabled) {
    gradcamImage.src = base64ToDataUrl(explanation.overlay_base64);
    gradcamImage.hidden = false;
    gradcamNote.textContent =
      "Highlighted regions show areas that influenced the model prediction.";
  } else {
    gradcamImage.hidden = true;
    gradcamNote.textContent =
      explanation && explanation.error
        ? `Grad-CAM explanation unavailable: ${explanation.error}`
        : "Grad-CAM explanation unavailable for this request.";
  }

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Init ------------------------------------------------------------------------
checkHealth();
