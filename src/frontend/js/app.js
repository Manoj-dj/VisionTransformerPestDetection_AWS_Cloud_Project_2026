// AgriVision PestGuard - frontend application logic.
// Plain HTML/CSS/JS by design so it is served from S3 + CloudFront without a build step.

// Cloud API (API Gateway HTTP API). No trailing slash.
//const API_BASE_URL = "https://nwz77svpn5.execute-api.ap-south-1.amazonaws.com";

// Local FastAPI development (only used when the page is opened locally and you uncomment this):
const API_BASE_URL = "http://127.0.0.1:8000";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB limit on the selected file
const ACCEPTED_MIME_TYPES = ["image/jpeg", "image/jpg", "image/png"];
const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png"];

// Images are resized in the browser before upload so the request stays small
// (the model resizes to 224x224 anyway).
const UPLOAD_MAX_SIDE_PX = 768;
const UPLOAD_JPEG_QUALITY = 0.9;
const REQUEST_TIMEOUT_MS = 35000;

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
  resetChat();
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
  resetChat();
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

// --- Client-side resize -> Base64 JPEG ---------------------------------------------
async function toResizedBase64(file, maxSide = UPLOAD_MAX_SIDE_PX) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff"; // avoids black background for transparent PNGs
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(bitmap, 0, 0, width, height);
  if (bitmap.close) bitmap.close();

  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", UPLOAD_JPEG_QUALITY)
  );
  if (!blob) throw new Error("Could not prepare the image for upload.");

  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

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

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const imageBase64 = await toResizedBase64(selectedFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let message = `Request failed with status ${response.status}.`;
      try {
        const errorBody = await response.json();
        message = extractErrorText(errorBody) || message;
      } catch (_) {
        // Ignore JSON parse failures; use the generic message above.
      }
      throw new Error(message);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    if (err && err.name === "AbortError") {
      showError("The request timed out. Please try again.");
    } else if (err instanceof TypeError) {
      showError("Could not reach the AgriVision PestGuard cloud API. Check your connection and try again.");
    } else {
      showError(err.message || "An unexpected error occurred.");
    }
  } finally {
    clearTimeout(timeoutId);
    loadingState.hidden = true;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

// Handles FastAPI's two error shapes: {"detail": "a string"} and
// {"detail": [{"loc": [...], "msg": "...", "type": "..."}]} (pydantic
// validation errors), plus plain {"error": "..."} / {"message": "..."}.
function extractErrorText(errorBody) {
  if (!errorBody || typeof errorBody !== "object") return "";
  const { detail, error, message } = errorBody;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (item && typeof item === "object" ? item.msg || JSON.stringify(item) : String(item)))
      .join(" ");
  }
  if (detail) return String(detail);
  if (error) return String(error);
  if (message) return String(message);
  return "";
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

  latencyValue.textContent = Number(data.latency_ms).toFixed(1);
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

  initChatForPrediction(data);
}

// --- Gemini advisory chatbot + voice input -----------------------------------------
// Reuses API_BASE_URL above; the backend holds the Gemini API key server-side and
// these calls never see it. Chat state resets whenever a new image is selected.

const chatSection = document.getElementById("chatSection");
const chatLog = document.getElementById("chatLog");
const chatLoading = document.getElementById("chatLoading");
const chatLoadingText = document.getElementById("chatLoadingText");
const chatErrorAlert = document.getElementById("chatError");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");
const micBtn = document.getElementById("micBtn");
const micIcon = document.getElementById("micIcon");

let chatHistory = []; // [{ role: "user" | "assistant", content: string }]
let predictionContext = null; // { predicted_class, top_k_predictions }
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function resetChat() {
  chatHistory = [];
  predictionContext = null;
  if (chatLog) chatLog.innerHTML = "";
  hideChatError();
  setChatLoading(false);
  if (chatSection) chatSection.hidden = true;
  stopRecording();
}

function appendChatBubble(role, text) {
  if (!chatLog) return;
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble-${role === "user" ? "user" : "assistant"}`;
  bubble.textContent = text;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setChatLoading(visible, text) {
  if (!chatLoading) return;
  chatLoading.hidden = !visible;
  if (visible && chatLoadingText) {
    chatLoadingText.textContent = text || "Thinking…";
  }
  if (chatSendBtn) chatSendBtn.disabled = visible;
  if (micBtn) micBtn.disabled = visible;
}

function showChatError(message) {
  if (!chatErrorAlert) return;
  chatErrorAlert.textContent = message;
  chatErrorAlert.hidden = false;
}

function hideChatError() {
  if (!chatErrorAlert) return;
  chatErrorAlert.hidden = true;
  chatErrorAlert.textContent = "";
}

async function parseErrorMessage(response, fallback) {
  try {
    const body = await response.json();
    return extractErrorText(body) || fallback;
  } catch (_) {
    return fallback;
  }
}

function initChatForPrediction(predictionData) {
  predictionContext = {
    predicted_class: predictionData.predicted_class,
    top_k_predictions: predictionData.top_k_predictions || [],
  };
  chatHistory = [];
  if (chatLog) chatLog.innerHTML = "";
  hideChatError();
  if (chatSection) chatSection.hidden = false;
  fetchInitialInsights();
}

async function fetchInitialInsights() {
  if (!predictionContext) return;
  setChatLoading(true, "Generating recommendations…");
  try {
    const response = await fetch(`${API_BASE_URL}/chat/insights`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(predictionContext),
    });
    if (!response.ok) {
      throw new Error(await parseErrorMessage(response, "Could not generate AI insights."));
    }
    const data = await response.json();
    chatHistory.push({ role: "assistant", content: data.message });
    appendChatBubble("assistant", data.message);
  } catch (err) {
    showChatError(err.message || "Could not generate AI insights.");
  } finally {
    setChatLoading(false);
  }
}

async function sendChatText(text) {
  const trimmed = text.trim();
  if (!trimmed || !predictionContext) return;

  hideChatError();
  chatHistory.push({ role: "user", content: trimmed });
  appendChatBubble("user", trimmed);

  setChatLoading(true, "Thinking…");
  try {
    const response = await fetch(`${API_BASE_URL}/chat/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        predicted_class: predictionContext.predicted_class,
        top_k_predictions: predictionContext.top_k_predictions,
        history: chatHistory.slice(0, -1),
        message: trimmed,
      }),
    });
    if (!response.ok) {
      throw new Error(await parseErrorMessage(response, "The AI assistant could not respond."));
    }
    const data = await response.json();
    chatHistory.push({ role: "assistant", content: data.message });
    appendChatBubble("assistant", data.message);
  } catch (err) {
    showChatError(err.message || "The AI assistant could not respond.");
  } finally {
    setChatLoading(false);
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = chatInput.value;
    chatInput.value = "";
    sendChatText(text);
  });
}

// --- Voice input (speech-to-text only; no text-to-speech) ---------------------------
function pickSupportedMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  if (typeof MediaRecorder === "undefined") return null;
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    try {
      mediaRecorder.stop();
    } catch (_) {
      // Already stopped.
    }
  }
  isRecording = false;
  if (micBtn) {
    micBtn.classList.remove("recording");
    micBtn.setAttribute("aria-pressed", "false");
  }
  if (micIcon) micIcon.textContent = "🎤";
}

async function startRecording() {
  if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
    showChatError("Voice recording is not supported in this browser.");
    return;
  }
  if (!predictionContext) {
    showChatError("Analyze an image first, then ask a question by voice.");
    return;
  }

  const mimeType = pickSupportedMimeType();
  if (mimeType === null) {
    showChatError("Voice recording is not supported in this browser.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      audioChunks = [];
      if (blob.size > 0) {
        transcribeAndSend(blob);
      }
    });

    mediaRecorder.start();
    isRecording = true;
    hideChatError();
    if (micBtn) {
      micBtn.classList.add("recording");
      micBtn.setAttribute("aria-pressed", "true");
    }
    if (micIcon) micIcon.textContent = "⏹";
  } catch (err) {
    showChatError("Microphone access was denied or is unavailable.");
  }
}

async function transcribeAndSend(audioBlob) {
  setChatLoading(true, "Transcribing voice note…");
  try {
    const formData = new FormData();
    const extension = audioBlob.type.includes("ogg") ? "ogg" : audioBlob.type.includes("mp4") ? "mp4" : "webm";
    formData.append("audio", audioBlob, `voice-note.${extension}`);

    const response = await fetch(`${API_BASE_URL}/chat/transcribe`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await parseErrorMessage(response, "Could not transcribe the voice note."));
    }
    const data = await response.json();
    setChatLoading(false);
    if (data.text && data.text.trim()) {
      await sendChatText(data.text);
    } else {
      showChatError("No speech was detected in the recording.");
    }
  } catch (err) {
    setChatLoading(false);
    showChatError(err.message || "Could not transcribe the voice note.");
  }
}

if (micBtn) {
  micBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

// --- Init ------------------------------------------------------------------------
checkHealth();