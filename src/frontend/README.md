# AgriVision PestGuard — Frontend

Plain HTML5/CSS3/vanilla JavaScript dashboard for the ViT-based IP102 pest
classifier. No framework, no build step — this can later be served
directly from an S3 bucket behind CloudFront.

## Files

```
src/frontend/
├── index.html      # Page structure and sections
├── css/styles.css  # Agriculture-themed green/teal design system
├── js/app.js       # Upload, validation, API calls, rendering
└── assets/         # Static assets (icons, images) if added later
```

## Configuration

The backend base URL is a single constant at the top of `js/app.js`:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

Update it if your backend runs on a different host/port.

## Run locally

Do **not** open `index.html` directly with `file://` — browsers apply
different CORS rules to `file://` pages, which can cause requests to the
API to fail unpredictably. Serve it over HTTP instead:

### Windows

```powershell
cd src/frontend
python -m http.server 5500
```

### Linux/macOS

```bash
cd src/frontend
python -m http.server 5500
```

Then open:

```
http://127.0.0.1:5500
```

Make sure the backend is running first (see `src/backend/README.md`) and
that its `ALLOWED_ORIGINS` includes `http://127.0.0.1:5500`.

## Usage

1. Drag and drop, or browse, a JPG/JPEG/PNG image (max 10 MB).
2. Click **Analyze image**.
3. Review the predicted pest, confidence, top-5 list, original image, and
   Grad-CAM overlay, plus the request latency.

The Grad-CAM overlay is an explanation heatmap over the 224×224 image the
model actually processed — it highlights influential regions and is not
an object-detection bounding box.
