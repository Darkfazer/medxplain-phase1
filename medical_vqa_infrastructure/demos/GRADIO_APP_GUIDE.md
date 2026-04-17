# 🏥 MedXplain Gradio App – Complete Guide

> **File:** `demos/gradio_app.py`  
> **Run:** `python demos/gradio_app.py`  
> **URL:** [http://localhost:7860](http://localhost:7860)

---

## 📌 What Is This App?

**MedXplain** is a web-based clinical decision support tool for **Medical Visual Question Answering (VQA)**.  
It allows a radiologist or clinician to:

1. Upload a chest X-ray image
2. Ask a clinical question in plain English
3. Receive an AI-generated answer with supporting evidence (heatmaps, confidence scores, differentials, and full reports)

The app runs entirely in **mock mode by default** — meaning it works and produces realistic outputs without requiring a trained model checkpoint. Once a real model is loaded, mock mode is automatically disabled.

---

## 🗂️ File Structure Overview

```
demos/
└── gradio_app.py          ← Main application file (this guide explains it)
    ├── Imports            ← All dependencies
    ├── _APP_THEME         ← Gradio visual theme (Soft blue)
    ├── _APP_CSS           ← Custom CSS styles
    ├── class MedicalVQADemo
    │   ├── __init__()           ← Loads config, model, all clinical modules
    │   ├── _mock_prediction()   ← Fallback when no real model is loaded
    │   ├── _apply_heatmap()     ← Overlays Grad-CAM on image
    │   ├── _log_session()       ← Records history of interactions
    │   ├── answer_with_heatmap()    ← Tab 1
    │   ├── report_aware_vqa()       ← Tab 2
    │   ├── context_aware_vqa()      ← Tab 3
    │   ├── longitudinal_analysis()  ← Tab 4
    │   ├── differential_diagnosis() ← Tab 5
    │   └── one_click_report()       ← Tab 6
    └── create_demo()      ← Builds the Gradio UI
    └── __main__           ← Launches the server
```

---

## ⚙️ How It Starts Up

When you run `python demos/gradio_app.py`, this happens in order:

1. **`sys.path` is patched** — the `medical_vqa_infrastructure/` root is added to Python's module search path so all sibling packages (`config/`, `inference/`, `clinical_features/`) are importable.
2. **`_APP_THEME` and `_APP_CSS`** are created at module level (Gradio 6 requires this).
3. **`MedicalVQADemo()` is instantiated** inside `create_demo()`:
   - Loads `config/inference_config.yaml`
   - Tries to load the real model → falls back to mock mode if it fails
   - Initialises all 5 clinical feature processors
4. **Gradio `Blocks` UI is built** with 6 tabs.
5. **`_find_free_port()`** scans ports 7860–7870 and picks the first available one.
6. **`app.launch()`** starts the local web server.

---

## 🧠 The `MedicalVQADemo` Class

This is the brain of the application. It holds all loaded components and exposes one method per tab.

### `__init__(config_name="inference_config")`

| Component | What it does |
|-----------|-------------|
| `load_config(config_name)` | Reads `config/inference_config.yaml` via OmegaConf |
| `VQAPredictor` | Real model wrapper — skipped in mock mode |
| `GradCAM` | Generates attention heatmaps — skipped in mock mode |
| `ReportAwareModule` | Embeds prior report text into the question |
| `ContextAwareModule` | Fuses clinical notes, labs, vitals with image features |
| `LongitudinalAnalyzer` | Computes pixel-level diff between two X-rays |
| `DifferentialDiagnoser` | Returns ranked differential diagnoses |
| `ReportDrafter` | Builds a structured Findings + Impression report |
| `session_history` | List that logs every inference call in memory |

---

### `_mock_prediction(image, question)` → dict

Called whenever the real model is unavailable (mock mode).

**Returns:**
```python
{
  "answer": "[MOCK] No acute cardiopulmonary process identified.",
  "confidence": 0.91,
  "attention_weights": np.ndarray(224, 224),  # smooth Gaussian blob
  "latency_ms": 200.0
}
```

The heatmap is generated using **4 random Gaussian blobs** — so it looks like a real attention map, not random noise.

---

### `_apply_heatmap(image, heatmap, alpha=0.5)` → np.ndarray

Overlays a grayscale attention map on top of the original X-ray using OpenCV's **JET colourmap** (blue=low attention, red=high attention).

**Steps:**
1. Resize heatmap to match image dimensions
2. Convert to uint8 and apply `cv2.COLORMAP_JET`
3. Blend with original image using `cv2.addWeighted()`

**Result:** A colour-highlighted X-ray showing which regions the model attended to.

---

### `_log_session(tab, question, answer)`

Records each prediction to `self.session_history` with a timestamp. Useful for auditing or future session export features.

---

## 🖥️ The Six Tabs — Detailed Explanation

---

### Tab 1 — 🔍 VQA & Heatmap

**Purpose:** The core feature — ask any clinical question about an X-ray and see both the answer and a visual explanation.

**Inputs:**
| Field | Type | Description |
|-------|------|-------------|
| Chest X-Ray | Image upload | The X-ray to analyse (numpy format) |
| Clinical Question | Textbox | e.g. *"Is there evidence of pneumonia?"* |
| Temperature | Slider (0.1–1.5) | Controls answer creativity — higher = more varied |

**Outputs:**
| Field | Description |
|-------|-------------|
| Answer | AI answer + temperature + confidence displayed together |
| Grad-CAM Heatmap | The X-ray with attention regions highlighted |
| Latency | How long the inference took (e.g. `⏱️ 0.203 s`) |
| Confidence | Model certainty (e.g. `🎯 91.0%`) |

**Method called:** `answer_with_heatmap(image, question, temperature)`

---

### Tab 2 — 📄 Report-Aware VQA

**Purpose:** Give the AI context from a **previous radiology report** so it can compare findings across time.

**Example use case:**  
*"The previous report noted mild consolidation in the right lower lobe. Has this progressed?"*

**Inputs:**
| Field | Description |
|-------|-------------|
| Chest X-Ray | Current X-ray to analyse |
| Clinical Question | The question to ask |
| Prior Radiology Report | Paste full text of a previous report |

**How it works:**  
`ReportAwareModule.embed_prior_report()` combines the prior report and the new question into a single enriched prompt:
```
[PRIOR REPORT]: <report text> [QUESTION]: <question>
```

**Output:** Answer with a citation showing which prior report was used.

**Method called:** `report_aware_vqa(image, question, prior_report)`

---

### Tab 3 — 🧪 Context-Aware VQA

**Purpose:** Enrich the AI's answer with the patient's full **clinical context** — notes, lab results, and vitals.

**Example use case:**  
*"Given WBC of 14.2 and fever of 38.9°C, is this consolidation consistent with bacterial pneumonia?"*

**Inputs:**
| Field | Example |
|-------|---------|
| Chest X-Ray | Current X-ray |
| Clinical Question | Free-text question |
| Clinical Notes | `65 yo male, fever 38.9°C, productive cough x3 days` |
| Lab Results | `WBC 14.2 K/µL, CRP 85 mg/L, Procalcitonin 0.8` |
| Vitals | `BP 128/82, HR 98, SpO2 94%, RR 22` |

**How it works:**  
`ContextAwareModule.fuse_context()` combines image features with the clinical dictionary. All context fields are also appended to the question prompt as structured tags.

**Output:** Contextualised answer with a summary of the clinical data provided.

**Method called:** `context_aware_vqa(image, question, clinical_notes, lab_results, vitals)`

---

### Tab 4 — 📅 Longitudinal Analysis

**Purpose:** Compare a **current X-ray** against a **prior X-ray** to detect interval changes (e.g., worsening effusion, resolving opacities).

**Inputs:**
| Field | Description |
|-------|-------------|
| Current X-Ray | Most recent imaging study |
| Prior X-Ray | Previous imaging study for comparison |
| Clinical Question | e.g. *"Has the pleural effusion changed?"* |

**How it works:**
1. Both images are resized to 224×224 and converted to grayscale tensors
2. `LongitudinalAnalyzer.compare()` computes `|current - prior|` pixel-wise
3. The difference map is normalised and overlaid on the current X-ray using the heatmap function
4. A **mean change index** percentage is reported

**Outputs:**
- Answer text with change index and confidence
- **Difference heatmap** — bright red/yellow = area of greatest change

**Method called:** `longitudinal_analysis(current_image, prior_image, question)`

---

### Tab 5 — 🩺 Differential Diagnosis

**Purpose:** Generate a **ranked list** of the most likely diagnoses with probability scores and explanations.

**Inputs:**
| Field | Description |
|-------|-------------|
| Chest X-Ray | Image to analyse |
| Clinical Question | e.g. *"What are the most likely diagnoses?"* |
| Top-K Slider (1–5) | How many differentials to show |

**How it works:**  
`DifferentialDiagnoser.get_differentials()` returns a sorted list of diagnoses. Each entry has:
- `diagnosis` — condition name
- `confidence` — probability (0–1)
- `explanation` — one-line radiological reasoning

The output is formatted with **ASCII progress bars** for visual clarity:
```
#1  Pneumonia
    Confidence: █████████████████░░░ 85%
    💬 Opacity in left lower lobe.
```

**Method called:** `differential_diagnosis(image, question, top_k)`

---

### Tab 6 — 📝 One-Click Report

**Purpose:** Generate a complete, **structured radiology report** with a single click — including Findings, Impression, and Recommendation sections.

**Inputs:**
| Field | Description |
|-------|-------------|
| Chest X-Ray | Image to analyse |
| Clinical Question | Report context (e.g. *"Generate a full report"*) |

**How it works:**
1. Runs mock prediction to get confidence
2. Gets differential diagnoses
3. Passes diagnoses to `ReportDrafter.generate_draft()` which formats them as structured findings
4. Appends a Recommendation and clinical disclaimer

**Output format:**
```
══════════════════════════════════════════════════
  🏥 MedXplain Radiology Report
  Generated: 2026-04-17 20:11:00
══════════════════════════════════════════════════

CLINICAL QUESTION
<question>

FINDINGS
- Pneumonia (p=85%): Opacity in left lower lobe.
- Atelectasis (p=10%): Volume loss suggested.

IMPRESSION: Refer to findings above.

RECOMMENDATION
  Primary diagnosis: Pneumonia
  Clinical correlation recommended.
  Follow-up imaging in 4–6 weeks if symptoms persist.
──────────────────────────────────────────────────
⚠️  For clinical decision support only.
──────────────────────────────────────────────────
```

**Method called:** `one_click_report(image, question)`

---

## 🔧 Infrastructure Modules Used

| Module | Class | What it provides |
|--------|-------|-----------------|
| `config/config_loader.py` | `load_config()` | Loads YAML config via OmegaConf |
| `inference/predictor.py` | `VQAPredictor` | Real model inference wrapper |
| `inference/gradcam.py` | `GradCAM` | Grad-CAM attention map generator |
| `clinical_features/report_aware.py` | `ReportAwareModule` | Embeds prior report into question |
| `clinical_features/context_aware.py` | `ContextAwareModule` | Fuses clinical notes with features |
| `clinical_features/longitudinal.py` | `LongitudinalAnalyzer` | Pixel-wise image comparison |
| `clinical_features/differential.py` | `DifferentialDiagnoser` | Ranked diagnosis list |
| `clinical_features/report_draft.py` | `ReportDrafter` | Structured report generation |

---

## 🟡 Mock Mode

Mock mode is **automatically enabled** when the real model fails to load.

| Feature | Mock behaviour |
|---------|---------------|
| Answer | Randomly selected from 5 realistic radiology sentences |
| Confidence | Random float between 0.72 and 0.97 |
| Heatmap | 4 smooth Gaussian blobs placed randomly |
| Latency | Simulated 200 ms sleep |
| Differentials | Fixed 3-item list (Pneumonia, Atelectasis, Pleural Effusion) |
| Report | Structured report built from the mock differentials |

You will see this warning in the terminal when mock mode is active:
```
⚠️  Using mock mode: Real model not yet wired – using mock mode.
```

---

## 🚀 How to Run

```powershell
# From medical_vqa_infrastructure/
python demos/gradio_app.py
```

Then open **http://localhost:7860** in your browser.

### Optional flags (edit the launch block in the file):

| Option | Default | Description |
|--------|---------|-------------|
| `server_port` | Auto (7860–7870) | Port to serve on |
| `share=True` | `False` | Creates a public Gradio link |
| `debug=True` | `False` | Shows detailed tracebacks in browser |
| `max_threads` | `4` | Max concurrent requests |

---

## ⚠️ Clinical Disclaimer

This application is for **research and decision-support only**.  
All AI-generated findings must be reviewed by a qualified radiologist.  
MedXplain does **not** replace professional medical judgment.

---

## 🔄 Adding a Real Model

To replace mock mode with a real model:

1. Edit `MedicalVQADemo.__init__()` in `gradio_app.py`
2. Replace the `raise NotImplementedError(...)` line with your actual model loading code:
```python
from models.vqa_model import MedVQAModel
model = MedVQAModel.from_pretrained("path/to/checkpoint")
self.predictor = VQAPredictor(model, device="cuda")
self.gradcam = GradCAM(model, target_layer=model.vision_encoder.layer4)
self.mock_mode = False
```
3. Update `_mock_prediction()` calls in each tab method to use `self.predictor.predict()` instead.
