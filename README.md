# ScrapLess AI

**Turn material waste into measurable savings.**

ScrapLess AI is a local Streamlit prototype for rectangular sheet-cutting decisions. It compares a deterministic First Fit Decreasing shelf baseline with a MaxRects Best Short Side Fit heuristic, reports area-based waste and illustrative cost, scores scrap risk using a RandomForest trained on synthetic proxy labels, and demonstrates experimental OpenCV contour detection.

## Features

- Editable built-in order or CSV upload.
- Validation of identifiers, dimensions, quantities, finite values, and limits.
- Configurable sheet dimensions, currency, cost, spacing/kerf, rotation, and sheet cap.
- Geometry-based layouts with explicit unplaced parts and validation.
- KPI comparison and placement CSV download.
- Synthetic ML evaluation and feature importance disclosure.
- Experimental image contour detection with pixel measurements.

## Requirements and quickstart

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Run tests with `python -m pytest tests -q`.

## CSV schema

Required columns are `part_id`, `length`, `width`, and `quantity`. Optional columns are `priority` and `material`.

```csv
part_id,length,width,quantity,priority,material
PANEL_A,600,400,4,High,Aluminium
```

Legacy `part_width`/`part_height` names are accepted and normalized for convenience.

## Architecture

```text
app.py                         Streamlit interface
scrapless/schemas.py           Input validation and normalization
scrapless/packing.py           FFD, MaxRects BSSF, and geometry validation
scrapless/metrics.py           Area, utilization, cost, and comparison metrics
scrapless/ml_model.py          Synthetic RandomForest risk demonstration
scrapless/vision.py            OpenCV contour demonstration
scrapless/reporting.py         CSV export helpers
data/generate_synthetic_data.py CLI generator
tests/                         pytest coverage
```

FFD sorts instances by descending area and fills shelves from the lower-left. MaxRects maintains free rectangles and chooses the candidate with the smallest short-side leftover, with deterministic tie breaks. Spacing is reserved in feasibility rectangles and is not subtracted from physical part area, so kerf is not counted twice.

## Metrics

`sheet area = sheet width × sheet length`; `utilization = placed part area / stock area × 100`; `unused area = stock area − placed part area`; `estimated cost = sheet count × configured sheet cost`. Unused area is a proxy for waste and can contain reusable offcuts or geometric gaps. Costs and savings are illustrative, not factory-certified.

## Synthetic data and ML

Generate reproducible model features with:

```powershell
python data\generate_synthetic_data.py --rows 5000 --seed 42 --output data\synthetic_scrap_risk_dataset.csv
```

The RandomForest learns transparent synthetic proxy labels. Its score is not a validated industrial probability and never replaces the geometric layout result.

## Docker

```powershell
docker build -t scrapless-ai .
docker run --rm -p 8501:8501 scrapless-ai
```

Open `http://localhost:8501`.

## Demonstration workflow

1. Load the sample order and explain the required quantities.
2. Set sheet size, cost, spacing, and rotation policy.
3. Run analysis and compare sheets, utilization, unplaced parts, and cost.
4. Inspect both layouts and download the placement CSV.
5. Show the synthetic risk score and disclose its limitations.
6. Upload a clean image in the vision tab and explain that detections are pixel-space candidates only.

## Limitations and future work

The heuristics are not mathematically optimal, material grain, saw sequencing, nesting across irregular shapes, and production constraints are not modeled. Vision is sensitive to lighting, shadows, touching parts, perspective, and reflective surfaces. Future work could add calibrated dimensions, kerf-aware tool paths, grain constraints, reusable offcut inventory, and validated factory data.

This project uses synthetic demonstration data and illustrative savings. It does not claim zero waste, optimal cutting, production certification, or guaranteed savings. No API key or external service is required.
