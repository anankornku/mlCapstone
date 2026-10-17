# Kidney Stone Second Reader

A detector that boxes kidney stones on coronal non-contrast CT slices, so the reading doctor misses fewer of them. It proposes; the doctor decides.

On 120 slices from 39 patients the model never saw, it finds **93.1% of stones** (95% CI 88–98%) with **0.13 false alarms per slice**. Red boxes alone find 87.9% of stones with 1 false red box in 120 slices.

## Run it

1. Upload `kidney.zip` to Google Drive at `MyDrive/kidney_capstone/kidney.zip`.
2. Open `kidney_stone_second_reader_v2.ipynb` in Google Colab and select a T4 GPU runtime.
3. Run all. The first run trains for about 30 minutes; later runs load the saved weights from Drive.
4. The last cell starts the Gradio demo and prints a `gradio.live` link. Try the slices in `demo_images/`, which come from patients the model never trained on.

## Files

| Path | What it is |
|---|---|
| `kidney_stone_second_reader_v2.ipynb` | Final notebook with outputs: data audit, split by patient, YOLOv8s training at 960 px, evaluation, error analysis, demo |
| `kidney_stone_second_reader.ipynb` | First run with outputs, kept for comparison (also trains a model on the published split) |
| `dataset/kidney/` | The dataset in YOLO format |
| `kidney.zip` | The same dataset, zipped for Colab |
| `demo_images/` | The 120 test slices; each file name gives the labeled stone count |
| `GLOSSARY.md` | Project terms: slice, scan session, hit, false alarm, show and red thresholds |
| `docs/adr/` | Why results use a split by patient instead of the published split |

## Method in brief

- **Split by patient.** Slice UIDs carry the acquisition time, which groups the 778 slices into 256 CT examinations. The published split puts slices of 93 examinations on both sides, so we split 70/15/15 by examination.
- **Doctor-oriented scoring.** A box hits a stone when its centre falls inside the stone or it overlaps it by IoU ≥ 0.1. We report stone sensitivity, false alarms per slice, precision, all-stones-found and FROC, with mAP50 for comparison.
- **Two thresholds chosen on validation.** The show threshold finds 90% of validation stones; the red threshold is where boxes find a stone 90% of the time.

## Limits

Every slice contains a stone, so specificity is unknown. The data comes from one centre and one scanner, in 2D slices without mm spacing. The model was evaluated on its own, not with doctors; showing a clinical benefit needs a reader study. Not for clinical use.

## Data

[TEZ_ROI_AUG v3](https://universe.roboflow.com/tez-nwkf5/tez_roi_aug/dataset/3) by a Roboflow user, CC BY 4.0.
