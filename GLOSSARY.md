# Kidney Stone Second Reader

A detector that flags kidney stones on coronal non-contrast CT for a radiologist or ER doctor to confirm, and an honest account of how well it does that.

## Imaging

**Slice**:
One 2D coronal non-contrast CT image of the abdomen, cropped around the kidneys.
_Avoid_: Image, scan, case, sample

**Scan Session**:
All slices acquired in one CT examination, inferred from acquisition timestamps; the closest available stand-in for a patient.
_Avoid_: Patient (no patient IDs exist), study, case

**Stone**:
A single labeled kidney stone within a slice; one slice can hold several.
_Avoid_: Object, lesion, calculus, positive

## Evaluation

**Hit**:
A detection whose center falls inside a labeled stone's box, or that overlaps it by IoU ≥ 0.1.
_Avoid_: True positive (ambiguous with the IoU ≥ 0.5 definition)

**Missed Stone**:
A labeled stone with no hit on it.
_Avoid_: False negative

**Duplicate**:
A hit on a stone that a more confident detection already found.
_Avoid_: Double detection, extra box

**False Alarm**:
A detection that is not a hit on any labeled stone.
_Avoid_: False positive, FP

**Show Threshold**:
The confidence below which boxes are hidden, chosen to find 90% of validation stones.
_Avoid_: Cutoff, conf

**Red Threshold**:
The confidence above which boxes find a stone at least 90% of the time on validation; boxes between it and the show threshold are orange ("look closer").
_Avoid_: Review threshold, REVIEW_BELOW

**Leaked Slice**:
A test slice whose scan session also has slices in the training set.
_Avoid_: Contaminated slice, duplicate

**Clean Slice**:
A test slice whose scan session has no slices in the training set.
_Avoid_: Unseen slice, held-out slice

**Operating Point**:
The confidence threshold chosen for clinical use, stated as stone sensitivity and false alarms per slice.
_Avoid_: Cutoff, threshold (alone)

## Product

**Second Reader**:
The role of the system: it proposes stones for the reading doctor to accept or reject, and never makes the diagnosis itself.
_Avoid_: Diagnostic tool, AI radiologist, classifier
