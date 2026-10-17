# Evaluate by scan session, not by the provided Roboflow split

The provided train/valid/test split mixes slices from the same CT examination: 93 of 256 scan sessions (inferred from the acquisition timestamps in the slice UIDs) span more than one split, and 68 of 123 test slices are leaked slices. We report the headline result on a split by scan session, and use the provided split only to show how much leakage inflates the score, because a doctor's question is "how does it do on a new patient", which the provided split cannot answer.

## Update after the first run

The first run also trained a model on the provided split. Leaked test slices scored mAP50 0.781 vs 0.711 for clean ones, but stone sensitivity was equal (0.90) and the leaked slices hold more stones per slice (2.1 vs 1.4), so the measured leak effect is small and confounded with difficulty. We dropped that model: only the session split answers "how does it do on a new patient", and the provided split now appears only in the data audit.

## Consequences

Sessions are inferred, not confirmed patient IDs: two examinations of one patient on different days would still land in different sessions.
