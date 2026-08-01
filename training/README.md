# Phase-by-phase FNO/PINO training

Run from this directory after activating the simulation virtual environment. Replace
`DATASET` with the Stage 1 dataset root.

```bash
python phase_01.py DATASET
python phase_02.py DATASET
python phase_03.py DATASET --output-dir DATASET/models/fno_baseline
python phase_04.py DATASET --output-dir DATASET/models/fno_search
python phase_05.py DATASET DATASET/models/fno_baseline/best.pt \
  --output-dir DATASET/models/pino
python phase_06.py DATASET \
  --fno DATASET/models/fno_baseline/best.pt \
  --pino DATASET/models/pino/best.pt \
  --output-dir DATASET/evaluation
```

Start Phase 01 with `--limit 3` to smoke-test preparation. See `phase.pdf` for the
theory, derivation, algorithm, parameter choices, and tuning protocol.
