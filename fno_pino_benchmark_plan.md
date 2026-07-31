# FNO vs PINO Full-Space Benchmark Plan

## 1. Objective

Build a reproducible, matched comparison of a time-conditioned Fourier Neural
Operator (FNO) and Physics-Informed Neural Operator (PINO) for predicting the
particle-velocity fields `vx(x,y,t)` and `vy(x,y,t)` produced by a Gaussian point
source in a homogeneous two-dimensional full space.

The work proceeds in two gated stages:

1. One-parameter sweep: source vertical position `y0` only.
2. Two-parameter sweep: Gaussian moment amplitude `M0` and period/duration `T`.

Stage 2 begins only after the complete Stage 1 data, training, inference, and
evaluation pipeline passes its acceptance checks.

## 2. Findings from the previous implementation

The existing `trained-FNO` and `trained-PINO` experiments provide a useful starting
point:

- The pilot used 96 training, 16 validation, and 32 interpolation-test cases.
- Targets were `vx` and `vy`; source location, duration, coordinates, and query time
  were encoded as model inputs.
- The enhanced FNO used spatial/source channels, normalized time, temporal Gaussian
  forcing, Fourier time features, and boundary distance.
- PINO warm-started from FNO and added the isotropic velocity-form Navier residual.
- Test relative-L2 errors were about 2.70% for FNO and 2.63% for PINO.
- PINO reduced the matched interior PDE residual by about 9.4%, but improved field
  error only modestly and did not resolve the large OOD error.

Reusable components include manifest handling, restart-safe simulation execution,
velocity extraction, enhanced FNO architecture, active-frame sampling, evaluation
metrics, and the PINO residual. Hard-coded grid spacing, source width, material
constants, time spacing, and boundary masks must be removed or derived from dataset
metadata.

## 3. Physical and numerical specification

### 3.1 Domain convention

Use a 15 km x 15 km total computational domain with the PML included inside that
extent on all four sides:

- Total computational domain: 15 km x 15 km.
- PML: 2 km on left, right, top, and bottom.
- PML-free physical interior: 11 km x 11 km.
- Computational coordinates: `x = [-7.5, 7.5] km`, `y = [0, 15] km`.
- PML-free coordinates: `x = [-5.5, 5.5] km`, `y = [2, 13] km`.
- Physical center/source default: `(x0, y0) = (0, 7.5) km`.
- Objective/training window: centered 10 km x 10 km,
  `x = [-5, 5] km`, `y = [2.5, 12.5] km`.

The objective window remains 0.5 km inside the PML-free interior. Sources at the
far-OOD limits lie on the objective edge and 0.5 km from the PML, which must be
reported as a deliberately difficult extrapolation condition.

### 3.2 Solver configuration

- Geometry: Cartesian, homogeneous, elastic, full space.
- Boundary treatment: PML on all four sides; no free surface.
- PML physical width: 2 km = 20 grid points.
- Pilot spatial resolution: 0.1 km (100 m). Retain 50 m as the later
  high-resolution production target after the pilot passes.
- Time step: 0.002 s.
- End time: 5.0 s.
- Solver steps: 2,500.
- Snapshot interval: 0.02 s (`iplot=10`), giving 250 frames per case.
- Discretization: sixth-order central diagonal-norm SBP, matching the benchmark.
- Source: Gaussian point source, `x0=0`, `t0=1.0 s`.
- Material baseline: `cp=6.0 km/s`, `cs=3.464 km/s`, `rho=2.6702 g/cm^3`.
- Simulation backend: CUDA/H100 for dataset production; validate representative
  cases against the CPU solver before bulk generation.
- Precision: solver `float64`; prepared ML tensors `float32` after verification.

The pilot computational grid has 151 points per axis (76 points in each horizontal
block, with the interface represented in both blocks). The centered objective
window has 101 x 101 points and excludes all PML cells.

## 4. Dataset organization and schema

Use one canonical dataset for both FNO and PINO:

```text
waveqlab2d_simulation/
  fno-pino-benchmark/
    stage1-y0/
      input/{train,validation,test}/
      raw/{train,validation,test}/
      prepared/{train,validation,test}/
      manifests/
      status/
      models/{fno,pino}/
      evaluation/
    stage2-amplitude-period/
      ...same layout...
```

The manifest is the source of truth and records:

- `case_id`, split, seed, and design index;
- `x0_km`, `y0_km`, `M0`, `T_s`, `t0_s`;
- physical, computational, objective, PML, grid, and time parameters;
- input filename, raw result path, prepared result path, and checksums;
- backend, precision, solver commit, run ID, completion status, and runtime.

Prepared cases contain only:

- `velocity`: `[time, x, y, 2]`, ordered as `vx`, `vy`;
- `time`, `x`, and `y` coordinates;
- source parameters and normalization metadata.

Prefer chunked HDF5 with compression for the canonical prepared dataset. NPZ can be
supported for small pilots, but a single monolithic NPZ per large split should be
avoided. Store each case independently or use chunked datasets so interrupted runs
are recoverable.

## 5. Stage 1: one-parameter `y0` sweep

### 5.1 Fixed and variable parameters

- Fixed `x0 = 0 km`.
- Fixed `M0 = 0.02824`.
- Fixed `T = 0.2 s`.
- Training distribution: `y0 = 7.5 +/- 3 km`, or `[4.5, 10.5] km`.
- OOD evaluation extends to `y0 = 7.5 +/- 5 km`, or `[2.5, 12.5] km`.

The solver evaluates its spatial Gaussian continuously, so off-grid `y0` values are
valid. Nevertheless, 1,500 samples in one dimension are strongly correlated. Treat
Stage 1 as a pipeline and interpolation benchmark, and report learning curves at
smaller subsets to quantify whether all 1,500 simulations add value.

### 5.2 Split

Use a deterministic, immutable split totaling 1,500 cases:

- Training: 1,050 ID cases with offsets in `[-3,3] km`.
- Validation: 100 ID plus 50 near-OOD cases with absolute offsets in `(3,4] km`.
- Test: 150 ID plus 150 far-OOD cases with absolute offsets in `(4,5] km`.

Generate a one-dimensional scrambled Sobol or stratified design independently
inside each interval. Assign splits before simulation with a fixed seed. Test
locations must be held out, not copied from training. Include `y0=7.5 km` explicitly
in the ID test set and `y0=2.5,12.5 km` in the OOD test set. Record nearest-training-
point distance for every validation/test case so extrapolation difficulty is clear.

Also define nested training subsets of 75, 150, 300, 600, and 1,050 cases. Train
both models on identical subsets to generate accuracy-versus-data learning curves.

To prioritize maximum OOD accuracy in the pilot, retain native 100 m spatial
resolution without further spatial downsampling and keep the full 0.02 s saved-time
resolution. Use edge-weighted training sampling near offsets
of +/-3 km, tune hyperparameters on the mixed ID/near-OOD validation split, train at
least three seeds, and evaluate a seed ensemble in addition to individual models.
Do not include the far-OOD test band in normalization, early stopping, or tuning.

### 5.3 Stage 1 gate

Proceed to Stage 2 only when:

- all manifests are deterministic and contain no duplicate parameter rows;
- representative CPU/GPU solver outputs agree within the established tolerance;
- every raw archive passes shape, finite-value, coordinate, and metadata checks;
- prepared `vx`/`vy` fields reproduce the raw objective-window values exactly;
- FNO and PINO train from the same splits, normalization, architecture, and seeds;
- evaluation produces complete field, physics, timing, and learning-curve reports.

## 6. Stage 2: amplitude-period sweep

Stage 2 fixes the source at the physical center and varies two parameters:

- `x0 = 0 km`, `y0 = 7.5 km`.
- Suggested `M0` range: `[0.01412, 0.04236]` (0.5x to 1.5x baseline).
- Suggested `T` range: `[0.15, 0.35] s`, matching the previous in-distribution pilot.

Confirm these ranges before generating Stage 2. Use a two-dimensional scrambled
Sobol design rather than a tensor grid, with the same 1,200/150/150 split. Include
the baseline `(M0,T)=(0.02824,0.2)` in the test set.

Add explicit OOD evaluation cases outside, but close to, the training rectangle;
keep them separate from the requested 1,500 ID samples. Recommended OOD boundaries
are `M0` at 0.4x and 1.6x baseline and `T` at 0.12 and 0.40 s, subject to solver
stability and physical relevance.

Because the elastic system is linear in source amplitude, Stage 2 must report both
raw-amplitude errors and amplitude-normalized errors. Include a linear-scaling
baseline; an operator model should outperform it mainly in the period dimension.

## 7. Scripts to implement in `waveqlab2d_simulation`

### 7.1 `generate_fno_pino_inputs.py`

Responsibilities:

- support `--stage y0` and `--stage amplitude-period`;
- generate deterministic train/validation/test designs;
- write full-space WaveQLab2D input decks and CSV/JSON manifests;
- set the external-PML computational geometry and objective output window;
- validate counts, uniqueness, ranges, PML width, grid dimensions, output cadence,
  and split isolation;
- provide `--dry-run`, `--seed`, `--count`, and `--overwrite` controls;
- never silently overwrite an existing manifest with a different design.

Implement Stage 1 first. Keep Stage 2 behind an explicit stage option and do not
generate it until the Stage 1 gate passes.

Stage 1 is implemented with a default no-write validation command:

```bash
python generate_fno_pino_inputs.py --stage y0 --dry-run
```

After reviewing the dry-run summary, generate the decks on Punakha with:

```bash
python generate_fno_pino_inputs.py --stage y0
```

Generated input/raw/prepared/status directories are excluded by `.gitignore`, while
the generator and plan remain trackable.

### 7.2 `generate_fno_pino_results.py`

Responsibilities:

- read the manifest rather than discovering cases by filename;
- execute the installed solver through the simulation-local runtime root;
- default to CUDA and support controlled CPU verification cases;
- be restart-safe and skip only archives that pass integrity validation;
- maintain an append-only JSONL status log;
- capture stdout/stderr per case, elapsed time, solver run ID, and checksums;
- stop cleanly on low disk space or repeated failures;
- support `--split`, `--limit`, `--jobs`, `--backend`, and `--resume`;
- write raw NPZ or streaming HDF5 results, then extract/crop only `vx` and `vy` into
  the 10 km x 10 km objective window;
- verify 250 frames, 101 x 101 points, two fields, finite values, and monotonic time;
- optionally remove bulky raw five-field output only after prepared-data checksum
  verification and only with an explicit cleanup flag.

Start with `--limit 3` smoke tests, then 15 cases, then the full Stage 1 campaign.

### 7.3 Follow-on scripts

- `prepare_fno_pino_dataset.py`: crop, field-select, normalize, checksum, and build
  chunked datasets without data leakage.
- `train_fno_benchmark.py`: metadata-driven time-conditioned FNO training.
- `train_pino_benchmark.py`: same architecture and initialization plus Navier
  residual fine-tuning.
- `evaluate_fno_pino.py`: matched accuracy, residual, timing, and plotting.
- `benchmark_fno_pino_inference.py`: warmed-up batch-1 and batched inference timing
  on the H100, including complete 250-frame trajectory generation.

## 8. Model inputs and outputs

### 8.1 Common supervised task

For a case parameter vector and arbitrary query time `t`, predict:

```text
[source representation, x, y, t, parameters] -> [vx(x,y,t), vy(x,y,t)]
```

Stage 1 parameters contain normalized `y0`; Stage 2 parameters contain normalized
`M0` and `T`. Derive the spatial source width from solver metadata (`2*dx = 0.2 km`)
instead of retaining the old hard-coded 0.2 km value.

Use identical input channels, velocity normalization, architecture width/modes,
training batches, optimizer budget, random seeds, and checkpoint selection for the
matched FNO/PINO comparison. PINO should warm-start from each matching FNO seed, but
report its additional fine-tuning time and total training cost.

### 8.2 PINO residual

- Use the full-space isotropic Navier velocity residual.
- Read `cp`, `cs`, `dx`, `dy`, and saved `dt` from metadata.
- Use consecutive saved frames for second time derivatives.
- Apply the residual only inside the objective domain.
- Mask the Gaussian source support, initially a radius of `4 sigma = 0.4 km`.
- Verify residual units, axis ordering, and derivatives on analytic and solver data.
- Sweep physics weights only after the baseline `0.01` run is reproducible.

## 9. Fair FNO/PINO evaluation

Run at least three shared seeds. Report median and dispersion for:

- global, `vx`, and `vy` relative L2;
- RMSE and normalized RMSE;
- spatial correlation;
- active-frame median and p90 relative L2;
- peak-amplitude error and wave-arrival-time error;
- error versus time and versus swept parameter;
- interior Navier residual on truth, FNO, and PINO;
- training time, peak GPU memory, parameter count, and checkpoint size;
- single-frame latency and complete 250-frame trajectory latency;
- speedup versus the H100 numerical solver;
- learning curves versus 75/150/300/600/1,050 training cases.

Use paired per-case statistical comparisons because FNO and PINO share the same test
cases. Do not select the winning model from training loss alone. Select checkpoints
using validation field relative L2 and report physics residual independently.

## 10. Storage and runtime controls

At 101 x 101 points, 250 frames, two `float32` fields require about 20.4 MB per case
before compression, or roughly 30.6 GB for 1,500 cases. Raw five-field `float64`
archives would be much larger. Therefore:

- retain only objective-window `vx` and `vy` for routine ML use;
- use chunked compression and per-case checksums;
- estimate free-space requirements before generation;
- run a compression/throughput pilot before choosing NPZ versus HDF5;
- keep raw solver files only until prepared outputs pass validation;
- never combine all cases into one failure-prone monolithic file.

## 11. Execution order

1. Confirm the external-PML domain convention and Stage 2 parameter ranges.
2. Implement and test the Stage 1 input generator.
3. Generate a three-case Stage 1 smoke manifest.
4. Implement the restart-safe simulation/result generator.
5. Compare one CPU/H100 pair and validate crop/field extraction.
6. Run 15 H100 cases and measure runtime, compression, and storage.
7. Freeze manifests and generate all 1,500 Stage 1 cases.
8. Adapt the previous FNO pipeline to metadata-driven 100 m pilot full-space data.
9. Train/evaluate FNO across shared seeds and nested data subsets.
10. Validate the PINO residual, warm-start, train, and evaluate PINO identically.
11. Publish the matched Stage 1 report and apply the gate.
12. Confirm `M0`/`T` ranges, then generate and run Stage 2.

## 12. Numerical-solver performance benchmarks

Use the same physical input for CPU and GPU performance tests, but generate derived
benchmark decks that retain the full computational work while reducing saved output
to a tiny source-centered window and one final frame. This prevents hundreds of
gigabytes of training-data I/O from obscuring solver throughput.

`benchmark_cpu_scaling.py` performs:

- one simulation in expected fastest-to-slowest order at 32, 16, 8, 4, 2, and 1
  Numba threads;
- fixed-32-thread throughput tests with these `(threads per simulation,
  concurrent simulations)` pairs, also in expected fastest-to-slowest order:
  `(8,4)`, `(4,8)`, `(2,16)`, `(1,32)`;
- per-process logs, CSV results, JSON results, speedup, parallel efficiency, batch
  wall time, and simulations per hour.

`benchmark_gpu_single.py` runs the identical simulation on one CUDA GPU, defaults to
three repetitions, and reports median/mean/min/max solver time. Compare its median
against the CPU single-simulation medians. Run both scripts inside an exclusive
SLURM allocation, record CPU affinity and NUMA policy, and avoid simultaneous user
workloads.
