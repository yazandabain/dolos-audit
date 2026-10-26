# Compute resources

The saved Dolos archive contains 20 generation runs: 14 CUDA runs recording an
NVIDIA L4 and six CPU runs. This includes the original eight GPU runs and the
later contrastive, rank-reconstruction, and steering experiments. Standalone CPU
weight scans, activation measurements, and next-token counterfactuals are
separate diagnostics.

## Hardware and execution limits

| Component | Configuration |
| --- | --- |
| GPU generation | One NVIDIA L4 per run; BF16 model parameters and CUDA execution |
| Hosted process bounds | At most two CPU cores and 8 GiB RAM; 3,600-second lifetime |
| Persistence | No persistent model volume; checkpoints and outputs saved as experiment artifacts |
| Projection spectra | CPU, four threads; fixed rank-16 randomized SVD with four power iterations |
| Activation-difference lens | CPU BF16 forwards, four threads; a 900-second diagnostic limit |
| Analysis reproduction | Standard-library analysis and the locked development environment; no model weights or GPU |

The generation limits and decoding settings are stored in each run's
`config.json`. Typical output caps are 128 tokens in the original audit, 300 in
contrastive discovery, and 192 in later rank/steering exploration. Smoke tests
use smaller caps. Exact configurations take precedence over this overview.
`run.json` records the actual device, precision, runtime versions, loading checks,
and timestamps. Model revisions and expected weight hashes are pinned in
[`configs/models.json`](../configs/models.json).

## Timing and storage

Observed CPU elapsed times are retained in the
[projection-spectrum report](../results/delta-spectrum-001.json) and
[activation report](../results/activation-lens-001.json). These are timings for the
recorded hardware and software, rather than hardware-normalized speed benchmarks.
Generation metadata and atomic batch files retain their own elapsed times.
Loading, inference, validation, and data movement can contribute different timing
components; batch timings do not measure total GPU occupancy.

The two checkpoints require approximately 9.3 GB of download/cache space, plus
runtime dependencies and experiment outputs. Model weights remain outside Git.
The public mean-activation JSON is approximately 2.4 MB and reconstructs a
492,952-byte derived activation artifact. It contains activation summaries rather
than model parameters.

The [model-diffing guide](model-diffing.md) separates saved-artifact checks from
optional new model computation and gives fresh output paths for reproduction.


## Continuous integration

For a public repository, the CPU workflow selects standard GitHub-hosted
`ubuntu-latest`. It runs on pushes and manual dispatches with a ten-minute job
limit. Local reproduction commands remain available independently of CI.
