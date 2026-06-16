# Results Artifact Convention

Every reported number in this project traces to a committed raw artifact in this directory.
If a number appears in docs, writeups, or README, the raw artifact that produced it lives here.

## Rules

1. **Raw artifacts are committed, not regenerated.** Re-runs get a versioned name (`_v2`, etc.); the original is never overwritten.
2. **Naming:** `day{N}_{type}.{ext}` — e.g. `day1_env.json`, `day3_timing.json`, `day4_timeline.nsys-rep`, `day4_kernel.ncu-rep`.
3. **Large binary artifacts** (`.nsys-rep`, `.ncu-rep`) are committed directly. Git LFS is required if any single artifact exceeds ~50 MB; add a `.gitattributes` entry at that point. Do not substitute a summary for the raw file.
4. **Never manually edit a raw artifact.** If a re-run is needed, rename the old artifact (`_v1`) before committing the new one.
5. **Timing artifacts must include the full distribution:** mean, p50, p95, p99, std, min, max, plus batch size, input shape, device, and driver/CUDA/PyTorch versions. Reporting only mean is a violation.
6. **env.md is the provenance anchor.** Every timing artifact references the `timestamp_utc` from `day1_env.json` so numbers trace to a verified environment, not an inherited assumption.
