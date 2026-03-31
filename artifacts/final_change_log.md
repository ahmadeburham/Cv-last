# Final Change Log

## Added
- `infer_id.py`: CLI alias entrypoint (`python infer_id.py ...`) mapped to pipeline runner.
- `.gitignore`: ignore generated runtime binaries and cache artifacts.

## Updated
- `run_pipeline.py`:
  - Added dependency-safe execution fallback.
  - Ensures per-image `result.json` is always written with explicit failure reason.
  - Preserves evaluation output generation in constrained environments.
- `artifacts/audit_summary.md`: updated with reused/replaced components and observed blockers.
- `README.md`: documented dependency fallback behavior.
- `artifacts/eval/*`: regenerated from real executed runs.

## Binary output handling
- Removed generated `.jpg/.png` runtime artifacts from git tracking.
- Runtime binaries are still produced under `output/` and `artifacts/runs_real/` during execution, but are excluded from commit per project rule.
