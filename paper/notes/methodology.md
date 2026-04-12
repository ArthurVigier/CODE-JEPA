# Methodology Notes

- Record every kill/go decision with the run id and config hash.
- Keep failed ablations and negative results in versioned JSON under `artifacts/`.
- Use `paper/results_manifest.json` as the canonical source for figures and tables.
- Prefer local logging: TensorBoard for train curves, JSON for summaries, PNG for paper figures.
