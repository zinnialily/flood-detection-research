# Flood Detection from Street-Level Imagery

**Research website** for an ongoing study on binary flood detection using transfer learning and hard negative mining, targeting *Computers & Geosciences* (Elsevier, ISSN 0098-3004).

**Live site:** https://zinnialily.github.io/flood-detection-research/

---

## What's on the site

- **The problem** — why street-level flood detection is hard and why swimming pools break naive classifiers
- **Research journey** — Phase 1 (CIBB 2026 conference paper) → Phase 2 (bug discovery) → Phase 3 (corrected pipeline)
- **Bugs found** — double-rescaling preprocessing bug, confounded architectural comparison, test-set threshold tuning
- **Corrected methodology** — six-step pipeline with partition integrity assertions, val-only threshold selection, PR-AUC as primary metric
- **Preliminary results** — CIBB results shown with caveats; corrected numbers pending GPU runs
- **What's next** — planned experiments in execution priority order
- **Code & reproducibility** — pipeline scripts, notebook execution order, reproducibility commitments

## Code repository

The actual training code lives at: https://github.com/CRIS-Hazard/imagevalidation

## Tech

Single-file static site — no build step required. Uses:
- [Tailwind CSS](https://tailwindcss.com/) (CDN)
- [Chart.js](https://www.chartjs.org/) (CDN)
- [Mermaid](https://mermaid.js.org/) (CDN, ESM)

## License

MIT
