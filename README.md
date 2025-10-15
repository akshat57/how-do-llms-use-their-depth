# How Do LLMs Use Their Depth? — Code & Figures

Lightweight code to reproduce analyses/plots for our paper on **layer-wise prediction dynamics** in LLMs (POS, multi-token facts, and option-constrained tasks). The workflow: **trace per-layer logits → write CSVs → make figures**.

---

## System Figure

```mermaid
flowchart LR
  A[Prompts & Datasets\n(POS / Facts / Downstream)] --> B[Model Forward Pass]
  B --> C[TunedLens / LogitLens\nper-layer logits]
  C --> D[Per-layer Metrics\n(ranks, earliest-layer ≤k)]
  D --> E[CSV Exports\nout/data/*.csv]
  E --> F[Plots\nout/plots/*.png]
