# How Do LLMs Use Their Depth? — Code & Figures

Lightweight code to reproduce analyses/plots for our paper on layer-wise prediction dynamics in LLMs (POS, multi-token facts, and option-constrained tasks). The workflow: trace per-layer logits → write CSVs → make figures.

---

## System Figure

```mermaid
flowchart LR
  A["Prompts & Datasets<br/>(POS / Facts / Downstream)"] --> B["Model Forward Pass"]
  B --> C["TunedLens / LogitLens<br/>per-layer logits"]
  C --> D["Per-layer Metrics<br/>(ranks, earliest-layer <= k)"]
  D --> E["CSV Exports<br/>out/data/*.csv"]
  E --> F["Plots<br/>out/plots/*.png"]
```

## **Environment Setup**




