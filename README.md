# How Do LLMs Use Their Depth? — Code to Generate Figures

Minimal instructions to reproduce the paper’s figures from code. Workflow: **trace per-layer logits → write CSVs → plot** (POS, multi-token facts, downstream options; plus frequency buckets and decision flips).

---

## System Figure
~~~mermaid
flowchart LR
  A["Prompts & Datasets<br/>(POS / Facts / Downstream)"] --> B["Model Forward Pass"]
  B --> C["TunedLens / LogitLens<br/>per-layer logits"]
  C --> D["Per-layer Metrics<br/>(ranks, earliest-layer <= k)"]
  D --> E["CSV Exports<br/>out/data/*.csv"]
  E --> F1["Earliest-layer vs k"]
  E --> F2["Decision flips (intermediate → final)"]
  E --> F3["Frequency buckets (Top-10/100/1000/Rest)"]
  F1 --> G["Plots<br/>out/plots"]
  F2 --> G
  F3 --> G
~~~

---

## Environment Setup
~~~bash
# Conda (recommended)
conda create -n depthstudy python=3.10 -y
conda activate depthstudy

# Core packages
pip install -U torch transformers tuned-lens pandas numpy matplotlib seaborn tqdm

# Plotly outputs (for frequency/flip figures)
pip install -U plotly kaleido

# POS tagging (optional, for POS case study)
pip install spacy && python -m spacy download en_core_web_sm
~~~

---

## Create Output Folders
~~~bash
mkdir -p out/data out/plots out/plot_data out/Layer_vs_Rank tokenizer_analysis
~~~

---

## Generate CSVs (run what you need)

**POS case study** → `out/data/gpt2-xl_{POS}_logit.csv`
~~~bash
python analyze_knowledge_layers_pos.py
~~~

**Facts (MQuAKE) for earliest-layer curves** → `out/data/fact_gpt2-xl_REASONING_logit.csv`
~~~bash
python analyze_knowledge_layers_facts.py
~~~

**Downstream option-constrained tasks** (MMLU/SST/NLI/MRPC) → per-task CSVs
~~~bash
python downstream_task_options_analysis.py
~~~

---

## Make Plots

**Facts — earliest layer vs rank (single figure)**
~~~bash
# Reads: out/data/fact_gpt2-xl_REASONING_logit.csv
python -u plot_facts2.py
# Output: out/plots/gpt2-xl_mquake_fact_logit.png
# (If your suffix differs, open plot_facts2.py and set: postfix="logit" or edit CSV_PATH)
~~~

**Frequency buckets across layers (Top-10/100/1000/Rest)**  
(Generates counters and saves stacked bar figures; uses Plotly/Kaleido.)
~~~bash
python top_ranked_token_at_each_layer.py
# Outputs (examples): tokenizer_analysis/gpt2-xl_*_final_plotly_TunedLens.png (and/or .html)
# Uses top_ranked_plot.py under the hood for pretty 100% stacked bars.
~~~

**Decision flips: how often early top-1 changes by final layer**  
(Bar/stacked views of “flipped / total” by frequency bucket.)
~~~bash
python decision_flip_intermediate_and_final_layer_comparison.py
# Outputs (examples): tokenizer_analysis/*flip*.png
# Uses helpers inside top_ranked_plot.py to render the figures.
~~~

**Notebooks (optional; additional publication figures & KDE guards)**
~~~bash
# Launch (GPU not required for plotting)
jupyter lab
# Open:
# - notebooks/graph.ipynb       (POS & general plots incl. KDE and earliest-layer views)
# - notebooks/graph_fact.ipynb  (facts; requires fact_* CSVs)
~~~

---

## Figure Mapping (Paper)

- **Figure 1 (Overview schematic)** — conceptual; built from outputs of frequency stacks + earliest-layer curves shown below.  
- **Figure 2 (Frequent tokens dominate early layers)** — run `top_ranked_token_at_each_layer.py` (calls `top_ranked_plot.py` to save stacked 100% bars under `tokenizer_analysis/`).  
- **Figure 3 (Decision flips with depth)** — run `decision_flip_intermediate_and_final_layer_comparison.py` (renders flip ratios via `top_ranked_plot.py`).  
- **Figure 4 (Earliest crossing thresholds, POS & Facts)** — generate CSVs with `analyze_knowledge_layers_pos.py` and `analyze_knowledge_layers_facts.py`; plot with `graph.ipynb` / `graph_fact.ipynb` (or use `plot_facts2.py` for the facts panel).

> Tip: plotting can run on CPU; CSV generation is faster with a GPU.

---

## Troubleshooting

- **CSV not found** → run the matching analysis script first, or edit `CSV_PATH` / `postfix` in the plotting script.  
- **“Cannot save file into a non-existent directory”** → create folders first:
  ~~~bash
  mkdir -p out/plots out/plot_data out/Layer_vs_Rank tokenizer_analysis
  ~~~
- **Plotly export errors** → ensure `pip install kaleido` (and restart the kernel/shell).  
- **KDE cell fails in notebooks** (empty/constant series) → guard before KDE:
  ~~~python
  s = df.groupby("prompt index")["layer"].min().dropna()
  if s.size >= 2 and s.nunique() >= 2:
      s.plot.kde(...)
  ~~~

---

## Suggested .gitignore
~~~gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
out/
logs/
results/
tokenizer_analysis/
~~~

---


## Citation
*How Do LLMs Use Their Depth?* — please cite if you use this code or figures.






