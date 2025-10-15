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


# =========================================
# 0) ENVIRONMENT SETUP (Conda, Python deps)
# add these lines to show users how to create/activate an env and install packages
# =========================================
# Conda (recommended)
conda create -n depthstudy python=3.10 -y
conda activate depthstudy
pip install -U torch transformers tuned-lens pandas numpy matplotlib seaborn tqdm
# optional (POS support):
pip install spacy && python -m spacy download en_core_web_sm


# =========================================
# 1) CREATE OUTPUT DIRECTORIES
# add this so runs don’t fail when saving files/figures
# =========================================
mkdir -p out/data out/plots out/plot_data out/Layer_vs_Rank


# =========================================
# 2) GENERATE CSVs — POS CASE STUDY
# add this to produce POS CSVs consumed by notebooks/plots
# writes: out/data/gpt2-xl_{POS}_logit.csv
# (edit scripts to change model/dataset if needed)
# =========================================
python analyze_knowledge_layers_pos.py


# =========================================
# 3) GENERATE CSVs — MULTI-TOKEN FACTS (MQuAKE)
# add this to produce fact CSVs for earliest-layer plots
# writes: out/data/fact_gpt2-xl_REASONING_logit.csv
# (if your suffix differs, edit `postfix` or `CSV_PATH` in plot_facts2.py)
# =========================================
python analyze_knowledge_layers_facts.py


# =========================================
# 4) GENERATE CSVs — OPTION-CONSTRAINED DOWNSTREAM TASKS
# add this for per-task CSVs used in downstream analyses
# writes: one CSV per task in the working directory
# =========================================
python downstream_task_options_analysis.py


# =========================================
# 5) PLOT — FACTS “EARLIEST LAYER VS RANK”
# add this to create the main facts figure
# reads: out/data/fact_gpt2-xl_REASONING_logit.csv (by default)
# writes: out/plots/gpt2-xl_mquake_fact_logit.png
# =========================================
python -u plot_facts2.py
# Output: out/plots/gpt2-xl_mquake_fact_logit.png
