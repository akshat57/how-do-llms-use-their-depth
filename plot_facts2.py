# -*- coding: utf-8 -*-
"""
Plot: Earliest layer where rank ≤ k (nice styling, same logic)
Assumes your CSV rows look like:

{
  'layer': layer_name,           # e.g., "transformer.h.12" or an int
  'prompt index': i,             # int
  'prompt': prompt,              # str (unused here)
  'answer': int(last_token),     # int (unused here)
  'answer_text': answer,         # str (unused here)
  'rank': rank,                  # int (1 = best)
  'token_num': token_num,        # 0/1/2 or 1/2/3  (normalized below)
  'answer_len': answer_len       # 1/2/3
}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# -------------------- CONFIG --------------------
# Set these two to point to your run
model = 'gpt2-xl'                # 'gpt2-xl' | 'Llama-2-7b' | 'pythia-6.9b' | 'pythia-6.9b-deduped' | 'Meta-Llama-3-8B'
dset_type = 'mquake_fact'             # just used in the filename you save
postfix = 'logit'
CSV_PATH = f"out/data/fact_{model}_REASONING_{postfix}.csv"  # <-- update if needed
print(model)

# Title / model display name
model_to_title = {
    'gpt2-xl': 'GPT2-XL',
    'Llama-2-7b': 'Llama2-7B',
    'pythia-6.9b-deduped': 'Pythia 6.9B',
    'pythia-6.9b': 'Pythia 6.9B',
    'Meta-Llama-3-8B': 'Meta-Llama-3-8B'
}

# x-axis limit by model (final layer index)
model_to_xlim = {
    'gpt2-xl': 48,
    'Llama-2-7b': 32,
    'pythia-6.9b-deduped': 32,
    'pythia-6.9b': 32,
    'Meta-Llama-3-8B': 32
}



# Axis / legend cosmetics
x_tick_size = 22
y_tick_size = 22
axis_fontsize = 28
legend_fontsize = 14

# k grid: log-spaced 1..10_000 (matches "essence" of old notebook)
K_VALUES = np.unique((np.logspace(0, 4, 200)).astype(int))

# Subtle global cosmetics (no logic changes)
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LINEWIDTH = 2.75  # thicker lines for clarity
# ------------------------------------------------


# ----------------- HELPERS (logic same) -----------------
def convert_layer_to_int(x):
    """Accept ints or strings like 'transformer.h.12'; return int or None."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    try:
        return int(s.split(".")[-1])
    except Exception:
        return None

def normalize_token_pos(series):
    """
    Normalize token_num to 1/2/3.
    If token_num is 0/1/2 -> convert to 1/2/3.
    If already 1/2/3, keep as-is.
    """
    vals = series.astype(int)
    return np.where(vals.min() == 0, vals + 1, vals)

def earliest_layer_curve(df_cat, k_values):
    """
    Given df with ['prompt index','layer','rank'], compute for each k:
      mean earliest layer (first layer where rank ≤ k per prompt), std, and count.
    Returns DataFrame: ['k','mean_layer','std_layer','n_prompts_used'].
    """
    if df_cat is None or df_cat.empty:
        return pd.DataFrame({"k": k_values, "mean_layer": np.nan, "std_layer": np.nan, "n_prompts_used": 0})

    df = df_cat.copy()
    df["layer"] = df["layer"].apply(convert_layer_to_int)
    df = df.dropna(subset=["layer", "rank", "prompt index"])
    if df.empty:
        return pd.DataFrame({"k": k_values, "mean_layer": np.nan, "std_layer": np.nan, "n_prompts_used": 0})

    df["layer"] = df["layer"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["prompt index"] = df["prompt index"].astype(int)
    df = df.sort_values(["prompt index", "layer"])

    by_prompt = dict(tuple(df.groupby("prompt index", sort=False)))

    rows = []
    for k in k_values:
        hits = []
        for pid, dfg in by_prompt.items():
            hit = dfg.loc[dfg["rank"] <= k]
            if not hit.empty:
                hits.append(int(hit["layer"].iloc[0]))
        if hits:
            arr = np.array(hits, dtype=int)
            rows.append({
                "k": int(k),
                "mean_layer": float(arr.mean()),
                "std_layer": float(arr.std(ddof=0)),
                "n_prompts_used": int(arr.size),
            })
        else:
            rows.append({
                "k": int(k),
                "mean_layer": np.nan,
                "std_layer": np.nan,
                "n_prompts_used": 0,
            })
    return pd.DataFrame(rows)

def _style_for(label: str):
    """
    Cosmetic only: solid for first token, dotted for later tokens.
    Works if labels contain 'second'/'third' OR '(pos 2)/(pos 3)' OR 'token 2/3'.
    """
    lower = label.lower()
    if ("second" in lower) or ("pos 2" in lower) or ("token 2" in lower):
        return ":"
    if ("third" in lower) or ("pos 3" in lower) or ("token 3" in lower):
        return ":"
    return "-"  # default: first token solid
# --------------------------------------------------------


# -------------------- LOAD DATA --------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df_all = pd.read_csv(CSV_PATH)

needed = {"prompt index", "layer", "rank", "token_num", "answer_len"}
missing = needed - set(df_all.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Normalize token positions to 1/2/3
df_all["token_pos"] = normalize_token_pos(df_all["token_num"].astype(int))
# --------------------------------------------------


# ------------- BUILD CATEGORIES (6 lines) -------------
cats = {
    "1-token (pos 1)": (df_all[(df_all["answer_len"] == 1) & (df_all["token_pos"] == 1)], dict()),
    "2-token (pos 1)": (df_all[(df_all["answer_len"] == 2) & (df_all["token_pos"] == 1)], dict()),
    "2-token (pos 2)": (df_all[(df_all["answer_len"] == 2) & (df_all["token_pos"] == 2)], dict()),
    "3-token (pos 1)": (df_all[(df_all["answer_len"] == 3) & (df_all["token_pos"] == 1)], dict()),
    "3-token (pos 2)": (df_all[(df_all["answer_len"] == 3) & (df_all["token_pos"] == 2)], dict()),
    "3-token (pos 3)": (df_all[(df_all["answer_len"] == 3) & (df_all["token_pos"] == 3)], dict()),
}
# -------------------------------------------------------


# --------- REPORT OVERALL PROMPT COUNTS ----------
print("=== Overall prompts per category (distinct 'prompt index') ===")
overall_counts = {}
for name, (df_cat, _) in cats.items():
    n = df_cat["prompt index"].nunique()
    overall_counts[name] = n
    print(f"[{name}] prompts available: {n}")
# -------------------------------------------------


# --------- COMPUTE EARLIEST-LAYER CURVES ----------
curves = {}
for name, (df_cat, _) in cats.items():
    curves[name] = earliest_layer_curve(df_cat, K_VALUES)
# --------------------------------------------------


# -------------------- PLOT (distinct pos-3 style) --------------------
# Round dot caps so dotted lines read clearly
mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.solid_capstyle"] = "round"

def _group_name(name: str) -> str:
    return name.split('(')[0].strip()  # "2-token (pos 1)" -> "2-token"

def _style_kwargs(label: str):
    """
    Solid for first token; distinct dotted for pos2; dotted+markers for pos3.
    All share color within a length group.
    """
    lower = label.lower()
    if ("second" in lower) or ("pos 2" in lower) or ("token 2" in lower):
        # Tight dotted
        return {"linestyle": (0, (1.0, 1.4))}
    if ("third" in lower) or ("pos 3" in lower) or ("token 3" in lower):
        # Looser dotted + subtle markers to ensure visible distinction
        return {
            "linestyle": (0, (1.0, 3.0)),
            "marker": "o",
            "markersize": 3.5,
            "markevery": 0.15,   # every ~15% of points
        }
    return {"linestyle": "-"}  # first token solid

# Determine active categories
active_names = []
for name in cats.keys():
    cur = curves.get(name)
    if cur is not None and not cur.empty and (cur["n_prompts_used"] > 0).any():
        active_names.append(name)

# Colorblind-friendly palette per group (1-, 2-, 3-token)
unique_groups = list(dict.fromkeys(_group_name(n) for n in active_names))
palette = sns.color_palette("colorblind", n_colors=max(3, len(unique_groups)))
group_to_color = {g: palette[i] for i, g in enumerate(unique_groups)}

# Plot with reversed axes: x = mean layer, y = rank (log)
for name in cats.keys():
    cur = curves.get(name)
    if cur is None or cur.empty:
        continue
    valid = cur[cur["n_prompts_used"] > 0]
    if valid.empty:
        continue

    # Print counts to terminal only (not in legend)
    print(f"{name}: n={overall_counts[name]}")

    plt.plot(
        valid["mean_layer"], valid["k"],
        color=group_to_color[_group_name(name)],
        linewidth=LINEWIDTH,
        label=name,                      # legend label WITHOUT counts
        **_style_kwargs(name)
    )

plt.yscale('log')
plt.xlabel('Layer', fontsize=axis_fontsize)
plt.ylabel('Rank', fontsize=axis_fontsize)
plt.tick_params(axis='x', labelsize=x_tick_size)
plt.tick_params(axis='y', labelsize=y_tick_size)
plt.xlim(0, model_to_xlim[model])

# No grid per request

# Legend (labels only)
leg = plt.legend(title='', fontsize=legend_fontsize, frameon=False, handlelength=3)
for line in leg.get_lines():
    line.set_linewidth(LINEWIDTH)

plt.tight_layout()
os.makedirs("out/plots", exist_ok=True)
plt.savefig(f'out/plots/{model}_{dset_type}_{postfix}.png', bbox_inches='tight')
plt.show()
# --------------------------------------------------------------------
