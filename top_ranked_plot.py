import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def plot_stack_plotly(
    frequency_counter: dict,
    save_location: str = "tokenizer_analysis/",
    model_filename: str = "model",
    postfix: str = "original",
    using_tuned: bool = True,
    title_prefix: str = "Top-ranked token bucket share by layer"
):
    label_converter = {
        "top10":  "Top 1–10",
        "top100": "Top 11–100",
        "top1000":"Top 101–1000",
        "mid10k": "Other Tokens",
    }
    # Choose some tasteful colors (feel free to adjust)
    colors = {
        "top10":   "#4F46E5",   # indigo
        "top100":  "#0EA5E9",   # light blue
        "top1000": "#10B981",   # green
        "mid10k":  "#9CA3AF",   # gray
    }

    # -------- Prepare data (skip layer 0) --------
    layers_sorted = sorted(frequency_counter.keys())
    layers = [L for L in layers_sorted if L != 0]

    def arr(bucket):
        return np.array([frequency_counter[L][bucket] for L in layers], dtype=float)

    c_top10   = arr("top10")
    c_top100  = arr("top100")
    c_top1000 = arr("top1000")
    c_mid10k  = arr("mid10k")

    den = c_top10 + c_top100 + c_top1000 + c_mid10k
    den = np.where(den == 0, 1, den)  # avoid divide-by-zero

    # percentages for text labels
    p_top10   = 100 * c_top10   / den
    p_top100  = 100 * c_top100  / den
    p_top1000 = 100 * c_top1000 / den
    p_mid10k  = 100 * c_mid10k  / den

    # -------- Build figure --------
    fig = go.Figure()

    def add_trace(name_key, counts, perc):
        fig.add_trace(go.Bar(
            name=label_converter[name_key],
            x=layers,
            y=counts,                     # raw counts; we'll use barnorm='percent'
            marker_color=colors[name_key],
            text=[f"{v:.1f}%" for v in perc],
            textposition="inside",
            textfont=dict(size=11),
            hovertemplate=(
                "<b>Layer %{x}</b><br>"
                + label_converter[name_key] + "<br>"
                "Count: %{y}<br>"
                "Share: %{text}<extra></extra>"
            ),
        ))

    add_trace("top10",   c_top10,   p_top10)
    add_trace("top100",  c_top100,  p_top100)
    add_trace("top1000", c_top1000, p_top1000)
    add_trace("mid10k",  c_mid10k,  p_mid10k)

    tuned_str = "TunedLens" if using_tuned else "LogitLens"
    fig.update_layout(
        template="plotly_white",
        title=f"{title_prefix} — {model_filename} ({tuned_str})",
        barmode="stack",
        barnorm="percent",            # ← 100% normalization per layer
        xaxis_title="Layer",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100], ticksuffix="%", dtick=20, gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(type="category"),  # keeps integer layers equally spaced
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=70, b=50),
        bargap=0.15,
    )

    # If text looks crowded, uncomment the next line to hide labels
    # fig.update_traces(text=None)

    # -------- Save outputs --------
    Path(save_location).mkdir(parents=True, exist_ok=True)
    base = Path(save_location) / f"{model_filename}_{postfix}_final_plotly_{tuned_str}"
    base = Path(save_location) / f"{model_filename}_{postfix}_final_plotly_{'TunedLens' if using_tuned else 'LogitLens'}"
    png_path = str(base) + ".png"

    # High-res PNG (set width/height or scale)
    fig.write_image(png_path, width=1600, height=800, scale=2)  # adjust as you like
    print(f"Saved PNG to: {png_path}")

    return fig


# 100% stacked bar chart with prominence-ordered colors (Plotly)
# - Most prominent: Top 1–10
# - Next: Top 11–100
# - Next: Top 101–1000
# - Light/ignorable: Other Tokens
# Requires: pip install -U plotly kaleido  (and ensure Chrome is installed for Kaleido)

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio
import re
import gc

def plot_stack_plotly_pretty(
    frequency_counter: dict,
    save_location: str = "tokenizer_analysis/",
    model_filename: str = "model",
    postfix: str = "original",
    using_tuned: bool = True,
    title_prefix: str = None,  # kept for API compatibility; ignored
):
    # --------- Tweakable font/format variables ---------
    LEGEND_FONT_SIZE   = 40
    LEGEND_SWATCH_WIDTH  = 60   # horizontal space per legend item
    LEGEND_MARKER_SIZE   = 40  
    X_AXIS_TITLE_SIZE    = 55
    Y_AXIS_TITLE_SIZE    = 55
    X_TICK_FONT_SIZE     = 20
    Y_TICK_FONT_SIZE   = 35
    INSIDE_LABEL_SIZE  = 11
    SHOW_TEXT_LABELS   = True   # set False if labels feel crowded
    BAR_LINE_WIDTH     = 0.6
    TEXT_COLOR         = "#2B2B2B"     # << same label color for ALL segments (e.g., "#000000" for black)

# --------- Labels & colors ---------
    labels = {
        "top10":   "Top 1–10",
        "top100":  "Top 11–100",
        "top1000": "Top 101–1000",
        "mid10k":  "Rest",
    }
    colors = {
        "top10":   "#F97316",  # vivid orange (most prominent)
        "top100":  "#2563EB",  # strong blue
        "top1000": "#10B981",  # emerald/teal
        "mid10k":  "#D1D5DB",  # light gray (ignorable)
    }
    LEGEND_ORDER = ["top10", "top100", "top1000", "mid10k"]  # legend left → right

    # --------- Prepare data (skip layer 0) ---------
    layers_sorted = sorted(frequency_counter.keys())
    layers = [L for L in layers_sorted if L != 0]

    def arr(bucket):
        return np.array([frequency_counter[L][bucket] for L in layers], dtype=float)

    c_top10   = arr("top10")
    c_top100  = arr("top100")
    c_top1000 = arr("top1000")
    c_mid10k  = arr("mid10k")

    den = c_top10 + c_top100 + c_top1000 + c_mid10k
    den = np.where(den == 0, 1, den)  # safety

    pct = {
        "top10":   100 * c_top10   / den,
        "top100":  100 * c_top100  / den,
        "top1000": 100 * c_top1000 / den,
        "mid10k":  100 * c_mid10k  / den,
    }

    # --------- Build figure ---------
    fig = go.Figure()

    def add_bar_trace(key, counts):
        fig.add_trace(go.Bar(
            name=labels[key],
            x=layers,
            y=counts,                        # barnorm='percent' normalizes to 100%
            marker_color=colors[key],
            marker_line_color="white",
            marker_line_width=BAR_LINE_WIDTH,
            text=[f"{v:.1f}%" for v in pct[key]] if SHOW_TEXT_LABELS else None,
            textposition="inside",
            textfont=dict(size=INSIDE_LABEL_SIZE, color=TEXT_COLOR),
            hovertemplate=(
                "<b>Layer %{x}</b><br>" + labels[key] +
                "<br>Count: %{y}<br>Share: %{text}<extra></extra>"
            ),
            showlegend=False,  # hide bars from legend; we'll add big legend markers below
        ))

    # Bottom -> top (visual stack)
    add_bar_trace("top10",   c_top10)
    add_bar_trace("top100",  c_top100)
    add_bar_trace("top1000", c_top1000)
    add_bar_trace("mid10k",  c_mid10k)

    # --- Custom legend entries with large squares ---
    for key in LEGEND_ORDER:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=LEGEND_MARKER_SIZE,
                        color=colors[key], line=dict(color="rgba(0,0,0,0.2)", width=1)),
            name=labels[key],
            hoverinfo="skip",
            showlegend=True,
        ))

    tuned_str = "TunedLens" if using_tuned else "LogitLens"

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        barnorm="percent",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=LEGEND_FONT_SIZE),
            itemsizing="trace",            # << let marker size control the swatch size
            itemwidth=LEGEND_SWATCH_WIDTH,
            traceorder="normal",
        ),
        margin=dict(l=60, r=20, t=20, b=50),
        bargap=0.15,
    )

    # Axes: larger titles & ticks
    fig.update_xaxes(
        title_text="Layer",
        title_font=dict(size=X_AXIS_TITLE_SIZE),
        tickfont=dict(size=X_TICK_FONT_SIZE),
        type="category",
    )
    fig.update_yaxes(
        title_text="Percentage",
        title_font=dict(size=Y_AXIS_TITLE_SIZE),
        tickfont=dict(size=Y_TICK_FONT_SIZE),
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="rgba(0,0,0,0.12)",
    )

    # --------- Save outputs ---------
    Path(save_location).mkdir(parents=True, exist_ok=True)
    base = Path(save_location) / f"{model_filename}_{postfix}_final_plotly_{tuned_str}"
    fig.write_html(str(base) + ".html", include_plotlyjs="cdn")
    fig.write_image(str(base) + ".png", width=1600, height=850, scale=2)

    # -------- Cleanup --------
    del fig
    gc.collect()


def plot_evolution_stack_plotly(
    frequency_flip_counter: dict,
    source_layer: int = 1,
    save_location: str = "tokenizer_analysis/",
    model_filename: str = "model",
    postfix: str = "evolution",
    using_tuned: bool = True,
    png_width: int = 1600,
    png_height: int = 850,
    png_scale: int = 2,
) -> None:
    """
    Build a 100% stacked bar chart of five outcomes relative to a fixed source layer
    and SAVE a PNG to disk (no return). Requires plotly + kaleido + Chrome.

    Stack order (bottom -> top):
      Changed → Other, Changed → Top 101–1000, Changed → Top 11–100, Changed → Top 1–10, Same Token
    """

    # ---------- Tweakable knobs ----------
    LEGEND_FONT_SIZE   = 30
    AXIS_TITLE_SIZE    = 55
    X_TICK_FONT_SIZE     = 20
    Y_TICK_FONT_SIZE   = 35
    INSIDE_LABEL_SIZE  = 11
    SHOW_TEXT_LABELS  = True
    BAR_LINE_WIDTH    = 0.6
    TEXT_COLOR        = "rgba(0,0,0,0.85)"  # single label color for all segments

    labels = {
        "exact":   "Unchanged",
        "top10":   "→ Top 1–10",
        "top100":  "→ Top 11–100",
        "top1000": "→ Top 101–1000",
        "others":  "→ Rest",
    }

    colors = {
        "exact":   "#FECACA",  # pale rose
        "top10":   "#F97316",  # vivid orange
        "top100":  "#2563EB",  # strong blue
        "top1000": "#10B981",  # emerald/teal
        "others":  "#D1D5DB",  # light gray
    }

    # ---- NEW: flipped stack & legend order (bottom -> top) ----
    STACK_ORDER  = ["others", "top1000", "top100", "top10", "exact"]
    LEGEND_ORDER = ["others", "top1000", "top100", "top10", "exact"]

    # -------- Collect data (layers > source_layer) --------
    layers_sorted = sorted(frequency_flip_counter.keys())
    layers = [L for L in layers_sorted if L > source_layer]
    if not layers:
        raise ValueError("No target layers found beyond source_layer; check your counters.")

    def arr(key):
        return np.array([frequency_flip_counter[L].get(key, 0) for L in layers], dtype=float)

    counts = {k: arr(k) for k in STACK_ORDER}
    denom = sum(counts[k] for k in STACK_ORDER)
    denom = np.where(denom == 0, 1, denom)  # safety per layer
    pct = {k: 100.0 * counts[k] / denom for k in STACK_ORDER}

    # -------- Build figure --------
    fig = go.Figure()
    for key in STACK_ORDER:  # bottom -> top
        fig.add_trace(go.Bar(
            name=labels[key],
            x=layers,
            y=counts[key],                   # barnorm='percent' normalizes to 100%
            marker_color=colors[key],
            marker_line_color="white",
            marker_line_width=BAR_LINE_WIDTH,
            text=[f"{v:.1f}%" for v in pct[key]] if SHOW_TEXT_LABELS else None,
            textposition="inside",
            textfont=dict(size=INSIDE_LABEL_SIZE, color=TEXT_COLOR),
            hovertemplate=(
                "<b>Layer %{x}</b><br>" + labels[key] +
                "<br>Count: %{y}<br>Share: %{text}<extra></extra>"
            ),
        ))

    tuned_str = "TunedLens" if using_tuned else "LogitLens"
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        barnorm="percent",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=LEGEND_FONT_SIZE),
            traceorder="normal",
        ),
        margin=dict(l=60, r=20, t=20, b=50),
        bargap=0.15,
    )

    # Enforce legend order
    rank = {name: i for i, name in enumerate(LEGEND_ORDER)}
    for tr in fig.data:
        # map displayed name back to our key
        for k, v in labels.items():
            if v == tr.name:
                tr.legendrank = rank[k]
                break

    # Axes
    fig.update_xaxes(
        title_text="Layer",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=X_TICK_FONT_SIZE),
        type="category",
    )
    fig.update_yaxes(
        title_text="Percentage",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=Y_TICK_FONT_SIZE),
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="rgba(0,0,0,0.12)",
    )

    # -------- Save PNG (no return) --------
    Path(save_location).mkdir(parents=True, exist_ok=True)
    base = Path(save_location) / f"{model_filename}_{postfix}_fromL{source_layer}_{tuned_str}"
    png_path = str(base) + ".png"
    fig.write_image(png_path, width=png_width, height=png_height, scale=png_scale)
    print(f"Saved PNG to: {png_path}")

    # -------- Cleanup --------
    del fig
    gc.collect()




#########################################
########################################

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_flip_ratios_from_counter(
    frequency_flip_counter,
    category_order=("top10", "top100", "top1000", "rest", "baseline"),  # baseline as 5th
    title="Decision Flips by Layer (flipped / total)",
    annotate=False,
    ylim=(0, 1),
    figsize=(12, 5),
    bar_width=0.18,
    rotation=0,
    save_path=None,
    start_layer=1,     # skip layer 0 by default
    end_layer=None,    # inclusive; None = last layer
    baseline_color="k" # black baseline bars
):
    """
    Expects:
      frequency_flip_counter[category] = list of length num_layers
      where each element is {'total': int, 'flipped': int}.

    Notes:
    - Baseline is plotted as a fifth bar per layer in black.
    - If 'baseline' key is missing but the four buckets exist, it will be synthesized
      as the per-layer sum over the four buckets.
    """

    # If baseline is missing but all four buckets exist, synthesize it by summing
    four = ("top10", "top100", "top1000", "rest")
    # if "baseline" not in frequency_flip_counter and all(k in frequency_flip_counter for k in four):
    #     num_layers = len(frequency_flip_counter[four[0]])
    #     baseline = []
    #     for i in range(num_layers):
    #         tot = sum(frequency_flip_counter[k][i].get("total", 0) or 0 for k in four)
    #         flp = sum(frequency_flip_counter[k][i].get("flipped", 0) or 0 for k in four)
    #         baseline.append({"total": tot, "flipped": flp})
    #     frequency_flip_counter = dict(frequency_flip_counter)  # shallow copy
    #     frequency_flip_counter["baseline"] = baseline

    # Validate categories present
    for c in category_order:
        if c not in frequency_flip_counter:
            raise KeyError(f"Missing category '{c}' in frequency_flip_counter")

    # Layer range
    num_layers = len(frequency_flip_counter[category_order[0]])
    if end_layer is None or end_layer > num_layers - 1:
        end_layer = num_layers - 1
    if start_layer < 0 or start_layer > end_layer:
        raise ValueError("Invalid start_layer/end_layer")
    layer_indices = list(range(start_layer, end_layer + 1))
    n_layers_plot = len(layer_indices)
    if n_layers_plot == 0:
        raise ValueError("No layers to plot with given start/end.")

    # Build ratios per category for selected layers
    def ratios_for_category(cat):
        vals = []
        for i in layer_indices:
            e = frequency_flip_counter[cat][i]
            total = e.get("total", 0) or 0
            flipped = e.get("flipped", 0) or 0
            vals.append((flipped / total) if total > 0 else 0.0)
        return np.array(vals, dtype=float)

    ratio_matrix = np.column_stack([ratios_for_category(cat) for cat in category_order])

    # Plot
    x = np.arange(n_layers_plot)
    n_cats = len(category_order)
    total_group_width = n_cats * bar_width
    start_offset = -0.5 * total_group_width + bar_width / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    for i, cat in enumerate(category_order):
        offsets = x + start_offset + i * bar_width
        # Only force color for baseline; others use default cycle
        if cat.lower() == "baseline":
            bars = ax.bar(offsets, ratio_matrix[:, i], width=bar_width, label=cat, color=baseline_color)
        else:
            bars = ax.bar(offsets, ratio_matrix[:, i], width=bar_width, label=cat)

        if annotate:
            for rect, val in zip(bars, ratio_matrix[:, i]):
                ax.annotate(
                    f"{val*100:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in layer_indices], rotation=rotation)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Flipped / Total")
    ax.set_title(title)
    ax.legend(title="Category")

    fig.tight_layout()

    # Save if requested (also create dir if needed)
    if save_path is not None:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig, ax



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # big square legend markers
from matplotlib.ticker import PercentFormatter, MultipleLocator

def plot_flip_ratios_from_counter_new(
    frequency_flip_counter,
    category_order=("top10", "top100", "top1000", "mid10k"),
    annotate=False,
    ylim=(0, 1),           # accepts 0..1; auto-converted to 0..100
    figsize=(18, 9),
    bar_width=0.24,        # thicker bars
    rotation=0,
    save_path=None,        # required; saves & closes internally
    start_layer=1,         # skip layer 0 by default
    end_layer=None,        # inclusive; None = last layer
    layer_step=2,          # plot every other layer by default
):
    """
    Saves a grouped bar chart (PERCENTAGES) and closes the figure. Returns nothing.
    """

    # ---- Fonts (your current settings) ----
    LEGEND_FONT_SIZE = 40
    X_AXIS_TITLE_SIZE = 50
    Y_AXIS_TITLE_SIZE = 50
    X_TICK_FONT_SIZE = 25
    Y_TICK_FONT_SIZE = 35
    LEGEND_MARKER_SIZE = 24   # ↑ match Plotly's big squares more closely

    # Legend placement (move slightly right)
    LEGEND_X = 0.6           # 0.5=center; increase to push right
    LEGEND_Y = 1.05

    # ---- Labels & colors (Plotly scheme) ----
    LABELS = {
        "top10":   "Top1–10",
        "top100":  "Top11–100",
        "top1000": "Top101–1000",
        "mid10k":  "Top1000+",
    }
    COLORS = {
        "top10":   "#F97316",
        "top100":  "#2563EB",
        "top1000": "#10B981",
        "mid10k":  "#D1D5DB",
    }
    LEGEND_ORDER = ["top10", "top100", "top1000", "mid10k"]

    # --- small spacing between layer groups ---
    group_padding = 0.06

    # Back-compat: 'rest' -> 'mid10k'
    if "mid10k" not in frequency_flip_counter and "rest" in frequency_flip_counter:
        frequency_flip_counter = dict(frequency_flip_counter)
        frequency_flip_counter["mid10k"] = frequency_flip_counter["rest"]

    # Validate presence & lengths
    for c in category_order:
        if c not in frequency_flip_counter:
            raise KeyError(f"Missing category '{c}' in frequency_flip_counter")
    lens = {c: len(frequency_flip_counter[c]) for c in category_order}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Inconsistent layer lengths: {lens}")

    # Layer range + thinning
    num_layers = len(frequency_flip_counter[category_order[0]])
    if end_layer is None or end_layer > num_layers - 1:
        end_layer = num_layers - 1
    if start_layer < 0 or start_layer > end_layer:
        raise ValueError("Invalid start_layer/end_layer")
    step = max(1, int(layer_step))
    layer_indices = list(range(start_layer, end_layer + 1, step))
    if not layer_indices:
        raise ValueError("No layers to plot with given range/step.")

    # Compute ratios -> percentages
    def ratios_for_category(cat):
        vals = []
        for i in layer_indices:
            e = frequency_flip_counter[cat][i]
            total = e.get("total", 0) or 0
            flipped = e.get("flipped", 0) or 0
            vals.append((flipped / total) if total > 0 else 0.0)
        return np.array(vals, dtype=float)

    ratio_matrix = np.column_stack([ratios_for_category(cat) for cat in category_order])
    pct_matrix = ratio_matrix * 100.0

    # ---- Plot (figure-level legend; no title) ----
    fig, ax = plt.subplots(figsize=figsize)

    # tiny padding between layer groups
    x = np.arange(len(layer_indices)) * (1.0 + group_padding)

    n_cats = len(category_order)
    total_group_width = n_cats * bar_width
    start_offset = -0.5 * total_group_width + bar_width / 2.0

    for i, cat in enumerate(category_order):
        offsets = x + start_offset + i * bar_width
        color = COLORS[cat]
        label = LABELS[cat]
        bars = ax.bar(offsets, pct_matrix[:, i], width=bar_width, label=label, color=color)

        if annotate:
            for rect, val in zip(bars, pct_matrix[:, i]):
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11, color="#2B2B2B"
                )

    # Axes labels/ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in layer_indices], rotation=rotation)

    # Convert any (0,1) ylim to (0,100) automatically
    y0, y1 = ylim
    if y1 <= 1.001:
        y0, y1 = y0 * 100.0, y1 * 100.0
    ax.set_ylim(y0, y1)

    ax.set_xlabel("Layer", fontsize=X_AXIS_TITLE_SIZE)
    ax.set_ylabel("Decision Flip Rate", fontsize=Y_AXIS_TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)

    # >>> Percent tick labels like Plotly <<<
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(20))  # 0%, 20%, 40%, ...

    # ---- Figure-level legend (no title), shifted right ----
    legend_handles = [
        Line2D([0], [0], marker='s', linestyle='None',
               markersize=LEGEND_MARKER_SIZE,
               markerfacecolor=COLORS[k],
               markeredgecolor=(0, 0, 0, 0.2),
               label=LABELS[k])
        for k in LEGEND_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(LEGEND_X, LEGEND_Y),  # move horizontally by changing LEGEND_X
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.8,
        handlelength=1.0,
        handletextpad=0.2,
    )

    # Let axes extend higher; keep a slim gap for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.975], pad=0.1)

    # ---- Save & close inside the function ----
    if save_path is None:
        raise ValueError("save_path must be provided.")
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # return None

from pathlib import Path
import gc
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# No HTML rendering
pio.renderers.default = "png"   # or "svg"

def plot_stack_plotly_pretty_new(
    frequency_counter: dict,
    save_location: str = "tokenizer_analysis/",
    model_filename: str = "model",
    postfix: str = "original",
    using_tuned: bool = True,
    title_prefix: str = None,  # kept for API compatibility; ignored
):
    # --------- Tweakable font/format variables ---------
    LEGEND_FONT_SIZE     = 50
    LEGEND_MARKER_SIZE   = 80   # big legend squares
    X_AXIS_TITLE_SIZE    = 55
    Y_AXIS_TITLE_SIZE    = 65
    X_TICK_FONT_SIZE     = 20
    Y_TICK_FONT_SIZE     = 35
    INSIDE_LABEL_SIZE    = 11
    SHOW_TEXT_LABELS     = True
    BAR_LINE_WIDTH       = 0.6
    TEXT_COLOR           = "#2B2B2B"

    # --------- Labels & colors ---------
    labels = {
        "top10":   "Top1–10",
        "top100":  "Top11–100",
        "top1000": "Top101–1000",
        "mid10k":  "Top1000+",
    }
    colors = {
        "top10":   "#F97316",
        "top100":  "#2563EB",
        "top1000": "#10B981",
        "mid10k":  "#D1D5DB",
    }
    LEGEND_ORDER = ["top10", "top100", "top1000", "mid10k"]

    # --------- Prepare data (skip layer 0) ---------
    layers_sorted = sorted(frequency_counter.keys())
    layers = [L for L in layers_sorted if L != 0]

    def arr(bucket):
        return np.array([frequency_counter[L][bucket] for L in layers], dtype=float)

    c_top10   = arr("top10")
    c_top100  = arr("top100")
    c_top1000 = arr("top1000")
    c_mid10k  = arr("mid10k")

    den = c_top10 + c_top100 + c_top1000 + c_mid10k
    den = np.where(den == 0, 1, den)  # safety

    pct = {
        "top10":   100 * c_top10   / den,
        "top100":  100 * c_top100  / den,
        "top1000": 100 * c_top1000 / den,
        "mid10k":  100 * c_mid10k  / den,
    }

    # --------- Build figure ---------
    fig = go.Figure()

    def add_bar_trace(key, counts):
        fig.add_trace(go.Bar(
            name=labels[key],
            x=layers,
            y=counts,
            marker_color=colors[key],
            marker_line_color="white",
            marker_line_width=BAR_LINE_WIDTH,
            text=[f"{v:.1f}%" for v in pct[key]] if SHOW_TEXT_LABELS else None,
            textposition="inside",
            textfont=dict(size=INSIDE_LABEL_SIZE, color=TEXT_COLOR),
            hovertemplate="<b>Layer %{x}</b><br>" + labels[key] +
                          "<br>Count: %{y}<br>Share: %{text}<extra></extra>",
            showlegend=False,
        ))

    # Bottom -> top (visual stack)
    add_bar_trace("top10",   c_top10)
    add_bar_trace("top100",  c_top100)
    add_bar_trace("top1000", c_top1000)
    add_bar_trace("mid10k",  c_mid10k)

    # --- Custom legend entries with large squares ---
    for key in LEGEND_ORDER:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=LEGEND_MARKER_SIZE,
                        color=colors[key], line=dict(color="rgba(0,0,0,0.2)", width=1)),
            name=labels[key],
            hoverinfo="skip",
            showlegend=True,
        ))

    tuned_str = "TunedLens" if using_tuned else "LogitLens"

    # Force a single horizontal legend that uses the full width (center-anchored)
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        barnorm="percent",
        legend=dict(
            orientation="h",
            x=0.50, xanchor="center",  # << center anchor so the whole row can span
            y=1.06, yanchor="bottom",
            font=dict(size=LEGEND_FONT_SIZE),
            itemsizing="trace",
            # IMPORTANT: do NOT set itemwidth (it can force wrapping)
            traceorder="normal",
        ),
        margin=dict(l=60, r=20, t=64, b=50),  # a bit more top margin for the single-row legend
        bargap=0.15,
    )

    # Axes (already show % on y with ticksuffix)
    fig.update_xaxes(
        title_text="Layer",
        title_font=dict(size=X_AXIS_TITLE_SIZE),
        tickfont=dict(size=X_TICK_FONT_SIZE),
        type="category",
    )
    fig.update_yaxes(
        title_text="Percentage",
        title_font=dict(size=Y_AXIS_TITLE_SIZE),
        tickfont=dict(size=Y_TICK_FONT_SIZE),
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="rgba(0,0,0,0.12)",
    )

    # --------- Save static image only (no HTML) ---------
    Path(save_location).mkdir(parents=True, exist_ok=True)
    base = Path(save_location) / f"{model_filename}_{postfix}_final_plotly_{tuned_str}"
    fig.write_image(str(base) + ".png", width=1600, height=850, scale=2)

    # -------- Cleanup --------
    del c_top10, c_top100, c_top1000, c_mid10k, den, pct
    del fig
    gc.collect()
    return


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MultipleLocator

def plot_flip_ratios_from_counter_lines(
    frequency_flip_counter,
    category_order=("top10", "top100", "top1000", "mid10k"),
    annotate=False,
    ylim=(0, 1),           # accepts 0..1; auto-converted to 0..100
    figsize=(18, 9),
    bar_width=0.24,        # unused (kept for API compatibility)
    rotation=0,
    save_path=None,        # required; saves & closes internally
    start_layer=1,         # skip layer 0 by default
    end_layer=None,        # inclusive; None = last layer
    layer_step=2,          # unused (lines show every layer)
):
    """
    Saves a LINE plot (percentages) of flip ratios per layer for four buckets.
    Uses same colors and legend style as the bar version. Returns nothing.
    """

    # ---- Fonts (match your current settings) ----
    LEGEND_FONT_SIZE = 40
    X_AXIS_TITLE_SIZE = 50
    Y_AXIS_TITLE_SIZE = 50
    X_TICK_FONT_SIZE = 25
    Y_TICK_FONT_SIZE = 35

    # Legend placement (right-shifted like your bar version)
    LEGEND_X = 0.60
    LEGEND_Y = 1.05

    # Line styling
    LINE_WIDTH = 3.0
    MARKER_SIZE = 6.0
    MARKER_STYLE = "o"  # small circles at each layer

    # ---- Labels & colors (exactly your scheme) ----
    LABELS = {
        "top10":   "Top 1–10",
        "top100":  "Top 11–100",
        "top1000": "Top 101–1000",
        "mid10k":  "Rest",
    }
    COLORS = {
        "top10":   "#F97316",
        "top100":  "#2563EB",
        "top1000": "#10B981",
        "mid10k":  "#D1D5DB",
    }
    LEGEND_ORDER = ["top10", "top100", "top1000", "mid10k"]

    # Back-compat: 'rest' -> 'mid10k'
    if "mid10k" not in frequency_flip_counter and "rest" in frequency_flip_counter:
        frequency_flip_counter = dict(frequency_flip_counter)
        frequency_flip_counter["mid10k"] = frequency_flip_counter["rest"]

    # Validate presence & lengths
    for c in category_order:
        if c not in frequency_flip_counter:
            raise KeyError(f"Missing category '{c}' in frequency_flip_counter")
    lens = {c: len(frequency_flip_counter[c]) for c in category_order}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Inconsistent layer lengths: {lens}")

    # Layer range (no skipping for lines)
    num_layers = len(frequency_flip_counter[category_order[0]])
    if end_layer is None or end_layer > num_layers - 1:
        end_layer = num_layers - 1
    if start_layer < 0 or start_layer > end_layer:
        raise ValueError("Invalid start_layer/end_layer")
    layer_indices = list(range(start_layer, end_layer + 1))
    if not layer_indices:
        raise ValueError("No layers to plot with given range.")

    # Compute ratios -> percentages
    def ratios_for_category(cat):
        vals = []
        for i in layer_indices:
            e = frequency_flip_counter[cat][i]
            total = e.get("total", 0) or 0
            flipped = e.get("flipped", 0) or 0
            vals.append((flipped / total) if total > 0 else 0.0)
        return np.array(vals, dtype=float)

    ratio_matrix = np.column_stack([ratios_for_category(cat) for cat in category_order])
    pct_matrix = ratio_matrix * 100.0

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layer_indices))  # one point per layer

    for i, cat in enumerate(category_order):
        y = pct_matrix[:, i]
        ax.plot(
            x, y,
            label=LABELS[cat],
            color=COLORS[cat],
            linewidth=LINE_WIDTH,
            marker=MARKER_STYLE,
            markersize=MARKER_SIZE,
        )
        if annotate:
            # annotate last point of each line with its value
            ax.annotate(
                f"{y[-1]:.1f}%",
                xy=(x[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=11,
                color=COLORS[cat]
            )

    # Axes labels/ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in layer_indices], rotation=rotation)

    # Convert any (0,1) ylim to (0,100) automatically
    y0, y1 = ylim
    if y1 <= 1.001:
        y0, y1 = y0 * 100.0, y1 * 100.0
    ax.set_ylim(y0, y1)

    ax.set_xlabel("Layer", fontsize=X_AXIS_TITLE_SIZE)
    ax.set_ylabel("Decision Flip Rate (%)", fontsize=Y_AXIS_TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)

    # Percent tick labels
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(20))  # 0%, 20%, ...

    # Legend (figure-level, right-shifted)
    fig.legend(
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(LEGEND_X, LEGEND_Y),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.8,
        handlelength=2.0,     # longer line segment in legend
        handletextpad=0.6,
    )

    # Give the plot room under the legend
    fig.tight_layout(rect=[0, 0, 1, 0.975], pad=0.2)

    # Save & close
    if save_path is None:
        raise ValueError("save_path must be provided.")
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # return None



# --- Global size controls (tweak these to change all plots) ---
FIG_WIDTH  = 1800   # default image width in px (slightly wider than before)
FIG_HEIGHT = 1350    # default image height in px (a touch taller to emphasize vertical differences)

def plot_stack_plotly_pretty_wide(
    frequency_counter: dict,
    save_location: str = "tokenizer_analysis/",
    model_filename: str = "model",
    postfix: str = "original",
    using_tuned: bool = True,
    title_prefix: str = None,     # kept for API compatibility; ignored
    fig_width: int | None = None, # override width per call (px)
    fig_height: int | None = None # override height per call (px)
):
    """
    Drop-in replacement for plot_stack_plotly_pretty_new that:
      - is slightly longer horizontally by default
      - exposes size controls via FIG_WIDTH / FIG_HEIGHT or per-call overrides
    """
    from pathlib import Path
    import gc
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    # No HTML rendering
    pio.renderers.default = "png"   # or "svg"

    # --------- Tweakable font/format variables ---------
    LEGEND_FONT_SIZE     = 60
    LEGEND_MARKER_SIZE   = 80   # big legend squares
    X_AXIS_TITLE_SIZE    = 75
    Y_AXIS_TITLE_SIZE    = 85
    X_TICK_FONT_SIZE     = 30
    Y_TICK_FONT_SIZE     = 55
    INSIDE_LABEL_SIZE    = 11
    SHOW_TEXT_LABELS     = True
    BAR_LINE_WIDTH       = 0.6
    TEXT_COLOR           = "#2B2B2B"

    # --------- Labels & colors ---------
    labels = {
        "top10":   "Top1–10",
        "top100":  "Top11–100",
        "top1000": "Top101–1000",
        "mid10k":  "Top1000+",
    }
    colors = {
        "top10":   "#F97316",
        "top100":  "#2563EB",
        "top1000": "#10B981",
        "mid10k":  "#D1D5DB",
    }
    LEGEND_ORDER = ["top10", "top100", "top1000", "mid10k"]

    # --------- Prepare data (skip layer 0) ---------
    layers_sorted = sorted(frequency_counter.keys())
    layers = [L for L in layers_sorted if L != 0]

    def arr(bucket):
        return np.array([frequency_counter[L][bucket] for L in layers], dtype=float)

    c_top10   = arr("top10")
    c_top100  = arr("top100")
    c_top1000 = arr("top1000")
    c_mid10k  = arr("mid10k")

    den = c_top10 + c_top100 + c_top1000 + c_mid10k
    den = np.where(den == 0, 1, den)  # safety

    pct = {
        "top10":   100 * c_top10   / den,
        "top100":  100 * c_top100  / den,
        "top1000": 100 * c_top1000 / den,
        "mid10k":  100 * c_mid10k  / den,
    }

    # --------- Build figure ---------
    fig = go.Figure()

    def add_bar_trace(key, counts):
        fig.add_trace(go.Bar(
            name=labels[key],
            x=layers,
            y=counts,
            marker_color=colors[key],
            marker_line_color="white",
            marker_line_width=BAR_LINE_WIDTH,
            text=[f"{v:.1f}%" for v in pct[key]] if SHOW_TEXT_LABELS else None,
            textposition="inside",
            textfont=dict(size=INSIDE_LABEL_SIZE, color=TEXT_COLOR),
            hovertemplate="<b>Layer %{x}</b><br>" + labels[key] +
                          "<br>Count: %{y}<br>Share: %{text}<extra></extra>",
            showlegend=False,
        ))

    # Bottom -> top (visual stack)
    add_bar_trace("top10",   c_top10)
    add_bar_trace("top100",  c_top100)
    add_bar_trace("top1000", c_top1000)
    add_bar_trace("mid10k",  c_mid10k)

    # --- Custom legend entries with large squares ---
    for key in LEGEND_ORDER:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=LEGEND_MARKER_SIZE,
                        color=colors[key], line=dict(color="rgba(0,0,0,0.2)", width=1)),
            name=labels[key],
            hoverinfo="skip",
            showlegend=True,
        ))

    tuned_str = "TunedLens" if using_tuned else "LogitLens"

    # Slightly more horizontal room; keep top margin ample for a single-row legend
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        barnorm="percent",
        legend=dict(
            orientation="h",
            x=0.50, xanchor="center",
            y=1.06, yanchor="bottom",
            font=dict(size=LEGEND_FONT_SIZE),
            itemsizing="trace",
            traceorder="normal",
        ),
        margin=dict(l=60, r=30, t=70, b=60),
        bargap=0.12,  # a hair tighter spacing
    )

    # Axes (emphasize vertical differences by giving a touch more height via defaults above)
    fig.update_xaxes(
        title_text="Layer",
        title_font=dict(size=X_AXIS_TITLE_SIZE),
        tickfont=dict(size=X_TICK_FONT_SIZE),
        type="category",
    )
    fig.update_yaxes(
        title_text="Percentage",
        title_font=dict(size=Y_AXIS_TITLE_SIZE),
        tickfont=dict(size=Y_TICK_FONT_SIZE),
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="rgba(0,0,0,0.12)",
    )

    # --------- Save static image only (no HTML) ---------
    Path(save_location).mkdir(parents=True, exist_ok=True)
    base = Path(save_location) / f"{model_filename}_{postfix}_final_plotly_{tuned_str}_long"

    # Use global defaults unless explicitly overridden
    out_w = int(fig_width  if fig_width  is not None else FIG_WIDTH)
    out_h = int(fig_height if fig_height is not None else FIG_HEIGHT)

    fig.write_image(str(base) + ".png", width=out_w, height=out_h, scale=2)

    # -------- Cleanup --------
    del c_top10, c_top100, c_top1000, c_mid10k, den, pct
    del fig
    gc.collect()
    return
