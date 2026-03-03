"""
CreditPath AI - Exploratory Data Analysis Report
=================================================
Reads from creditpathai.db (processed_loans + raw_loans tables)
Outputs:  EDA_Report.pdf
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from scipy import stats
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT1   = "#58a6ff"   # blue
ACCENT2   = "#ff7b72"   # red/orange
ACCENT3   = "#3fb950"   # green
ACCENT4   = "#d2a8ff"   # purple
ACCENT5   = "#ffa657"   # amber
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
GRID_COL  = "#21262d"

PALETTE   = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
             "#79c0ff", "#ffab70", "#56d364", "#bc8cff", "#f78166"]

sns.set_theme(style="dark", rc={
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_PRI,
    "axes.titlecolor":   TEXT_PRI,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "grid.color":        GRID_COL,
    "text.color":        TEXT_PRI,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COL,
})

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data from creditpathai.db …")
conn = sqlite3.connect("creditpathai.db")
df_raw  = pd.read_sql("SELECT * FROM raw_loans",       conn)
df_proc = pd.read_sql("SELECT * FROM processed_loans", conn)
conn.close()
print(f"  raw_loans:       {df_raw.shape}")
print(f"  processed_loans: {df_proc.shape}")

TARGET = "loanStatus"   # 0 = no default, 1 = default

# helper label
def lbl(v):
    return "Default (1)" if v == 1 else "No Default (0)"

# ── PDF setup ─────────────────────────────────────────────────────────────────
PDF_PATH = "EDA_Report.pdf"
pdf = PdfPages(PDF_PATH)

def save(fig):
    pdf.savefig(fig, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)

def std_fig(rows=1, cols=1, w=16, h=9):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    return fig, axes

def title_text(ax, txt, sub=""):
    ax.set_title(txt, color=TEXT_PRI, fontsize=13, fontweight="bold", pad=12)
    if sub:
        ax.set_xlabel(sub, color=TEXT_SEC, fontsize=9)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 0 – Cover
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(DARK_BG)
ax  = fig.add_subplot(111)
ax.set_facecolor(DARK_BG)
ax.axis("off")

ax.text(0.5, 0.72, "CreditPath AI", ha="center", va="center",
        fontsize=52, fontweight="bold", color=ACCENT1,
        transform=ax.transAxes)
ax.text(0.5, 0.58, "Exploratory Data Analysis Report",
        ha="center", va="center", fontsize=28, color=TEXT_PRI,
        transform=ax.transAxes)
ax.text(0.5, 0.46, f"Dataset: processed_loans  |  Rows: {len(df_proc):,}  |  Features: {df_proc.shape[1]}",
        ha="center", va="center", fontsize=14, color=TEXT_SEC,
        transform=ax.transAxes)
ax.text(0.5, 0.38, "Target  →  loanStatus  (0 = No Default  |  1 = Default)",
        ha="center", va="center", fontsize=13, color=ACCENT2,
        transform=ax.transAxes)

# decorative line
for y in [0.30, 0.85]:
    ax.axhline(y, xmin=0.1, xmax=0.9, color=ACCENT1, linewidth=1.5, alpha=0.4)

save(fig)
print("Page 0 – Cover done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dataset Overview
# ════════════════════════════════════════════════════════════════════════════
fig, axes = std_fig(1, 2, 16, 8)
fig.suptitle("Dataset Overview", color=TEXT_PRI, fontsize=16,
             fontweight="bold", y=1.01)

# --- Missing-value heatmap (processed) ---
ax = axes[0]
miss_pct = (df_proc.isnull().mean() * 100).sort_values(ascending=False)
miss_cols = miss_pct[miss_pct > 0]
if miss_cols.empty:
    ax.text(0.5, 0.5, "No Missing Values\nin processed_loans ✓",
            ha="center", va="center", fontsize=16, color=ACCENT3,
            transform=ax.transAxes)
    ax.set_facecolor(CARD_BG)
    ax.axis("off")
else:
    bars = ax.barh(miss_cols.index, miss_cols.values, color=ACCENT2, alpha=0.85)
    ax.set_xlabel("% Missing", color=TEXT_SEC)
    ax.set_title("Missing Values – processed_loans", color=TEXT_PRI,
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors=TEXT_SEC)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5)
    for bar, val in zip(bars, miss_cols.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=TEXT_PRI, fontsize=8)

# --- Missing-value heatmap (raw) ---
ax = axes[1]
miss_raw = (df_raw.isnull().mean() * 100).sort_values(ascending=False)
miss_raw_cols = miss_raw[miss_raw > 0]
if miss_raw_cols.empty:
    ax.text(0.5, 0.5, "No Missing Values\nin raw_loans ✓",
            ha="center", va="center", fontsize=16, color=ACCENT3,
            transform=ax.transAxes)
    ax.set_facecolor(CARD_BG)
    ax.axis("off")
else:
    bars = ax.barh(miss_raw_cols.index, miss_raw_cols.values,
                   color=ACCENT5, alpha=0.85)
    ax.set_xlabel("% Missing", color=TEXT_SEC)
    ax.set_title("Missing Values – raw_loans", color=TEXT_PRI,
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors=TEXT_SEC)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5)
    for bar, val in zip(bars, miss_raw_cols.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=TEXT_PRI, fontsize=8)

fig.tight_layout()
save(fig)
print("Page 1 – Dataset Overview done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Target Variable Analysis
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Target Variable Analysis  –  loanStatus (0=No Default, 1=Default)",
             color=TEXT_PRI, fontsize=15, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

vc       = df_proc[TARGET].value_counts().sort_index()
labels   = [lbl(i) for i in vc.index]
counts   = vc.values
pcts     = counts / counts.sum() * 100
colors   = [ACCENT3, ACCENT2]

# Donut
ax0 = fig.add_subplot(gs[0, 0])
wedges, texts, autotexts = ax0.pie(
    counts, labels=labels, autopct="%1.1f%%",
    colors=colors, startangle=90,
    wedgeprops=dict(width=0.5, edgecolor=DARK_BG, linewidth=2))
for t in texts:
    t.set_color(TEXT_PRI)
    t.set_fontsize(9)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax0.set_title("Class Distribution (Donut)", color=TEXT_PRI,
              fontsize=11, fontweight="bold")

# Bar
ax1 = fig.add_subplot(gs[0, 1])
bars = ax1.bar(labels, counts, color=colors, edgecolor=DARK_BG,
               linewidth=1.5, width=0.5)
for bar, pct, cnt in zip(bars, pcts, counts):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 200,
             f"{cnt:,}\n({pct:.1f}%)",
             ha="center", color=TEXT_PRI, fontsize=10, fontweight="bold")
ax1.set_ylabel("Count", color=TEXT_SEC)
ax1.set_title("Absolute Counts", color=TEXT_PRI, fontsize=11, fontweight="bold")
ax1.grid(axis="y", color=GRID_COL, linewidth=0.5)
ax1.tick_params(colors=TEXT_SEC)
ax1.set_facecolor(CARD_BG)

# Key stats
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
ax2.set_facecolor(CARD_BG)
default_rate = pcts[1] if len(pcts) > 1 else pcts[0]
imbalance_ratio = counts[0] / counts[1] if len(counts) > 1 else 1
stats_txt = (
    f"Default Rate:      {default_rate:.2f}%\n"
    f"No-Default Rate:   {pcts[0]:.2f}%\n\n"
    f"Total Records:     {counts.sum():,}\n"
    f"Defaults:          {counts[1] if len(counts)>1 else 0:,}\n"
    f"Non-Defaults:      {counts[0]:,}\n\n"
    f"Imbalance Ratio:   1 : {imbalance_ratio:.1f}"
)
ax2.text(0.1, 0.85, "Key Statistics", color=ACCENT1, fontsize=13,
        fontweight="bold", transform=ax2.transAxes, va="top")
ax2.text(0.1, 0.70, stats_txt, color=TEXT_PRI, fontsize=10,
        transform=ax2.transAxes, va="top", family="monospace")

# loanStatus in raw (string categories)
raw_ls = df_raw["loanStatus"].value_counts()
ax3 = fig.add_subplot(gs[1, :])
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(raw_ls))]
bars = ax3.barh(raw_ls.index, raw_ls.values, color=bar_colors,
                edgecolor=DARK_BG, linewidth=0.8)
ax3.set_title("Raw loanStatus Categories (raw_loans table)", color=TEXT_PRI,
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Count", color=TEXT_SEC)
ax3.grid(axis="x", color=GRID_COL, linewidth=0.5)
ax3.set_facecolor(CARD_BG)
ax3.tick_params(colors=TEXT_SEC)
for bar, val in zip(bars, raw_ls.values):
    pct_val = val / raw_ls.sum() * 100
    ax3.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
             f"{val:,}  ({pct_val:.1f}%)", va="center",
             color=TEXT_PRI, fontsize=8)

save(fig)
print("Page 2 – Target Variable done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Descriptive Statistics
# ════════════════════════════════════════════════════════════════════════════
numeric_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

desc = df_proc[numeric_cols + [TARGET]].describe().T
desc["skewness"] = df_proc[numeric_cols].skew()
desc["kurtosis"] = df_proc[numeric_cols].kurtosis()

fig, ax = plt.subplots(figsize=(20, max(10, len(desc)*0.35)))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis("off")
ax.set_title("Descriptive Statistics (processed_loans – numeric features)",
             color=TEXT_PRI, fontsize=13, fontweight="bold", pad=12)

show_cols = ["count","mean","std","min","25%","50%","75%","max","skewness","kurtosis"]
show_cols = [c for c in show_cols if c in desc.columns]
tbl_data  = [[f"{v:.3g}" if isinstance(v, (int,float)) else str(v)
              for v in desc.loc[idx, show_cols]] for idx in desc.index]

tbl = ax.table(
    cellText=tbl_data,
    rowLabels=desc.index.tolist(),
    colLabels=show_cols,
    cellLoc="center", rowLoc="right",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.3)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(CARD_BG if r % 2 == 0 else "#1c2129")
    cell.set_edgecolor(GRID_COL)
    cell.set_text_props(color=TEXT_PRI)
    if r == 0:
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color=ACCENT1, fontweight="bold")
    if c == -1:
        cell.set_facecolor("#1a1f27")
        cell.set_text_props(color=ACCENT4, fontweight="bold")

fig.tight_layout()
save(fig)
print("Page 3 – Descriptive Statistics done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 & 5 – Histograms of all numeric features
# ════════════════════════════════════════════════════════════════════════════
COLS_PER_PAGE = 12
pages_hist = [numeric_cols[i:i+COLS_PER_PAGE]
              for i in range(0, len(numeric_cols), COLS_PER_PAGE)]

for pidx, page_cols in enumerate(pages_hist):
    n = len(page_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Feature Distributions (Page {pidx+1}/{len(pages_hist)})",
                 color=TEXT_PRI, fontsize=15, fontweight="bold", y=1.01)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(page_cols):
        ax = axes_flat[i]
        ax.set_facecolor(CARD_BG)
        data0 = df_proc.loc[df_proc[TARGET] == 0, col].dropna()
        data1 = df_proc.loc[df_proc[TARGET] == 1, col].dropna()
        bins  = min(40, max(10, int(len(data0)**0.4)))
        lo, hi = df_proc[col].quantile(0.01), df_proc[col].quantile(0.99)
        rng = (lo, hi) if lo < hi else None
        ax.hist(data0, bins=bins, color=ACCENT3, alpha=0.65, label="No Default",
                range=rng, density=True)
        ax.hist(data1, bins=bins, color=ACCENT2, alpha=0.65, label="Default",
                range=rng, density=True)
        ax.set_title(col, color=TEXT_PRI, fontsize=8, fontweight="bold")
        ax.tick_params(colors=TEXT_SEC, labelsize=6)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
        ax.set_yticks([])
        skew_val = df_proc[col].skew()
        ax.text(0.98, 0.95, f"skew={skew_val:.2f}", ha="right", va="top",
                transform=ax.transAxes, color=TEXT_SEC, fontsize=6)
        if i == 0:
            ax.legend(fontsize=6, framealpha=0.3)

    for j in range(len(page_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    save(fig)
    print(f"Page – Histograms batch {pidx+1} done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Box-plots: key features vs Target
# ════════════════════════════════════════════════════════════════════════════
key_features = [
    "loanAmount", "interestRate", "annualIncome", "dtiRatio",
    "monthlyPayment", "revolvingUtilizationRate", "numDelinquency2Years",
    "numInquiries6Mon", "numDerogatoryRec", "loan_to_income_ratio",
    "grade_score", "lengthCreditHistory"
]
key_features = [c for c in key_features if c in df_proc.columns]

ncols = 4
nrows = int(np.ceil(len(key_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Key Feature Distributions vs Target (Box-plots)",
             color=TEXT_PRI, fontsize=15, fontweight="bold", y=1.01)
axes_flat = axes.flatten()

for i, col in enumerate(key_features):
    ax = axes_flat[i]
    ax.set_facecolor(CARD_BG)
    grp0 = df_proc.loc[df_proc[TARGET] == 0, col].dropna()
    grp1 = df_proc.loc[df_proc[TARGET] == 1, col].dropna()
    bp = ax.boxplot([grp0, grp1],
                    patch_artist=True,
                    notch=True,
                    medianprops=dict(color="#fff", linewidth=2),
                    whiskerprops=dict(color=TEXT_SEC),
                    capprops=dict(color=TEXT_SEC),
                    flierprops=dict(marker="o", markersize=1.5,
                                   alpha=0.2, color=TEXT_SEC))
    bp["boxes"][0].set_facecolor(ACCENT3 + "99")
    if len(bp["boxes"]) > 1:
        bp["boxes"][1].set_facecolor(ACCENT2 + "99")
    ax.set_xticklabels(["No Default", "Default"], color=TEXT_PRI, fontsize=8)
    ax.set_title(col, color=TEXT_PRI, fontsize=8, fontweight="bold")
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.4)

    # t-test
    t_stat, p_val = stats.ttest_ind(grp0, grp1, equal_var=False)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax.text(0.98, 0.97, sig, ha="right", va="top",
            transform=ax.transAxes, color=ACCENT5, fontsize=10, fontweight="bold")

for j in range(len(key_features), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.tight_layout()
save(fig)
print("Page – Box-plots done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Correlation Heatmap (full processed_loans)
# ════════════════════════════════════════════════════════════════════════════
corr_df = df_proc[numeric_cols + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(22, 18))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr_df, dtype=bool))

sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0,
            vmin=-1, vmax=1,
            annot=len(corr_df) <= 25,
            fmt=".2f", annot_kws={"size": 6},
            linewidths=0.3, linecolor=DARK_BG,
            square=True, ax=ax,
            cbar_kws={"shrink": 0.6})

ax.set_title("Correlation Heatmap – Pairwise Pearson Correlation",
             color=TEXT_PRI, fontsize=15, fontweight="bold", pad=15)
ax.tick_params(colors=TEXT_PRI, labelsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                   color=TEXT_SEC, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   color=TEXT_SEC, fontsize=7)

fig.tight_layout()
save(fig)
print("Page – Correlation Heatmap done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Feature Correlations with Target (sorted bar)
# ════════════════════════════════════════════════════════════════════════════
target_corr = df_proc[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()

fig, ax = plt.subplots(figsize=(16, max(8, len(target_corr)*0.28)))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

bar_colors = [ACCENT2 if v > 0 else ACCENT1 for v in target_corr.values]
bars = ax.barh(target_corr.index, target_corr.values,
               color=bar_colors, edgecolor=DARK_BG, linewidth=0.5)
ax.axvline(0, color=TEXT_SEC, linewidth=0.8, linestyle="--")
ax.set_title("Feature Correlation with Target (loanStatus)",
             color=TEXT_PRI, fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation Coefficient", color=TEXT_SEC)
ax.tick_params(colors=TEXT_PRI, labelsize=8)
ax.grid(axis="x", color=GRID_COL, linewidth=0.5)

for bar, val in zip(bars, target_corr.values):
    ax.text(val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}",
            ha="left" if val >= 0 else "right",
            va="center", color=TEXT_PRI, fontsize=7)

fig.tight_layout()
save(fig)
print("Page – Feature Correlation with Target done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Categorical features (raw_loans) vs Default
# ════════════════════════════════════════════════════════════════════════════
# Map loanStatus in raw to binary
status_map = {s: (0 if s.lower() in ["current","fully paid","does not meet the credit policy. status:fully paid"] else 1)
              for s in df_raw["loanStatus"].unique()}
df_raw_tmp = df_raw.copy()
df_raw_tmp["default"] = df_raw_tmp["loanStatus"].map(status_map)

cat_cols = ["purpose","grade","homeOwnership","term","residentialState"]
cat_cols = [c for c in cat_cols if c in df_raw_tmp.columns]

ncols = 2
nrows = int(np.ceil(len(cat_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Categorical Features vs Default Rate",
             color=TEXT_PRI, fontsize=15, fontweight="bold", y=1.01)
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, col in enumerate(cat_cols):
    ax = axes_flat[i]
    ax.set_facecolor(CARD_BG)
    grp = df_raw_tmp.groupby(col)["default"].agg(["mean","count"]).reset_index()
    grp.columns = [col, "default_rate", "count"]
    grp["default_rate"] *= 100
    grp = grp.sort_values("default_rate", ascending=True)

    # Limit to top 20 for residentialState
    if col == "residentialState":
        grp = grp.tail(20)

    bar_c = [PALETTE[j % len(PALETTE)] for j in range(len(grp))]
    bars  = ax.barh(grp[col].astype(str), grp["default_rate"],
                    color=bar_c, edgecolor=DARK_BG, linewidth=0.5)
    ax.set_title(f"Default Rate by {col}", color=TEXT_PRI,
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Default Rate (%)", color=TEXT_SEC)
    ax.tick_params(colors=TEXT_PRI, labelsize=7)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.4)

    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height()/2,
                f"{row['default_rate']:.1f}%  (n={int(row['count']):,})",
                va="center", color=TEXT_PRI, fontsize=6.5)

for j in range(len(cat_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.tight_layout()
save(fig)
print("Page – Categorical vs Default done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Interest Rate & Loan Amount deep-dive
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Interest Rate & Loan Amount Deep-Dive",
             color=TEXT_PRI, fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Loan Amount KDE by target
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(CARD_BG)
for tgt, col, lbl_txt in [(0, ACCENT3, "No Default"), (1, ACCENT2, "Default")]:
    data = df_proc.loc[df_proc[TARGET] == tgt, "loanAmount"].dropna()
    data.plot.kde(ax=ax, color=col, label=lbl_txt, linewidth=2)
ax.set_title("Loan Amount KDE", color=TEXT_PRI, fontsize=10, fontweight="bold")
ax.set_xlabel("Loan Amount", color=TEXT_SEC)
ax.legend(fontsize=8)
ax.grid(color=GRID_COL, linewidth=0.4)
ax.tick_params(colors=TEXT_SEC)

# 2. Interest Rate KDE by target
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(CARD_BG)
for tgt, col, lbl_txt in [(0, ACCENT3, "No Default"), (1, ACCENT2, "Default")]:
    data = df_proc.loc[df_proc[TARGET] == tgt, "interestRate"].dropna()
    data.plot.kde(ax=ax, color=col, label=lbl_txt, linewidth=2)
ax.set_title("Interest Rate KDE", color=TEXT_PRI, fontsize=10, fontweight="bold")
ax.set_xlabel("Interest Rate", color=TEXT_SEC)
ax.legend(fontsize=8)
ax.grid(color=GRID_COL, linewidth=0.4)
ax.tick_params(colors=TEXT_SEC)

# 3. DTI Ratio histogram
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(CARD_BG)
for tgt, col, lbl_txt in [(0, ACCENT3, "No Default"), (1, ACCENT2, "Default")]:
    data = df_proc.loc[df_proc[TARGET] == tgt, "dtiRatio"].dropna()
    ax.hist(data, bins=40, color=col, alpha=0.6, label=lbl_txt, density=True)
ax.set_title("DTI Ratio Distribution", color=TEXT_PRI, fontsize=10, fontweight="bold")
ax.set_xlabel("DTI Ratio", color=TEXT_SEC)
ax.legend(fontsize=8)
ax.grid(color=GRID_COL, linewidth=0.4)
ax.tick_params(colors=TEXT_SEC)

# 4. Loan Amount vs Interest Rate scatter
ax = fig.add_subplot(gs[1, :2])
ax.set_facecolor(CARD_BG)
sample = df_proc.sample(min(5000, len(df_proc)), random_state=42)
for tgt, col, lbl_txt in [(0, ACCENT3, "No Default"), (1, ACCENT2, "Default")]:
    mask = sample[TARGET] == tgt
    ax.scatter(sample.loc[mask, "loanAmount"],
               sample.loc[mask, "interestRate"],
               c=col, alpha=0.35, s=8, label=lbl_txt)
ax.set_xlabel("Loan Amount", color=TEXT_SEC)
ax.set_ylabel("Interest Rate", color=TEXT_SEC)
ax.set_title("Loan Amount vs Interest Rate (sample 5K)", color=TEXT_PRI,
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(color=GRID_COL, linewidth=0.4)
ax.tick_params(colors=TEXT_SEC)

# 5. Annual Income vs Loan Amount
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor(CARD_BG)
for tgt, col, lbl_txt in [(0, ACCENT3, "No Default"), (1, ACCENT2, "Default")]:
    mask = sample[TARGET] == tgt
    ax.scatter(sample.loc[mask, "annualIncome"],
               sample.loc[mask, "loanAmount"],
               c=col, alpha=0.35, s=8, label=lbl_txt)
ax.set_xlabel("Annual Income", color=TEXT_SEC)
ax.set_ylabel("Loan Amount", color=TEXT_SEC)
ax.set_title("Annual Income vs Loan Amount", color=TEXT_PRI,
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(color=GRID_COL, linewidth=0.4)
ax.tick_params(colors=TEXT_SEC)

save(fig)
print("Page – Interest Rate / Loan Amount Deep-Dive done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Engineered Feature Analysis
# ════════════════════════════════════════════════════════════════════════════
eng_feats = [c for c in df_proc.columns if c not in [
    "loanAmount","interestRate","annualIncome","dtiRatio",
    "monthlyPayment","revolvingUtilizationRate",
    "numDelinquency2Years","numInquiries6Mon",
    "numDerogatoryRec","lengthCreditHistory",
    "numTotalCreditLines","numOpenCreditLines",
    "numOpenCreditLines1Year","revolvingBalance",
    "numChargeoff1year","isJointApplication","incomeVerified",
    "yearsEmployment","term_months","grade_score",
    TARGET
] and df_proc[c].dtype in [np.float64, np.int64]]

eng_feats = eng_feats[:12]  # first 12 engineered
if eng_feats:
    ncols = 4
    nrows = int(np.ceil(len(eng_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Engineered Features vs Target",
                 color=TEXT_PRI, fontsize=15, fontweight="bold", y=1.01)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(eng_feats):
        ax = axes_flat[i]
        ax.set_facecolor(CARD_BG)
        data0 = df_proc.loc[df_proc[TARGET] == 0, col].dropna()
        data1 = df_proc.loc[df_proc[TARGET] == 1, col].dropna()
        bins  = min(40, 15)
        lo, hi = df_proc[col].quantile(0.01), df_proc[col].quantile(0.99)
        rng = (lo, hi) if lo < hi else None
        ax.hist(data0, bins=bins, range=rng, color=ACCENT3, alpha=0.65,
                label="No Default", density=True)
        ax.hist(data1, bins=bins, range=rng, color=ACCENT2, alpha=0.65,
                label="Default", density=True)
        ax.set_title(col, color=TEXT_PRI, fontsize=8, fontweight="bold")
        ax.tick_params(colors=TEXT_SEC, labelsize=6)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
        ax.set_yticks([])
        t_stat, p_val = stats.ttest_ind(data0, data1, equal_var=False)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "ns")
        ax.text(0.98, 0.95, sig, ha="right", va="top",
                transform=ax.transAxes, color=ACCENT5, fontsize=10)
        if i == 0:
            ax.legend(fontsize=6, framealpha=0.3)

    for j in range(len(eng_feats), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    save(fig)
    print("Page – Engineered Features done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Loan Purpose & Grade Analysis
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Purpose & Grade Analysis",
             color=TEXT_PRI, fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

# Purpose cols (one-hot in processed)
purpose_cols = [c for c in df_proc.columns if c.startswith("purpose_")]
if purpose_cols:
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(CARD_BG)
    def_rates = {}
    counts_all = {}
    for pc in purpose_cols:
        sub = df_proc[df_proc[pc] == 1]
        if len(sub) > 0:
            def_rates[pc.replace("purpose_", "")] = sub[TARGET].mean() * 100
            counts_all[pc.replace("purpose_", "")] = len(sub)
    dr_series = pd.Series(def_rates).sort_values()
    bar_c = [PALETTE[i % len(PALETTE)] for i in range(len(dr_series))]
    bars = ax.bar(dr_series.index, dr_series.values,
                  color=bar_c, edgecolor=DARK_BG, width=0.5)
    ax.set_title("Default Rate by Loan Purpose", color=TEXT_PRI,
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Default Rate (%)", color=TEXT_SEC)
    ax.tick_params(colors=TEXT_PRI)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
    for bar, key in zip(bars, dr_series.index):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}%\n(n={counts_all[key]:,})",
                ha="center", color=TEXT_PRI, fontsize=8)

# Grade distribution
if "grade_score" in df_proc.columns:
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(CARD_BG)
    gr_df = df_raw.groupby("grade")["loanStatus"].apply(
        lambda x: (x.map(status_map) == 1).mean() * 100
    ).reset_index()
    gr_df.columns = ["grade", "default_rate"]
    gr_df = gr_df.sort_values("grade")
    bars = ax.bar(gr_df["grade"], gr_df["default_rate"],
                  color=PALETTE[:len(gr_df)], edgecolor=DARK_BG, width=0.5)
    ax.set_title("Default Rate by Grade", color=TEXT_PRI,
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Default Rate (%)", color=TEXT_SEC)
    ax.tick_params(colors=TEXT_PRI)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}%",
                ha="center", color=TEXT_PRI, fontsize=8)

# Home ownership
home_cols = [c for c in df_proc.columns if c.startswith("homeOwnership_")]
if home_cols:
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(CARD_BG)
    ho_rates = {}
    for hc in home_cols + ["neither_own_nor_rent"]:
        if hc == "neither_own_nor_rent":
            mask = (df_proc.get("homeOwnership_own", pd.Series(0)) == 0) & \
                   (df_proc.get("homeOwnership_rent", pd.Series(0)) == 0)
            sub  = df_proc[mask]
            key  = "mortgage/other"
        else:
            sub = df_proc[df_proc[hc] == 1]
            key = hc.replace("homeOwnership_", "")
        if len(sub) > 0:
            ho_rates[key] = sub[TARGET].mean() * 100
    ho_series = pd.Series(ho_rates).sort_values()
    bars = ax.bar(ho_series.index, ho_series.values,
                  color=[ACCENT3, ACCENT2, ACCENT4][:len(ho_series)],
                  edgecolor=DARK_BG, width=0.4)
    ax.set_title("Default Rate by Home Ownership", color=TEXT_PRI,
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Default Rate (%)", color=TEXT_SEC)
    ax.tick_params(colors=TEXT_PRI)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}%",
                ha="center", color=TEXT_PRI, fontsize=9)

save(fig)
print("Page – Purpose & Grade Analysis done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – Credit History & Delinquency
# ════════════════════════════════════════════════════════════════════════════
fig, axes = std_fig(2, 3, 18, 10)
fig.suptitle("Credit History & Delinquency Analysis",
             color=TEXT_PRI, fontsize=15, fontweight="bold")
axes_flat = axes.flatten()

credit_feats = [
    "lengthCreditHistory", "numTotalCreditLines", "numOpenCreditLines",
    "numDelinquency2Years", "numDerogatoryRec", "numChargeoff1year"
]
credit_feats = [c for c in credit_feats if c in df_proc.columns]

for i, col in enumerate(credit_feats):
    ax = axes_flat[i]
    ax.set_facecolor(CARD_BG)
    d0 = df_proc.loc[df_proc[TARGET] == 0, col].dropna()
    d1 = df_proc.loc[df_proc[TARGET] == 1, col].dropna()
    lo = min(d0.quantile(0.01), d1.quantile(0.01))
    hi = max(d0.quantile(0.99), d1.quantile(0.99))
    rng = (lo, hi) if lo < hi else None
    ax.hist(d0, bins=30, color=ACCENT3, alpha=0.7, range=rng,
            label="No Default", density=True)
    ax.hist(d1, bins=30, color=ACCENT2, alpha=0.7, range=rng,
            label="Default", density=True)
    ax.set_title(col, color=TEXT_PRI, fontsize=9, fontweight="bold")
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.4)
    ax.set_yticks([])
    m0, m1 = d0.median(), d1.median()
    ax.axvline(m0, color=ACCENT3, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.axvline(m1, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.text(0.02, 0.95, f"Med(0)={m0:.1f}\nMed(1)={m1:.1f}",
            ha="left", va="top", transform=ax.transAxes,
            color=TEXT_SEC, fontsize=7, family="monospace")
    if i == 0:
        ax.legend(fontsize=7, framealpha=0.3)

fig.tight_layout()
save(fig)
print("Page – Credit History done")

# ════════════════════════════════════════════════════════════════════════════
# PAGE – KPI Summary
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(DARK_BG)
ax = fig.add_subplot(111)
ax.set_facecolor(DARK_BG)
ax.axis("off")
fig.suptitle("EDA Summary – Key Findings", color=TEXT_PRI,
             fontsize=18, fontweight="bold", y=0.98)

# Compute top positive/negative correlations with target
top_pos = target_corr.nlargest(5)
top_neg = target_corr.nsmallest(5)

default_pct = df_proc[TARGET].mean() * 100
avg_int_def = df_proc.loc[df_proc[TARGET]==1, "interestRate"].mean()
avg_int_ndf = df_proc.loc[df_proc[TARGET]==0, "interestRate"].mean()
avg_dti_def = df_proc.loc[df_proc[TARGET]==1, "dtiRatio"].mean()
avg_dti_ndf = df_proc.loc[df_proc[TARGET]==0, "dtiRatio"].mean()

kfis = [
    ("Overall Default Rate",  f"{default_pct:.2f}%",  ACCENT2),
    ("Avg Interest Rate (Default)",   f"{avg_int_def:.2f}%",  ACCENT2),
    ("Avg Interest Rate (No Default)", f"{avg_int_ndf:.2f}%", ACCENT3),
    ("Avg DTI Ratio (Default)",   f"{avg_dti_def:.2f}",  ACCENT2),
    ("Avg DTI Ratio (No Default)", f"{avg_dti_ndf:.2f}", ACCENT3),
    ("Total Records",  f"{len(df_proc):,}",  ACCENT4),
]

for idx, (label, value, color) in enumerate(kfis):
    row, col_idx = divmod(idx, 3)
    x = 0.05 + col_idx * 0.32
    y = 0.82 - row * 0.28

    rect = FancyBboxPatch((x, y-0.16), 0.28, 0.20,
                          transform=ax.transAxes,
                          boxstyle="round,pad=0.01",
                          facecolor=CARD_BG, edgecolor=color,
                          linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + 0.14, y - 0.01, value, ha="center", va="center",
            fontsize=20, fontweight="bold", color=color,
            transform=ax.transAxes, zorder=3)
    ax.text(x + 0.14, y - 0.10, label, ha="center", va="center",
            fontsize=9, color=TEXT_SEC, transform=ax.transAxes, zorder=3)

# Top correlations note
note_y = 0.20
ax.text(0.05, note_y, "Top Positive Correlations with Default:",
        color=ACCENT2, fontsize=10, fontweight="bold",
        transform=ax.transAxes, va="top")
for i, (feat, val) in enumerate(top_pos.items()):
    ax.text(0.05, note_y - 0.05 - i*0.04,
            f"  {feat}: {val:+.3f}",
            color=TEXT_PRI, fontsize=9, family="monospace",
            transform=ax.transAxes, va="top")

ax.text(0.55, note_y, "Top Negative Correlations with Default:",
        color=ACCENT3, fontsize=10, fontweight="bold",
        transform=ax.transAxes, va="top")
for i, (feat, val) in enumerate(top_neg.items()):
    ax.text(0.55, note_y - 0.05 - i*0.04,
            f"  {feat}: {val:+.3f}",
            color=TEXT_PRI, fontsize=9, family="monospace",
            transform=ax.transAxes, va="top")

save(fig)
print("Page – KPI Summary done")

# ── Close PDF ─────────────────────────────────────────────────────────────────
pdf.close()
print(f"\n✅  EDA_Report.pdf saved  →  {PDF_PATH}")
