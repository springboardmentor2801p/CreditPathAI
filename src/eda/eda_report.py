import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")
REPORTS_PATH = "data/reports/eda/"
os.makedirs(REPORTS_PATH, exist_ok=True)

# Style
sns.set_theme(style="whitegrid")
PALETTE = ["#2ecc71", "#e74c3c"]


def load_data() -> pd.DataFrame:
    logger.info("Loading data from SQLite...")
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df = pd.read_sql("SELECT * FROM loan_engineered", engine)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── 1. OVERVIEW ──────────────────────────────────────────────────────────────
def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Distribution — Loan Default Status", fontsize=14, fontweight='bold')

    counts = df['status'].value_counts()
    labels = ['No Default (0)', 'Default (1)']

    axes[0].pie(counts, labels=labels, autopct='%1.1f%%',
                colors=PALETTE, startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title("Class Balance")

    axes[1].bar(labels, counts.values, color=PALETTE, edgecolor='white', linewidth=1.5)
    axes[1].set_title("Count by Class")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 500, f"{v:,}", ha='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "01_target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 2. NUMERIC DISTRIBUTIONS ─────────────────────────────────────────────────
def plot_numeric_distributions(df):
    numeric_cols = ['loan_amount', 'income', 'credit_score',
                    'ltv', 'dtir1', 'property_value',
                    'credit_risk_score', 'income_loan_ratio']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Numeric Feature Distributions by Default Status",
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for status, color, label in zip([0, 1], PALETTE, ['No Default', 'Default']):
            subset = df[df['status'] == status][col].dropna()
            axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                        label=label, density=True)
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "02_numeric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 3. CATEGORICAL FEATURES ───────────────────────────────────────────────────
def plot_categorical_features(df):
    cat_cols = ['loan_type', 'loan_purpose', 'credit_worthiness',
                'business_or_commercial', 'neg_ammortization',
                'lump_sum_payment', 'occupancy_type', 'total_units']
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Default Rate by Categorical Features",
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        default_rate = df.groupby(col)['status'].mean().sort_values(ascending=False)
        bars = axes[i].bar(default_rate.index.astype(str),
                          default_rate.values * 100,
                          color='#e74c3c', alpha=0.8, edgecolor='white')
        axes[i].set_title(f"Default Rate: {col.replace('_', ' ').title()}")
        axes[i].set_ylabel("Default Rate (%)")
        axes[i].set_xlabel(col)
        axes[i].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, default_rate.values):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f'{val*100:.1f}%', ha='center', fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "03_categorical_default_rates.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 4. CORRELATION HEATMAP ────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=np.number).drop(
        columns=['id', 'year'], errors='ignore'
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "04_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 5. LTV & CREDIT SCORE ANALYSIS ───────────────────────────────────────────
def plot_risk_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Risk Factor Analysis", fontsize=14, fontweight='bold')

    # LTV bands vs default rate
    ltv_default = df.groupby('ltv_band')['status'].mean() * 100
    axes[0].bar(ltv_default.index.astype(str), ltv_default.values,
                color='#e74c3c', alpha=0.8, edgecolor='white')
    axes[0].set_title("Default Rate by LTV Band")
    axes[0].set_xlabel("LTV Band (0=Low Risk → 4=High Risk)")
    axes[0].set_ylabel("Default Rate (%)")
    for i, v in enumerate(ltv_default.values):
        axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

    # Credit score distribution
    for status, color, label in zip([0, 1], PALETTE, ['No Default', 'Default']):
        subset = df[df['status'] == status]['credit_score'].dropna()
        axes[1].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[1].set_title("Credit Score Distribution")
    axes[1].set_xlabel("Credit Score")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # High risk flag
    hr_counts = df.groupby(['high_risk_flag', 'status']).size().unstack(fill_value=0)
    hr_counts.plot(kind='bar', ax=axes[2], color=PALETTE,
                   edgecolor='white', rot=0)
    axes[2].set_title("High Risk Flag vs Default")
    axes[2].set_xlabel("High Risk Flag (0=No, 1=Yes)")
    axes[2].set_ylabel("Count")
    axes[2].legend(['No Default', 'Default'])

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "05_risk_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 6. MISSING VALUES ─────────────────────────────────────────────────────────
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(missing) == 0:
        ax.text(0.5, 0.5, 'No Missing Values Found!\nData is Clean.',
                ha='center', va='center', fontsize=16,
                color='#2ecc71', fontweight='bold',
                transform=ax.transAxes)
        ax.set_title("Missing Values Analysis", fontsize=14, fontweight='bold')
    else:
        ax.barh(missing.index, missing.values, color='#e74c3c', alpha=0.8)
        ax.set_title("Missing Values by Column", fontsize=14, fontweight='bold')
        ax.set_xlabel("Missing Count")

    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, "06_missing_values.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ── 7. SUMMARY STATS ──────────────────────────────────────────────────────────
def save_summary_stats(df):
    stats = df.describe().T
    stats['missing'] = df.isnull().sum()
    stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    stats['dtype'] = df.dtypes

    path = os.path.join(REPORTS_PATH, "summary_statistics.csv")
    stats.to_csv(path)
    logger.info(f"Saved summary stats to: {path}")

    logger.info("\nDATASET OVERVIEW:")
    logger.info(f"  Total Records  : {len(df):,}")
    logger.info(f"  Total Features : {df.shape[1]}")
    logger.info(f"  Default Rate   : {df['status'].mean()*100:.1f}%")
    logger.info(f"  Missing Values : {df.isnull().sum().sum()}")


if __name__ == "__main__":
    df = load_data()

    logger.info("Generating EDA report...")

    save_summary_stats(df)
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_features(df)
    plot_correlation_heatmap(df)
    plot_risk_analysis(df)
    plot_missing_values(df)

    logger.info(f"EDA complete! All charts saved to: {REPORTS_PATH}")
    logger.info("Files generated:")
    for f in sorted(os.listdir(REPORTS_PATH)):
        logger.info(f"  - {f}")