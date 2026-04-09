import io
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────


# ── Name helpers ──────────────────────────────────────────────────────────────

def get_name_key(full_name) -> str:
    if pd.isna(full_name) or str(full_name).strip() == "":
        return ""
    parts = str(full_name).strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[-1]} {parts[0][0]}."


# ── Data loading ──────────────────────────────────────────────────────────────

def load_js_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tourney_date_dt"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df["winner_key"] = df["winner_name"].apply(get_name_key)
    df["loser_key"]  = df["loser_name"].apply(get_name_key)
    log.info("Loaded JS data: %d rows", len(df))
    return df


def download_odds_year(year: int, session: requests.Session) -> pd.DataFrame | None:
    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    log.info("Downloading %d ...", year)
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed %d: %s", year, exc)
        return None
    df = pd.read_excel(io.BytesIO(r.content))
    wanted = ["Date", "Winner", "Loser", "PSW", "PSL", "AvgW", "AvgL"]
    df = df[[c for c in wanted if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Winner", "Loser", "Date"])
    log.info("  %d usable rows", len(df))
    return df


def download_all_odds(years: list[int]) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; tennis-backtest/1.0)"})
    frames = []
    for year in years:
        df = download_odds_year(year, session)
        if df is not None:
            frames.append(df)
        time.sleep(1)
    combined = pd.concat(frames, ignore_index=True)
    log.info("Total odds rows: %d", len(combined))
    return combined


# ── Matching ──────────────────────────────────────────────────────────────────

def merge_datasets(js_df: pd.DataFrame, odds_df: pd.DataFrame, DATE_WINDOW) -> pd.DataFrame:
    log.info("Merging datasets...")
    odds_small = odds_df[["Date", "Winner", "Loser", "PSW", "PSL", "AvgW", "AvgL"]].copy()
    merged = pd.merge(
        js_df, odds_small,
        left_on=["winner_key", "loser_key"],
        right_on=["Winner", "Loser"],
        how="left",
    )
    merged["date_diff"] = (merged["Date"] - merged["tourney_date_dt"]).dt.days
    result = merged[(merged["date_diff"] >= -1) & (merged["date_diff"] <= DATE_WINDOW)].copy()
    result = result.drop_duplicates(subset=["match_id"])
    log.info(
        "Merge complete: %d / %d rows matched (%.1f%%)",
        len(result), len(js_df), 100 * len(result) / max(len(js_df), 1),
    )
    return result


# ── Odds assignment ───────────────────────────────────────────────────────────

def assign_all_odds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w_best = df[["PSW", "AvgW"]].max(axis=1)
    l_best = df[["PSL", "AvgL"]].max(axis=1)
    df["odds_A_pin"]  = np.where(df["outcome"] == 1, df["PSW"],  df["PSL"])
    df["odds_B_pin"]  = np.where(df["outcome"] == 0, df["PSW"],  df["PSL"])
    df["odds_A_avg"]  = np.where(df["outcome"] == 1, df["AvgW"], df["AvgL"])
    df["odds_B_avg"]  = np.where(df["outcome"] == 0, df["AvgW"], df["AvgL"])
    df["odds_A_best"] = np.where(df["outcome"] == 1, w_best, l_best)
    df["odds_B_best"] = np.where(df["outcome"] == 0, w_best, l_best)
    return df


# ── Strategy computation ──────────────────────────────────────────────────────

def compute_bets(df: pd.DataFrame, edge: float) -> pd.DataFrame:
    df = df.copy()
    df["bet_on_A"] = df["Trigger_Window1_pb"] > (1 / df["odds_A"] + edge)
    df["bet_on_B"] = (1 - df["Trigger_Window1_pb"]) > (1 / df["odds_B"] + edge)
    df["pnl"] = 0.0
    df.loc[df["bet_on_A"] & (df["outcome"] == 1), "pnl"] =  df["odds_A"] - 1
    df.loc[df["bet_on_A"] & (df["outcome"] == 0), "pnl"] = -1
    df.loc[df["bet_on_B"] & (df["outcome"] == 0), "pnl"] =  df["odds_B"] - 1
    df.loc[df["bet_on_B"] & (df["outcome"] == 1), "pnl"] = -1
    return df


def run_grid(
    df: pd.DataFrame,
    edges: list[float],
    thresholds: np.ndarray,

) -> tuple[np.ndarray, np.ndarray]:
    pnl_mat  = np.zeros((len(edges), len(thresholds)))
    bets_mat = np.zeros((len(edges), len(thresholds)))
    for i, edge in enumerate(edges):
        df_e = compute_bets(df, edge)
        for j, t in enumerate(thresholds):
            valid  = abs(df_e["Trigger_Window1_pb"] - 0.5) >= t
            n_bets = int((df_e["bet_on_A"] & valid).sum() + (df_e["bet_on_B"] & valid).sum())
            profit = df_e.loc[valid, "pnl"].sum()
            pnl_mat[i, j]  = profit
            bets_mat[i, j] = n_bets
           
    return pnl_mat, bets_mat


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_2d_panels(edges, thresholds, pnl_mat, bets_mat, output_dir, scenario_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(edges)))
    for i, edge in enumerate(edges):
        label = f"Edge: {edge*100:.0f}%"
        ax1.plot(thresholds, pnl_mat[i],  marker="o", linewidth=2, color=colors[i], label=label)
        ax2.plot(thresholds, bets_mat[i], marker="s", linewidth=2, color=colors[i], label=label, linestyle="--")
    ax1.set_title(f"[{scenario_name} Odds] Profit by abstention threshold & expected edge")
    ax1.set_ylabel("Total profit / loss (units)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Model edge", loc="best")
    ax2.set_title(f"[{scenario_name} Odds] Number of bets by abstention threshold & edge")
    ax2.set_xlabel("Abstention threshold (distance from 0.5)")
    ax2.set_ylabel("Total number of bets")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Model edge", loc="best")
    fig.tight_layout()
    out = output_dir / f"{scenario_name.lower()}_edge_abstention_2D_panels.png"
    fig.savefig(out, dpi=300)
    log.info("Saved %s", out)
    plt.close(fig)


def plot_3d_surface(edges, thresholds, pnl_mat, output_dir, scenario_name):
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(thresholds, edges)
    surf = ax.plot_surface(X, Y, pnl_mat, cmap="viridis", edgecolor="k", alpha=0.8)
    ax.set_title(f"[{scenario_name} Odds] 3D view: profit vs abstention threshold vs edge")
    ax.set_xlabel("Abstention threshold (dist from 0.5)")
    ax.set_ylabel("Model edge required")
    ax.set_zlabel("Total profit (units)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Profit (units)")
    fig.tight_layout()
    out = output_dir / f"{scenario_name.lower()}_edge_abstention_3D_surface.png"
    fig.savefig(out, dpi=300)
    log.info("Saved %s", out)
    plt.close(fig)


def plot_scenario_all_strategies(
    df_scenario: pd.DataFrame,
    edges: list[float],
    thresholds: np.ndarray,
    years: list[int],
    output_dir: Path,
    scenario_name: str,
) -> None:
    """
    One PNG per scenario.
    Grid layout: rows = edges (4), cols = thresholds (11).
    Each cell = yearly profit bar chart with a Δ overall balance badge.
    """
    n_rows      = len(edges)
    n_cols      = len(thresholds)
    year_labels = [str(y) for y in years]
    x           = np.arange(len(years))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 3.2),
        sharey="row",   # same y-scale per edge row for easy comparison
    )

    fig.suptitle(
        f"{scenario_name.upper()} ODDS  —  Profit per Year  "
        f"(rows = edge requirement  |  cols = abstention threshold)",
        fontsize=14, fontweight="bold", y=1.002,
    )

    for r, edge in enumerate(edges):
        df_e = compute_bets(df_scenario, edge)

        for c, t in enumerate(thresholds):
            ax    = axes[r][c]
            valid = abs(df_e["Trigger_Window1_pb"] - 0.5) >= t

            yearly_pnl = (
                df_e.loc[valid]
                .groupby("year")["pnl"]
                .sum()
                .reindex(years, fill_value=0.0)
                .values
            )
            total_pnl = float(yearly_pnl.sum())
            n_bets    = int(
                (df_e["bet_on_A"] & valid).sum()
                + (df_e["bet_on_B"] & valid).sum()
            )

            # ── Bars ──────────────────────────────────────────────────────
            bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in yearly_pnl]
            ax.bar(x, yearly_pnl, color=bar_colors, edgecolor="white", linewidth=0.4, width=0.72)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

            # ── Axis formatting ───────────────────────────────────────────
            ax.set_xticks(x)
            ax.set_xticklabels([y[-2:] for y in year_labels], fontsize=6.5)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            # ── Column header: threshold (top row only) ───────────────────
            if r == 0:
                ax.set_title(f"T = {t:.2f}", fontsize=8.5, fontweight="bold", pad=5)

            # ── Row label: edge (first column only) ───────────────────────
            if c == 0:
                ax.set_ylabel(f"Edge {edge*100:.0f}%\nProfit (u)", fontsize=8)

            # ── Δ overall balance badge (bottom-centre of each cell) ──────
            delta_color = "#1a7a1a" if total_pnl >= 0 else "#b30000"
            ax.text(
                0.5, 0.04,
                f"Δ {total_pnl:+.1f}  |  {n_bets}b",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=delta_color,
                    edgecolor="none",
                    alpha=0.88,
                ),
            )

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = output_dir / f"{scenario_name.lower()}_all_strategies_profit_per_year.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("Saved %s", out)
    plt.close(fig)