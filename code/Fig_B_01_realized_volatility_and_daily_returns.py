from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RETURN_FILE = ROOT / "data/raw/Return.csv"
RV_FILE = ROOT / "data/raw/RV.csv"
OUTPUT_FILE = ROOT / "results/generated/Fig_B_01_realized_volatility_and_daily_returns.pdf"


def main():
    returns_full = pd.read_csv(RETURN_FILE, parse_dates=["date"])
    rv_full = pd.read_csv(RV_FILE, parse_dates=["date"])
    cutoff_date = pd.Timestamp("2010-09-14")
    returns_full = returns_full.loc[returns_full["date"] >= cutoff_date].copy()
    rv_full = rv_full.loc[rv_full["date"] >= cutoff_date].copy()
    markets = [column for column in returns_full.columns if column != "date"]
    missing = [market for market in markets[:18] if market not in rv_full.columns]
    if missing:
        raise ValueError("Missing realized volatility columns: " + ", ".join(missing))

    figure = plt.figure(figsize=(18, 30))
    grid = gridspec.GridSpec(6, 3, figure=figure)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    year_ticks = [
        datetime(2011, 11, 19),
        datetime(2014, 11, 24),
        datetime(2017, 12, 14),
        datetime(2021, 1, 4),
    ]

    for index, market in enumerate(markets[:18]):
        row = index // 3
        column = index % 3
        returns = returns_full[["date", market]]
        realized_volatility = rv_full[["date", market]]
        merged = pd.merge(returns, realized_volatility, on="date")
        merged = merged.set_index("date")
        merged.columns = ["returns", "realized_volatility"]

        left_axis = figure.add_subplot(grid[row, column])
        left_axis.margins(x=0)
        left_axis.plot(
            merged.index,
            merged["returns"],
            color="tab:blue",
            label="Return",
            linewidth=0.8,
        )
        left_axis.tick_params(axis="y", labelcolor="tab:blue")
        returns_min = min(merged["returns"].min(), -0.08)
        returns_max = max(merged["returns"].max(), 0.06)
        left_axis.set_ylim(returns_min, returns_max)
        left_axis.yaxis.set_major_locator(MaxNLocator(5))
        left_axis.set_xticks(year_ticks)
        left_axis.set_xticklabels(["2011", "2014", "2017", "2021"], fontsize=8)

        right_axis = left_axis.twinx()
        right_axis.margins(x=0)
        right_axis.plot(
            merged.index,
            merged["realized_volatility"],
            color=(0.9, 0.3, 0.0),
            label="RV",
            linewidth=0.8,
        )
        right_axis.tick_params(axis="y", labelcolor=(0.9, 0.3, 0.0))
        right_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        rv_max = max(merged["realized_volatility"].max(), 0.004)
        right_axis.set_ylim(0, rv_max)
        right_axis.yaxis.set_major_locator(MaxNLocator(5))

        left_axis.set_title(market.lstrip("."), fontsize=10)
        if index == 0:
            left_lines, left_labels = left_axis.get_legend_handles_labels()
            right_lines, right_labels = right_axis.get_legend_handles_labels()
            left_axis.legend(
                left_lines + right_lines,
                left_labels + right_labels,
                loc="upper right",
                fontsize=8,
            )
        left_axis.tick_params(axis="both", which="major", labelsize=8)
        right_axis.tick_params(axis="both", which="major", labelsize=8)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
