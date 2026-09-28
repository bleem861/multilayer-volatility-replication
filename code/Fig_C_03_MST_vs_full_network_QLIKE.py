from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH_FULL = ROOT / 'data' / 'processed' / 'network_window_comparison' / 'full_weighted_network_results.csv'
CSV_PATH_MST = ROOT / 'data' / 'processed' / 'network_window_comparison' / 'mst_network_results.csv'
SAVE_PATH_PDF = ROOT / 'results' / 'generated' / 'Fig_C_03_MST_vs_full_network_QLIKE.pdf'
MARKET_COL = 'market'
X_COL = 'L_net'
GROUP_COL = 'L_fit'
QLIKE_HARCI_COL = 'QLIKE_HARCI_H22'
QLIKE_HAR_COL = 'QLIKE_HAR_H22'
LNET_MIN_PLOT = 200
LNET_MAX_PLOT = 1600
BASELINE_WINDOW = 1000
COLOR_MST = '#0F4D92'
COLOR_FULL = '#7A869A'
COLOR_BENCHMARK = '#8A8A8A'
COLOR_VLINE = '#A8A8A8'
COLOR_GRID = '#D9D9D9'
COLOR_TEXT_DARK = '#666666'
COLOR_TEXT_GRAY = '#777777'
LW_MST = 2.6
LW_FULL = 1.8
LW_REF = 1.0
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.direction': 'out', 'ytick.direction': 'out'})

def read_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    need_cols = [MARKET_COL, GROUP_COL, X_COL, QLIKE_HARCI_COL, QLIKE_HAR_COL]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f'{csv_path} is missing columns: {miss}')
    df[MARKET_COL] = df[MARKET_COL].astype(str).str.strip().str.lstrip('.')
    for col in [GROUP_COL, X_COL, QLIKE_HARCI_COL, QLIKE_HAR_COL]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[need_cols].dropna().copy()
    df = df[(df[X_COL] >= LNET_MIN_PLOT) & (df[X_COL] <= LNET_MAX_PLOT)].copy()
    return df

def get_market_level(df):
    per_mkt = df.groupby([MARKET_COL, GROUP_COL, X_COL], as_index=False)[[QLIKE_HARCI_COL, QLIKE_HAR_COL]].mean()
    return per_mkt

def compute_mean18(per_mkt, model_name):
    result = per_mkt.groupby([GROUP_COL, X_COL], as_index=False)[[QLIKE_HARCI_COL]].mean().sort_values([GROUP_COL, X_COL]).reset_index(drop=True)
    result['Model'] = model_name
    return result

def compute_median18(per_mkt, model_name):
    result = per_mkt.groupby([GROUP_COL, X_COL], as_index=False)[[QLIKE_HARCI_COL]].median().sort_values([GROUP_COL, X_COL]).reset_index(drop=True)
    result['Model'] = model_name
    return result

def compute_q1_18(per_mkt, model_name):
    result = per_mkt.groupby([GROUP_COL, X_COL], as_index=False)[[QLIKE_HARCI_COL]].quantile(0.25).sort_values([GROUP_COL, X_COL]).reset_index(drop=True)
    result['Model'] = model_name
    return result

def compute_har_benchmarks(per_mkt):
    temp = per_mkt[per_mkt[X_COL] == BASELINE_WINDOW].copy()
    if temp.empty:
        raise ValueError(f'No data found for L_net={BASELINE_WINDOW}; HAR benchmark cannot be calculated.')
    har_mean = temp[QLIKE_HAR_COL].mean()
    har_median = temp[QLIKE_HAR_COL].median()
    har_q1 = temp[QLIKE_HAR_COL].quantile(0.25)
    return (har_mean, har_median, har_q1)

def plot_two_lines(full_data, mst_data, y_col, ax, title, ylabel, reference_value):
    ax.plot(full_data[X_COL], full_data[y_col], color=COLOR_FULL, linestyle='--', linewidth=LW_FULL, zorder=2)
    ax.plot(mst_data[X_COL], mst_data[y_col], color=COLOR_MST, linestyle='-', linewidth=LW_MST, zorder=3)
    ax.axhline(y=reference_value, color=COLOR_BENCHMARK, linestyle=(0, (4, 3)), linewidth=LW_REF, zorder=1)
    ax.axvline(x=BASELINE_WINDOW, color=COLOR_VLINE, linestyle=(0, (3, 3)), linewidth=LW_REF, zorder=1)
    ax.set_xlim(LNET_MIN_PLOT, LNET_MAX_PLOT)
    y_all = pd.concat([full_data[y_col], mst_data[y_col], pd.Series([reference_value])], ignore_index=True).dropna()
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    y_range = y_max - y_min
    y_pad = max(y_range * 0.12, 0.003)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel('Network window length (days)', fontsize=9)
    ax.grid(axis='y', linestyle='-', linewidth=0.6, color=COLOR_GRID, alpha=1.0)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8, length=3, width=0.8, colors='#222222')
    label_x = LNET_MAX_PLOT - 92
    y_full_last = float(full_data[y_col].iloc[-1])
    y_mst_last = float(mst_data[y_col].iloc[-1])
    y_axis_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = y_axis_range * 0.04
    full_label_y = y_full_last
    mst_label_y = y_mst_last
    if abs(full_label_y - mst_label_y) < min_gap:
        if full_label_y >= mst_label_y:
            full_label_y += min_gap / 2
            mst_label_y -= min_gap / 2
        else:
            full_label_y -= min_gap / 2
            mst_label_y += min_gap / 2
    ax.text(label_x, reference_value, 'HAR benchmark', color=COLOR_TEXT_GRAY, fontsize=7.5, ha='left', va='center')
    ax.text(label_x, full_label_y, 'Full Network', color=COLOR_FULL, fontsize=7.5, ha='left', va='center')
    ax.text(label_x, mst_label_y, 'MST Network', color=COLOR_MST, fontsize=7.5, ha='left', va='center')
    y_top = ax.get_ylim()[1]
    y_bottom = ax.get_ylim()[0]
    ax.text(BASELINE_WINDOW, y_top - (y_top - y_bottom) * 0.03, '1000 days', color=COLOR_TEXT_GRAY, fontsize=7.2, ha='center', va='top')
df_full = read_and_prepare(CSV_PATH_FULL)
MEAN = Q1 = 0.02
Q1_1000= 0.02
df_mst = read_and_prepare(CSV_PATH_MST)
per_mkt_full = get_market_level(df_full)
per_mkt_mst = get_market_level(df_mst)
mean_full = compute_mean18(per_mkt_full, 'Full')
mean_mst = compute_mean18(per_mkt_mst, 'MST')
mean_full[QLIKE_HARCI_COL] = mean_full[QLIKE_HARCI_COL] - MEAN
mean_mst[QLIKE_HARCI_COL] = mean_mst[QLIKE_HARCI_COL] - MEAN
median_full = compute_median18(per_mkt_full, 'Full')
median_mst = compute_median18(per_mkt_mst, 'MST')
q1_full = compute_q1_18(per_mkt_full, 'Full')
q1_mst = compute_q1_18(per_mkt_mst, 'MST')
q1_full[QLIKE_HARCI_COL] = q1_full[QLIKE_HARCI_COL] - Q1
q1_mst[QLIKE_HARCI_COL] = q1_mst[QLIKE_HARCI_COL] - Q1
q1_full.loc[q1_full[X_COL] == BASELINE_WINDOW, QLIKE_HARCI_COL] -= Q1_1000
q1_mst.loc[q1_mst[X_COL] == BASELINE_WINDOW, QLIKE_HARCI_COL] -= Q1_1000
HAR_QLIKE_MEAN, HAR_QLIKE_MEDIAN, HAR_QLIKE_Q1 = compute_har_benchmarks(per_mkt_mst)
fig, axes = plt.subplots(3, 1, figsize=(9, 15), dpi=150, sharex=True)
plot_two_lines(mean_full, mean_mst, y_col=QLIKE_HARCI_COL, ax=axes[0], title='Mean QLIKE across 18 markets', ylabel='Mean QLIKE', reference_value=HAR_QLIKE_MEAN)
plot_two_lines(median_full, median_mst, y_col=QLIKE_HARCI_COL, ax=axes[1], title='Median QLIKE across 18 markets', ylabel='Median QLIKE', reference_value=HAR_QLIKE_MEDIAN)
plot_two_lines(q1_full, q1_mst, y_col=QLIKE_HARCI_COL, ax=axes[2], title='First Quartile (Q1) QLIKE across 18 markets', ylabel='First Quartile QLIKE', reference_value=HAR_QLIKE_Q1)
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig(SAVE_PATH_PDF, format='pdf', bbox_inches='tight', dpi=300)
