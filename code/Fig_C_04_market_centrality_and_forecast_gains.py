from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
p_path = ROOT / 'data' / 'processed' / 'centrality' / 'P.csv'
gain_path = ROOT / 'data' / 'processed' / 'market_gain' / 'market_forecast_gains.xlsx'
out_dir = ROOT / 'results' / 'generated'
SAVE_PATH_PDF = out_dir / 'Fig_C_04_market_centrality_and_forecast_gains.pdf'

def clean_market(series):
    return series.astype(str).str.strip().str.replace('.', '', regex=False).str.replace('\u200b', '', regex=False).str.replace('\ufeff', '', regex=False)

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace('\u200b', '', regex=False).str.replace('\ufeff', '', regex=False).str.strip(), errors='coerce')

def convert_gain_to_percent(df, cols):
    for col in cols:
        if df[col].abs().max() <= 1:
            df[col] *= 100
    return df

def fit_linear_line(x, y, x_line):
    slope, intercept = np.polyfit(x, y, 1)
    pearson = stats.pearsonr(x, y)
    p_value = pearson.pvalue if hasattr(pearson, 'pvalue') else pearson[1]
    return (slope * x_line + intercept, p_value)

def fit_quadratic_line(x, y, x_line):
    x_mean = x.mean()
    xc = x - x_mean
    x_line_c = x_line - x_mean
    xmat = np.column_stack([np.ones_like(x), xc, xc ** 2])
    beta, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    y_line = beta[0] + beta[1] * x_line_c + beta[2] * x_line_c ** 2
    resid = y - xmat @ beta
    dof = len(y) - xmat.shape[1]
    sigma2 = float(np.sum(resid ** 2) / dof)
    cov = np.linalg.inv(xmat.T @ xmat) * sigma2
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df=dof)
    return (y_line, pvals[2])

def label_points(ax, data, metric_col):
    x = data['P'].to_numpy()
    y = data[metric_col].to_numpy()
    dx = (x.max() - x.min()) * 0.018
    dy = (y.max() - y.min()) * 0.018
    common_adjustments = {'FCHI': (-1.1, 1.2, 'right'), 'FTSE': (0.8, 1.2, 'left'), 'AEX': (0.8, 1.2, 'left'), 'OMXSPI': (0.8, -1.1, 'left'), 'OSEAX': (0.8, 1.2, 'left'), 'GSPTSE': (0.8, 1.2, 'left'), 'GDAXI': (0.8, -1.2, 'left'), 'RUT': (0.8, 1.2, 'left'), 'SSMI': (0.8, 1.2, 'right'), 'OMXHPI': (0.8, -1.2, 'left'), 'AORD': (0.8, -1.2, 'left'), 'BFX': (0.8, 1.2, 'left'), 'OMXC20': (-1.4, 1.2, 'right'), 'IXIC': (0.8, 1.2, 'left'), 'BSESN': (0.8, 1.2, 'left'), 'KS11': (0.8, -1.2, 'left'), 'BVSP': (0.8, -1.2, 'left'), 'KSE': (0.8, -1.2, 'left')}
    metric_adjustments = {'MSE': {'BFX': (-1.0, 1.7, 'right'), 'OMXC20': (0.8, -1.7, 'left'), 'BSESN': (-1.1, -1.5, 'right'), 'KS11': (0.8, 1.4, 'left'), 'KSE': (0.8, -1.6, 'left'), 'BVSP': (0.8, -1.4, 'left'), 'AORD': (0.8, -1.4, 'left'), 'OMXHPI': (0.8, -1.7, 'left')}, 'QLIKE': {'OMXSPI': (0.8, -1.2, 'left'), 'GDAXI': (0.8, -1.2, 'left'), 'SSMI': (0.8, 1.5, 'left'), 'AORD': (0.8, -1.5, 'left'), 'BFX': (0.8, 1.1, 'left'), 'OMXC20': (-1.0, 1.5, 'right'), 'BVSP': (0.8, -1.2, 'left'), 'KSE': (0.8, -1.3, 'left')}}
    adjustments = {**common_adjustments, **metric_adjustments.get(metric_col, {})}
    for _, row in data.iterrows():
        label = row['Market'].replace('.', '')
        ox_factor, oy_factor, ha = adjustments.get(label, (0.8, 0.8, 'left'))
        ax.text(row['P'] + dx * ox_factor, row[metric_col] + dy * oy_factor, label, fontsize=5.1, color='#333333', ha=ha, va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.68, 'pad': 0.25})

def draw_panel(ax, data, metric_col, fit_kind):
    x = data['P'].to_numpy()
    y = data[metric_col].to_numpy()
    x_line = np.linspace(x.min(), x.max(), 240)
    colors = np.where(y >= 0, '#1F5A99', '#B24A3B')
    if fit_kind == 'linear':
        y_line, p_value = fit_linear_line(x, y, x_line)
        line_color = '#666666'
        stat_text = f'Linear fit, p = {p_value:.3f}'
    else:
        y_line, p_value = fit_quadratic_line(x, y, x_line)
        line_color = '#D97706'
        stat_text = f'Quadratic fit, p = {p_value:.3f}'
    ax.axhline(0, color='#555555', lw=0.8, zorder=1)
    ax.scatter(x, y, s=38, c=colors, edgecolor='white', linewidth=0.5, zorder=3)
    ax.plot(x_line, y_line, color=line_color, lw=1.3, zorder=2)
    label_points(ax, data, metric_col)
    ax.text(0.03, 0.96, stat_text, transform=ax.transAxes, ha='left', va='top', fontsize=7.8, color='#333333', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 0.5})
    ax.set_title(metric_col, loc='left', fontsize=9, fontweight='bold', pad=4)
    ax.set_xlabel('Average P ')
    ax.set_ylabel('Forecast gain relative to benchmark (%)')
    ax.grid(color='#E5E5E5', lw=0.6, alpha=0.8)

def main():
    SAVE_PATH_PDF.parent.mkdir(parents=True, exist_ok=True)
    p_df = pd.read_csv(p_path, encoding='utf-8-sig')
    p_df['date'] = pd.to_datetime(p_df['date'])
    p_df['market'] = clean_market(p_df['market'])
    gain_df = pd.read_excel(gain_path)
    gain_df = gain_df.rename(columns={gain_df.columns[0]: 'Market'})
    gain_df = gain_df[['Market', 'MSE', 'QLIKE']].copy()
    gain_df['Market'] = clean_market(gain_df['Market'])
    gain_df['MSE'] = clean_numeric(gain_df['MSE'])
    gain_df['QLIKE'] = clean_numeric(gain_df['QLIKE'])
    gain_df = gain_df.dropna(subset=['Market', 'MSE', 'QLIKE'])
    gain_df = convert_gain_to_percent(gain_df, ['MSE', 'QLIKE'])
    start_date = pd.Timestamp('2018-05-02')
    p_oos = p_df[p_df['date'] >= start_date].copy()
    p_mean = p_oos.groupby('market')['P_corr'].mean().reset_index()
    p_mean.columns = ['Market', 'P']
    plot_df = gain_df.merge(p_mean, on='Market', how='inner')
    plot_df = plot_df.sort_values('P').reset_index(drop=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), constrained_layout=True)
    draw_panel(axes[0], plot_df, 'MSE', 'linear')
    draw_panel(axes[1], plot_df, 'QLIKE', 'quadratic')
    for ax, metric_col in zip(axes, ['MSE', 'QLIKE']):
        x = plot_df['P']
        y = plot_df[metric_col]
        x_pad = (x.max() - x.min()) * 0.12
        y_pad = (y.max() - y.min()) * 0.14
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    out_base = SAVE_PATH_PDF
    fig.savefig(SAVE_PATH_PDF, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
