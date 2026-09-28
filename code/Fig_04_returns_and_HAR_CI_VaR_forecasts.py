import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / 'results' / 'generated'
OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--recomputed', action='store_true', help='Plot regenerated VaR data instead of the supplied intermediate files.')
args = parser.parse_args()
INPUT_DIR = OUT_DIR / 'Fig_04_VaR' if args.recomputed else PROJECT_ROOT / 'data' / 'processed' / 'Fig_04_VaR'



def load_market_var(market):
    base = INPUT_DIR


    if not market.startswith('.'):
        market = '.' + market

    df_99 = pd.read_csv(f'{base}/VaR_0.011/HAR-CI-new/{market}_daily.csv')
    df_95 = pd.read_csv(f'{base}/VaR_0.051/HAR-CI-new/{market}_daily.csv')
    df_90 = pd.read_csv(f'{base}/VaR_0.11/HAR-CI-new/{market}_daily.csv')


    for df in [df_99, df_95, df_90]:
        df['date'] = pd.to_datetime(df['date'])


    df = df_99[['date', 'R_22_mean_true', 'VaR_22_mean_pred']].copy()
    df = df.rename(columns={
        'R_22_mean_true': 'return',
        'VaR_22_mean_pred': 'VaR99'
    })


    df = df.merge(
        df_95[['date', 'VaR_22_mean_pred']].rename(columns={'VaR_22_mean_pred': 'VaR95'}),
        on='date',
        how='inner'
    )

    df = df.merge(
        df_90[['date', 'VaR_22_mean_pred']].rename(columns={'VaR_22_mean_pred': 'VaR90'}),
        on='date',
        how='inner'
    )


    for col in ['return', 'VaR99', 'VaR95', 'VaR90']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().sort_values('date').reset_index(drop=True)


    start_date = pd.Timestamp('2018-11-01')
    end_date = pd.Timestamp('2021-11-30')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    return df


df_ixic = load_market_var('.IXIC')
df_ks11 = load_market_var('.KS11')


fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)


axes[0].scatter(df_ixic['date'], df_ixic['return'], marker='x', color='black', s=12, label='Returns')
axes[0].plot(df_ixic['date'], df_ixic['VaR99'], color='blue', linewidth=1.5, label='99% VaR')
axes[0].plot(df_ixic['date'], df_ixic['VaR95'], color='red', linewidth=1.5, label='95% VaR')
axes[0].plot(df_ixic['date'], df_ixic['VaR90'], color='green', linewidth=1.5, label='90% VaR')
axes[0].set_title('IXIC', fontsize=14)
axes[0].set_ylabel('Return')
axes[0].legend(loc='lower right')
axes[0].grid(True, linestyle='--', alpha=0.7)


axes[1].scatter(df_ks11['date'], df_ks11['return'], marker='x', color='black', s=12, label='Returns')
axes[1].plot(df_ks11['date'], df_ks11['VaR99'], color='blue', linewidth=1.5, label='99% VaR')
axes[1].plot(df_ks11['date'], df_ks11['VaR95'], color='red', linewidth=1.5, label='95% VaR')
axes[1].plot(df_ks11['date'], df_ks11['VaR90'], color='green', linewidth=1.5, label='90% VaR')
axes[1].set_title('KS11', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Return')
axes[1].legend(loc='lower right')
axes[1].grid(True, linestyle='--', alpha=0.7)


axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


for ax, df in zip(axes, [df_ixic, df_ks11]):
    ax.set_xlim(df['date'].min(), df['date'].max())

plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig_04_returns_and_HAR_CI_VaR_forecasts.pdf', dpi=900, bbox_inches='tight')
plt.show()
