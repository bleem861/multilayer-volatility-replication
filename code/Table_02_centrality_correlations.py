from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'data/processed/centrality/P.csv'
df = pd.read_csv(path)
p_cols = ['P_corr', 'P_kendall', 'P_partial', 'P_tail']
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['market'] = df['market'].astype(str)
df[p_cols] = df[p_cols].apply(pd.to_numeric, errors='coerce')
markets = sorted(df['market'].dropna().unique().tolist())
per_market_corr = {}
for m in markets:
    subp = df.loc[df['market'] == m, p_cols].dropna()
    if len(subp) < 3:
        corr = pd.DataFrame(np.nan, index=p_cols, columns=p_cols)
    else:
        corr = subp.corr(method='pearson')
    per_market_corr[m] = corr
abs_avg_corr = pd.concat({m: per_market_corr[m].abs() for m in markets}, names=['market', 'row']).groupby(level=1).mean()
out_csv = ROOT / 'results/generated/Table_02_centrality_correlations.csv'
out_csv.parent.mkdir(parents=True, exist_ok=True)
abs_avg_corr.round(6).to_csv(out_csv, float_format='%.6f')
