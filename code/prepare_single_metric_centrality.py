from pathlib import Path
from io import StringIO
from functools import partial
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm as _tqdm
from pycop import empirical

tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
ROOT = Path(__file__).resolve().parents[1]
RETURN_FILE = ROOT / "data/raw/Return.csv"
OUTPUT_DIR = ROOT / "results/generated/centrality/single_metric"
L_NET = 1000
PRED_END = pd.Timestamp("2022-02-25")
KINDS = ("Corr", "kendall", "partial", "tail")
METRICS = {"DC": "NS", "BC": "BC", "CC": "CC", "EC": "EC", "Ecc": "Ecc"}

def read_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str[:10], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    df = df.set_index('date').dropna(how='all')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def build_network_weighted(corr_matrix: pd.DataFrame):
    G = nx.Graph()
    for n1, row in corr_matrix.iterrows():
        for n2, value in row.items():
            if n1 == n2:
                continue
            if pd.isna(value):
                continue
            value = float(np.clip(value, -1.0, 1.0))
            distance = np.sqrt(2 * (1 - value))
            if np.isnan(distance) or np.isinf(distance):
                continue
            G.add_edge(n1, n2, weight=distance)
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        nodes = list(corr_matrix.columns)
        nan_dict = {k: np.nan for k in nodes}
        return (nan_dict, nan_dict, nan_dict, nan_dict, nan_dict)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    MST_C = nx.Graph()
    for u, v, data in mst.edges(data=True):
        distance = data.get('weight', np.nan)
        if pd.isna(distance) or np.isinf(distance):
            continue
        value = abs(1 - distance ** 2 / 2)
        if pd.isna(value) or np.isinf(value):
            continue
        MST_C.add_edge(u, v, weight=float(value))
    try:
        NS = dict(MST_C.degree(weight='weight'))
    except Exception:
        NS = {k: np.nan for k in corr_matrix.columns}
    try:
        EC = nx.eigenvector_centrality(MST_C, weight='weight', tol=0.001)
    except Exception:
        EC = {k: np.nan for k in corr_matrix.columns}
    try:
        CC = nx.closeness_centrality(mst, distance='weight')
    except Exception:
        CC = {k: np.nan for k in corr_matrix.columns}
    try:
        BC = nx.betweenness_centrality(mst, weight='weight')
    except Exception:
        BC = {k: np.nan for k in corr_matrix.columns}
    try:
        Ecc = nx.eccentricity(mst, weight='weight')
    except Exception:
        Ecc = {k: np.nan for k in corr_matrix.columns}
    return (NS, EC, CC, BC, Ecc)

def build_network_unweighted(corr_matrix: pd.DataFrame):
    G = nx.Graph()
    for i, row in corr_matrix.iterrows():
        for j, value in row.items():
            if i == j:
                continue
            if pd.isna(value):
                continue
            value = float(np.clip(value, -1.0, 1.0))
            distance = np.sqrt(2 * (1 - value))
            if np.isnan(distance) or np.isinf(distance):
                continue
            G.add_edge(i, j, weight=distance)
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        nodes = list(corr_matrix.columns)
        nan_dict = {k: np.nan for k in nodes}
        return (nan_dict, nan_dict, nan_dict, nan_dict, nan_dict)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    unweighted_mst = nx.Graph()
    for u, v in mst.edges():
        unweighted_mst.add_edge(u, v)
    try:
        NS = dict(unweighted_mst.degree())
    except Exception:
        NS = {k: np.nan for k in corr_matrix.columns}
    try:
        EC = nx.eigenvector_centrality(unweighted_mst, tol=0.01)
    except Exception:
        EC = {k: np.nan for k in corr_matrix.columns}
    try:
        CC = nx.closeness_centrality(unweighted_mst)
    except Exception:
        CC = {k: np.nan for k in corr_matrix.columns}
    try:
        BC = nx.betweenness_centrality(unweighted_mst)
    except Exception:
        BC = {k: np.nan for k in corr_matrix.columns}
    try:
        Ecc = nx.eccentricity(unweighted_mst)
    except Exception:
        Ecc = {k: np.nan for k in corr_matrix.columns}
    return (NS, EC, CC, BC, Ecc)

def compute_centralities(df_ret: pd.DataFrame, L_net: int, start_date, end_date):
    dates = df_ret.loc[start_date:end_date].index
    used_dates = []
    W = {k: {m: [] for m in ['NS', 'EC', 'CC', 'BC', 'Ecc']} for k in ['Corr', 'kendall', 'partial', 'tail']}
    U = {k: {m: [] for m in ['NS', 'EC', 'CC', 'BC', 'Ecc']} for k in ['Corr', 'kendall', 'partial', 'tail']}
    for d in tqdm(dates):
        data_rolling = df_ret.loc[:d].tail(L_net)
        data_rolling = data_rolling.dropna(axis=1, how='all')
        std = data_rolling.std(axis=0, skipna=True)
        good_cols = std[std > 0].index
        data_rolling = data_rolling[good_cols]
        if data_rolling.shape[1] < 3:
            continue
        used_dates.append(d)
        corr_mat = data_rolling.corr(method='pearson').fillna(0.0).clip(-1.0, 1.0)
        kendall_mat = data_rolling.corr(method='kendall').fillna(0.0).clip(-1.0, 1.0)
        try:
            inv_corr = np.linalg.inv(corr_mat.values)
        except np.linalg.LinAlgError:
            inv_corr = np.linalg.pinv(corr_mat.values)
        partial_raw = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
        partial_mat = pd.DataFrame(partial_raw, index=corr_mat.index, columns=corr_mat.columns)
        partial_mat = partial_mat.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        tail_mat = pd.DataFrame(0.0, index=data_rolling.columns, columns=data_rolling.columns)
        cols = list(data_rolling.columns)
        for a in range(len(cols)):
            for b in range(a, len(cols)):
                stock_a = cols[a]
                stock_b = cols[b]
                try:
                    tail_corr = empirical(data_rolling[[stock_a, stock_b]].T.values).LTDC(0.1)
                except Exception:
                    tail_corr = np.nan
                if pd.isna(tail_corr) or np.isinf(tail_corr):
                    tail_corr = 0.0
                tail_corr = float(np.clip(tail_corr, -1.0, 1.0))
                tail_mat.loc[stock_a, stock_b] = tail_corr
                tail_mat.loc[stock_b, stock_a] = tail_corr
        np.fill_diagonal(tail_mat.values, 1.0)
        mats = {'Corr': corr_mat, 'kendall': kendall_mat, 'partial': partial_mat, 'tail': tail_mat}
        for kind, mat in mats.items():
            ns_w, ec_w, cc_w, bc_w, ecc_w = build_network_weighted(mat)
            ns_u, ec_u, cc_u, bc_u, ecc_u = build_network_unweighted(mat)
            W[kind]['NS'].append(ns_w)
            W[kind]['EC'].append(ec_w)
            W[kind]['CC'].append(cc_w)
            W[kind]['BC'].append(bc_w)
            W[kind]['Ecc'].append(ecc_w)
            U[kind]['NS'].append(ns_u)
            U[kind]['EC'].append(ec_u)
            U[kind]['CC'].append(cc_u)
            U[kind]['BC'].append(bc_u)
            U[kind]['Ecc'].append(ecc_u)
    return (W, U, pd.DatetimeIndex(used_dates))

def load_ranked_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if 'date' in df.columns:
        date_col = 'date'
    elif 'Unnamed: 0' in df.columns:
        date_col = 'Unnamed: 0'
    else:
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors='coerce')
        if parsed.notna().mean() < 0.8:
            raise ValueError(f'Missing date-like column in: {fp}. Columns={list(df.columns)}')
        date_col = first_col
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index.name = 'date'
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.apply(lambda row: row.rank(ascending=False), axis=1)


def make_single_metric(weighted, unweighted):
    rw = load_ranked_csv(StringIO(weighted.to_csv()))
    ru = load_ranked_csv(StringIO(unweighted.to_csv()))
    rw, ru = rw.align(ru, join="inner", axis=0)
    rw, ru = rw.align(ru, join="inner", axis=1)
    if rw.empty:
        raise ValueError("No aligned centrality observations")
    N = rw.shape[1]
    return (rw + ru - 2.0) / (2.0 * (N - 1))


def save_single_metrics(weighted, unweighted, dates, output_dir=OUTPUT_DIR):
    for metric, raw_metric in METRICS.items():
        directory = Path(output_dir) / ("P_" + metric)
        directory.mkdir(parents=True, exist_ok=True)
        for kind in KINDS:
            w = pd.DataFrame(weighted[kind][raw_metric], index=dates)
            u = pd.DataFrame(unweighted[kind][raw_metric], index=dates)
            result = make_single_metric(w, u)
            result.to_csv(directory / (kind + ".csv"))


def main():
    if not RETURN_FILE.is_file():
        raise FileNotFoundError(f"Required input not found: {RETURN_FILE}")
    returns = read_returns(RETURN_FILE).sort_index()
    if len(returns) < L_NET:
        raise ValueError("Insufficient observations for the network window")
    start_date = returns.index[L_NET - 1]
    end_date = min(PRED_END, returns.index.max())
    weighted, unweighted, dates = compute_centralities(returns, L_NET, start_date, end_date)
    save_single_metrics(weighted, unweighted, dates)


if __name__ == "__main__":
    main()
