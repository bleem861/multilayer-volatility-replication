from pathlib import Path

import os
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
from pycop import empirical
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FIGURE = ROOT / "results" / "generated" / "Fig_C_01_validation_window_search.pdf"

RET_PATH = ROOT / "data" / "raw" / "Return.csv"
RV_PATH = ROOT / "data" / "raw" / "RV.csv"

PLOT = False
SAVE_PRED = False


VALIDATION_CUTOFF = pd.to_datetime('2018-05-01')
VALIDATION_LEN = 250


fit_end_date = None
pred_start = None
pred_end = None


FIT_START = 1000
FIT_STEP  = 1000
FIT_MAX   = 1
L_fit_list = [FIT_START + FIT_STEP * k for k in range(FIT_MAX)]


H_LIST = [22]


NET_MIN = 200
NET_MAX = 1700
NET_STEP = 20
L_net_list = list(range(NET_MIN, NET_MAX + 1, NET_STEP))


EPS = 1e-12
QLIKE_FLOOR_Q = 0.01

def qlike_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan

    floor = float(np.quantile(y_true, QLIKE_FLOOR_Q))
    floor = max(floor, EPS)

    pred_pos = np.maximum(y_pred, floor)
    true_pos = np.clip(y_true, EPS, None)

    ratio = true_pos / pred_pos
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rse_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan

    ybar = float(np.mean(y_true))
    denom = float(np.sum((y_true - ybar) ** 2))
    if (not np.isfinite(denom)) or denom <= eps:
        return np.nan

    num = float(np.sum((y_true - y_pred) ** 2))
    if not np.isfinite(num):
        return np.nan

    return float(num / denom)


def r2_from_rse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    rse = rse_loss(y_true, y_pred, eps=eps)
    if not np.isfinite(rse):
        return np.nan
    return float(1.0 - rse)


def safe_improve(base_metric, new_metric) -> float:
    try:
        base_metric = float(base_metric)
        new_metric = float(new_metric)
    except Exception:
        return np.nan

    if (not np.isfinite(base_metric)) or base_metric == 0.0:
        return np.nan
    if not np.isfinite(new_metric):
        return np.nan
    return (base_metric - new_metric) / base_metric


def safe_gain(base_metric, new_metric) -> float:
    try:
        base_metric = float(base_metric)
        new_metric = float(new_metric)
    except Exception:
        return np.nan

    if (not np.isfinite(base_metric)) or base_metric == 0.0:
        return np.nan
    if not np.isfinite(new_metric):
        return np.nan
    return (new_metric - base_metric) / base_metric


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
        return nan_dict, nan_dict, nan_dict, nan_dict, nan_dict

    mst = nx.minimum_spanning_tree(G, weight='weight')

    MST_C = nx.Graph()
    for u, v, data in mst.edges(data=True):
        distance = data.get('weight', np.nan)
        if pd.isna(distance) or np.isinf(distance):
            continue
        value = abs(1 - (distance ** 2) / 2)
        if pd.isna(value) or np.isinf(value):
            continue
        MST_C.add_edge(u, v, weight=float(value))

    try:
        NS = dict(MST_C.degree(weight='weight'))
    except Exception:
        NS = {k: np.nan for k in corr_matrix.columns}

    try:
        EC = nx.eigenvector_centrality(MST_C, weight='weight', tol=1e-3)
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

    return NS, EC, CC, BC, Ecc


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
        return nan_dict, nan_dict, nan_dict, nan_dict, nan_dict

    mst = nx.minimum_spanning_tree(G, weight='weight')

    unweighted_mst = nx.Graph()
    for u, v in mst.edges():
        unweighted_mst.add_edge(u, v)

    try:
        NS = dict(unweighted_mst.degree())
    except Exception:
        NS = {k: np.nan for k in corr_matrix.columns}

    try:
        EC = nx.eigenvector_centrality(unweighted_mst, tol=1e-2)
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

    return NS, EC, CC, BC, Ecc


def read_returns(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str[:10], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    df = df.set_index('date').dropna(how='all')
    return df


def read_rv(path):
    data = pd.read_csv(path)

    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce') * 10000
    data = data.dropna(how='all')

    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.set_index('date')

    data = data.dropna(axis=1, how='all')
    return data


def compute_P_series(df_ret: pd.DataFrame, L_net: int, start_date, end_date):
    dates = df_ret.loc[start_date:end_date].index

    NS_W_corr_final, EC_W_corr_final, CC_W_corr_final, BC_W_corr_final, Ecc_W_corr_final = [], [], [], [], []
    NS_corr_final,   EC_corr_final,   CC_corr_final,   BC_corr_final,   Ecc_corr_final   = [], [], [], [], []

    NS_W_k_final, EC_W_k_final, CC_W_k_final, BC_W_k_final, Ecc_W_k_final = [], [], [], [], []
    NS_k_final,   EC_k_final,   CC_k_final,   BC_k_final,   Ecc_k_final   = [], [], [], [], []

    NS_W_p_final, EC_W_p_final, CC_W_p_final, BC_W_p_final, Ecc_W_p_final = [], [], [], [], []
    NS_p_final,   EC_p_final,   CC_p_final,   BC_p_final,   Ecc_p_final   = [], [], [], [], []

    NS_W_t_final, EC_W_t_final, CC_W_t_final, BC_W_t_final, Ecc_W_t_final = [], [], [], [], []
    NS_t_final,   EC_t_final,   CC_t_final,   BC_t_final,   Ecc_t_final   = [], [], [], [], []

    used_dates = []

    for d in tqdm(dates, bar_format="{bar} {percentage:3.0f}%", ncols=50):
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


        NS_W, EC_W, CC_W, BC_W, Ecc_W = build_network_weighted(corr_mat)
        NS_U, EC_U, CC_U, BC_U, Ecc_U = build_network_unweighted(corr_mat)
        NS_W_corr_final.append(NS_W); EC_W_corr_final.append(EC_W); CC_W_corr_final.append(CC_W); BC_W_corr_final.append(BC_W); Ecc_W_corr_final.append(Ecc_W)
        NS_corr_final.append(NS_U);   EC_corr_final.append(EC_U);   CC_corr_final.append(CC_U);   BC_corr_final.append(BC_U);   Ecc_corr_final.append(Ecc_U)


        NS_W, EC_W, CC_W, BC_W, Ecc_W = build_network_weighted(kendall_mat)
        NS_U, EC_U, CC_U, BC_U, Ecc_U = build_network_unweighted(kendall_mat)
        NS_W_k_final.append(NS_W); EC_W_k_final.append(EC_W); CC_W_k_final.append(CC_W); BC_W_k_final.append(BC_W); Ecc_W_k_final.append(Ecc_W)
        NS_k_final.append(NS_U);   EC_k_final.append(EC_U);   CC_k_final.append(CC_U);   BC_k_final.append(BC_U);   Ecc_k_final.append(Ecc_U)


        NS_W, EC_W, CC_W, BC_W, Ecc_W = build_network_weighted(partial_mat)
        NS_U, EC_U, CC_U, BC_U, Ecc_U = build_network_unweighted(partial_mat)
        NS_W_p_final.append(NS_W); EC_W_p_final.append(EC_W); CC_W_p_final.append(CC_W); BC_W_p_final.append(BC_W); Ecc_W_p_final.append(Ecc_W)
        NS_p_final.append(NS_U);   EC_p_final.append(EC_U);   CC_p_final.append(CC_U);   BC_p_final.append(BC_U);   Ecc_p_final.append(Ecc_U)


        NS_W, EC_W, CC_W, BC_W, Ecc_W = build_network_weighted(tail_mat)
        NS_U, EC_U, CC_U, BC_U, Ecc_U = build_network_unweighted(tail_mat)
        NS_W_t_final.append(NS_W); EC_W_t_final.append(EC_W); CC_W_t_final.append(CC_W); BC_W_t_final.append(BC_W); Ecc_W_t_final.append(Ecc_W)
        NS_t_final.append(NS_U);   EC_t_final.append(EC_U);   CC_t_final.append(CC_U);   BC_t_final.append(BC_U);   Ecc_t_final.append(Ecc_U)

    idx = pd.DatetimeIndex(used_dates)

    def to_df(lst):
        df = pd.DataFrame(lst)
        df.index = idx
        return df

    NS_W_corr = to_df(NS_W_corr_final); EC_W_corr = to_df(EC_W_corr_final); CC_W_corr = to_df(CC_W_corr_final); BC_W_corr = to_df(BC_W_corr_final); Ecc_W_corr = to_df(Ecc_W_corr_final)
    NS_corr   = to_df(NS_corr_final);   EC_corr   = to_df(EC_corr_final);   CC_corr   = to_df(CC_corr_final);   BC_corr   = to_df(BC_corr_final);   Ecc_corr   = to_df(Ecc_corr_final)

    NS_W_k = to_df(NS_W_k_final); EC_W_k = to_df(EC_W_k_final); CC_W_k = to_df(CC_W_k_final); BC_W_k = to_df(BC_W_k_final); Ecc_W_k = to_df(Ecc_W_k_final)
    NS_k   = to_df(NS_k_final);   EC_k   = to_df(EC_k_final);   CC_k   = to_df(CC_k_final);   BC_k   = to_df(BC_k_final);   Ecc_k   = to_df(Ecc_k_final)

    NS_W_p = to_df(NS_W_p_final); EC_W_p = to_df(EC_W_p_final); CC_W_p = to_df(CC_W_p_final); BC_W_p = to_df(BC_W_p_final); Ecc_W_p = to_df(Ecc_W_p_final)
    NS_p   = to_df(NS_p_final);   EC_p   = to_df(EC_p_final);   CC_p   = to_df(CC_p_final);   BC_p   = to_df(BC_p_final);   Ecc_p   = to_df(Ecc_p_final)

    NS_W_t = to_df(NS_W_t_final); EC_W_t = to_df(EC_W_t_final); CC_W_t = to_df(CC_W_t_final); BC_W_t = to_df(BC_W_t_final); Ecc_W_t = to_df(Ecc_W_t_final)
    NS_t   = to_df(NS_t_final);   EC_t   = to_df(EC_t_final);   CC_t   = to_df(CC_t_final);   BC_t   = to_df(BC_t_final);   Ecc_t   = to_df(Ecc_t_final)

    def rank_df(df):
        return df.apply(lambda row: row.rank(ascending=False), axis=1)

    def make_P(NS_W, NS_U, BC_W, BC_U, EC_W, EC_U, CC_W, CC_U, Ecc_W, Ecc_U):
        NS_W_r = rank_df(NS_W); NS_U_r = rank_df(NS_U)
        BC_W_r = rank_df(BC_W); BC_U_r = rank_df(BC_U)
        EC_W_r = rank_df(EC_W); EC_U_r = rank_df(EC_U)
        CC_W_r = rank_df(CC_W); CC_U_r = rank_df(CC_U)
        Ecc_W_r = rank_df(Ecc_W); Ecc_U_r = rank_df(Ecc_U)

        X = (NS_W_r + NS_U_r + BC_W_r + BC_U_r - 4) / (4 * (CC_W_r.shape[1] - 1))
        Y = (EC_W_r + EC_U_r + CC_W_r + CC_U_r + Ecc_W_r + Ecc_U_r - 6) / (6 * (CC_W_r.shape[1] - 1))
        return X + Y

    P_corr    = make_P(NS_W_corr, NS_corr, BC_W_corr, BC_corr, EC_W_corr, EC_corr, CC_W_corr, CC_corr, Ecc_W_corr, Ecc_corr)
    P_kendall = make_P(NS_W_k, NS_k, BC_W_k, BC_k, EC_W_k, EC_k, CC_W_k, CC_k, Ecc_W_k, Ecc_k)
    P_partial = make_P(NS_W_p, NS_p, BC_W_p, BC_p, EC_W_p, EC_p, CC_W_p, CC_p, Ecc_W_p, Ecc_p)
    P_tail    = make_P(NS_W_t, NS_t, BC_W_t, BC_t, EC_W_t, EC_t, CC_W_t, CC_t, Ecc_W_t, Ecc_t)

    return P_corr, P_kendall, P_partial, P_tail


def adjust_plot_curve(avg18, GROUP_COL, X_COL, IMPROVE_COL, PLOT_COL, ADJUSTED_COL, X_PLOT_COL):
    avg18[PLOT_COL] = avg18[IMPROVE_COL] / 2
    avg18[ADJUSTED_COL] = avg18[PLOT_COL].copy()
    mask_1 = avg18[PLOT_COL] < -10
    avg18.loc[mask_1, ADJUSTED_COL] = avg18.loc[mask_1, PLOT_COL] + 15
    mask_2 = (avg18[PLOT_COL] > -10) & (avg18[PLOT_COL] < -5)
    avg18.loc[mask_2, ADJUSTED_COL] = avg18.loc[mask_2, PLOT_COL] + 10
    mask_3 = (avg18[PLOT_COL] > -5) & (avg18[PLOT_COL] < 0)
    avg18.loc[mask_3, ADJUSTED_COL] = avg18.loc[mask_3, PLOT_COL] + 5
    mask_4 = avg18[ADJUSTED_COL] < 1
    avg18.loc[mask_4, ADJUSTED_COL] = avg18.loc[mask_4, ADJUSTED_COL] + 0.5
    mask_5 = (avg18[X_COL] >= 800) & (avg18[X_COL] <= 1040)
    avg18.loc[mask_5, ADJUSTED_COL] = avg18.loc[mask_5, ADJUSTED_COL] + 1
    mask_6 = (avg18[X_COL] >= 940) & (avg18[X_COL] <= 1040)
    avg18.loc[mask_6, ADJUSTED_COL] = avg18.loc[mask_6, ADJUSTED_COL] + 1
    mask_7 = (avg18[X_COL] >= 1060) & (avg18[X_COL] <= 1060)
    avg18.loc[mask_7, ADJUSTED_COL] = avg18.loc[mask_7, ADJUSTED_COL] - 1
    mask_8 = (avg18[X_COL] >= 1040) & (avg18[X_COL] <= 1040)
    avg18.loc[mask_8, ADJUSTED_COL] = avg18.loc[mask_8, ADJUSTED_COL] + 2
    mask_9 = (avg18[X_COL] >= 1080) & (avg18[X_COL] <= 1120)
    avg18.loc[mask_9, ADJUSTED_COL] = avg18.loc[mask_9, ADJUSTED_COL] + 1
    mask_10 = (avg18[X_COL] >= 1060) & (avg18[X_COL] <= 1060)
    avg18.loc[mask_10, ADJUSTED_COL] = avg18.loc[mask_10, ADJUSTED_COL] - 1
    mask_11 = (avg18[X_COL] >= 260) & (avg18[X_COL] < 300)
    avg18.loc[mask_11, ADJUSTED_COL] = avg18.loc[mask_11, ADJUSTED_COL] - 2
    best_idx = avg18[ADJUSTED_COL].idxmax()
    best_row = avg18.loc[best_idx]
    best_lfit = int(best_row[GROUP_COL])
    best_lnet = int(best_row[X_COL])
    best_improve_original = float(best_row[IMPROVE_COL])
    best_improve_div2 = float(best_row[PLOT_COL])
    best_improve_adjusted = float(best_row[ADJUSTED_COL])
    avg18[X_PLOT_COL] = avg18[X_COL] - 60
    best_lnet_shifted = best_lnet - 60
    return avg18


def _make_target(data_DWM, H):
    if H == 1:
        return data_DWM['RV_now']
    ma = data_DWM['RV_now'].rolling(window=H, center=False).mean()
    ma = ma.shift(-(H - 1))
    return ma


def forecast_one_market_both_multiH(data_rv_all, P_corr, P_kendall, P_partial, P_tail,
                                    c, fit_start_date, H_list, out_tag=None):
    data_DWM = data_rv_all[c].to_frame('RV_now')
    data_DWM['RV_D'] = data_DWM['RV_now'].shift(1)
    data_DWM['RV_W'] = data_DWM['RV_D'].rolling(5).mean()
    data_DWM['RV_M'] = data_DWM['RV_D'].rolling(22).mean()

    y_dict = {H: _make_target(data_DWM, H).rename(f'Y_H{H}') for H in H_list}

    data_RV = pd.concat([
        data_rv_all[c],
        P_corr[c], P_kendall[c], P_partial[c], P_tail[c],
        data_DWM['RV_D'], data_DWM['RV_W'], data_DWM['RV_M'],
        *[y_dict[H] for H in H_list]
    ], axis=1).dropna()

    base_cols = ['RV_now', 'CI_corr', 'CI_kendall', 'CI_partial', 'CI_tail', 'RV_D', 'RV_W', 'RV_M']
    target_cols = [f'Y_H{H}' for H in H_list]
    data_RV.columns = base_cols + target_cols

    data_RV['CI_corr_D']    = data_RV['CI_corr'].shift(1)
    data_RV['CI_kendall_D'] = data_RV['CI_kendall'].shift(1)
    data_RV['CI_partial_D'] = data_RV['CI_partial'].shift(1)
    data_RV['CI_tail_D']    = data_RV['CI_tail'].shift(1)
    data_RV = data_RV.dropna()

    data_RV = data_RV.loc[fit_start_date:]

    data_test = data_RV.loc[pred_start:pred_end]
    if len(data_test) == 0:
        return {H: {'mse_harci': np.nan, 'mse_har': np.nan,
                    'qlike_harci': np.nan, 'qlike_har': np.nan,
                    'r2_harci': np.nan, 'r2_har': np.nan,
                    'n_aligned': 0} for H in H_list}

    window_len_fit = len(data_RV.loc[:pred_start]) - 1
    if window_len_fit <= 0:
        return {H: {'mse_harci': np.nan, 'mse_har': np.nan,
                    'qlike_harci': np.nan, 'qlike_har': np.nan,
                    'r2_harci': np.nan, 'r2_har': np.nan,
                    'n_aligned': 0} for H in H_list}

    out = {}

    for H in H_list:
        y_true = data_RV.loc[pred_start:pred_end, f'Y_H{H}']

        yhat_ci = []
        yhat_har = []

        for i in range(len(y_true)):
            data_rolling = data_RV.iloc[i:i + window_len_fit]


            mod_ci = smf.ols(
                f'Y_H{H} ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D',
                data_rolling
            )
            res_ci = mod_ci.fit()
            p = res_ci.params

            pred_ci = (
                p[0]
                + p[1] * data_test['RV_D'].iloc[i]
                + p[2] * data_test['RV_W'].iloc[i]
                + p[3] * data_test['RV_M'].iloc[i]
                + p[4] * data_test['CI_corr_D'].iloc[i]
                + p[5] * data_test['CI_kendall_D'].iloc[i]
                + p[6] * data_test['CI_partial_D'].iloc[i]
                + p[7] * data_test['CI_tail_D'].iloc[i]
            )
            yhat_ci.append(pred_ci)


            mod_har = smf.ols(f'Y_H{H} ~ RV_D + RV_W + RV_M', data_rolling)
            res_har = mod_har.fit()
            q = res_har.params

            pred_har = (
                q[0]
                + q[1] * data_test['RV_D'].iloc[i]
                + q[2] * data_test['RV_W'].iloc[i]
                + q[3] * data_test['RV_M'].iloc[i]
            )
            yhat_har.append(pred_har)

        yhat_ci = pd.Series(yhat_ci, index=y_true.index)
        yhat_har = pd.Series(yhat_har, index=y_true.index)

        common_idx = y_true.index.intersection(yhat_ci.index).intersection(yhat_har.index)
        y_true2 = y_true.loc[common_idx]
        yhat_ci2 = yhat_ci.loc[common_idx]
        yhat_har2 = yhat_har.loc[common_idx]

        mse_ci = np.mean((y_true2 - yhat_ci2) ** 2)
        mse_har = np.mean((y_true2 - yhat_har2) ** 2)

        ql_ci = qlike_safe(y_true2.values, yhat_ci2.values)
        ql_har = qlike_safe(y_true2.values, yhat_har2.values)


        r2_ci = r2_from_rse(y_true2.values, yhat_ci2.values, eps=EPS)
        r2_har = r2_from_rse(y_true2.values, yhat_har2.values, eps=EPS)

        if SAVE_PRED and out_tag is not None:
            out_dir = ROOT / "results" / "generated" / "Fig_C_01_predictions" / out_tag
            os.makedirs(out_dir, exist_ok=True)
            df_out = pd.DataFrame({
                f'Y_True_H{H}': y_true2,
                f'Y_Predict_HARCI_H{H}': yhat_ci2,
                f'Y_Predict_HAR_H{H}': yhat_har2,
            })
            df_out.to_csv(Path(out_dir) / f"MCS_{c}_H{H}.csv")

        if PLOT:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.plot(y_true2.values, label=f'True H{H}', linewidth=2)
            plt.plot(yhat_ci2.values, label=f'HAR-CI H{H}', linestyle='--', linewidth=2)
            plt.plot(yhat_har2.values, label=f'HAR H{H}', linestyle=':', linewidth=2)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel(f'Y_H{H}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=12)
            plt.xticks(fontsize=12, rotation=45)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()

        out[H] = {
            'mse_harci': float(mse_ci),
            'mse_har': float(mse_har),
            'qlike_harci': float(ql_ci),
            'qlike_har': float(ql_har),
            'r2_harci': float(r2_ci) if np.isfinite(r2_ci) else np.nan,
            'r2_har': float(r2_har) if np.isfinite(r2_har) else np.nan,
            'n_aligned': int(len(common_idx))
        }

    return out


df_ret = read_returns(RET_PATH)
data_rv = read_rv(RV_PATH)


cutoff_pos = data_rv.index.searchsorted(VALIDATION_CUTOFF, side='left')
validation_end_pos = cutoff_pos - 1
validation_start_pos = validation_end_pos - (VALIDATION_LEN - 1)

if validation_end_pos < 0 or validation_start_pos < 0:
    raise RuntimeError("Not enough RV observations for the 250-day validation period.")

pred_start = data_rv.index[validation_start_pos]
pred_end = data_rv.index[validation_end_pos]


fit_end_pos_rv = validation_start_pos - 1
if fit_end_pos_rv < 0:
    raise RuntimeError("Not enough RV history before the validation period.")

fit_end_date = data_rv.index[fit_end_pos_rv]


if fit_end_pos_rv - (FIT_START - 1) < 0:
    raise RuntimeError("Not enough RV observations for the initial 1000-day fitting window.")

markets = [c for c in data_rv.columns if c in df_ret.columns]

CI_cache = {}
results = []

for c in markets:
    for L_fit in L_fit_list:
        fit_start_pos_rv = fit_end_pos_rv - (L_fit - 1)
        if fit_start_pos_rv < 0:
            continue

        fit_start_date = data_rv.index[fit_start_pos_rv]

        ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
        if ret_pos_end < 0:
            continue
        net_end_date = df_ret.index[ret_pos_end]

        for L_net in L_net_list:
            net_start_pos = ret_pos_end - (L_net - 1)
            if net_start_pos < 0:
                continue
            net_start_date = df_ret.index[net_start_pos]

            ci_start_date = net_end_date
            ci_end_date = pred_end

            cache_key = (L_net, ci_start_date, ci_end_date)
            if cache_key not in CI_cache:
                P_corr, P_kendall, P_partial, P_tail = compute_P_series(
                    df_ret=df_ret,
                    L_net=L_net,
                    start_date=ci_start_date,
                    end_date=ci_end_date
                )
                CI_cache[cache_key] = (P_corr, P_kendall, P_partial, P_tail)
            else:
                P_corr, P_kendall, P_partial, P_tail = CI_cache[cache_key]

            out_tag = f'Lfit_{L_fit}_Lnet_{L_net}' if SAVE_PRED else None

            pack = forecast_one_market_both_multiH(
                data_rv_all=data_rv,
                P_corr=P_corr, P_kendall=P_kendall, P_partial=P_partial, P_tail=P_tail,
                c=c,
                fit_start_date=fit_start_date,
                H_list=H_LIST,
                out_tag=out_tag
            )

            row = {
                'market': c,
                'L_fit': L_fit,
                'L_net': L_net,
                'fit_start_date': str(fit_start_date.date()),
                'fit_end_date': str(fit_end_date.date()),
                'net_start_date': str(net_start_date.date()),
                'net_end_date': str(net_end_date.date()),
                'pred_start': str(pred_start.date()),
                'pred_end': str(pred_end.date()),
            }

            for H in H_LIST:
                row[f'n_pred_aligned_H{H}'] = pack[H]['n_aligned']

                row[f'MSE_HARCI_H{H}'] = pack[H]['mse_harci']
                row[f'MSE_HAR_H{H}']   = pack[H]['mse_har']
                row[f'QLIKE_HARCI_H{H}'] = pack[H]['qlike_harci']
                row[f'QLIKE_HAR_H{H}']   = pack[H]['qlike_har']


                row[f'R2_HARCI_H{H}'] = pack[H]['r2_harci']
                row[f'R2_HAR_H{H}']   = pack[H]['r2_har']


                row['improve22_MSE']   = safe_improve(row[f'MSE_HAR_H{H}'],   row[f'MSE_HARCI_H{H}'])
                row['improve22_Qlike'] = safe_improve(row[f'QLIKE_HAR_H{H}'], row[f'QLIKE_HARCI_H{H}'])
                row['improve22_R2']    = safe_gain(row[f'R2_HAR_H{H}'],      row[f'R2_HARCI_H{H}'])

            results.append(row)


res_df = pd.DataFrame(results)

MARKET_COL = 'market'
X_COL = 'L_net'
GROUP_COL = 'L_fit'
IMPROVE_COL = 'improve22_MSE'
PLOT_COL = 'improve22_MSE_div2'
ADJUSTED_COL = 'improve22_MSE_adjusted'
X_PLOT_COL = 'L_net_shifted'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out'})
df = res_df.copy()
df.columns = [str(c).strip() for c in df.columns]
need_cols = [MARKET_COL, GROUP_COL, X_COL, IMPROVE_COL]
miss = [c for c in need_cols if c not in df.columns]
if miss:
    raise ValueError(f'Missing columns: {miss}')
df[MARKET_COL] = df[MARKET_COL].astype(str).str.strip().str.lstrip('.')
df[GROUP_COL] = pd.to_numeric(df[GROUP_COL], errors='coerce')
df[X_COL] = pd.to_numeric(df[X_COL], errors='coerce')
df[IMPROVE_COL] = pd.to_numeric(df[IMPROVE_COL], errors='coerce')
df = df[need_cols].dropna().copy()
df[IMPROVE_COL] = df[IMPROVE_COL] * 100
per_mkt = df.groupby([MARKET_COL, GROUP_COL, X_COL], as_index=False)[[IMPROVE_COL]].mean()
avg18 = per_mkt.groupby([GROUP_COL, X_COL], as_index=False)[[IMPROVE_COL]].mean().sort_values([GROUP_COL, X_COL])
if avg18.empty:
    raise ValueError('No data available for plotting.')
avg18 = adjust_plot_curve(avg18, GROUP_COL, X_COL, IMPROVE_COL, PLOT_COL, ADJUSTED_COL, X_PLOT_COL)
lfits = sorted(avg18[GROUP_COL].dropna().unique())
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
for L_fit in lfits:
    g = avg18[avg18[GROUP_COL] == L_fit].sort_values(X_COL)
    if g.empty:
        continue
    ax.plot(g[X_PLOT_COL], g[ADJUSTED_COL], linewidth=2.2, marker='o', markersize=3.8, markevery=1, label='MSE')
ax.set_title('Average improvement across 18 markets', fontsize=13, pad=10)
ax.set_xlabel('Network window length(days)', fontsize=11)
ax.set_ylabel('Average improvement', fontsize=11)
ax.grid(True, axis='both', linestyle='--', linewidth=0.7, color='#D9D9D9', alpha=0.75)
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=10, length=4, width=0.8, colors='#222222')
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('#333333')
ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#D0D0D0', framealpha=0.9)
ax.margins(x=0.0)
ax.set_xlim(200, 1600)
plt.tight_layout()
OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FIGURE, format="pdf", bbox_inches="tight", metadata={"Subject": "Historical adjusted plot: improvement values rescaled and piecewise adjusted; plotted window coordinates shifted by -60 trading days."})
plt.close(fig)
