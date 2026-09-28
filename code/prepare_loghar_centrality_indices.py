from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
from pycop import empirical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RET_PATH = PROJECT_ROOT / 'data' / 'raw' / 'Return.csv'
RV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'RV.csv'
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'generated' / 'centrality' / 'P_log.csv'
L_NET = 700
L_FIT = 1000
FIT_END_DATE = pd.Timestamp('2018-05-01')
PRED_END = pd.Timestamp('2022-02-25')


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

    for d in tqdm(dates, desc='Computing LogHAR centrality indices'):
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


if __name__ == '__main__':
    df_ret = read_returns(RET_PATH)
    data_rv = read_rv(RV_PATH)
    if FIT_END_DATE not in data_rv.index:
        raise ValueError('The estimation end date is missing from the RV data.')
    fit_end_pos = data_rv.index.get_loc(FIT_END_DATE)
    fit_start_pos = fit_end_pos - (L_FIT - 1)
    if fit_start_pos < 0:
        raise ValueError('Insufficient RV history for the estimation window.')
    fit_start_date = data_rv.index[fit_start_pos]
    ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
    if ret_pos_end - (L_NET - 1) < 0:
        raise ValueError('Insufficient return history for the network window.')
    ci_start_date = df_ret.index[ret_pos_end]
    ci_end_date = PRED_END
    P_corr, P_kendall, P_partial, P_tail = compute_P_series(
        df_ret=df_ret,
        L_net=L_NET,
        start_date=ci_start_date,
        end_date=ci_end_date
    )
    output = pd.concat({
        'P_corr': P_corr.stack(),
        'P_kendall': P_kendall.stack(),
        'P_partial': P_partial.stack(),
        'P_tail': P_tail.stack(),
    }, axis=1).reset_index()
    output.columns = ['date', 'market', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    output['L_net'] = L_NET
    output['ci_start_date'] = str(pd.to_datetime(ci_start_date).date())
    output['ci_end_date'] = str(pd.to_datetime(ci_end_date).date())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
