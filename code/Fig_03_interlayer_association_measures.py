from pathlib import Path
import os
import math
import time
import scipy as sc
from scipy.stats import norm
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
import seaborn as sns
import scipy.stats as ss
from scipy import stats
import statistics
from pycop import empirical
import statsmodels.formula.api as smf
from matplotlib.dates import DateFormatter
from matplotlib.pylab import mpl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["axes.grid"] = False
plt.rc('font', family='Times New Roman')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

RET_PATH = PROJECT_ROOT / 'data' / 'raw' / 'Return.csv'
RV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'RV.csv'


LA = 240
LB = 3
LC = 120
LD = 30

START_DATE_PLOT = '2014-07-04'
END_DATE_PLOT   = '2018-05-02'
MIN_MARKETS_NET = 5
SAVE_FIG = True


USE_ABS_ASSOC_STRENGTH = True


OUT_DIR = PROJECT_ROOT / 'results' / 'generated'
os.makedirs(OUT_DIR, exist_ok=True)


def _norm_date_series(s):

    s = pd.to_datetime(s, errors='coerce', utc=True)
    return s.dt.tz_localize(None).dt.normalize()

def _norm_date_index(idx):

    x = pd.to_datetime(idx, errors='coerce', utc=True)
    x = pd.DatetimeIndex(x)
    x = x.tz_localize(None).normalize()
    return x

def read_returns(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]

    if 'date' not in df.columns:
        raise RuntimeError("The returns file must contain a 'date' column.")

    df['date'] = pd.to_datetime(
        df['date'].astype(str).str.strip().str[:10],
        errors='coerce'
    )
    df = df.dropna(subset=['date']).copy()
    df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    df = df.set_index('date').sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(axis=1, how='all')
    return df

def clean_colnames(df):

    out = df.copy()
    out.columns = [str(c).strip().replace('.', '') for c in out.columns]

    out = out.T.groupby(level=0).mean().T
    return out


def build_network_weighted(corr_matrix: pd.DataFrame):


    G = nx.Graph()
    for n1, row in corr_matrix.iterrows():
        for n2, value in row.items():
            if n1 == n2:
                continue
            if pd.isna(value):
                continue
            value = float(np.clip(value, -1.0, 1.0))

            distance = np.sqrt(2 * (1 - (value)))
            if np.isnan(distance) or np.isinf(distance):
                continue
            G.add_edge(n1, n2, weight=distance)

    nodes = list(corr_matrix.columns)
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        nan_dict = {k: np.nan for k in nodes}
        return None, None, {'NS': nan_dict, 'EC': nan_dict, 'CC': nan_dict, 'BC': nan_dict, 'Ecc': nan_dict}

    mst = nx.minimum_spanning_tree(G, weight='weight')


    mst_sim = nx.Graph()
    mst_sim.add_nodes_from(mst.nodes())
    for u, v, data in mst.edges(data=True):
        distance = data.get('weight', np.nan)
        if pd.isna(distance) or np.isinf(distance):
            continue
        sim = (1 - (distance ** 2) / 2)
        if pd.isna(sim) or np.isinf(sim):
            continue
        mst_sim.add_edge(u, v, weight=float(sim))


    for n in nodes:
        if n not in mst:
            mst.add_node(n)
        if n not in mst_sim:
            mst_sim.add_node(n)

    try:
        NS = dict(mst_sim.degree(weight='weight'))
    except Exception:
        NS = {k: np.nan for k in nodes}
    try:
        EC = nx.eigenvector_centrality(mst_sim, weight='weight', tol=1e-3)
    except Exception:
        EC = {k: np.nan for k in nodes}
    try:
        CC = nx.closeness_centrality(mst, distance='weight')
    except Exception:
        CC = {k: np.nan for k in nodes}
    try:
        BC = nx.betweenness_centrality(mst, weight='weight')
    except Exception:
        BC = {k: np.nan for k in nodes}
    try:
        Ecc = nx.eccentricity(mst, weight='weight')
    except Exception:
        Ecc = {k: np.nan for k in nodes}

    return mst, mst_sim, {'NS': NS, 'EC': EC, 'CC': CC, 'BC': BC, 'Ecc': Ecc}


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

    nodes = list(corr_matrix.columns)
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        nan_dict = {k: np.nan for k in nodes}
        return None, None, {'NS': nan_dict, 'EC': nan_dict, 'CC': nan_dict, 'BC': nan_dict, 'Ecc': nan_dict}

    mst = nx.minimum_spanning_tree(G, weight='weight')

    unweighted_mst = nx.Graph()
    unweighted_mst.add_nodes_from(mst.nodes())
    for u, v in mst.edges():
        unweighted_mst.add_edge(u, v)

    for n in nodes:
        if n not in mst:
            mst.add_node(n)
        if n not in unweighted_mst:
            unweighted_mst.add_node(n)

    try:
        NS = dict(unweighted_mst.degree())
    except Exception:
        NS = {k: np.nan for k in nodes}
    try:
        EC = nx.eigenvector_centrality(unweighted_mst, tol=1e-2)
    except Exception:
        EC = {k: np.nan for k in nodes}
    try:
        CC = nx.closeness_centrality(unweighted_mst)
    except Exception:
        CC = {k: np.nan for k in nodes}
    try:
        BC = nx.betweenness_centrality(unweighted_mst)
    except Exception:
        BC = {k: np.nan for k in nodes}
    try:
        Ecc = nx.eccentricity(unweighted_mst)
    except Exception:
        Ecc = {k: np.nan for k in nodes}

    return mst, unweighted_mst, {'NS': NS, 'EC': EC, 'CC': CC, 'BC': BC, 'Ecc': Ecc}


def compute_four_dependency_matrices(data_rolling: pd.DataFrame):


    corr_mat = data_rolling.corr(method='pearson').replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


    kendall_mat = data_rolling.corr(method='kendall').replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


    try:
        inv_corr = np.linalg.inv(corr_mat.values)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr_mat.values)

    denom = np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    with np.errstate(divide='ignore', invalid='ignore'):
        partial_raw = -inv_corr / denom

    partial_mat = pd.DataFrame(partial_raw, index=corr_mat.index, columns=corr_mat.columns)
    partial_mat = partial_mat.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    np.fill_diagonal(partial_mat.values, 1.0)


    tail_mat = pd.DataFrame(0.0, index=data_rolling.columns, columns=data_rolling.columns)
    cols = list(data_rolling.columns)
    for a in range(len(cols)):
        for b in range(a, len(cols)):
            stock_a = cols[a]
            stock_b = cols[b]
            if a == b:
                tail_corr = 1.0
            else:
                try:
                    pair = data_rolling[[stock_a, stock_b]].dropna()
                    if len(pair) < 10:
                        tail_corr = 0.0
                    else:
                        tail_corr = empirical(pair.T.values).LTDC(0.1)
                except Exception:
                    tail_corr = np.nan

                if pd.isna(tail_corr) or np.isinf(tail_corr):
                    tail_corr = 0.0
                tail_corr = float(np.clip(tail_corr, -1.0, 1.0))

            tail_mat.loc[stock_a, stock_b] = tail_corr
            tail_mat.loc[stock_b, stock_a] = tail_corr
    np.fill_diagonal(tail_mat.values, 1.0)

    return {
        'Pearson': corr_mat,
        'Kendall': kendall_mat,
        'Partial': partial_mat,
        'Tail': tail_mat
    }


def graph_to_adjacency_df(G, nodes):


    A = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    if G is None:
        return A
    for u, v in G.edges():
        if (u in A.index) and (v in A.columns):
            A.loc[u, v] = 1
            A.loc[v, u] = 1
    np.fill_diagonal(A.values, 0)
    return A


def metric_avg_association_strength(dep_mats_dict, graph_for_edges_dict, use_abs=True):


    out = {}
    for layer_name, M in dep_mats_dict.items():
        G = graph_for_edges_dict.get(layer_name, None)
        if G is None or G.number_of_edges() == 0:
            out[layer_name] = np.nan
            continue

        vals = []
        for u, v in G.edges():
            if (u in M.index) and (v in M.columns):
                w = M.loc[u, v]
                if pd.notna(w):
                    vals.append(abs(w) if use_abs else float(w))

        out[layer_name] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
    return out


def metric_average_edge_overlap(adj_dict):


    layer_names = list(adj_dict.keys())
    if len(layer_names) == 0:
        return np.nan

    all_nodes = sorted(set().union(*[set(A.index) for A in adj_dict.values()]))
    A_list = [
        adj_dict[name].reindex(index=all_nodes, columns=all_nodes, fill_value=0).astype(int)
        for name in layer_names
    ]

    n = len(all_nodes)
    if n < 2:
        return np.nan

    overlap_count = np.zeros((n, n), dtype=int)
    for A in A_list:
        overlap_count += A.values

    iu = np.triu_indices(n, k=1)
    overlap_upper = overlap_count[iu]

    k_union = int(np.sum(overlap_upper >= 1))
    if k_union == 0:
        return np.nan

    O_avg = float(np.sum(overlap_upper) / k_union)
    return O_avg


def metric_unique_edge_coefficient(adj_dict):


    layer_names = list(adj_dict.keys())
    if len(layer_names) == 0:
        return {}

    all_nodes = sorted(set().union(*[set(A.index) for A in adj_dict.values()]))
    A_std = {
        name: adj_dict[name].reindex(index=all_nodes, columns=all_nodes, fill_value=0).astype(int)
        for name in layer_names
    }

    n = len(all_nodes)
    iu = np.triu_indices(n, k=1)

    out = {}
    for alpha in layer_names:
        A_alpha = A_std[alpha].values
        a_alpha = A_alpha[iu]

        k_alpha = int(np.sum(a_alpha))
        if k_alpha == 0:
            out[alpha] = np.nan
            continue

        prod_other = np.ones_like(a_alpha, dtype=int)
        for beta in layer_names:
            if beta == alpha:
                continue
            prod_other *= (1 - A_std[beta].values[iu])

        unique_num = int(np.sum(a_alpha * prod_other))
        out[alpha] = unique_num / k_alpha

    return out


def metric_interlayer_degree_rank_correlation(unweighted_graph_dict):


    layer_order = ['Pearson', 'Kendall', 'Partial', 'Tail']
    layer_names = [x for x in layer_order if x in unweighted_graph_dict]

    if len(layer_names) < 2:
        return {}

    all_nodes = sorted(set().union(*[
        set(unweighted_graph_dict[name].nodes()) if unweighted_graph_dict[name] is not None else set()
        for name in layer_names
    ]))

    if len(all_nodes) < 3:
        return {f'{a}-{b}': np.nan for i, a in enumerate(layer_names) for b in layer_names[i+1:]}

    degree_df = pd.DataFrame(index=all_nodes)
    for name in layer_names:
        G = unweighted_graph_dict[name]
        if G is None:
            degree_df[name] = np.nan
        else:
            degs = dict(G.degree())
            degree_df[name] = pd.Series(degs).reindex(all_nodes)

    degree_df = degree_df.fillna(0.0)


    rank_df = degree_df.rank(axis=0, ascending=False, method='average')

    pair_values = {}
    for i in range(len(layer_names)):
        for j in range(i + 1, len(layer_names)):
            a, b = layer_names[i], layer_names[j]
            x = rank_df[a]
            y = rank_df[b]

            if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
                pair_values[f'{a}-{b}'] = np.nan
            else:
                pair_values[f'{a}-{b}'] = float(x.corr(y, method='pearson'))

    return pair_values


def compute_multilayer_timeseries_metrics(df_ret, L_net, start_date=None, end_date=None, min_markets=5):


    df_ret = df_ret.sort_index().copy()

    if start_date is not None or end_date is not None:
        dates = df_ret.loc[start_date:end_date].index
    else:
        dates = df_ret.index

    panel_a_records = []
    panel_b_records = []
    panel_c_records = []
    panel_d_records = []
    used_dates = []

    for d in tqdm(dates, desc='Computing network measures'):

        data_rolling = df_ret.loc[:d].tail(L_net).copy()


        data_rolling = data_rolling.dropna(axis=1, how='all')
        std = data_rolling.std(axis=0, skipna=True)
        good_cols = std[std > 0].index
        data_rolling = data_rolling[good_cols]

        if data_rolling.shape[1] < min_markets:
            continue


        dep_mats = compute_four_dependency_matrices(data_rolling)


        weighted_mst_dist = {}
        weighted_mst_sim = {}
        unweighted_mst_dist = {}
        unweighted_mst = {}
        centrality_weighted = {}
        centrality_unweighted = {}


        for layer_name, M in dep_mats.items():
            mst_w_dist, mst_w_sim, cent_w = build_network_weighted(M)
            mst_u_dist, mst_u, cent_u = build_network_unweighted(M)

            weighted_mst_dist[layer_name] = mst_w_dist
            weighted_mst_sim[layer_name] = mst_w_sim
            unweighted_mst_dist[layer_name] = mst_u_dist
            unweighted_mst[layer_name] = mst_u
            centrality_weighted[layer_name] = cent_w
            centrality_unweighted[layer_name] = cent_u


        all_nodes_union = sorted(set().union(*[
            set(g.nodes()) if g is not None else set()
            for g in unweighted_mst.values()
        ]))

        if len(all_nodes_union) == 0:
            continue

        adj_dict = {
            layer_name: graph_to_adjacency_df(unweighted_mst[layer_name], all_nodes_union)
            for layer_name in ['Pearson', 'Kendall', 'Partial', 'Tail']
            if layer_name in unweighted_mst
        }


        a_vals = metric_avg_association_strength(
            dep_mats_dict=dep_mats,
            graph_for_edges_dict=unweighted_mst,
            use_abs=USE_ABS_ASSOC_STRENGTH
        )


        O_avg = metric_average_edge_overlap(adj_dict)
        b_vals = {'overall': O_avg}


        c_vals = metric_unique_edge_coefficient(adj_dict)


        d_pairs = metric_interlayer_degree_rank_correlation(unweighted_mst)

        used_dates.append(d)
        panel_a_records.append(a_vals)
        panel_b_records.append(b_vals)
        panel_c_records.append(c_vals)
        panel_d_records.append(d_pairs)

    idx = _norm_date_index(pd.DatetimeIndex(used_dates))

    panel_a_df = pd.DataFrame(panel_a_records, index=idx).sort_index()
    panel_b_df = pd.DataFrame(panel_b_records, index=idx).sort_index()
    panel_c_df = pd.DataFrame(panel_c_records, index=idx).sort_index()
    panel_d_df = pd.DataFrame(panel_d_records, index=idx).sort_index()

    layer_order = ['Pearson', 'Kendall', 'Partial', 'Tail']
    if len(panel_a_df) > 0:
        panel_a_df = panel_a_df[[c for c in layer_order if c in panel_a_df.columns]]
    if len(panel_c_df) > 0:
        panel_c_df = panel_c_df[[c for c in layer_order if c in panel_c_df.columns]]

    pair_order = [
        'Pearson-Kendall',
        'Pearson-Partial',
        'Pearson-Tail',
        'Kendall-Partial',
        'Kendall-Tail',
        'Partial-Tail'
    ]
    if len(panel_d_df) > 0:
        panel_d_df = panel_d_df[[c for c in pair_order if c in panel_d_df.columns]]

    return panel_a_df, panel_b_df, panel_c_df, panel_d_df


def compute_single_panel_by_lnet(df_ret, panel_name, L_net, start_date=None, end_date=None, min_markets=5):


    pa, pb, pc, pd_ = compute_multilayer_timeseries_metrics(
        df_ret=df_ret,
        L_net=L_net,
        start_date=start_date,
        end_date=end_date,
        min_markets=min_markets
    )

    panel_name = str(panel_name).lower()
    if panel_name == 'a':
        return pa
    elif panel_name == 'b':
        return pb
    elif panel_name == 'c':
        return pc
    elif panel_name == 'd':
        return pd_
    else:
        raise ValueError("panel_name must be one of {'a','b','c','d'}")


def plot_figure1_style(panel_a_df, panel_b_df, panel_c_df, panel_d_df,
                       save_path=None, title=None):


    layer_colors = {
        'Pearson': '#1f77b4',
        'Kendall': '#d2691e',
        'Partial': '#DAA520',
        'Tail': '#7D3C98'
    }

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    ax1.set_ylabel(r'$\langle C^{[\alpha]} \rangle$', fontsize=12)
    ax2.set_ylabel(r'$\langle O \rangle$', fontsize=12)
    ax3.set_ylabel(r'$U^{[\alpha]}$', fontsize=12)


    if panel_a_df is not None and len(panel_a_df) > 0:
        for col in ['Pearson', 'Kendall', 'Partial', 'Tail']:
            if col not in panel_a_df.columns:
                continue
            s = panel_a_df[col].astype(float)
            ax1.plot(
                s.index, s.values,
                lw=1.2,
                color=layer_colors.get(col),
                alpha=0.95,
                label=col
            )
        ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.set_title('(a) Average Correlation Strength', fontsize=12, y=-0.28)
    ax1.tick_params(axis='x', rotation=45)


    if panel_b_df is not None and len(panel_b_df) > 0 and 'overall' in panel_b_df.columns:
        s = panel_b_df['overall'].astype(float)
        ax2.plot(s.index, s.values, lw=1.0, alpha=0.9, color='#1f77b4')
    ax2.set_title('(b) Average Edge Overlap', fontsize=12, y=-0.28)
    ax2.tick_params(axis='x', rotation=45)


    if panel_c_df is not None and len(panel_c_df) > 0:
        for col in ['Pearson', 'Kendall', 'Partial', 'Tail']:
            if col not in panel_c_df.columns:
                continue
            s = panel_c_df[col].astype(float)
            ax3.plot(
                s.index, s.values,
                lw=1.0,
                color=layer_colors.get(col),
                alpha=0.95,
                label=col
            )
        ax3.legend(loc='upper right', fontsize=9, frameon=True)
    ax3.set_title('(c) Unique Edge Ratio', fontsize=12, y=-0.28)
    ax3.tick_params(axis='x', rotation=45)


    pair_order = [
        'Pearson-Kendall',
        'Pearson-Partial',
        'Pearson-Tail',
        'Kendall-Partial',
        'Kendall-Tail',
        'Partial-Tail'
    ]
    if panel_d_df is not None and len(panel_d_df) > 0:
        hm_df = panel_d_df.copy()
        hm_df = hm_df[[c for c in pair_order if c in hm_df.columns]]
        hm_plot = hm_df.T

        sns.heatmap(
            hm_plot,
            ax=ax4,
            cmap='hot_r',
            cbar=True,
            xticklabels=False,
            yticklabels=True,
            vmin=-0.2, vmax=0.8
        )
        ax4.set_xlabel('')
        ax4.set_ylabel('')


        if hm_df.shape[0] > 1:
            n_ticks = min(8, hm_df.shape[0])
            xtick_pos = np.linspace(0, hm_df.shape[0] - 1, n_ticks, dtype=int)
            ax4.set_xticks(xtick_pos + 0.5)
            ax4.set_xticklabels(
                [hm_df.index[i].strftime('%Y-%m-%d') for i in xtick_pos],
                rotation=45, ha='right', fontsize=8
            )

    ax4.set_title('(d) Degree Correlation Between Layers', fontsize=12, y=-0.28)


    time_dfs = [df for df in [panel_a_df, panel_b_df, panel_c_df] if df is not None and len(df) > 0]
    if len(time_dfs) > 0:
        xmin = min(df.index.min() for df in time_dfs)
        xmax = max(df.index.max() for df in time_dfs)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([xmin, xmax])

    for ax in [ax1, ax2, ax3]:
        ax.grid(False)

    if title is not None:
        fig.suptitle(title, fontsize=13, y=0.99)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=600)

    plt.show()


if __name__ == '__main__':

    df_ret = read_returns(RET_PATH)
    df_ret = clean_colnames(df_ret)


    panel_a_df = compute_single_panel_by_lnet(
        df_ret=df_ret,
        panel_name='a',
        L_net=LA,
        start_date=START_DATE_PLOT,
        end_date=END_DATE_PLOT,
        min_markets=MIN_MARKETS_NET
    )

    panel_b_df = compute_single_panel_by_lnet(
        df_ret=df_ret,
        panel_name='b',
        L_net=LB,
        start_date=START_DATE_PLOT,
        end_date=END_DATE_PLOT,
        min_markets=MIN_MARKETS_NET
    )

    panel_c_df = compute_single_panel_by_lnet(
        df_ret=df_ret,
        panel_name='c',
        L_net=LC,
        start_date=START_DATE_PLOT,
        end_date=END_DATE_PLOT,
        min_markets=MIN_MARKETS_NET
    )

    panel_d_df = compute_single_panel_by_lnet(
        df_ret=df_ret,
        panel_name='d',
        L_net=LD,
        start_date=START_DATE_PLOT,
        end_date=END_DATE_PLOT,
        min_markets=MIN_MARKETS_NET
    )




    fig_path = (
        os.path.join(
            OUT_DIR,
            'Fig_03_interlayer_association_measures.pdf'
        )
        if SAVE_FIG else None
    )

    plot_figure1_style(
        panel_a_df=panel_a_df,
        panel_b_df=panel_b_df,
        panel_c_df=panel_c_df,
        panel_d_df=panel_d_df,
        save_path=fig_path,
    )
