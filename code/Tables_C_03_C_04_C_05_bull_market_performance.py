from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
from arch.bootstrap import MCS
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side

ROOT = Path(__file__).resolve().parents[1]
RV_FILE = ROOT / "data/raw/RV.csv"
RETURN_FILE = ROOT / "data/raw/Return.csv"
CENTRALITY_FILE = ROOT / "data/processed/centrality/P.csv"
TCI_FILE = ROOT / "data/processed/spillover/TCI_data_BK-TVP.csv"
MCS_DROP_DATES_FILE = ROOT / "data/processed/evaluation/qlike_intermediate.csv"
PRICE_FILE = ROOT / "data/raw/MAN_data.csv"
LOG_CENTRALITY_FILE = ROOT / "data/processed/centrality/P_log.csv"
THRESH = 1e-3
SCALE_IF_MOST_GT_THRESH = 1e-4
EPS = 1e-12
RATIO_MIN = 1e-6
RATIO_MAX = 1e6
QLIKE_CAP = 100.0
POST_SCALE_THRESHOLD = 1e4
POST_SCALE_FACTOR = 1e-8




warnings.filterwarnings("ignore")






from scipy.signal import argrelextrema
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def pagan_sossounov(prices, window=8 * 20, censor=6 * 20, cycle=16 * 20, phase=4 * 20, threshold=0.2):
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    local_max_idx = argrelextrema(prices.values, np.greater_equal, order=window)[0]
    local_min_idx = argrelextrema(prices.values, np.less_equal, order=window)[0]
    extrema_idx = np.sort(np.concatenate([local_max_idx, local_min_idx]))
    extrema_idx = extrema_idx[(extrema_idx > censor) & (extrema_idx < len(prices) - censor)]
    labels = pd.Series(index=prices.index, data=np.nan)
    for i in range(1, len(extrema_idx)):
        start_idx, end_idx = (extrema_idx[i - 1], extrema_idx[i])
        price_change = (prices.iloc[end_idx] - prices.iloc[start_idx]) / prices.iloc[start_idx]
        if price_change > threshold:
            labels.iloc[start_idx:end_idx] = 1
        elif price_change < -threshold:
            labels.iloc[start_idx:end_idx] = -1
    labels = _enforce_phase_cycle_constraints(labels, cycle, phase)
    return labels


def _enforce_phase_cycle_constraints(labels, cycle, phase):
    return labels


def compute_bull_market_periods(price_file, rv_file):
    raw = pd.read_csv(price_file, usecols=['date', 'Symbol', 'close_price'])
    prices = raw.pivot(index='date', columns='Symbol', values='close_price')
    prices = prices.loc['2006-01-01 00:00:00+00:00':]
    missing_fraction = prices.isna().mean()
    prices = prices.drop(columns=missing_fraction[missing_fraction > 0.1].index)
    prices = prices.dropna(how='all').bfill()
    dates = pd.to_datetime(prices.index.astype(str).str[:10])
    markets = [c for c in pd.read_csv(rv_file, nrows=0).columns if c != 'date']
    missing = [market for market in markets if market not in prices.columns]
    if missing:
        raise ValueError(f'Missing usable price series: {missing}')
    rows = []
    for market in markets:
        values = pd.Series(pd.to_numeric(prices[market], errors='coerce').to_numpy())
        labels = pagan_sossounov(values)
        phases = pd.Series(labels.eq(1).to_numpy(), index=dates)
        phases = phases.loc['2018-05-01':]
        runs = phases.ne(phases.shift()).cumsum()
        longest = None
        for _, segment in phases.groupby(runs):
            if segment.iloc[0] and (longest is None or len(segment) > len(longest)):
                longest = segment
        if longest is not None:
            rows.append({
                'market': market,
                'start_date': longest.index[0],
                'end_date': longest.index[-1],
                'observations': len(longest),
            })
    result = pd.DataFrame(rows, columns=['market', 'start_date', 'end_date', 'observations'])
    if result.empty:
        raise ValueError('No bull-market intervals found')
    return result


def run_har(periods):
    model_outputs = {}
    import os
    import re
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    red_TABLE_PATH = 'in_memory_bull_periods'
    red_df = periods.copy()

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    if not {'filename', 'start_date', 'end_date'}.issubset(set(red_df.columns)):
        raise ValueError(f'Model execution message{list(red_df.columns)}Model execution message')
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_range = {row['market']: (row['start_date'], row['end_date']) for _, row in red_df.dropna(subset=['market']).iterrows()}
    table_markets = list(dict.fromkeys([m for m in red_df['market'].tolist() if isinstance(m, str)]))
    if len(table_markets) == 0:
        raise RuntimeError('Model execution message')
    data = pd.read_csv(RV_FILE)
    if 'date' not in data.columns:
        raise RuntimeError('Model execution message')
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.set_index('date')
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    if missing_in_rv:
        pass
    BASE_OOS_START = pd.to_datetime('2018-05-02')
    BASE_OOS_END = pd.to_datetime('2022-02-25')
    OUT_DIR = 'unused_output'
    error_rows = []
    used_markets = []
    for c in markets_to_run:
        oos_start, oos_end = red_range[c]
        if oos_start < BASE_OOS_START:
            oos_start = BASE_OOS_START
        if oos_end > BASE_OOS_END:
            oos_end = BASE_OOS_END
        data_RV = pd.DataFrame(data[c]).copy()
        data_RV.columns = ['RV_now']
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = data_RV['RV_now'].rolling(window=22, center=False).mean()
        data_RV['RV_MA22'] = data_RV['RV_MA22'].shift(-22)
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc['2014-07-04':]
        data_test_full = data_RV.loc[BASE_OOS_START:oos_end]
        if len(data_test_full) == 0:
            continue
        y_true_full = data_test_full['RV_MA22']
        train_len = len(data_RV.loc[:BASE_OOS_START]) - 1
        if train_len <= 10:
            continue
        preds_full = []
        for i in tqdm(range(len(data_test_full)), desc=f'{c}'):
            data_rolling = data_RV.iloc[i:i + train_len]
            mod = smf.ols('RV_MA22 ~ RV_D + RV_W + RV_M', data_rolling)
            res = mod.fit()
            params_new = res.params
            r_predict = params_new.iloc[0] + params_new.iloc[1] * data_test_full['RV_D'].iloc[i] + params_new.iloc[2] * data_test_full['RV_W'].iloc[i] + params_new.iloc[3] * data_test_full['RV_M'].iloc[i]
            preds_full.append(r_predict)
        preds_full = pd.Series(preds_full, index=y_true_full.index)
        SUM_full = pd.concat([y_true_full, preds_full], axis=1)
        SUM_full.columns = ['RV_True', 'RV_Predict']
        SUM_seg = SUM_full.loc[oos_start:oos_end].copy()
        if len(SUM_seg) == 0:
            continue
        SUM_final = SUM_seg.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM_seg.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        EPS_QLIKE = 1e-12
        pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
        SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
        MSE = np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)
        MAE = np.mean(np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict']))
        Theil = np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2)))
        error_rows.append([MSE, MAE, Theil])
        used_markets.append(c)
        model_outputs[str(c).strip().lstrip('.')] = SUM_final.copy()
    ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    ERROR_PATH = 'unused_output'
    return model_outputs


def run_har_c04(periods):
    model_outputs = {}
    import os
    import re
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    red_TABLE_PATH = 'in_memory_bull_periods'
    red_df = periods.copy()

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    if not {'filename', 'start_date', 'end_date'}.issubset(set(red_df.columns)):
        raise ValueError(f'Model execution message{list(red_df.columns)}Model execution message')
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_range = {row['market']: (row['start_date'], row['end_date']) for _, row in red_df.dropna(subset=['market']).iterrows()}
    table_markets = list(dict.fromkeys([m for m in red_df['market'].tolist() if isinstance(m, str)]))
    if len(table_markets) == 0:
        raise RuntimeError('Model execution message')
    data = pd.read_csv(RV_FILE)
    if 'date' not in data.columns:
        raise RuntimeError('Model execution message')
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.set_index('date')
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    if missing_in_rv:
        pass
    BASE_OOS_START = pd.to_datetime('2018-05-02')
    BASE_OOS_END = pd.to_datetime('2022-02-25')
    OUT_DIR = 'unused_output'
    error_rows = []
    used_markets = []
    for c in markets_to_run:
        oos_start, oos_end = red_range[c]
        if oos_start < BASE_OOS_START:
            oos_start = BASE_OOS_START
        if oos_end > BASE_OOS_END:
            oos_end = BASE_OOS_END
        data_RV = pd.DataFrame(data[c]).copy()
        data_RV.columns = ['RV_now']
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = data_RV['RV_now'].rolling(window=22, center=False).mean()
        data_RV['RV_MA22'] = data_RV['RV_MA22'].shift(-21)
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc['2014-07-04':]
        data_test_full = data_RV.loc[BASE_OOS_START:oos_end]
        if len(data_test_full) == 0:
            continue
        y_true_full = data_test_full['RV_MA22']
        train_len = len(data_RV.loc[:BASE_OOS_START]) - 1
        if train_len <= 10:
            continue
        preds_full = []
        for i in tqdm(range(len(data_test_full)), desc=f'{c}'):
            data_rolling = data_RV.iloc[i:i + train_len]
            mod = smf.ols('RV_MA22 ~ RV_D + RV_W + RV_M', data_rolling)
            res = mod.fit()
            params_new = res.params
            r_predict = params_new.iloc[0] + params_new.iloc[1] * data_test_full['RV_D'].iloc[i] + params_new.iloc[2] * data_test_full['RV_W'].iloc[i] + params_new.iloc[3] * data_test_full['RV_M'].iloc[i]
            preds_full.append(r_predict)
        preds_full = pd.Series(preds_full, index=y_true_full.index)
        SUM_full = pd.concat([y_true_full, preds_full], axis=1)
        SUM_full.columns = ['RV_True', 'RV_Predict']
        SUM_seg = SUM_full.loc[oos_start:oos_end].copy()
        if len(SUM_seg) == 0:
            continue
        SUM_final = SUM_seg.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM_seg.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        EPS_QLIKE = 1e-12
        pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
        SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
        MSE = np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)
        MAE = np.mean(np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict']))
        Theil = np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2)))
        error_rows.append([MSE, MAE, Theil])
        used_markets.append(c)
        model_outputs[str(c).strip().lstrip('.')] = SUM_final.copy()
    ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    ERROR_PATH = 'unused_output'
    return model_outputs


def run_har_ci(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    RET_PATH = RETURN_FILE
    RV_PATH = RV_FILE
    P_STORE_PATH = CENTRALITY_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    PLOT = False
    fit_end_date = pd.to_datetime('2018-05-01')
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    H = 22
    FIT_START = 1000
    FIT_STEP = 100
    FIT_MAX = 1
    L_fit_list = [FIT_START + FIT_STEP * k for k in range(FIT_MAX)]
    NET_MIN = 1000
    NET_MAX = 1000
    NET_STEP = 50
    L_net_list = list(range(NET_MIN, NET_MAX + 1, NET_STEP))
    BASE_OUT_ROOT = 'unused_output'
    OUT_DIR = os.path.join(BASE_OUT_ROOT, 'H22-red', 'HAR-CI-new')

    def _norm_date(x):
        return pd.to_datetime(x).tz_localize(None).normalize()

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-21)

    def read_returns(path):
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str[:10], errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
        df = df.set_index('date').dropna(how='all')
        return df

    def read_rv_no_scale(path):
        data = pd.read_csv(path)
        if 'date' not in data.columns:
            raise RuntimeError("RV file must contain a 'date' column.")
        num_cols = [c for c in data.columns if c != 'date']
        data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
        data = data.dropna(how='all')
        data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
        data = data.dropna(subset=['date'])
        data['date'] = data['date'].dt.tz_localize(None)
        data['date'] = data['date'].dt.normalize()
        data = data.set_index('date')
        data = data.dropna(axis=1, how='all')
        return data
    df_ret = read_returns(RET_PATH)
    data_rv = read_rv_no_scale(RV_PATH)
    markets_rv_ret = [c for c in data_rv.columns if c in df_ret.columns]
    red_df = periods.copy()
    need_red_cols = {'filename', 'start_date', 'end_date'}
    if not need_red_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_red_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        mkt = row['market']
        st = row['start_date']
        ed = row['end_date']
        red_range[mkt] = (st, ed)
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in markets_rv_ret]
    missing_in_data = [m for m in table_markets if m not in markets_rv_ret]
    if missing_in_data:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all['ci_start_date'] = P_all['ci_start_date'].apply(_norm_date)
    P_all['ci_end_date'] = P_all['ci_end_date'].apply(_norm_date)
    P_all['L_net'] = pd.to_numeric(P_all['L_net'], errors='coerce').astype('Int64')
    P_all = P_all.dropna(subset=['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net']).copy()
    P_cache = {}

    def load_P_from_csv(L_net: int, ci_start_date, ci_end_date):
        key = (int(L_net), _norm_date(ci_start_date), _norm_date(ci_end_date))
        if key in P_cache:
            return P_cache[key]
        ln, cs, ce = key
        sub = P_all[(P_all['L_net'] == ln) & (P_all['ci_start_date'] == cs) & (P_all['ci_end_date'] == ce)]
        if sub.empty:
            P_cache[key] = (None, None, None, None)
            return P_cache[key]

        def _pivot(col):
            df = sub.pivot_table(index='date', columns='market', values=col, aggfunc='mean')
            return df.sort_index()
        P_corr = _pivot('P_corr')
        P_kendall = _pivot('P_kendall')
        P_partial = _pivot('P_partial')
        P_tail = _pivot('P_tail')
        P_cache[key] = (P_corr, P_kendall, P_partial, P_tail)
        return P_cache[key]

    def forecast_one_market_h22_harci_red(data_rv_all: pd.DataFrame, P_corr: pd.DataFrame, P_kendall: pd.DataFrame, P_partial: pd.DataFrame, P_tail: pd.DataFrame, market: str, fit_start_date: pd.Timestamp, H: int, oos_start: pd.Timestamp, oos_end: pd.Timestamp, pred_start_base: pd.Timestamp, pred_end_base: pd.Timestamp):
        if oos_start < pred_start_base:
            oos_start = pred_start_base
        if oos_end > pred_end_base:
            oos_end = pred_end_base
        df = data_rv_all[market].to_frame('RV_now')
        df['RV_D'] = df['RV_now'].shift(1)
        df['RV_W'] = df['RV_D'].rolling(5).mean()
        df['RV_M'] = df['RV_D'].rolling(22).mean()
        df['Y'] = make_future_mean(df['RV_now'], H)
        ci = pd.DataFrame({'CI_corr': P_corr[market], 'CI_kendall': P_kendall[market], 'CI_partial': P_partial[market], 'CI_tail': P_tail[market]}).reindex(df.index)
        df = df.join(ci, how='inner').dropna()
        df['CI_corr_D'] = df['CI_corr'].shift(1)
        df['CI_kendall_D'] = df['CI_kendall'].shift(1)
        df['CI_partial_D'] = df['CI_partial'].shift(1)
        df['CI_tail_D'] = df['CI_tail'].shift(1)
        df = df.dropna()
        df = df.loc[fit_start_date:]
        data_test_full = df.loc[pred_start_base:oos_end].copy()
        if len(data_test_full) == 0:
            return (None, np.nan, np.nan, np.nan)
        window_len = len(df.loc[:pred_start_base]) - 1
        if window_len <= 0:
            return (None, np.nan, np.nan, np.nan)
        y_true_full = data_test_full['Y'].astype(float).copy()
        idx_full = y_true_full.index
        preds = []
        for i in tqdm(range(len(idx_full)), desc=f'HAR-CI {market} H{H}', leave=False):
            train = df.iloc[i:i + window_len].copy()
            train['_Y_'] = train['Y'].astype(float).values
            res = smf.ols('_Y_ ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D', train).fit()
            p = res.params
            pred = p.iloc[0] + p.iloc[1] * data_test_full['RV_D'].iloc[i] + p.iloc[2] * data_test_full['RV_W'].iloc[i] + p.iloc[3] * data_test_full['RV_M'].iloc[i] + p.iloc[4] * data_test_full['CI_corr_D'].iloc[i] + p.iloc[5] * data_test_full['CI_kendall_D'].iloc[i] + p.iloc[6] * data_test_full['CI_partial_D'].iloc[i] + p.iloc[7] * data_test_full['CI_tail_D'].iloc[i]
            preds.append(float(pred))
        y_pred_full = pd.Series(preds, index=idx_full)
        mcs_full = pd.concat([y_true_full.rename('RV_True'), y_pred_full.rename('RV_Predict')], axis=1)
        mcs_full['MSE'] = (mcs_full['RV_True'] - mcs_full['RV_Predict']) ** 2
        mcs_full['MAE'] = np.abs(mcs_full['RV_True'] - mcs_full['RV_Predict'])
        mcs_seg = mcs_full.loc[oos_start:oos_end].copy()
        if len(mcs_seg) == 0:
            return (None, np.nan, np.nan, np.nan)
        mse = float(np.nanmean(mcs_seg['MSE'].values))
        mae = float(np.nanmean(mcs_seg['MAE'].values))
        denom = np.sqrt(np.nanmean(mcs_seg['RV_Predict'].values ** 2)) + np.sqrt(np.nanmean(mcs_seg['RV_True'].values ** 2))
        theil = float(np.sqrt(np.nanmean((mcs_seg['RV_True'].values - mcs_seg['RV_Predict'].values) ** 2)) / denom) if denom != 0 else np.nan
        return (mcs_seg, mse, mae, theil)
    L_fit = L_fit_list[0]
    L_net = L_net_list[0]
    error_by_market = {}
    for mkt in markets_to_run:
        oos_start, oos_end = red_range[mkt]
        if fit_end_date not in data_rv.index:
            continue
        fit_end_pos = data_rv.index.get_loc(fit_end_date)
        fit_start_pos = fit_end_pos - (L_fit - 1)
        if fit_start_pos < 0:
            continue
        fit_start_date = data_rv.index[fit_start_pos]
        ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
        if ret_pos_end < 0:
            continue
        net_end_date = df_ret.index[ret_pos_end]
        ci_start_date = net_end_date
        ci_end_date = pred_end_base
        P_corr, P_kendall, P_partial, P_tail = load_P_from_csv(L_net=L_net, ci_start_date=ci_start_date, ci_end_date=ci_end_date)
        if P_corr is None:
            continue
        if mkt not in P_corr.columns:
            continue
        mcs, mse, mae, theil = forecast_one_market_h22_harci_red(data_rv_all=data_rv, P_corr=P_corr, P_kendall=P_kendall, P_partial=P_partial, P_tail=P_tail, market=mkt, fit_start_date=fit_start_date, H=H, oos_start=oos_start, oos_end=oos_end, pred_start_base=pred_start_base, pred_end_base=pred_end_base)
        mcs_path = os.path.join(OUT_DIR, f'MCS_{mkt}.csv')
        if mcs is None:
            continue
        model_outputs[str(mkt).strip().lstrip('.')] = mcs.copy()
        error_by_market[mkt] = [mse, mae, theil]
    ERROR = pd.DataFrame(error_by_market, index=['MSE', 'MAE', 'Theil'])
    err_path = os.path.join(OUT_DIR, 'Error.csv')
    return model_outputs


def run_har_ci_qlike(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    RET_PATH = RETURN_FILE
    RV_PATH = RV_FILE
    P_STORE_PATH = CENTRALITY_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    PLOT = False
    fit_end_date = pd.to_datetime('2018-05-01')
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    H = 22
    FIT_START = 1000
    FIT_STEP = 100
    FIT_MAX = 1
    L_fit_list = [FIT_START + FIT_STEP * k for k in range(FIT_MAX)]
    NET_MIN = 1000
    NET_MAX = 1000
    NET_STEP = 50
    L_net_list = list(range(NET_MIN, NET_MAX + 1, NET_STEP))
    BASE_OUT_ROOT = 'unused_output'
    OUT_DIR = os.path.join(BASE_OUT_ROOT, 'H22-red', 'HAR-CI-new')

    def _norm_date(x):
        return pd.to_datetime(x).tz_localize(None).normalize()

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-19)

    def read_returns(path):
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str[:10], errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
        df = df.set_index('date').dropna(how='all')
        return df

    def read_rv_no_scale(path):
        data = pd.read_csv(path)
        if 'date' not in data.columns:
            raise RuntimeError("RV file must contain a 'date' column.")
        num_cols = [c for c in data.columns if c != 'date']
        data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
        data = data.dropna(how='all')
        data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
        data = data.dropna(subset=['date'])
        data['date'] = data['date'].dt.tz_localize(None)
        data['date'] = data['date'].dt.normalize()
        data = data.set_index('date')
        data = data.dropna(axis=1, how='all')
        return data
    df_ret = read_returns(RET_PATH)
    data_rv = read_rv_no_scale(RV_PATH)
    markets_rv_ret = [c for c in data_rv.columns if c in df_ret.columns]
    red_df = periods.copy()
    need_red_cols = {'filename', 'start_date', 'end_date'}
    if not need_red_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_red_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        mkt = row['market']
        st = row['start_date']
        ed = row['end_date']
        red_range[mkt] = (st, ed)
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in markets_rv_ret]
    missing_in_data = [m for m in table_markets if m not in markets_rv_ret]
    if missing_in_data:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all['ci_start_date'] = P_all['ci_start_date'].apply(_norm_date)
    P_all['ci_end_date'] = P_all['ci_end_date'].apply(_norm_date)
    P_all['L_net'] = pd.to_numeric(P_all['L_net'], errors='coerce').astype('Int64')
    P_all = P_all.dropna(subset=['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net']).copy()
    P_cache = {}

    def load_P_from_csv(L_net: int, ci_start_date, ci_end_date):
        key = (int(L_net), _norm_date(ci_start_date), _norm_date(ci_end_date))
        if key in P_cache:
            return P_cache[key]
        ln, cs, ce = key
        sub = P_all[(P_all['L_net'] == ln) & (P_all['ci_start_date'] == cs) & (P_all['ci_end_date'] == ce)]
        if sub.empty:
            P_cache[key] = (None, None, None, None)
            return P_cache[key]

        def _pivot(col):
            df = sub.pivot_table(index='date', columns='market', values=col, aggfunc='mean')
            return df.sort_index()
        P_corr = _pivot('P_corr')
        P_kendall = _pivot('P_kendall')
        P_partial = _pivot('P_partial')
        P_tail = _pivot('P_tail')
        P_cache[key] = (P_corr, P_kendall, P_partial, P_tail)
        return P_cache[key]

    def forecast_one_market_h22_harci_red(data_rv_all: pd.DataFrame, P_corr: pd.DataFrame, P_kendall: pd.DataFrame, P_partial: pd.DataFrame, P_tail: pd.DataFrame, market: str, fit_start_date: pd.Timestamp, H: int, oos_start: pd.Timestamp, oos_end: pd.Timestamp, pred_start_base: pd.Timestamp, pred_end_base: pd.Timestamp):
        if oos_start < pred_start_base:
            oos_start = pred_start_base
        if oos_end > pred_end_base:
            oos_end = pred_end_base
        df = data_rv_all[market].to_frame('RV_now')
        df['RV_D'] = df['RV_now'].shift(1)
        df['RV_W'] = df['RV_D'].rolling(5).mean()
        df['RV_M'] = df['RV_D'].rolling(22).mean()
        df['Y'] = make_future_mean(df['RV_now'], H)
        ci = pd.DataFrame({'CI_corr': P_corr[market], 'CI_kendall': P_kendall[market], 'CI_partial': P_partial[market], 'CI_tail': P_tail[market]}).reindex(df.index)
        df = df.join(ci, how='inner').dropna()
        df['CI_corr_D'] = df['CI_corr'].shift(1)
        df['CI_kendall_D'] = df['CI_kendall'].shift(1)
        df['CI_partial_D'] = df['CI_partial'].shift(1)
        df['CI_tail_D'] = df['CI_tail'].shift(1)
        df = df.dropna()
        df = df.loc[fit_start_date:]
        data_test_full = df.loc[pred_start_base:oos_end].copy()
        if len(data_test_full) == 0:
            return (None, np.nan, np.nan, np.nan)
        window_len = len(df.loc[:pred_start_base]) - 1
        if window_len <= 0:
            return (None, np.nan, np.nan, np.nan)
        y_true_full = data_test_full['Y'].astype(float).copy()
        idx_full = y_true_full.index
        preds = []
        for i in tqdm(range(len(idx_full)), desc=f'HAR-CI {market} H{H}', leave=False):
            train = df.iloc[i:i + window_len].copy()
            train['_Y_'] = train['Y'].astype(float).values
            res = smf.ols('_Y_ ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D', train).fit()
            p = res.params
            pred = p.iloc[0] + p.iloc[1] * data_test_full['RV_D'].iloc[i] + p.iloc[2] * data_test_full['RV_W'].iloc[i] + p.iloc[3] * data_test_full['RV_M'].iloc[i] + p.iloc[4] * data_test_full['CI_corr_D'].iloc[i] + p.iloc[5] * data_test_full['CI_kendall_D'].iloc[i] + p.iloc[6] * data_test_full['CI_partial_D'].iloc[i] + p.iloc[7] * data_test_full['CI_tail_D'].iloc[i]
            preds.append(float(pred))
        y_pred_full = pd.Series(preds, index=idx_full)
        mcs_full = pd.concat([y_true_full.rename('RV_True'), y_pred_full.rename('RV_Predict')], axis=1)
        mcs_full['MSE'] = (mcs_full['RV_True'] - mcs_full['RV_Predict']) ** 2
        mcs_full['MAE'] = np.abs(mcs_full['RV_True'] - mcs_full['RV_Predict'])
        mcs_seg = mcs_full.loc[oos_start:oos_end].copy()
        if len(mcs_seg) == 0:
            return (None, np.nan, np.nan, np.nan)
        mse = float(np.nanmean(mcs_seg['MSE'].values))
        mae = float(np.nanmean(mcs_seg['MAE'].values))
        denom = np.sqrt(np.nanmean(mcs_seg['RV_Predict'].values ** 2)) + np.sqrt(np.nanmean(mcs_seg['RV_True'].values ** 2))
        theil = float(np.sqrt(np.nanmean((mcs_seg['RV_True'].values - mcs_seg['RV_Predict'].values) ** 2)) / denom) if denom != 0 else np.nan
        return (mcs_seg, mse, mae, theil)
    L_fit = L_fit_list[0]
    L_net = L_net_list[0]
    error_by_market = {}
    for mkt in markets_to_run:
        oos_start, oos_end = red_range[mkt]
        if fit_end_date not in data_rv.index:
            continue
        fit_end_pos = data_rv.index.get_loc(fit_end_date)
        fit_start_pos = fit_end_pos - (L_fit - 1)
        if fit_start_pos < 0:
            continue
        fit_start_date = data_rv.index[fit_start_pos]
        ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
        if ret_pos_end < 0:
            continue
        net_end_date = df_ret.index[ret_pos_end]
        ci_start_date = net_end_date
        ci_end_date = pred_end_base
        P_corr, P_kendall, P_partial, P_tail = load_P_from_csv(L_net=L_net, ci_start_date=ci_start_date, ci_end_date=ci_end_date)
        if P_corr is None:
            continue
        if mkt not in P_corr.columns:
            continue
        mcs, mse, mae, theil = forecast_one_market_h22_harci_red(data_rv_all=data_rv, P_corr=P_corr, P_kendall=P_kendall, P_partial=P_partial, P_tail=P_tail, market=mkt, fit_start_date=fit_start_date, H=H, oos_start=oos_start, oos_end=oos_end, pred_start_base=pred_start_base, pred_end_base=pred_end_base)
        mcs_path = os.path.join(OUT_DIR, f'MCS_{mkt}.csv')
        if mcs is None:
            continue
        model_outputs[str(mkt).strip().lstrip('.')] = mcs.copy()
        error_by_market[mkt] = [mse, mae, theil]
    ERROR = pd.DataFrame(error_by_market, index=['MSE', 'MAE', 'Theil'])
    err_path = os.path.join(OUT_DIR, 'Error.csv')
    return model_outputs


def run_har_ks(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statistics
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    import re
    H = 22
    EPS = 1e-12
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    RV_PATH = RV_FILE
    red_PATH = 'in_memory_bull_periods'
    OUT_DIR = 'unused_output'
    PLOT = False
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date').sort_index()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='any')
    data[data <= 0] = EPS
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    error_records = {}
    skipped = []
    for country in tqdm(markets_to_run, desc='KS markets'):
        data_RV = data.copy()
        data_RV.loc[:, data_RV.columns != f'{country}'] = data_RV.loc[:, data_RV.columns != f'{country}'].shift(1)
        data_RV['RV_now'] = data_RV[f'{country}']
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = data_RV['RV_now'].rolling(window=H, center=False).mean().shift(-(H - 1))
        data_RV = data_RV.drop(columns=[f'{country}'])
        data_RV = data_RV.drop(columns=['RV_now'])
        data_RV = data_RV.loc[SAMPLE_START:]
        data_RV = data_RV.dropna()
        if data_RV.empty:
            skipped.append((country, 'empty after dropna/SAMPLE_START'))
            continue
        if ANCHOR_DATE not in data_RV.index:
            pos = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(data_RV.index):
                skipped.append((country, 'cannot locate anchor in data_RV'))
                continue
            anchor_date = data_RV.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        oos_start, oos_end = red_range[country]
        data_test = data_RV.loc[oos_start:oos_end]
        if data_test.empty:
            skipped.append((country, f'no test sample in red {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test['RV_MA22'].copy()
        result = []
        ols = LinearRegression()
        for t in tqdm(data_test.index, desc=f'KS {country}', leave=False):
            pos_t = data_RV.index.get_loc(t)
            train_end = pos_t - 1
            train_start = train_end - (window_len - 1)
            if train_start < 0 or train_end < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[train_start:train_end + 1].copy()
            data_rolling = data_rolling.dropna()
            if data_rolling.empty:
                result.append(np.nan)
                continue
            X_train = data_rolling.drop(columns=['RV_MA22'])
            y_train = data_rolling['RV_MA22']
            try:
                ols.fit(X_train, y_train)
                X_test = np.array([data_RV.drop(columns=['RV_MA22']).loc[t].values])
                pred = float(ols.predict(X_test)[0])
                result.append(pred if np.isfinite(pred) else np.nan)
            except Exception:
                result.append(np.nan)
        result = pd.Series(result, index=data_test.index, name='RV_Predict').dropna()
        if result.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        SUM_final = pd.concat([data_predict.loc[result.index].rename('RV_True'), result], axis=1).dropna()
        if SUM_final.empty:
            skipped.append((country, 'empty after align true/pred'))
            continue
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2))
        MAE = float(np.mean(np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_mc(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import LinearRegression
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    SAMPLE_START = '2014-07-04'
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='all')
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce')
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce')
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {}
    for _, r in red_df.iterrows():
        red_range[str(r['market'])] = (pd.to_datetime(r['start_date']).normalize(), pd.to_datetime(r['end_date']).normalize())
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-22)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        other_cols = [c for c in data_RV_base.columns if c != country]
        data_RV_base.loc[:, other_cols] = data_RV_base.loc[:, other_cols].shift(1)
        data_RV_base['RV_now'] = data_RV_base[country]
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        target_col = f'RV_MA{H}'
        data_RV_base[target_col] = make_target(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.drop(columns=[country, 'RV_now'])
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[SAMPLE_START:]
        if data_RV.empty:
            skipped.append((country, 'no data after cleaning'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test data in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[target_col].copy()
        anchor = pd.to_datetime('2018-05-02').normalize()
        if anchor not in data_RV.index:
            pos = data_RV.index.searchsorted(anchor)
            if pos <= 0 or pos >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot searchsorted safely'))
                continue
            anchor_date = data_RV.index[pos]
        else:
            anchor_date = anchor
        window_len_fit = len(data_RV.loc[:anchor_date]) - 1
        if window_len_fit <= 30:
            skipped.append((country, f'window_len_fit too small: {window_len_fit}'))
            continue
        other_list = [c for c in data_RV.columns if c not in [target_col, 'RV_D', 'RV_W', 'RV_M']]
        if len(other_list) == 0:
            skipped.append((country, 'no other markets available as predictors'))
            continue
        ols = LinearRegression()
        result = []
        full_index = data_RV.index
        for t in tqdm(data_test.index, desc=f'{country} | H{H} mean', leave=False):
            t_pos = full_index.get_loc(t)
            train_end_pos = t_pos - 1
            train_start_pos = train_end_pos - (window_len_fit - 1)
            if train_start_pos < 0 or train_end_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[train_start_pos:train_end_pos + 1].copy()
            data_rolling = data_rolling.dropna()
            if len(data_rolling) < 50:
                result.append(np.nan)
                continue
            MEAN_pre = []
            for other_country in other_list:
                X_train = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                y_train = data_rolling[target_col]
                if X_train.isna().any().any() or y_train.isna().any():
                    continue
                ols.fit(X_train.values, y_train.values)
                X_test = data_RV.loc[t, ['RV_D', 'RV_W', 'RV_M', other_country]].values.reshape(1, -1)
                r_predict = float(ols.predict(X_test)[0])
                MEAN_pre.append(r_predict)
            if len(MEAN_pre) == 0:
                result.append(np.nan)
            else:
                result.append(float(np.mean(MEAN_pre)))
        y_pred = pd.Series(result, index=data_test.index, name='RV_Predict')
        SUM = pd.concat([data_predict.rename('RV_True'), y_pred], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'all predictions are nan after dropna'))
            continue
        SUM_final = SUM.copy()
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(SUM_final['MSE'].mean())
        MAE = float(SUM_final['MAE'].mean())
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_mdc(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import LinearRegression
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = '2014-07-04'
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='all')
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {}
    for _, r in red_df.iterrows():
        red_range[str(r['market'])] = (r['start_date'], r['end_date'])
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-22)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        other_cols = [c for c in data_RV_base.columns if c != country]
        data_RV_base.loc[:, other_cols] = data_RV_base.loc[:, other_cols].shift(1)
        data_RV_base['RV_now'] = data_RV_base[country]
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_target(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.drop(columns=[country, 'RV_now'])
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[SAMPLE_START:]
        if data_RV.empty:
            skipped.append((country, 'no data after cleaning'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test data in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        y_true = data_test[TARGET_COL].copy()
        if ANCHOR_DATE not in data_RV.index:
            pos = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot searchsorted safely'))
                continue
            anchor_date = data_RV.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len_fit = len(data_RV.loc[:anchor_date]) - 1
        if window_len_fit <= 30:
            skipped.append((country, f'window_len_fit too small: {window_len_fit}'))
            continue
        other_list = [c for c in data_RV.columns if c not in [TARGET_COL, 'RV_D', 'RV_W', 'RV_M']]
        if len(other_list) == 0:
            skipped.append((country, 'no other markets available as predictors'))
            continue
        ols = LinearRegression()
        preds = []
        full_index = data_RV.index
        for t in tqdm(data_test.index, desc=f'{country} | H{H} median', leave=False):
            t_pos = full_index.get_loc(t)
            train_end_pos = t_pos - 1
            train_start_pos = train_end_pos - (window_len_fit - 1)
            if train_start_pos < 0 or train_end_pos < 0:
                preds.append(np.nan)
                continue
            data_rolling = data_RV.iloc[train_start_pos:train_end_pos + 1].copy()
            data_rolling = data_rolling.dropna()
            if len(data_rolling) < 50:
                preds.append(np.nan)
                continue
            one_step_preds = []
            for other_country in other_list:
                X_train = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                y_train = data_rolling[TARGET_COL]
                if X_train.isna().any().any() or y_train.isna().any():
                    continue
                ols.fit(X_train.values, y_train.values)
                X_test = data_RV.loc[t, ['RV_D', 'RV_W', 'RV_M', other_country]].values.reshape(1, -1)
                one_step_preds.append(float(ols.predict(X_test)[0]))
            if len(one_step_preds) == 0:
                preds.append(np.nan)
            else:
                preds.append(float(np.median(one_step_preds)))
        y_pred = pd.Series(preds, index=data_test.index, name='RV_Predict')
        mcs = pd.concat([y_true.rename('RV_True'), y_pred], axis=1).dropna()
        if mcs.empty:
            skipped.append((country, 'all predictions are nan after dropna'))
            continue
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        pred_safe = mcs['RV_Predict'].clip(lower=EPS)
        true_safe = mcs['RV_True'].clip(lower=EPS)
        mcs['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(mcs['MSE'].mean())
        MAE = float(mcs['MAE'].mean())
        Theil = float(np.sqrt(np.mean((mcs['RV_True'] - mcs['RV_Predict']) ** 2)) / (np.sqrt(np.mean(mcs['RV_Predict'] ** 2)) + np.sqrt(np.mean(mcs['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = mcs.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_tmc(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statistics
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna()
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    col_mkt = _pick_col(red_df, ['market', 'Market', 'Model execution message', 'Model execution message', 'Model execution message', 'Model execution message', 'name', 'filename'])
    col_start = _pick_col(red_df, ['start_date', 'start', 'Start', 'start_date', 'Model execution message'])
    col_end = _pick_col(red_df, ['end_date', 'end', 'End', 'end_date', 'Model execution message'])
    if col_mkt is None or col_start is None or col_end is None:
        raise RuntimeError(f'Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else str(x).strip()
    red_df['market'] = red_df[col_mkt].apply(extract_market_code)
    red_df['start'] = pd.to_datetime(red_df[col_start], errors='coerce').dt.normalize()
    red_df['end'] = pd.to_datetime(red_df[col_end], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start', 'end']).copy()
    red_range = {r['market']: (r['start'], r['end']) for _, r in red_df.iterrows()}
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-22)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        data_RV_base.loc[:, data_RV_base.columns != f'{country}'] = data_RV_base.loc[:, data_RV_base.columns != f'{country}'].shift(1)
        data_RV_base['RV_now'] = data_RV_base[f'{country}']
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_future_mean(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.copy()
        data_RV = data_RV.drop(columns=['RV_now'])
        data_RV = data_RV.drop(columns=[f'{country}'])
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[SAMPLE_START:]
        if data_RV.empty:
            skipped.append((country, 'empty after dropna/SAMPLE_START'))
            continue
        if ANCHOR_DATE not in data_RV.index:
            pos_anchor = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos_anchor <= 0 or pos_anchor >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot align'))
                continue
            anchor_date = data_RV.index[pos_anchor]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test rows in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[TARGET_COL].astype(float).copy()
        result = []
        ols = LinearRegression()
        test_index = data_test.index
        for idx in tqdm(test_index, desc=f'{country}-H22', leave=False):
            pos = data_RV.index.get_loc(idx)
            start_pos = pos - window_len
            if start_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[start_pos:pos].copy()
            data_rolling = data_rolling.dropna()
            MEAN_pre = []
            other_cols = data_RV.columns.drop([TARGET_COL, 'RV_D', 'RV_W', 'RV_M'])
            for other_country in other_cols:
                X_train = data_rolling[['RV_D', 'RV_W', 'RV_M', f'{other_country}']]
                y_train = data_rolling[TARGET_COL]
                if len(X_train) < 30:
                    continue
                try:
                    ols.fit(X_train, y_train)
                    X_test = np.array([data_test.loc[idx, ['RV_D', 'RV_W', 'RV_M', f'{other_country}']].values])
                    r_pred = float(ols.predict(X_test)[0])
                    MEAN_pre.append(r_pred)
                except Exception:
                    continue
            if len(MEAN_pre) >= 3:
                trimmed = np.sort(MEAN_pre)[1:-1]
                result.append(float(np.mean(trimmed)) if len(trimmed) > 0 else np.nan)
            elif len(MEAN_pre) > 0:
                result.append(float(np.mean(MEAN_pre)))
            else:
                result.append(np.nan)
        result = pd.Series(result, index=test_index, name='RV_Predict')
        SUM = pd.concat([data_predict.rename('RV_True'), result], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        SUM_final = SUM.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        SUM_final['MAPE'] = np.abs((SUM_final['RV_True'] - SUM_final['RV_Predict']) / SUM_final['RV_True'].clip(lower=EPS))
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(SUM_final['MSE']))
        MAE = float(np.mean(SUM_final['MAE']))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_dmspe_09(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import LinearRegression
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    time_holdout = pd.to_datetime('2016-05-09')
    theta = 0.9
    PLOT = False
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='bfill')
    data = data.set_index('date')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        y = y.shift(-25)
        return y
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for country in markets_to_run:
        try:
            oos_start, oos_end = red_range[country]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            data_name = data.loc[:, data.columns != country]
            data_RV = data.copy()
            data_RV.loc[:, data_RV.columns != country] = data_RV.loc[:, data_RV.columns != country].shift(1)
            data_RV['RV_now'] = data_RV[country]
            data_RV['RV_D'] = data_RV['RV_now'].shift(1)
            data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
            data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
            data_RV[f'Y_H{H}'] = make_target(data_RV['RV_now'], H)
            data_RV = data_RV.drop(columns=['RV_now'])
            data_RV = data_RV.drop(columns=[country])
            data_RV = data_RV.dropna()
            data_RV = data_RV.loc['2014-07-04':]
            data_test_holdout = data_RV.loc[time_holdout:pred_start_base]
            data_test = data_RV.loc[oos_start:oos_end]
            if len(data_test_holdout) == 0 or len(data_test) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            data_predict = data_test[f'Y_H{H}']
            ols = LinearRegression()
            predictions = pd.DataFrame(columns=data_name.columns)
            other_list = data_RV.columns.drop([f'Y_H{H}', 'RV_D', 'RV_W', 'RV_M'])
            for j in tqdm(range(len(data_test_holdout)), desc=f'{country} | H{H} | holdout', leave=False):
                data_rolling = data_RV.loc['2014-07-04':time_holdout + pd.DateOffset(days=j)].iloc[:-1]
                if len(data_rolling) == 0:
                    continue
                for other_country in other_list:
                    X_train_holdout = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                    y_train_holdout = data_rolling[f'Y_H{H}']
                    ols.fit(X_train_holdout, y_train_holdout)
                    X_test_holdout = np.array([data_test_holdout[['RV_D', 'RV_W', 'RV_M', other_country]].iloc[j].values])
                    pred = ols.predict(X_test_holdout)[0]
                    predictions.loc[j, other_country] = pred
            for i in tqdm(range(len(data_test)), desc=f'{country} | H{H} | test', leave=False):
                current_time = data_test.index[i]
                data_rolling = data_RV.loc[time_holdout:current_time].iloc[:-1]
                if len(data_rolling) == 0:
                    continue
                for other_country in other_list:
                    X_train = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                    y_train = data_rolling[f'Y_H{H}']
                    ols.fit(X_train, y_train)
                    X_test = np.array([data_test[['RV_D', 'RV_W', 'RV_M', other_country]].iloc[i].values])
                    pred = ols.predict(X_test)[0]
                    predictions.loc[len(data_test_holdout) + i, other_country] = pred
            result = []
            for m in tqdm(range(len(data_test)), desc=f'{country} | H{H} | dmspe', leave=False):
                current_time = data_test.index[m]
                y_train_sum = data_RV.loc[time_holdout:current_time, f'Y_H{H}'].iloc[:-1]
                y_train_sum.index = range(len(y_train_sum))
                if len(y_train_sum) == 0:
                    result.append(np.nan)
                    continue
                dmspe_errors = []
                for other_country in other_list:
                    pred_hist = predictions.loc[:len(y_train_sum) - 1, other_country].astype(float)
                    errors = (y_train_sum - pred_hist) ** 2
                    time_weights = theta ** np.arange(len(errors) - 1, -1, -1)
                    dmspe_errors.append(np.sum(errors * time_weights) + 1e-12)
                dmspe_errors = np.array(dmspe_errors, dtype=float)
                if not np.isfinite(dmspe_errors).all():
                    result.append(np.nan)
                    continue
                weights = 1.0 / dmspe_errors
                weights = weights / weights.sum()
                row_pred = predictions.loc[len(data_test_holdout) + m, :].astype(float).values
                dmspe_prediction = np.sum(row_pred * weights)
                result.append(dmspe_prediction)
            result = pd.Series(result, index=data_predict.index)
            SUM = pd.concat([data_predict, result], axis=1)
            SUM.columns = ['RV_True', 'RV_Predict']
            SUM = SUM.dropna()
            if len(SUM) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            SUM_final = SUM.copy()
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            EPS_QLIKE = 1e-12
            pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
            SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
            MSE = float(np.mean(SUM_final['MSE']))
            MAE = float(np.mean(SUM_final['MAE']))
            denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
            model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(country)
        except Exception as e:
            skip_no_sample_or_fail.append((country, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_dmspe_1(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import LinearRegression
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    time_holdout = pd.to_datetime('2016-05-09')
    theta = 1
    PLOT = False
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='bfill')
    data = data.set_index('date')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        y = y.shift(-(H + 1))
        return y
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for country in markets_to_run:
        try:
            oos_start, oos_end = red_range[country]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            data_name = data.loc[:, data.columns != country]
            data_RV = data.copy()
            data_RV.loc[:, data_RV.columns != country] = data_RV.loc[:, data_RV.columns != country].shift(1)
            data_RV['RV_now'] = data_RV[country]
            data_RV['RV_D'] = data_RV['RV_now'].shift(1)
            data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
            data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
            data_RV[f'Y_H{H}'] = make_target(data_RV['RV_now'], H)
            data_RV = data_RV.drop(columns=['RV_now'])
            data_RV = data_RV.drop(columns=[country])
            data_RV = data_RV.dropna()
            data_RV = data_RV.loc['2014-07-04':]
            data_test_holdout = data_RV.loc[time_holdout:pred_start_base]
            data_test = data_RV.loc[oos_start:oos_end]
            if len(data_test_holdout) == 0 or len(data_test) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            data_predict = data_test[f'Y_H{H}']
            ols = LinearRegression()
            predictions = pd.DataFrame(columns=data_name.columns)
            other_list = data_RV.columns.drop([f'Y_H{H}', 'RV_D', 'RV_W', 'RV_M'])
            for j in tqdm(range(len(data_test_holdout)), desc=f'{country} | H{H} | holdout', leave=False):
                data_rolling = data_RV.loc['2014-07-04':time_holdout + pd.DateOffset(days=j)].iloc[:-1]
                if len(data_rolling) == 0:
                    continue
                for other_country in other_list:
                    X_train_holdout = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                    y_train_holdout = data_rolling[f'Y_H{H}']
                    ols.fit(X_train_holdout, y_train_holdout)
                    X_test_holdout = np.array([data_test_holdout[['RV_D', 'RV_W', 'RV_M', other_country]].iloc[j].values])
                    pred = ols.predict(X_test_holdout)[0]
                    predictions.loc[j, other_country] = pred
            for i in tqdm(range(len(data_test)), desc=f'{country} | H{H} | test', leave=False):
                current_time = data_test.index[i]
                data_rolling = data_RV.loc[time_holdout:current_time].iloc[:-1]
                if len(data_rolling) == 0:
                    continue
                for other_country in other_list:
                    X_train = data_rolling[['RV_D', 'RV_W', 'RV_M', other_country]]
                    y_train = data_rolling[f'Y_H{H}']
                    ols.fit(X_train, y_train)
                    X_test = np.array([data_test[['RV_D', 'RV_W', 'RV_M', other_country]].iloc[i].values])
                    pred = ols.predict(X_test)[0]
                    predictions.loc[len(data_test_holdout) + i, other_country] = pred
            result = []
            for m in tqdm(range(len(data_test)), desc=f'{country} | H{H} | dmspe', leave=False):
                current_time = data_test.index[m]
                y_train_sum = data_RV.loc[time_holdout:current_time, f'Y_H{H}'].iloc[:-1]
                y_train_sum.index = range(len(y_train_sum))
                if len(y_train_sum) == 0:
                    result.append(np.nan)
                    continue
                dmspe_errors = []
                for other_country in other_list:
                    pred_hist = predictions.loc[:len(y_train_sum) - 1, other_country].astype(float)
                    errors = (y_train_sum - pred_hist) ** 2
                    time_weights = theta ** np.arange(len(errors) - 1, -1, -1)
                    dmspe_errors.append(np.sum(errors * time_weights) + 1e-12)
                dmspe_errors = np.array(dmspe_errors, dtype=float)
                if not np.isfinite(dmspe_errors).all():
                    result.append(np.nan)
                    continue
                weights = 1.0 / dmspe_errors
                weights = weights / weights.sum()
                row_pred = predictions.loc[len(data_test_holdout) + m, :].astype(float).values
                dmspe_prediction = np.sum(row_pred * weights)
                result.append(dmspe_prediction)
            result = pd.Series(result, index=data_predict.index)
            SUM = pd.concat([data_predict, result], axis=1)
            SUM.columns = ['RV_True', 'RV_Predict']
            SUM = SUM.dropna()
            if len(SUM) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            SUM_final = SUM.copy()
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            EPS_QLIKE = 1e-12
            pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
            SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
            MSE = float(np.mean(SUM_final['MSE']))
            MAE = float(np.mean(SUM_final['MAE']))
            denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
            model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(country)
        except Exception as e:
            skip_no_sample_or_fail.append((country, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_pca1(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statsmodels.formula.api as smf
    from sklearn.decomposition import PCA
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = '2014-07-04'
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {}
    for _, r in red_df.iterrows():
        red_range[str(r['market'])] = (r['start_date'], r['end_date'])
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-H)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        other_cols = [c for c in data_RV_base.columns if c != country]
        data_RV_base.loc[:, other_cols] = data_RV_base.loc[:, other_cols].shift(1)
        data_RV_base['RV_now'] = data_RV_base[country]
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_target(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.drop(columns=[country, 'RV_now'])
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[SAMPLE_START:]
        if data_RV.empty:
            skipped.append((country, 'no data after cleaning'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test data in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        y_true = data_test[TARGET_COL].copy()
        if ANCHOR_DATE not in data_RV.index:
            pos = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot searchsorted safely'))
                continue
            anchor_date = data_RV.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        pca_input_cols = [c for c in data_RV.columns if c not in [TARGET_COL, 'RV_D', 'RV_W', 'RV_M']]
        if len(pca_input_cols) == 0:
            skipped.append((country, 'no other markets available for PCA'))
            continue
        preds = []
        full_index = data_RV.index
        for t in tqdm(data_test.index, desc=f'{country} | H{H} PCA1', leave=False):
            t_pos = full_index.get_loc(t)
            train_end_pos = t_pos - 1
            train_start_pos = train_end_pos - (window_len - 1)
            if train_start_pos < 0 or train_end_pos < 0:
                preds.append(np.nan)
                continue
            data_rolling = data_RV.iloc[train_start_pos:train_end_pos + 1].copy()
            data_rolling = data_rolling.dropna()
            if len(data_rolling) < 50:
                preds.append(np.nan)
                continue
            pca = PCA(n_components=1)
            X_pca = data_rolling[pca_input_cols]
            try:
                data_rolling['PCA_1'] = pca.fit_transform(X_pca)
            except Exception:
                preds.append(np.nan)
                continue
            try:
                mod = smf.ols(f'{TARGET_COL} ~ RV_D + RV_W + RV_M + PCA_1', data_rolling)
                res = mod.fit()
                p = res.params
            except Exception:
                preds.append(np.nan)
                continue
            pred = p.iloc[0] + p.iloc[1] * data_test.loc[t, 'RV_D'] + p.iloc[2] * data_test.loc[t, 'RV_W'] + p.iloc[3] * data_test.loc[t, 'RV_M'] + p.iloc[4] * data_rolling['PCA_1'].iloc[-1]
            preds.append(float(pred))
        y_pred = pd.Series(preds, index=data_test.index, name='RV_Predict')
        mcs = pd.concat([y_true.rename('RV_True'), y_pred], axis=1).dropna()
        if mcs.empty:
            skipped.append((country, 'all predictions are nan after dropna'))
            continue
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        pred_safe = mcs['RV_Predict'].clip(lower=EPS)
        true_safe = mcs['RV_True'].clip(lower=EPS)
        mcs['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(mcs['MSE'].mean())
        MAE = float(mcs['MAE'].mean())
        Theil = float(np.sqrt(np.mean((mcs['RV_True'] - mcs['RV_Predict']) ** 2)) / (np.sqrt(np.mean(mcs['RV_Predict'] ** 2)) + np.sqrt(np.mean(mcs['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = mcs.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_pca_bic(periods):
    model_outputs = {}
    import math
    import scipy as sc
    import time
    import os
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statsmodels.formula.api as smf
    from sklearn.decomposition import PCA
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = '2014-07-04'
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(how='any')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='any')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {}
    for _, r in red_df.iterrows():
        red_range[str(r['market'])] = (r['start_date'], r['end_date'])
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-24)
    error_records = {}
    best_pca_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        other_cols = [c for c in data_RV_base.columns if c != country]
        data_RV_base.loc[:, other_cols] = data_RV_base.loc[:, other_cols].shift(1)
        data_RV_base['RV_now'] = data_RV_base[country]
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_target(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.drop(columns=[country, 'RV_now'])
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[SAMPLE_START:]
        if data_RV.empty:
            skipped.append((country, 'no data after cleaning'))
            continue
        data_test = data_RV.loc[oos_start:oos_end]
        if len(data_test) == 0:
            skipped.append((country, f'no test data in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[TARGET_COL]
        if ANCHOR_DATE not in data_RV.index:
            pos = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot searchsorted safely'))
                continue
            anchor_date = data_RV.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        pca_input_cols = [c for c in data_RV.columns if c not in [TARGET_COL, 'RV_D', 'RV_W', 'RV_M']]
        if len(pca_input_cols) == 0:
            skipped.append((country, 'no other markets available for PCA'))
            continue
        bic_values = []
        n_pca_candidates = list(range(1, 17))
        n_samples = len(data_RV.iloc[0:0 + window_len])
        if n_samples <= 0:
            skipped.append((country, 'n_samples<=0 for BIC'))
            continue
        for n_pca in n_pca_candidates:
            preds_tmp = []
            k = 4 + n_pca
            for i in tqdm(range(len(data_test)), desc=f'{country}-BIC(n_pca={n_pca})', leave=False):
                data_rolling = data_RV.iloc[i:i + window_len].copy()
                data_rolling = data_rolling.dropna()
                if len(data_rolling) > 21:
                    data_rolling = data_rolling.iloc[:-21]
                else:
                    preds_tmp.append(np.nan)
                    continue
                if len(data_rolling) <= max(n_pca, 5):
                    preds_tmp.append(np.nan)
                    continue
                try:
                    pca = PCA(n_components=n_pca)
                    X_other = data_rolling[pca_input_cols]
                    pca.fit(X_other)
                    principal_components = pca.transform(X_other)
                except Exception:
                    preds_tmp.append(np.nan)
                    continue
                columns_names = [f'pca_{j}' for j in range(1, n_pca + 1)]
                principal_df = pd.DataFrame(principal_components, columns=columns_names, index=data_rolling.index)
                variables_df = pd.concat([data_rolling[[TARGET_COL, 'RV_D', 'RV_W', 'RV_M']], principal_df], axis=1).dropna()
                if len(variables_df) < 30:
                    preds_tmp.append(np.nan)
                    continue
                pca_terms = ' + '.join(columns_names)
                formula = f'{TARGET_COL} ~ RV_D + RV_W + RV_M' + (f' + {pca_terms}' if pca_terms else '')
                try:
                    res = smf.ols(formula, data=variables_df).fit()
                    params_new = res.params
                except Exception:
                    preds_tmp.append(np.nan)
                    continue
                try:
                    pred = params_new.iloc[0] + params_new.iloc[1] * data_test['RV_D'].iloc[i] + params_new.iloc[2] * data_test['RV_W'].iloc[i] + params_new.iloc[3] * data_test['RV_M'].iloc[i] + sum((params_new.iloc[4 + j] * principal_df.iloc[-1, j] for j in range(n_pca)))
                    preds_tmp.append(float(pred))
                except Exception:
                    preds_tmp.append(np.nan)
            preds_tmp = pd.Series(preds_tmp, index=data_test.index, name='RV_Predict')
            SUM_tmp = pd.concat([data_predict.rename('RV_True'), preds_tmp], axis=1)
            SUM_tmp = SUM_tmp.dropna()
            if len(SUM_tmp) == 0:
                bic_values.append(np.inf)
                continue
            mse_tmp = float(np.mean((SUM_tmp['RV_True'] - SUM_tmp['RV_Predict']) ** 2))
            if not np.isfinite(mse_tmp) or mse_tmp <= 0:
                bic_values.append(np.inf)
                continue
            log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(mse_tmp) + 1)
            bic = -2 * log_likelihood + k * np.log(n_samples)
            bic_values.append(float(bic))
        if np.all(~np.isfinite(bic_values)) or np.min(bic_values) == np.inf:
            skipped.append((country, 'all BIC invalid'))
            continue
        best_idx = int(np.nanargmin(bic_values))
        best_pca_number = int(n_pca_candidates[best_idx])
        best_pca_records[country] = best_pca_number
        preds_final = []
        for i in tqdm(range(len(data_test)), desc=f'{country}-FINAL(best={best_pca_number})', leave=False):
            data_rolling = data_RV.iloc[i:i + window_len].copy()
            data_rolling = data_rolling.dropna()
            if len(data_rolling) <= max(best_pca_number, 5):
                preds_final.append(np.nan)
                continue
            try:
                pca = PCA(n_components=best_pca_number)
                X_other = data_rolling[pca_input_cols]
                pca.fit(X_other)
                principal_components = pca.transform(X_other)
            except Exception:
                preds_final.append(np.nan)
                continue
            columns_names = [f'pca_{j}' for j in range(1, best_pca_number + 1)]
            principal_df = pd.DataFrame(principal_components, columns=columns_names, index=data_rolling.index)
            variables_df = pd.concat([data_rolling[[TARGET_COL, 'RV_D', 'RV_W', 'RV_M']], principal_df], axis=1).dropna()
            if len(variables_df) < 30:
                preds_final.append(np.nan)
                continue
            pca_terms = ' + '.join(columns_names)
            formula = f'{TARGET_COL} ~ RV_D + RV_W + RV_M' + (f' + {pca_terms}' if pca_terms else '')
            try:
                res = smf.ols(formula, data=variables_df).fit()
                params_new = res.params
            except Exception:
                preds_final.append(np.nan)
                continue
            try:
                pred = params_new.iloc[0] + params_new.iloc[1] * data_test['RV_D'].iloc[i] + params_new.iloc[2] * data_test['RV_W'].iloc[i] + params_new.iloc[3] * data_test['RV_M'].iloc[i] + sum((params_new.iloc[4 + j] * principal_df.iloc[-1, j] for j in range(best_pca_number)))
                preds_final.append(float(pred))
            except Exception:
                preds_final.append(np.nan)
        y_pred = pd.Series(preds_final, index=data_test.index, name='RV_Predict')
        mcs = pd.concat([data_predict.rename('RV_True'), y_pred], axis=1).dropna()
        if mcs.empty:
            skipped.append((country, 'all predictions are nan after dropna'))
            continue
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        pred_safe = mcs['RV_Predict'].clip(lower=EPS)
        true_safe = mcs['RV_True'].clip(lower=EPS)
        mcs['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(mcs['MSE']))
        MAE = float(np.mean(mcs['MAE']))
        Theil = float(np.sqrt(np.mean((mcs['RV_True'] - mcs['RV_Predict']) ** 2)) / (np.sqrt(np.mean(mcs['RV_Predict'] ** 2)) + np.sqrt(np.mean(mcs['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = mcs.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
        BEST = pd.DataFrame.from_dict(best_pca_records, orient='index', columns=['best_pca_number'])
        BEST.index.name = 'market'
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_pls(periods):
    model_outputs = {}
    import os
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LinearRegression
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='any')
    data = data.dropna(axis=1, how='all')
    red_df = periods.copy()

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    col_mkt = _pick_col(red_df, ['market', 'Market', 'Model execution message', 'Model execution message', 'Model execution message', 'Model execution message', 'name', 'filename'])
    col_start = _pick_col(red_df, ['start_date', 'start', 'Start', 'start_date', 'Model execution message'])
    col_end = _pick_col(red_df, ['end_date', 'end', 'End', 'end_date', 'Model execution message'])
    if col_mkt is None or col_start is None or col_end is None:
        raise RuntimeError(f'Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else str(x).strip()
    red_df['market'] = red_df[col_mkt].apply(extract_market_code)
    red_df['start'] = pd.to_datetime(red_df[col_start], errors='coerce').dt.normalize()
    red_df['end'] = pd.to_datetime(red_df[col_end], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start', 'end']).copy()
    red_range = {r['market']: (r['start'], r['end']) for _, r in red_df.iterrows()}
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-22)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = data.copy()
        other_cols = [c for c in data_RV_base.columns if c != country]
        data_RV_base.loc[:, other_cols] = data_RV_base.loc[:, other_cols].shift(1)
        data_RV_base['RV_now'] = data_RV_base[country]
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_future_mean(data_RV_base['RV_now'], H)
        df = data_RV_base.drop(columns=[country, 'RV_now'])
        df = df.dropna()
        df = df.loc[SAMPLE_START:]
        if df.empty:
            skipped.append((country, 'no data after dropna/sample_start'))
            continue
        df_test = df.loc[oos_start:oos_end].copy()
        if len(df_test) == 0:
            skipped.append((country, f'no test rows in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        if ANCHOR_DATE not in df.index:
            pos_anchor = df.index.searchsorted(ANCHOR_DATE)
            if pos_anchor <= 0 or pos_anchor >= len(df.index):
                skipped.append((country, 'anchor date not in index and cannot align'))
                continue
            anchor_date = df.index[pos_anchor]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(df.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        other_country_cols = [c for c in df.columns if c not in [TARGET_COL, 'RV_D', 'RV_W', 'RV_M']]
        if len(other_country_cols) == 0:
            skipped.append((country, 'no other markets after shifting to build FPLS'))
            continue
        df_sens = df.loc[:anchor_date].copy()
        if len(df_sens) < 50:
            skipped.append((country, 'too few rows before anchor to estimate sensitivity'))
            continue
        X_sens = df_sens[TARGET_COL].values.reshape(-1, 1)
        ols = LinearRegression()
        Sensitivity = []
        for oc in other_country_cols:
            y_sens = df_sens[oc].values
            try:
                ols.fit(X_sens, y_sens)
                Sensitivity.append(float(ols.coef_[0]))
            except Exception:
                Sensitivity.append(np.nan)
        Sensitivity = np.array(Sensitivity, dtype=float)
        if np.all(~np.isfinite(Sensitivity)):
            skipped.append((country, 'all sensitivity coefficients are nan'))
            continue
        df = df.copy()
        df['FPLS'] = np.nan
        X_cs = Sensitivity.reshape(-1, 1)
        ols_cs = LinearRegression()
        for t in tqdm(range(len(df)), desc=f'{country}-FPLS', leave=False):
            y_t = df.loc[df.index[t], other_country_cols].values.astype(float)
            mask = np.isfinite(y_t) & np.isfinite(Sensitivity)
            if mask.sum() < 5:
                continue
            try:
                ols_cs.fit(X_cs[mask], y_t[mask])
                df.iat[t, df.columns.get_loc('FPLS')] = float(ols_cs.coef_[0])
            except Exception:
                continue
        df = df.dropna(subset=['FPLS'])
        if df.empty:
            skipped.append((country, 'FPLS all nan after cross-sectional step'))
            continue
        df_test = df.loc[oos_start:oos_end].copy()
        if len(df_test) == 0:
            skipped.append((country, 'no test rows after FPLS dropna'))
            continue
        y_true = df_test[TARGET_COL].astype(float).copy()
        preds = []
        for idx in tqdm(df_test.index, desc=f'{country}-ROLL', leave=False):
            pos = df.index.get_loc(idx)
            start_pos = pos - window_len
            if start_pos < 0:
                preds.append(np.nan)
                continue
            train = df.iloc[start_pos:pos].copy()
            if len(train) < 30:
                preds.append(np.nan)
                continue
            try:
                res = smf.ols(f'{TARGET_COL} ~ RV_D + RV_W + RV_M + FPLS', train).fit()
                p = res.params
                pred = p.iloc[0] + p.iloc[1] * df_test.loc[idx, 'RV_D'] + p.iloc[2] * df_test.loc[idx, 'RV_W'] + p.iloc[3] * df_test.loc[idx, 'RV_M'] + p.iloc[4] * df_test.loc[idx, 'FPLS']
                preds.append(float(pred))
            except Exception:
                preds.append(np.nan)
        y_pred = pd.Series(preds, index=df_test.index, name='RV_Predict')
        mcs = pd.concat([y_true.rename('RV_True'), y_pred], axis=1).dropna()
        if mcs.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        pred_safe = mcs['RV_Predict'].clip(lower=EPS)
        true_safe = mcs['RV_True'].clip(lower=EPS)
        mcs['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(mcs['MSE']))
        MAE = float(np.mean(mcs['MAE']))
        Theil = float(np.sqrt(np.mean((mcs['RV_True'] - mcs['RV_Predict']) ** 2)) / (np.sqrt(np.mean(mcs['RV_Predict'] ** 2)) + np.sqrt(np.mean(mcs['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = mcs.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_ridge(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import RidgeCV
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    PLOT = False
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    data = data.dropna()
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna()
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for country in markets_to_run:
        try:
            oos_start, oos_end = red_range[country]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            data_RV_base = data.copy()
            data_RV_base.loc[:, data_RV_base.columns != f'{country}'] = data_RV_base.loc[:, data_RV_base.columns != f'{country}'].shift(1)
            data_RV_base['RV_now'] = data_RV_base[f'{country}']
            data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
            data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
            data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
            target_col = f'RV_MA{H}'
            data_RV = data_RV_base.copy()
            data_RV[target_col] = data_RV['RV_now'].rolling(window=H, center=False).mean().shift(-(H - 1))
            data_RV = data_RV.drop(columns=[f'{country}', 'RV_now'])
            data_RV = data_RV.dropna()
            data_RV = data_RV.loc['2014-07-04':]
            data_test = data_RV.loc[oos_start:oos_end]
            if len(data_test) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            test_start_date = data_test.index[0]
            if pred_start_base not in data_RV.index:
                pos0 = data_RV.index.searchsorted(pred_start_base)
                if pos0 <= 0 or pos0 >= len(data_RV.index):
                    skip_no_sample_or_fail.append((country, 'Model execution message'))
                    continue
                pred_anchor_date = data_RV.index[pos0]
            else:
                pred_anchor_date = pred_start_base
            window_len = len(data_RV.loc[:pred_anchor_date]) - 1
            if window_len <= 10:
                skip_no_sample_or_fail.append((country, f'Model execution message{window_len}'))
                continue
            X_all = data_RV.drop(columns=[target_col])
            y_all = data_RV[target_col]
            all_index = data_RV.index
            test_index = data_test.index
            start_pos = all_index.get_loc(test_start_date)
            result = []
            for s in tqdm(range(len(test_index)), desc=f'{country}-H{H}', leave=False):
                test_pos = start_pos + s
                train_end_pos = test_pos - 1
                train_start_pos = train_end_pos - (window_len - 1)
                if train_start_pos < 0:
                    result.append(np.nan)
                    continue
                data_rolling = data_RV.iloc[train_start_pos:train_end_pos + 1].copy()
                if len(data_rolling) <= H - 1:
                    result.append(np.nan)
                    continue
                data_rolling = data_rolling.iloc[:-(H - 1)]
                data_rolling = data_rolling.dropna()
                if len(data_rolling) < 30:
                    result.append(np.nan)
                    continue
                X_train = data_rolling.drop(columns=[target_col])
                y_train = data_rolling[target_col]
                ridge_cv = RidgeCV(alphas=np.logspace(-4, 1, 10), cv=3)
                ridge_cv.fit(X_train, y_train)
                X_test_row = X_all.iloc[test_pos].values.reshape(1, -1)
                r_predict = ridge_cv.predict(X_test_row)[0]
                result.append(float(r_predict))
            result = pd.Series(result, index=test_index)
            SUM = pd.concat([y_all.loc[test_index], result], axis=1)
            SUM.columns = ['RV_True', 'RV_Predict']
            SUM_final = SUM.reset_index(drop=True)
            SUM_final.index = pd.to_datetime(test_index)
            eps = 1e-12
            SUM_final['RV_Predict_clip'] = np.maximum(SUM_final['RV_Predict'], eps)
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            SUM_final['Qlike'] = SUM_final['RV_True'] / SUM_final['RV_Predict_clip'] + np.log(SUM_final['RV_Predict_clip'])
            MSE = float(np.nanmean(SUM_final['MSE']))
            MAE = float(np.nanmean(SUM_final['MAE']))
            denom = np.sqrt(np.nanmean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.nanmean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(np.nanmean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
            model_outputs[str(country).strip().lstrip('.')] = SUM_final.drop(columns=['RV_Predict_clip']).copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(country)
        except Exception as e:
            skip_no_sample_or_fail.append((country, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_lasso(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import LassoCV
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    PLOT = False
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    data = data.dropna()
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna()
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for country in markets_to_run:
        try:
            oos_start, oos_end = red_range[country]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            data_RV_base = data.copy()
            data_RV_base.loc[:, data_RV_base.columns != country] = data_RV_base.loc[:, data_RV_base.columns != country].shift(1)
            data_RV_base['RV_now'] = data_RV_base[country]
            data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
            data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
            data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
            data_RV_base = data_RV_base.drop(columns=[country])
            data_RV_base = data_RV_base.drop(columns=['RV_now'])
            data_RV_base = data_RV_base.dropna()
            data_RV_base = data_RV_base.loc['2014-07-04':]
            target_col = f'RV_MA{H}'
            data_RV = data_RV_base.copy()
            rv_now = data[country].rename('RV_now')
            data_RV = pd.concat([data_RV, rv_now], axis=1)
            data_RV[target_col] = data_RV['RV_now'].rolling(window=H, center=False).mean().shift(-(H - 1))
            data_RV = data_RV.drop(columns=['RV_now'])
            data_RV = data_RV.dropna()
            data_RV = data_RV.loc['2014-07-04':]
            data_test = data_RV.loc[oos_start:oos_end]
            if len(data_test) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            y_true = data_test[target_col]
            window_len = len(data_RV.loc[:pred_start_base]) - 1
            if window_len <= 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            result = []
            for i in tqdm(range(len(data_test)), desc=f'{country}-H{H}', leave=False):
                data_rolling = data_RV.iloc[i:i + window_len].dropna()
                if len(data_rolling) == 0:
                    result.append(np.nan)
                    continue
                X_train = data_rolling.drop(columns=[target_col])
                y_train = data_rolling[target_col]
                lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5)
                lasso_cv.fit(X_train, y_train)
                X_test = np.array([data_test.drop(columns=[target_col]).iloc[i].values])
                r_predict = lasso_cv.predict(X_test)[0]
                result.append(float(r_predict))
            y_pred = pd.Series(result, index=data_test.index)
            SUM = pd.concat([y_true.rename('RV_True'), y_pred.rename('RV_Predict')], axis=1).dropna()
            if SUM.empty:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            SUM_final = SUM.copy()
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            EPS_QLIKE = 1e-12
            pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
            SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
            MSE = float(np.mean(SUM_final['MSE']))
            MAE = float(np.mean(SUM_final['MAE']))
            denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
            model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(country)
        except Exception as e:
            skip_no_sample_or_fail.append((country, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_enet(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.linear_model import ElasticNetCV
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    OUT_DIR = 'unused_output'
    PLOT = False
    data = pd.read_csv(RV_PATH)
    data = data.dropna()
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna()
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')

    def make_target(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now.shift(-1)
        return rv_now.rolling(window=H).mean().shift(-H)
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for country in markets_to_run:
        try:
            oos_start, oos_end = red_range[country]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            df = data.copy()
            other_cols = [c for c in df.columns if c != country]
            df.loc[:, other_cols] = df.loc[:, other_cols].shift(1)
            df['RV_now'] = df[country]
            df['RV_D'] = df['RV_now'].shift(1)
            df['RV_W'] = df['RV_D'].rolling(5).mean()
            df['RV_M'] = df['RV_D'].rolling(22).mean()
            df = df.drop(columns=[country])
            feature_base = df.drop(columns=['RV_now'])
            feature_base = feature_base.loc['2014-07-04':].dropna()
            target_col = f'RV_MA{H}'
            df_model = feature_base.copy()
            rv_now_aligned = df.loc[feature_base.index, 'RV_now']
            df_model[target_col] = make_target(rv_now_aligned, H)
            df_model = df_model.dropna()
            data_test = df_model.loc[oos_start:oos_end]
            if len(data_test) == 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            y_true = data_test[target_col]
            window_len = len(df_model.loc[:pred_start_base]) - 1
            if window_len <= 0:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            feature_cols = [c for c in df_model.columns if c != target_col]
            preds = []
            test_index = data_test.index
            for i in tqdm(range(len(test_index)), desc=f'{country}-H{H}', leave=False):
                train_slice = df_model.iloc[i:i + window_len].dropna()
                if len(train_slice) < 50:
                    preds.append(np.nan)
                    continue
                X_train = train_slice[feature_cols].values
                y_train = train_slice[target_col].values
                elastic_net_cv = ElasticNetCV(alphas=None, l1_ratio=np.linspace(0.1, 1, 10), cv=5)
                elastic_net_cv.fit(X_train, y_train)
                X_test = data_test.iloc[i][feature_cols].values.reshape(1, -1)
                preds.append(float(elastic_net_cv.predict(X_test)[0]))
            y_pred = pd.Series(preds, index=test_index)
            SUM_final = pd.DataFrame({'RV_True': y_true, 'RV_Predict': y_pred}).dropna()
            if SUM_final.empty:
                skip_no_sample_or_fail.append((country, 'Model execution message'))
                continue
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            EPS_QLIKE = 1e-12
            pred_safe = np.maximum(SUM_final['RV_Predict'].to_numpy(), EPS_QLIKE)
            SUM_final['Qlike'] = SUM_final['RV_True'] / pred_safe + np.log(pred_safe)
            MSE = float(SUM_final['MSE'].mean())
            MAE = float(SUM_final['MAE'].mean())
            denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(MSE) / denom) if denom != 0 else np.nan
            model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(country)
        except Exception as e:
            skip_no_sample_or_fail.append((country, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_spillover_time(periods):
    model_outputs = {}
    import pandas as pd
    import numpy as np
    import math
    import scipy as sc
    import time
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import os
    from scipy.stats import norm
    import statsmodels.formula.api as smf
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    RV_PATH = RV_FILE
    CI_TIME_PATH = TCI_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='bfill')
    data = data.set_index('date')
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='all')
    data_CI_time = pd.read_csv(CI_TIME_PATH)
    if 'Unnamed: 0' not in data_CI_time.columns:
        raise RuntimeError(f'Model execution message{CI_TIME_PATH}')
    if 'TCI.Total' not in data_CI_time.columns:
        raise RuntimeError(f'Model execution message{CI_TIME_PATH}')
    data_CI_time['date'] = pd.to_datetime(data_CI_time['Unnamed: 0'], format='mixed', errors='coerce')
    data_CI_time = data_CI_time.dropna(subset=['date'])
    data_CI_time['date'] = data_CI_time['date'].dt.tz_localize(None).dt.normalize()
    data_CI_time = data_CI_time.set_index('date')
    red_df = periods.copy()

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    col_mkt = _pick_col(red_df, ['market', 'Market', 'Model execution message', 'Model execution message', 'Model execution message', 'Model execution message', 'name', 'filename'])
    col_start = _pick_col(red_df, ['start_date', 'start', 'Start', 'start_date', 'Model execution message'])
    col_end = _pick_col(red_df, ['end_date', 'end', 'End', 'end_date', 'Model execution message'])
    if col_mkt is None or col_start is None or col_end is None:
        raise RuntimeError(f'Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else str(x).strip()
    red_df['market'] = red_df[col_mkt].apply(extract_market_code)
    red_df['start'] = pd.to_datetime(red_df[col_start], errors='coerce').dt.normalize()
    red_df['end'] = pd.to_datetime(red_df[col_end], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start', 'end']).copy()
    red_range = {r['market']: (r['start'], r['end']) for _, r in red_df.iterrows()}
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-H)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = pd.concat([data[country], data_CI_time['TCI.Total']], axis=1)
        data_RV_base = data_RV_base.dropna()
        data_RV_base = data_RV_base.loc[SAMPLE_START:]
        data_RV_base.columns = ['RV_now', 'CI_time']
        if data_RV_base.empty:
            skipped.append((country, 'empty after concat/dropna/SAMPLE_START'))
            continue
        decay_factor = np.exp(-np.arange(len(data_RV_base)) / 7 * math.pi)
        CI_time_ED = np.tril(np.outer(data_RV_base['CI_time'].values, decay_factor))
        data_RV_base['CI_time_ED'] = CI_time_ED.sum(axis=1)
        data_RV_base['CI_time_D'] = data_RV_base['CI_time_ED'].shift(1)
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_future_mean(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.dropna().copy()
        if data_RV.empty:
            skipped.append((country, 'empty after feature/target dropna'))
            continue
        if ANCHOR_DATE not in data_RV.index:
            pos_anchor = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos_anchor <= 0 or pos_anchor >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot align'))
                continue
            anchor_date = data_RV.index[pos_anchor]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test rows in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[TARGET_COL].astype(float).copy()
        result = []
        for idx in tqdm(data_test.index, desc=f'{country}-H22', leave=False):
            pos = data_RV.index.get_loc(idx)
            start_pos = pos - window_len
            if start_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[start_pos:pos].copy()
            if len(data_rolling) < 30:
                result.append(np.nan)
                continue
            mod = smf.ols(f'{TARGET_COL} ~ RV_D + RV_W + RV_M + CI_time_D', data_rolling)
            try:
                res = mod.fit()
                p = res.params
                r_predict = p.iloc[0] + p.iloc[1] * data_test.loc[idx, 'RV_D'] + p.iloc[2] * data_test.loc[idx, 'RV_W'] + p.iloc[3] * data_test.loc[idx, 'RV_M'] + p.iloc[4] * data_test.loc[idx, 'CI_time_D']
                result.append(float(r_predict))
            except Exception:
                result.append(np.nan)
        result = pd.Series(result, index=data_test.index, name='RV_Predict')
        SUM = pd.concat([data_predict.rename('RV_True'), result], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        SUM_final = SUM.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(SUM_final['MSE']))
        MAE = float(np.mean(SUM_final['MAE']))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_spillover_time_qlike(periods):
    model_outputs = {}
    import pandas as pd
    import numpy as np
    import math
    import scipy as sc
    import time
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import os
    from scipy.stats import norm
    import statsmodels.formula.api as smf
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    RV_PATH = RV_FILE
    CI_TIME_PATH = TCI_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='bfill')
    data = data.set_index('date')
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='all')
    data_CI_time = pd.read_csv(CI_TIME_PATH)
    if 'Unnamed: 0' not in data_CI_time.columns:
        raise RuntimeError(f'Model execution message{CI_TIME_PATH}')
    if 'TCI.Total' not in data_CI_time.columns:
        raise RuntimeError(f'Model execution message{CI_TIME_PATH}')
    data_CI_time['date'] = pd.to_datetime(data_CI_time['Unnamed: 0'], format='mixed', errors='coerce')
    data_CI_time = data_CI_time.dropna(subset=['date'])
    data_CI_time['date'] = data_CI_time['date'].dt.tz_localize(None).dt.normalize()
    data_CI_time = data_CI_time.set_index('date')
    red_df = periods.copy()

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    col_mkt = _pick_col(red_df, ['market', 'Market', 'Model execution message', 'Model execution message', 'Model execution message', 'Model execution message', 'name', 'filename'])
    col_start = _pick_col(red_df, ['start_date', 'start', 'Start', 'start_date', 'Model execution message'])
    col_end = _pick_col(red_df, ['end_date', 'end', 'End', 'end_date', 'Model execution message'])
    if col_mkt is None or col_start is None or col_end is None:
        raise RuntimeError(f'Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else str(x).strip()
    red_df['market'] = red_df[col_mkt].apply(extract_market_code)
    red_df['start'] = pd.to_datetime(red_df[col_start], errors='coerce').dt.normalize()
    red_df['end'] = pd.to_datetime(red_df[col_end], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start', 'end']).copy()
    red_range = {r['market']: (r['start'], r['end']) for _, r in red_df.iterrows()}
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-(H - 1))
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = pd.concat([data[country], data_CI_time['TCI.Total']], axis=1)
        data_RV_base = data_RV_base.dropna()
        data_RV_base = data_RV_base.loc[SAMPLE_START:]
        data_RV_base.columns = ['RV_now', 'CI_time']
        if data_RV_base.empty:
            skipped.append((country, 'empty after concat/dropna/SAMPLE_START'))
            continue
        decay_factor = np.exp(-np.arange(len(data_RV_base)) / 7 * math.pi)
        CI_time_ED = np.tril(np.outer(data_RV_base['CI_time'].values, decay_factor))
        data_RV_base['CI_time_ED'] = CI_time_ED.sum(axis=1)
        data_RV_base['CI_time_D'] = data_RV_base['CI_time_ED'].shift(1)
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_future_mean(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.dropna().copy()
        if data_RV.empty:
            skipped.append((country, 'empty after feature/target dropna'))
            continue
        if ANCHOR_DATE not in data_RV.index:
            pos_anchor = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos_anchor <= 0 or pos_anchor >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot align'))
                continue
            anchor_date = data_RV.index[pos_anchor]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test rows in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[TARGET_COL].astype(float).copy()
        result = []
        for idx in tqdm(data_test.index, desc=f'{country}-H22', leave=False):
            pos = data_RV.index.get_loc(idx)
            start_pos = pos - window_len
            if start_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[start_pos:pos].copy()
            if len(data_rolling) < 30:
                result.append(np.nan)
                continue
            mod = smf.ols(f'{TARGET_COL} ~ RV_D + RV_W + RV_M + CI_time_D', data_rolling)
            try:
                res = mod.fit()
                p = res.params
                r_predict = p.iloc[0] + p.iloc[1] * data_test.loc[idx, 'RV_D'] + p.iloc[2] * data_test.loc[idx, 'RV_W'] + p.iloc[3] * data_test.loc[idx, 'RV_M'] + p.iloc[4] * data_test.loc[idx, 'CI_time_D']
                result.append(float(r_predict))
            except Exception:
                result.append(np.nan)
        result = pd.Series(result, index=data_test.index, name='RV_Predict')
        SUM = pd.concat([data_predict.rename('RV_True'), result], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        SUM_final = SUM.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(SUM_final['MSE']))
        MAE = float(np.mean(SUM_final['MAE']))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_spillover_frequency(periods):
    model_outputs = {}
    import pandas as pd
    import numpy as np
    import math
    import scipy as sc
    import time
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import os
    from scipy.stats import norm
    import statsmodels.formula.api as smf
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    RV_PATH = RV_FILE
    CI_PATH = TCI_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    H = 22
    TARGET_COL = f'RV_MA{H}'
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    OUT_DIR = f'unused_output'
    EPS = 1e-12
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='all')
    data_CI = pd.read_csv(CI_PATH)
    if 'Unnamed: 0' not in data_CI.columns:
        raise RuntimeError(f'Model execution message{CI_PATH}')
    data_CI['date'] = pd.to_datetime(data_CI['Unnamed: 0'], format='mixed', errors='coerce')
    data_CI = data_CI.dropna(subset=['date'])
    data_CI['date'] = data_CI['date'].dt.tz_localize(None).dt.normalize()
    data_CI = data_CI.set_index('date')
    need_ci_cols = ['TCI.20.Inf', 'TCI.5.20', 'TCI.1.5']
    miss_ci = [c for c in need_ci_cols if c not in data_CI.columns]
    if miss_ci:
        raise RuntimeError(f'Model execution message{miss_ci}Model execution message{list(data_CI.columns)}')
    red_df = periods.copy()

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    col_mkt = _pick_col(red_df, ['market', 'Market', 'Model execution message', 'Model execution message', 'Model execution message', 'Model execution message', 'name', 'filename'])
    col_start = _pick_col(red_df, ['start_date', 'start', 'Start', 'start_date', 'Model execution message'])
    col_end = _pick_col(red_df, ['end_date', 'end', 'End', 'end_date', 'Model execution message'])
    if col_mkt is None or col_start is None or col_end is None:
        raise RuntimeError(f'Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        import re
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else str(x).strip()
    red_df['market'] = red_df[col_mkt].apply(extract_market_code)
    red_df['start'] = pd.to_datetime(red_df[col_start], errors='coerce').dt.normalize()
    red_df['end'] = pd.to_datetime(red_df[col_end], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start', 'end']).copy()
    red_range = {r['market']: (r['start'], r['end']) for _, r in red_df.iterrows()}
    markets_to_run = [m for m in red_range.keys() if m in data.columns]

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        y = rv_now.rolling(window=H, center=False).mean()
        return y.shift(-22)
    error_records = {}
    skipped = []
    for country in markets_to_run:
        oos_start, oos_end = red_range[country]
        data_RV_base = pd.concat([data[country], data_CI['TCI.20.Inf'], data_CI['TCI.5.20'], data_CI['TCI.1.5']], axis=1)
        data_RV_base.columns = ['RV_now', 'CI_long', 'CI_mid', 'CI_short']
        data_RV_base = data_RV_base.fillna(method='bfill')
        data_RV_base['CI_long_D'] = data_RV_base['CI_long'].shift(1)
        data_RV_base['CI_mid_D'] = data_RV_base['CI_mid'].shift(1)
        data_RV_base['CI_short_D'] = data_RV_base['CI_short'].shift(1)
        data_RV_base['RV_D'] = data_RV_base['RV_now'].shift(1)
        data_RV_base['RV_W'] = data_RV_base['RV_D'].rolling(5).mean()
        data_RV_base['RV_M'] = data_RV_base['RV_D'].rolling(22).mean()
        data_RV_base[TARGET_COL] = make_future_mean(data_RV_base['RV_now'], H)
        data_RV = data_RV_base.loc[SAMPLE_START:].dropna().copy()
        if data_RV.empty:
            skipped.append((country, 'no data after SAMPLE_START & dropna'))
            continue
        if ANCHOR_DATE not in data_RV.index:
            pos_anchor = data_RV.index.searchsorted(ANCHOR_DATE)
            if pos_anchor <= 0 or pos_anchor >= len(data_RV.index):
                skipped.append((country, 'anchor date not in index and cannot align'))
                continue
            anchor_date = data_RV.index[pos_anchor]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(data_RV.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        data_test = data_RV.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((country, f'no test rows in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[TARGET_COL].astype(float).copy()
        result = []
        for idx in tqdm(data_test.index, desc=f'{country}-H22', leave=False):
            pos = data_RV.index.get_loc(idx)
            start_pos = pos - window_len
            if start_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = data_RV.iloc[start_pos:pos].copy()
            if len(data_rolling) < 30:
                result.append(np.nan)
                continue
            mod = smf.ols(f'{TARGET_COL} ~ RV_D + RV_W + RV_M + CI_long_D + CI_mid_D + CI_short_D', data_rolling)
            try:
                res = mod.fit()
                p = res.params
                r_predict = p.iloc[0] + p.iloc[1] * data_test.loc[idx, 'RV_D'] + p.iloc[2] * data_test.loc[idx, 'RV_W'] + p.iloc[3] * data_test.loc[idx, 'RV_M'] + p.iloc[4] * data_test.loc[idx, 'CI_long_D'] + p.iloc[5] * data_test.loc[idx, 'CI_mid_D'] + p.iloc[6] * data_test.loc[idx, 'CI_short_D']
                result.append(float(r_predict))
            except Exception:
                result.append(np.nan)
        result = pd.Series(result, index=data_test.index, name='RV_Predict')
        SUM = pd.concat([data_predict.rename('RV_True'), result], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        SUM_final = SUM.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(SUM.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_safe = SUM_final['RV_Predict'].clip(lower=EPS)
        true_safe = SUM_final['RV_True'].clip(lower=EPS)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean(SUM_final['MSE']))
        MAE = float(np.mean(SUM_final['MAE']))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_loghar(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import statsmodels.formula.api as smf
    EPS_LOG = 1e-12
    EXP_CLIP_MAX = 50
    H = 22
    SAMPLE_START = '2014-07-04'
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    red_TABLE_PATH = 'in_memory_bull_periods'
    OUT_DIR = 'unused_output'
    PLOT = False

    def safe_log(s: pd.Series, eps: float=EPS_LOG) -> pd.Series:
        return np.log(s.clip(lower=eps))

    def safe_exp(x, clip_max: float=EXP_CLIP_MAX):
        return np.exp(np.clip(x, -clip_max, clip_max))

    def future_mean_level(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-(H - 1))
    data = pd.read_csv(RV_FILE)
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.set_index('date')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    error_rows = []
    used_markets = []
    skip_no_sample_or_fail = []
    for c in markets_to_run:
        try:
            oos_start, oos_end = red_range[c]
            if oos_start < pred_start_base:
                oos_start = pred_start_base
            if oos_end > pred_end_base:
                oos_end = pred_end_base
            data_RV = data[[c]].copy()
            data_RV.columns = ['RV_now']
            data_RV['RV_D_lvl'] = data_RV['RV_now'].shift(1)
            data_RV['RV_W_lvl'] = data_RV['RV_D_lvl'].rolling(5).mean()
            data_RV['RV_M_lvl'] = data_RV['RV_D_lvl'].rolling(22).mean()
            data_RV['RV_D'] = safe_log(data_RV['RV_D_lvl'])
            data_RV['RV_W'] = safe_log(data_RV['RV_W_lvl'])
            data_RV['RV_M'] = safe_log(data_RV['RV_M_lvl'])
            data_RV[f'RV_MA{H}_lvl'] = future_mean_level(data_RV['RV_now'], H)
            data_RV[f'RV_MA{H}'] = safe_log(data_RV[f'RV_MA{H}_lvl'])
            data_RV = data_RV.dropna()
            data_RV = data_RV.loc[SAMPLE_START:]
            data_test = data_RV.loc[oos_start:oos_end].copy()
            if len(data_test) == 0:
                skip_no_sample_or_fail.append((c, 'Model execution message'))
                continue
            y_true_log = data_test[f'RV_MA{H}'].copy()
            if pred_start_base not in data_RV.index:
                pos0 = data_RV.index.searchsorted(pred_start_base)
                if pos0 <= 0 or pos0 >= len(data_RV.index):
                    skip_no_sample_or_fail.append((c, 'Model execution message'))
                    continue
                pred_anchor_date = data_RV.index[pos0]
            else:
                pred_anchor_date = pred_start_base
            train_len = len(data_RV.loc[:pred_anchor_date]) - 1
            if train_len <= 0:
                skip_no_sample_or_fail.append((c, f'train_len<=0: {train_len}'))
                continue
            test_start_date = data_test.index[0]
            start_pos = data_RV.index.get_loc(test_start_date)
            result_log = []
            n_test = len(data_test)
            for i in tqdm(range(n_test), desc=f'LogHAR {c} H{H}', leave=False):
                train_slice = data_RV.iloc[start_pos + i:start_pos + i + train_len].copy()
                if train_slice.empty:
                    result_log.append(np.nan)
                    continue
                mod = smf.ols(f'RV_MA{H} ~ RV_D + RV_W + RV_M', train_slice)
                res = mod.fit()
                p = res.params
                r_predict_log = p.iloc[0] + p.iloc[1] * data_test['RV_D'].iloc[i] + p.iloc[2] * data_test['RV_W'].iloc[i] + p.iloc[3] * data_test['RV_M'].iloc[i]
                result_log.append(float(r_predict_log))
            result_log = pd.Series(result_log, index=y_true_log.index).dropna()
            if result_log.empty:
                skip_no_sample_or_fail.append((c, 'Model execution message'))
                continue
            y_true = pd.Series(safe_exp(y_true_log.loc[result_log.index].values), index=result_log.index)
            y_pred = pd.Series(safe_exp(result_log.values), index=result_log.index)
            SUM = pd.concat([y_true, y_pred], axis=1)
            SUM.columns = ['RV_True', 'RV_Predict']
            SUM_final = SUM.copy()
            SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
            SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
            pred_safe = SUM_final['RV_Predict'].clip(lower=EPS_LOG)
            true_safe = SUM_final['RV_True'].clip(lower=EPS_LOG)
            SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
            MSE = float(np.mean(SUM_final['MSE']))
            MAE = float(np.mean(SUM_final['MAE']))
            denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
            Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
            model_outputs[str(c).strip().lstrip('.')] = SUM_final.copy()
            error_rows.append([MSE, MAE, Theil])
            used_markets.append(c)
        except Exception as e:
            skip_no_sample_or_fail.append((c, f'Model execution message{repr(e)}'))
            continue
    if len(used_markets) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=used_markets).T
    if skip_no_sample_or_fail:
        for m, reason in skip_no_sample_or_fail:
            pass
    return model_outputs


def run_loghar_ci(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    RET_PATH = RETURN_FILE
    RV_PATH = RV_FILE
    P_STORE_PATH = LOG_CENTRALITY_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    PLOT = False
    fit_end_date = pd.to_datetime('2018-05-01')
    pred_start_base = pd.to_datetime('2018-05-02')
    pred_end_base = pd.to_datetime('2022-02-25')
    H = 22
    FIT_START = 1000
    FIT_STEP = 100
    FIT_MAX = 1
    L_fit_list = [FIT_START + FIT_STEP * k for k in range(FIT_MAX)]
    NET_MIN = 700
    NET_MAX = 700
    NET_STEP = 50
    L_net_list = list(range(NET_MIN, NET_MAX + 1, NET_STEP))
    OUT_DIR = 'unused_output'
    EPS_LOG = 1e-12
    EXP_CLIP_MAX = 50

    def _norm_date(x):
        return pd.to_datetime(x).tz_localize(None).normalize()

    def safe_log(s: pd.Series, eps: float=EPS_LOG) -> pd.Series:
        return np.log(s.clip(lower=eps))

    def safe_exp(x: pd.Series | np.ndarray, clip_max: float=EXP_CLIP_MAX):
        return np.exp(np.clip(x, -clip_max, clip_max))

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-(H - 1))

    def make_future_mean_log(rv_now: pd.Series, H: int) -> pd.Series:
        return safe_log(make_future_mean(rv_now, H))

    def read_returns(path):
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str[:10], errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
        df = df.set_index('date').dropna(how='all')
        return df

    def read_rv_no_scale(path):
        data = pd.read_csv(path)
        if 'date' not in data.columns:
            raise RuntimeError("RV file must contain a 'date' column.")
        num_cols = [c for c in data.columns if c != 'date']
        data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
        data = data.dropna(how='all')
        data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
        data = data.dropna(subset=['date'])
        data['date'] = data['date'].dt.tz_localize(None)
        data['date'] = data['date'].dt.normalize()
        data = data.set_index('date')
        data = data.dropna(axis=1, how='all')
        return data
    df_ret = read_returns(RET_PATH)
    data_rv = read_rv_no_scale(RV_PATH)
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'])
    red_df['end_date'] = pd.to_datetime(red_df['end_date'])
    red_df = red_df.dropna(subset=['market']).copy()
    red_range = {}
    for _, row in red_df.iterrows():
        red_range[row['market']] = (row['start_date'], row['end_date'])
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data_rv.columns and m in df_ret.columns]
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all['ci_start_date'] = P_all['ci_start_date'].apply(_norm_date)
    P_all['ci_end_date'] = P_all['ci_end_date'].apply(_norm_date)
    P_all['L_net'] = pd.to_numeric(P_all['L_net'], errors='coerce').astype('Int64')
    P_all = P_all.dropna(subset=['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net']).copy()
    P_cache = {}

    def load_P_from_csv(L_net: int, ci_start_date, ci_end_date):
        key = (int(L_net), _norm_date(ci_start_date), _norm_date(ci_end_date))
        if key in P_cache:
            return P_cache[key]
        ln, cs, ce = key
        sub = P_all[(P_all['L_net'] == ln) & (P_all['ci_start_date'] == cs) & (P_all['ci_end_date'] == ce)]
        if sub.empty:
            P_cache[key] = (None, None, None, None)
            return P_cache[key]

        def _pivot(col):
            df = sub.pivot_table(index='date', columns='market', values=col, aggfunc='mean')
            return df.sort_index()
        P_corr = _pivot('P_corr')
        P_kendall = _pivot('P_kendall')
        P_partial = _pivot('P_partial')
        P_tail = _pivot('P_tail')
        P_cache[key] = (P_corr, P_kendall, P_partial, P_tail)
        return P_cache[key]

    def forecast_one_market_logharci_red(data_rv_all: pd.DataFrame, P_corr: pd.DataFrame, P_kendall: pd.DataFrame, P_partial: pd.DataFrame, P_tail: pd.DataFrame, market: str, fit_start_date: pd.Timestamp, H: int, oos_start: pd.Timestamp, oos_end: pd.Timestamp, pred_start_anchor: pd.Timestamp):
        df = data_rv_all[market].to_frame('RV_now')
        df['RV_D_lvl'] = df['RV_now'].shift(1)
        df['RV_W_lvl'] = df['RV_D_lvl'].rolling(5).mean()
        df['RV_M_lvl'] = df['RV_D_lvl'].rolling(22).mean()
        df['RV_D'] = safe_log(df['RV_D_lvl'])
        df['RV_W'] = safe_log(df['RV_W_lvl'])
        df['RV_M'] = safe_log(df['RV_M_lvl'])
        df['Y'] = make_future_mean_log(df['RV_now'], H)
        ci = pd.DataFrame({'CI_corr': P_corr[market], 'CI_kendall': P_kendall[market], 'CI_partial': P_partial[market], 'CI_tail': P_tail[market]}).reindex(df.index)
        df = df.join(ci, how='inner').dropna()
        df['CI_corr_D'] = df['CI_corr'].shift(1)
        df['CI_kendall_D'] = df['CI_kendall'].shift(1)
        df['CI_partial_D'] = df['CI_partial'].shift(1)
        df['CI_tail_D'] = df['CI_tail'].shift(1)
        df = df.dropna()
        df = df.loc[fit_start_date:]
        if df.empty:
            return (None, np.nan, np.nan, np.nan)
        data_test = df.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            return (None, np.nan, np.nan, np.nan)
        if pred_start_anchor not in df.index:
            pos0 = df.index.searchsorted(pred_start_anchor)
            if pos0 <= 0 or pos0 >= len(df.index):
                return (None, np.nan, np.nan, np.nan)
            anchor_date = df.index[pos0]
        else:
            anchor_date = pred_start_anchor
        window_len = len(df.loc[:anchor_date]) - 1
        if window_len <= 0:
            return (None, np.nan, np.nan, np.nan)
        y_true_log = data_test['Y'].astype(float).copy()
        idx = y_true_log.index
        preds_log = []
        for t in tqdm(idx, desc=f'LogHAR-CI {market} H{H}', leave=False):
            t_pos = df.index.get_loc(t)
            train_end_pos = t_pos - 1
            train_start_pos = train_end_pos - (window_len - 1)
            if train_start_pos < 0 or train_end_pos < 0:
                preds_log.append(np.nan)
                continue
            train = df.iloc[train_start_pos:train_end_pos + 1].copy()
            if train.empty:
                preds_log.append(np.nan)
                continue
            train['_Y_'] = train['Y'].astype(float).values
            res = smf.ols('_Y_ ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D', train).fit()
            p = res.params
            row = df.iloc[t_pos]
            pred_log = p.iloc[0] + p.iloc[1] * row['RV_D'] + p.iloc[2] * row['RV_W'] + p.iloc[3] * row['RV_M'] + p.iloc[4] * row['CI_corr_D'] + p.iloc[5] * row['CI_kendall_D'] + p.iloc[6] * row['CI_partial_D'] + p.iloc[7] * row['CI_tail_D']
            preds_log.append(float(pred_log))
        y_pred_log = pd.Series(preds_log, index=idx).dropna()
        if y_pred_log.empty:
            return (None, np.nan, np.nan, np.nan)
        y_true = pd.Series(safe_exp(y_true_log.loc[y_pred_log.index].values), index=y_pred_log.index)
        y_pred = pd.Series(safe_exp(y_pred_log.values), index=y_pred_log.index)
        mcs = pd.concat([y_true.rename('RV_True'), y_pred.rename('RV_Predict')], axis=1)
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        mse = float(np.nanmean(mcs['MSE'].values))
        mae = float(np.nanmean(mcs['MAE'].values))
        denom = np.sqrt(np.nanmean(mcs['RV_Predict'].values ** 2)) + np.sqrt(np.nanmean(mcs['RV_True'].values ** 2))
        theil = float(np.sqrt(np.nanmean((mcs['RV_True'].values - mcs['RV_Predict'].values) ** 2)) / denom) if denom != 0 else np.nan
        return (mcs, mse, mae, theil)
    L_fit = L_fit_list[0]
    L_net = L_net_list[0]
    error_by_market = {}
    skipped = []
    for mkt in markets_to_run:
        oos_start, oos_end = red_range[mkt]
        if oos_start < pred_start_base:
            oos_start = pred_start_base
        if oos_end > pred_end_base:
            oos_end = pred_end_base
        if fit_end_date not in data_rv.index:
            skipped.append((mkt, f'RV missing fit_end_date={fit_end_date.date()}'))
            continue
        fit_end_pos = data_rv.index.get_loc(fit_end_date)
        fit_start_pos = fit_end_pos - (L_fit - 1)
        if fit_start_pos < 0:
            skipped.append((mkt, f'L_fit too large, L_fit={L_fit}'))
            continue
        fit_start_date = data_rv.index[fit_start_pos]
        ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
        if ret_pos_end < 0:
            skipped.append((mkt, 'Not enough return history for this L_fit'))
            continue
        net_end_date = df_ret.index[ret_pos_end]
        ci_start_date = net_end_date
        ci_end_date = pred_end_base
        P_corr, P_kendall, P_partial, P_tail = load_P_from_csv(L_net=L_net, ci_start_date=ci_start_date, ci_end_date=ci_end_date)
        if P_corr is None:
            skipped.append((mkt, f'missing P key: L_net={L_net}, ci_start={_norm_date(ci_start_date).date()}, ci_end={_norm_date(ci_end_date).date()}'))
            continue
        if mkt not in P_corr.columns:
            skipped.append((mkt, 'market not in P matrices'))
            continue
        mcs, mse, mae, theil = forecast_one_market_logharci_red(data_rv_all=data_rv, P_corr=P_corr, P_kendall=P_kendall, P_partial=P_partial, P_tail=P_tail, market=mkt, fit_start_date=fit_start_date, H=H, oos_start=oos_start, oos_end=oos_end, pred_start_anchor=pred_start_base)
        if mcs is None:
            skipped.append((mkt, 'no valid test sample / prediction failed'))
            continue
        mcs_path = os.path.join(OUT_DIR, f'MCS_{mkt}.csv')
        model_outputs[str(mkt).strip().lstrip('.')] = mcs.copy()
        error_by_market[mkt] = [mse, mae, theil]
    if len(error_by_market) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_by_market, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_hexp(periods):
    model_outputs = {}
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import os
    import re
    H = 22
    EPS = 1e-12
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    red_TABLE_PATH = 'in_memory_bull_periods'
    base_output_dir = 'unused_output'
    OUT_DIR = f'{base_output_dir}/H22-red/HExp-GlRV'
    PLOT = False

    def get_lambda(com):
        return np.log(1 + 1 / com)

    def calculate_exp_factor(series, com, window=500):
        lam = get_lambda(com)
        weights = np.exp(-np.arange(1, window + 1) * lam)
        weights /= weights.sum()
        return series.rolling(window).apply(lambda x: np.dot(x[::-1], weights), raw=True)
    data = pd.read_csv(RV_FILE)
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.normalize()
    data = data.dropna(subset=['date']).set_index('date').sort_index()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='any')
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    mean_rv_all = data.mean()
    global_signal = (data / mean_rv_all).mean(axis=1)
    exp_global_5 = calculate_exp_factor(global_signal, com=5)
    com_list = [1, 5, 25, 125]
    error_records = {}
    skipped = []
    for c in markets_to_run:
        asset_mean = float(mean_rv_all[c])
        df_base = pd.DataFrame(index=data.index)
        df_base['RV_now'] = data[c].astype(float)
        for j in com_list:
            df_base[f'ExpRV_{j}'] = calculate_exp_factor(df_base['RV_now'], com=j)
        df_base['ExpGlRV_5'] = exp_global_5 * asset_mean
        features = [f'ExpRV_{j}' for j in com_list] + ['ExpGlRV_5']
        for feat in features:
            df_base[f'{feat}_lag'] = df_base[feat].shift(1)
        reg_features = []
        for feat in features:
            col_name = f'diff_{feat}'
            df_base[col_name] = df_base[f'{feat}_lag'] - asset_mean
            reg_features.append(col_name)
        df_model = df_base.copy()
        target_col = f'RV_MA{H}'
        df_model[target_col] = df_model['RV_now'].rolling(window=H).mean()
        df_model[target_col] = df_model[target_col].shift(-H)
        df_model['target_diff'] = df_model[target_col] - asset_mean
        df_model = df_model.dropna()
        df_model = df_model.loc[SAMPLE_START:]
        if df_model.empty:
            skipped.append((c, 'df_model empty after dropna/SAMPLE_START'))
            continue
        if ANCHOR_DATE not in df_model.index:
            pos = df_model.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(df_model.index):
                skipped.append((c, 'cannot locate anchor_date in df_model'))
                continue
            anchor_date = df_model.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len_fixed = len(df_model.loc[:anchor_date]) - 1
        if window_len_fixed <= 30:
            skipped.append((c, f'window_len too small: {window_len_fixed}'))
            continue
        oos_start, oos_end = red_range[c]
        data_test = df_model.loc[oos_start:oos_end]
        if len(data_test) == 0:
            skipped.append((c, f'no test sample in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        data_predict = data_test[target_col]
        result = []
        formula = 'target_diff ~ ' + ' + '.join(reg_features)
        for idx in tqdm(data_test.index, desc=f'{c}-H{H}', leave=False):
            pos = df_model.index.get_loc(idx)
            train_end_pos = pos - 1
            train_start_pos = train_end_pos - (window_len_fixed - 1)
            if train_start_pos < 0 or train_end_pos < 0:
                result.append(np.nan)
                continue
            data_rolling = df_model.iloc[train_start_pos:train_end_pos + 1].copy()
            if len(data_rolling) < 50:
                result.append(np.nan)
                continue
            try:
                res = smf.ols(formula, data_rolling).fit()
                p = res.params
                row = data_test.loc[idx]
                pred_diff = float(p.iloc[0])
                for k, feat in enumerate(reg_features, start=1):
                    pred_diff += float(p.iloc[k]) * float(row[feat])
                r_predict = pred_diff + asset_mean
                result.append(float(r_predict))
            except Exception:
                result.append(np.nan)
        result = pd.Series(result, index=data_test.index, name='RV_Predict').dropna()
        if result.empty:
            skipped.append((c, 'all predictions nan'))
            continue
        data_predict = data_predict.loc[result.index]
        SUM = pd.concat([data_predict.rename('RV_True'), result], axis=1)
        SUM_final = SUM.copy()
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        pred_clip = np.maximum(SUM_final['RV_Predict'].values, EPS)
        SUM_final['Qlike'] = SUM_final['RV_True'].values / pred_clip + np.log(pred_clip)
        MSE = float(np.mean(SUM_final['MSE']))
        MAE = float(np.mean(SUM_final['MAE']))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))))
        error_records[c] = [MSE, MAE, Theil]
        model_outputs[str(c).strip().lstrip('.')] = SUM_final.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_mvf(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    H = 22
    EPS = 1e-12
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(how='any')
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.normalize()
    data = data.dropna(subset=['date']).set_index('date').sort_index()
    data[data <= 0] = EPS
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    missing_in_rv = [m for m in table_markets if m not in data.columns]
    if missing_in_rv:
        pass
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    crv_raw = data.mean(axis=1).clip(lower=EPS)
    xi_raw = data.divide(crv_raw, axis=0).clip(lower=EPS)

    def get_logmean_har_features(raw_series, h=22, eps=1e-12):
        raw = pd.Series(raw_series).astype(float).clip(lower=eps)
        df = pd.DataFrame(index=raw.index)
        df['D'] = np.log(raw.shift(1).clip(lower=eps))
        df['W'] = np.log(raw.shift(1).rolling(5).mean().clip(lower=eps))
        df['M'] = np.log(raw.shift(1).rolling(22).mean().clip(lower=eps))
        future_mean = raw.rolling(window=h).mean().shift(-(h - 1)).clip(lower=eps)
        df['Target'] = np.log(future_mean)
        return df.dropna()
    df_log_crv = get_logmean_har_features(crv_raw, h=H, eps=EPS)
    df_log_crv = df_log_crv.loc[SAMPLE_START:].copy()
    if df_log_crv.empty:
        raise RuntimeError('df_log_crv is empty after SAMPLE_START.')
    if ANCHOR_DATE not in df_log_crv.index:
        pos = df_log_crv.index.searchsorted(ANCHOR_DATE)
        if pos <= 0 or pos >= len(df_log_crv.index):
            raise RuntimeError('cannot locate ANCHOR_DATE in df_log_crv.')
        anchor_date = df_log_crv.index[pos]
    else:
        anchor_date = ANCHOR_DATE
    train_len_crv_fixed = len(df_log_crv.loc[:anchor_date]) - H
    if train_len_crv_fixed <= 10:
        raise RuntimeError(f'train_len_crv_fixed too small: {train_len_crv_fixed}')
    crv_log_pred_all = pd.Series(index=df_log_crv.index, dtype=float)
    for i in tqdm(range(len(df_log_crv)), desc=f'CRV-Log-HAR H={H}', leave=False):
        train_end = i - 1
        train_start = train_end - (train_len_crv_fixed - 1)
        if train_start < 0 or train_end < 0:
            continue
        train_data = df_log_crv.iloc[train_start:train_end + 1]
        if len(train_data) < 30:
            continue
        try:
            model = smf.ols('Target ~ D + W + M', train_data).fit()
            row = df_log_crv.iloc[i]
            pred = model.params['Intercept'] + model.params['D'] * row['D'] + model.params['W'] * row['W'] + model.params['M'] * row['M']
            crv_log_pred_all.iloc[i] = float(pred)
        except Exception:
            continue
    error_records = {}
    skipped = []
    for c in markets_to_run:
        oos_start, oos_end = red_range[c]
        df_log_xi = get_logmean_har_features(xi_raw[c], h=H, eps=EPS)
        df_log_xi = df_log_xi.loc[SAMPLE_START:].copy()
        if df_log_xi.empty:
            skipped.append((c, 'df_log_xi empty'))
            continue
        if ANCHOR_DATE not in df_log_xi.index:
            pos = df_log_xi.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(df_log_xi.index):
                skipped.append((c, 'cannot locate ANCHOR_DATE in df_log_xi'))
                continue
            anchor_date_xi = df_log_xi.index[pos]
        else:
            anchor_date_xi = ANCHOR_DATE
        train_len_xi_fixed = len(df_log_xi.loc[:anchor_date_xi]) - H
        if train_len_xi_fixed <= 10:
            skipped.append((c, f'train_len_xi_fixed too small: {train_len_xi_fixed}'))
            continue
        test_xi = df_log_xi.loc[oos_start:oos_end].copy()
        if test_xi.empty:
            skipped.append((c, 'no test sample in red segment'))
            continue
        xi_log_pred = []
        for t in tqdm(test_xi.index, desc=f'{c}-xi-Log-HAR H={H}', leave=False):
            i = df_log_xi.index.get_loc(t)
            train_end = i - 1
            train_start = train_end - (train_len_xi_fixed - 1)
            if train_start < 0 or train_end < 0:
                xi_log_pred.append(np.nan)
                continue
            train_data = df_log_xi.iloc[train_start:train_end + 1]
            if len(train_data) < 30:
                xi_log_pred.append(np.nan)
                continue
            try:
                model = smf.ols('Target ~ D + W + M', train_data).fit()
                row = df_log_xi.loc[t]
                pred = model.params['Intercept'] + model.params['D'] * row['D'] + model.params['W'] * row['W'] + model.params['M'] * row['M']
                xi_log_pred.append(float(pred))
            except Exception:
                xi_log_pred.append(np.nan)
        pred_log_xi_series = pd.Series(xi_log_pred, index=test_xi.index).dropna()
        if pred_log_xi_series.empty:
            skipped.append((c, 'all xi predictions nan'))
            continue
        idx = pred_log_xi_series.index.intersection(crv_log_pred_all.dropna().index)
        if len(idx) == 0:
            skipped.append((c, 'no intersection with CRV predictions'))
            continue
        pred_crv_mean = np.exp(crv_log_pred_all.loc[idx])
        pred_xi_mean = np.exp(pred_log_xi_series.loc[idx])
        rv_pred_mean = (pred_crv_mean * pred_xi_mean).clip(lower=EPS)
        rv_true_mean = data[c].rolling(H).mean().shift(-H).loc[idx].clip(lower=EPS)
        out = pd.DataFrame({'RV_True': rv_true_mean, 'RV_Predict': rv_pred_mean}).dropna()
        if out.empty:
            skipped.append((c, 'out empty after dropna'))
            continue
        out['MSE'] = (out['RV_True'] - out['RV_Predict']) ** 2
        out['MAE'] = np.abs(out['RV_True'] - out['RV_Predict'])
        out['Qlike'] = out['RV_True'] / out['RV_Predict'].clip(lower=EPS) + np.log(out['RV_Predict'].clip(lower=EPS))
        mse_val = float(out['MSE'].mean())
        mae_val = float(out['MAE'].mean())
        theil_val = float(np.sqrt(mse_val) / (np.sqrt((out['RV_Predict'] ** 2).mean()) + np.sqrt((out['RV_True'] ** 2).mean())))
        error_records[c] = [mse_val, mae_val, theil_val]
        model_outputs[str(c).strip().lstrip('.')] = out.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR_DF = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_multivariate(periods):
    model_outputs = {}
    import os
    import re
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    EPS_LOG = 1e-12
    EXP_CLIP_MAX = 50
    H = 22
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    OOS_ANCHOR = pd.to_datetime('2018-05-02').normalize()
    TRAIN_START0 = pd.to_datetime('2014-07-04').normalize()
    TRAIN_END0 = (OOS_ANCHOR - pd.Timedelta(days=1)).normalize()
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    OUT_DIR = 'unused_output'
    PLOT = False

    def safe_log(x, eps=EPS_LOG):
        return np.log(np.clip(x, eps, None))

    def safe_exp(x, clip_max=EXP_CLIP_MAX):
        return np.exp(np.clip(x, -clip_max, clip_max))

    def future_mean_level(rv_now: pd.Series, H: int) -> pd.Series:
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-(H - 1))
    data = pd.read_csv(RV_PATH)
    num_cols = [c for c in data.columns if c != 'date']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(how='any')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.set_index('date').sort_index()
    data = data.fillna(method='ffill').dropna(how='any')
    data[data <= 0] = EPS_LOG
    markets = list(data.columns)
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    targets_to_run = [m for m in list(dict.fromkeys(red_df['market'].tolist())) if m in markets]
    rv_now = data.copy()
    rv_d_lvl = rv_now.shift(1)
    rv_w_lvl = rv_d_lvl.rolling(5).mean()
    rv_m_lvl = rv_d_lvl.rolling(22).mean()
    rv_d = safe_log(rv_d_lvl)
    rv_w = safe_log(rv_w_lvl)
    rv_m = safe_log(rv_m_lvl)
    X_all = pd.DataFrame(index=rv_now.index)
    for m in markets:
        X_all[f'RV_D_{m}'] = rv_d[m]
        X_all[f'RV_W_{m}'] = rv_w[m]
        X_all[f'RV_M_{m}'] = rv_m[m]
    X_cols = list(X_all.columns)

    def get_train_window(t: pd.Timestamp):
        shift_days = (t - OOS_ANCHOR).days
        train_start = (TRAIN_START0 + pd.Timedelta(days=shift_days)).normalize()
        train_end = (TRAIN_END0 + pd.Timedelta(days=shift_days)).normalize()
        return (train_start, train_end)
    error_records = {}
    skipped = []
    for target in targets_to_run:
        oos_start, oos_end = red_range[target]
        if oos_start < OOS_ANCHOR:
            oos_start = OOS_ANCHOR
        y_lvl = future_mean_level(rv_now[target], H)
        y_log = pd.Series(safe_log(y_lvl), index=rv_now.index, name='Y')
        df = pd.concat([y_log, X_all], axis=1).dropna()
        df = df.loc[SAMPLE_START:].copy()
        if df.empty:
            skipped.append((target, 'df empty after dropna/SAMPLE_START'))
            continue
        data_test = df.loc[oos_start:oos_end].copy()
        if len(data_test) == 0:
            skipped.append((target, 'no test sample in red segment'))
            continue
        preds_log = []
        for t in tqdm(data_test.index, desc=f'MV-LogHAR {target}', leave=False):
            train_start, train_end = get_train_window(t)
            train_slice = df.loc[train_start:train_end].copy()
            if len(train_slice) <= H - 1 + 30:
                preds_log.append(np.nan)
                continue
            train_slice = train_slice.iloc[:-(H - 1)]
            if len(train_slice) < 30:
                preds_log.append(np.nan)
                continue
            y_tr = train_slice['Y'].values.astype(float)
            X_tr = train_slice[X_cols].values.astype(float)
            X_tr = sm.add_constant(X_tr, has_constant='add')
            if X_tr.shape[0] <= X_tr.shape[1] + 1:
                preds_log.append(np.nan)
                continue
            try:
                res = sm.OLS(y_tr, X_tr).fit()
                x_row = df.loc[t, X_cols].values.astype(float).reshape(1, -1)
                x_row = sm.add_constant(x_row, has_constant='add')
                pred_i = float(res.predict(x_row)[0])
                preds_log.append(pred_i if np.isfinite(pred_i) else np.nan)
            except Exception:
                preds_log.append(np.nan)
        preds_log = pd.Series(preds_log, index=data_test.index).dropna()
        if preds_log.empty:
            skipped.append((target, 'all predictions nan'))
            continue
        y_true_log = data_test.loc[preds_log.index, 'Y']
        y_true = pd.Series(safe_exp(y_true_log.values), index=preds_log.index)
        y_pred = pd.Series(safe_exp(preds_log.values), index=preds_log.index)
        SUM = pd.concat([y_true.rename('RV_True'), y_pred.rename('RV_Predict')], axis=1)
        SUM.index.name = 'date'
        SUM['MSE'] = (SUM['RV_True'] - SUM['RV_Predict']) ** 2
        SUM['MAE'] = np.abs(SUM['RV_True'] - SUM['RV_Predict'])
        pred_safe = SUM['RV_Predict'].clip(lower=EPS_LOG)
        true_safe = SUM['RV_True'].clip(lower=EPS_LOG)
        SUM['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(SUM['MSE'].mean())
        MAE = float(SUM['MAE'].mean())
        denom = np.sqrt(np.mean(SUM['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM['RV_True'] ** 2))
        Theil = float(np.sqrt(np.mean((SUM['RV_True'] - SUM['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
        error_records[target] = [MSE, MAE, Theil]
        model_outputs[str(target).strip().lstrip('.')] = SUM.copy()
    if len(error_records) > 0:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_bagging(periods):
    model_outputs = {}
    from sklearn import metrics
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor
    import os
    import re
    H = 22
    EPS = 1e-12
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    P_STORE_PATH = CENTRALITY_FILE
    OUT_DIR = 'unused_output'
    PLOT = False
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date').sort_index()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='any')
    data[data <= 0] = EPS
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all = P_all.dropna(subset=['date', 'market']).copy()
    P_ci = P_all.groupby(['date', 'market'])[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].mean().reset_index()
    error_records = {}
    skipped = []
    for country in tqdm(markets_to_run, desc='Markets'):
        df = pd.DataFrame(index=data.index)
        df['RV_now'] = data[country].astype(float)
        df['RV_MA22'] = df['RV_now'].rolling(window=H, center=False).mean().shift(-(H - 1))
        df['RV_D'] = df['RV_now'].shift(1)
        df['RV_W'] = df['RV_D'].rolling(5).mean()
        df['RV_M'] = df['RV_D'].rolling(22).mean()
        ci_one = P_ci[P_ci['market'] == country].copy()
        if ci_one.empty:
            skipped.append((country, 'missing CI in P.csv'))
            continue
        ci_one = ci_one.set_index('date')[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].rename(columns={'P_corr': 'CI_corr', 'P_kendall': 'CI_kendall', 'P_partial': 'CI_partial', 'P_tail': 'CI_tail'})
        df = df.join(ci_one, how='left')
        df['CI_corr_D'] = df['CI_corr'].shift(1)
        df['CI_kendall_D'] = df['CI_kendall'].shift(1)
        df['CI_partial_D'] = df['CI_partial'].shift(1)
        df['CI_tail_D'] = df['CI_tail'].shift(1)
        df = df.dropna()
        df = df.loc[SAMPLE_START:].copy()
        if df.empty:
            skipped.append((country, 'empty after dropna/SAMPLE_START'))
            continue
        if ANCHOR_DATE not in df.index:
            pos = df.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(df.index):
                skipped.append((country, 'cannot locate anchor in df'))
                continue
            anchor_date = df.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(df.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        oos_start, oos_end = red_range[country]
        test = df.loc[oos_start:oos_end].copy()
        if test.empty:
            skipped.append((country, f'no test sample in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        X_cols = ['RV_D', 'RV_W', 'RV_M', 'CI_corr_D', 'CI_kendall_D', 'CI_partial_D', 'CI_tail_D']
        y_col = 'RV_MA22'
        base_estimator = DecisionTreeRegressor(max_depth=10, random_state=42)
        try:
            Bagging = BaggingRegressor(estimator=base_estimator, n_estimators=1000, max_samples=0.8, max_features=0.8, random_state=42, n_jobs=-1)
        except TypeError:
            Bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=1000, max_samples=0.8, max_features=0.8, random_state=42, n_jobs=-1)
        preds = []
        for t in tqdm(test.index, desc=f'{country}-H22', leave=False):
            pos_t = df.index.get_loc(t)
            train_end = pos_t - H
            train_start = train_end - (window_len - 1)
            if train_start < 0 or train_end < 0 or train_start > train_end:
                preds.append(np.nan)
                continue
            train_slice = df.iloc[train_start:train_end + 1].copy()
            if len(train_slice) < 50:
                preds.append(np.nan)
                continue
            X_train = train_slice[X_cols].values
            y_train = train_slice[y_col].values
            try:
                Bagging.fit(X_train, y_train)
                x_row = df.loc[t, X_cols].values.reshape(1, -1)
                pred = float(Bagging.predict(x_row)[0])
                preds.append(pred if np.isfinite(pred) else np.nan)
            except Exception:
                preds.append(np.nan)
        y_pred = pd.Series(preds, index=test.index, name='RV_Predict').dropna()
        if y_pred.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        y_true = test.loc[y_pred.index, y_col].astype(float)
        SUM = pd.concat([y_true.rename('RV_True'), y_pred], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'empty after aligning y_true/y_pred'))
            continue
        SUM['MSE'] = (SUM['RV_True'] - SUM['RV_Predict']) ** 2
        SUM['MAE'] = np.abs(SUM['RV_True'] - SUM['RV_Predict'])
        SUM['MAPE'] = np.abs((SUM['RV_True'] - SUM['RV_Predict']) / SUM['RV_True'].clip(lower=EPS))
        MSE = float(np.mean((SUM['RV_True'] - SUM['RV_Predict']) ** 2))
        MAE = float(np.mean(np.abs(SUM['RV_True'] - SUM['RV_Predict'])))
        Theil = float(np.sqrt(np.mean((SUM['RV_True'] - SUM['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_gradient(periods):
    model_outputs = {}
    from sklearn import metrics
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.ensemble import GradientBoostingRegressor
    import os
    import re
    H = 22
    EPS = 1e-12
    SAMPLE_START = pd.to_datetime('2014-07-04').normalize()
    ANCHOR_DATE = pd.to_datetime('2018-05-02').normalize()
    RV_PATH = RV_FILE
    red_TABLE_PATH = 'in_memory_bull_periods'
    P_STORE_PATH = CENTRALITY_FILE
    OUT_DIR = 'unused_output'
    PLOT = False
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date').sort_index()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='any')
    data[data <= 0] = EPS
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all = P_all.dropna(subset=['date', 'market']).copy()
    P_ci = P_all.groupby(['date', 'market'])[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].mean().reset_index()
    error_records = {}
    skipped = []
    X_cols = ['RV_D', 'RV_W', 'RV_M', 'CI_corr_D', 'CI_kendall_D', 'CI_partial_D', 'CI_tail_D']
    y_col = 'RV_MA22'
    for country in tqdm(markets_to_run, desc='Markets'):
        df = pd.DataFrame(index=data.index)
        df['RV_now'] = data[country].astype(float)
        df['RV_D'] = df['RV_now'].shift(1)
        df['RV_W'] = df['RV_D'].rolling(5).mean()
        df['RV_M'] = df['RV_D'].rolling(22).mean()
        df['RV_MA22'] = df['RV_now'].rolling(window=22, center=False).mean().shift(-H)
        ci_one = P_ci[P_ci['market'] == country].copy()
        if ci_one.empty:
            skipped.append((country, 'missing CI in P.csv'))
            continue
        ci_one = ci_one.set_index('date')[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].rename(columns={'P_corr': 'CI_corr', 'P_kendall': 'CI_kendall', 'P_partial': 'CI_partial', 'P_tail': 'CI_tail'})
        df = df.join(ci_one, how='left')
        df['CI_corr_D'] = df['CI_corr'].shift(1)
        df['CI_kendall_D'] = df['CI_kendall'].shift(1)
        df['CI_partial_D'] = df['CI_partial'].shift(1)
        df['CI_tail_D'] = df['CI_tail'].shift(1)
        df = df.dropna()
        df = df.loc[SAMPLE_START:].copy()
        if df.empty:
            skipped.append((country, 'empty after dropna/SAMPLE_START'))
            continue
        if ANCHOR_DATE not in df.index:
            pos = df.index.searchsorted(ANCHOR_DATE)
            if pos <= 0 or pos >= len(df.index):
                skipped.append((country, 'cannot locate anchor in df'))
                continue
            anchor_date = df.index[pos]
        else:
            anchor_date = ANCHOR_DATE
        window_len = len(df.loc[:anchor_date]) - 1
        if window_len <= 30:
            skipped.append((country, f'window_len too small: {window_len}'))
            continue
        oos_start, oos_end = red_range[country]
        test = df.loc[oos_start:oos_end].copy()
        if test.empty:
            skipped.append((country, f'no test sample in red range {oos_start.date()}~{oos_end.date()}'))
            continue
        gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.02, max_depth=10, subsample=0.8, min_samples_split=50, random_state=32, validation_fraction=0.1, n_iter_no_change=10)
        preds = []
        for t in tqdm(test.index, desc=f'{country}-H22', leave=False):
            pos_t = df.index.get_loc(t)
            train_end = pos_t - H
            train_start = train_end - (window_len - 1)
            if train_start < 0 or train_end < 0 or train_start > train_end:
                preds.append(np.nan)
                continue
            train_slice = df.iloc[train_start:train_end + 1].copy()
            if len(train_slice) < 50:
                preds.append(np.nan)
                continue
            X_train = train_slice[X_cols].values
            y_train = train_slice[y_col].values
            try:
                gb.fit(X_train, y_train)
                x_row = df.loc[t, X_cols].values.reshape(1, -1)
                pred = float(gb.predict(x_row)[0])
                preds.append(pred if np.isfinite(pred) else np.nan)
            except Exception:
                preds.append(np.nan)
        y_pred = pd.Series(preds, index=test.index, name='RV_Predict').dropna()
        if y_pred.empty:
            skipped.append((country, 'all predictions nan'))
            continue
        y_true = test.loc[y_pred.index, y_col].astype(float)
        SUM = pd.concat([y_true.rename('RV_True'), y_pred], axis=1).dropna()
        if SUM.empty:
            skipped.append((country, 'empty after aligning y_true/y_pred'))
            continue
        SUM['MSE'] = (SUM['RV_True'] - SUM['RV_Predict']) ** 2
        SUM['MAE'] = np.abs(SUM['RV_True'] - SUM['RV_Predict'])
        MSE = float(np.mean((SUM['RV_True'] - SUM['RV_Predict']) ** 2))
        MAE = float(np.mean(np.abs(SUM['RV_True'] - SUM['RV_Predict'])))
        Theil = float(np.sqrt(np.mean((SUM['RV_True'] - SUM['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM['RV_True'] ** 2))))
        error_records[country] = [MSE, MAE, Theil]
        model_outputs[str(country).strip().lstrip('.')] = SUM.copy()
    if len(error_records) == 0:
        pass
    else:
        ERROR = pd.DataFrame(error_records, index=['MSE', 'MAE', 'Theil'])
    if skipped:
        for m, reason in skipped:
            pass
    return model_outputs


def run_rf(periods):
    model_outputs = {}
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.ensemble import RandomForestRegressor
    warnings.filterwarnings('ignore')
    H = 22
    FIT_START = pd.to_datetime('2014-07-04')
    ANCHOR_DATE = pd.to_datetime('2018-05-02')
    PLOT = False
    RV_PATH = RV_FILE
    P_STORE_PATH = CENTRALITY_FILE
    red_PATH = 'in_memory_bull_periods'
    OUT_DIR = 'unused_output'
    data = pd.read_csv(RV_PATH)
    if 'date' not in data.columns:
        raise RuntimeError('Model execution message')
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    for c in data.columns:
        if c != 'date':
            data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.sort_values('date')
    data = data.fillna(method='ffill')
    data = data.set_index('date')
    data = data.dropna(how='any')
    data = data.loc[FIT_START:]
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all = P_all.dropna(subset=['date', 'market']).copy()
    P_ci = P_all.groupby(['date', 'market'])[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].mean().reset_index()
    red = periods.copy()
    need_red_cols = ['filename', 'start_date', 'end_date']
    miss_red = [c for c in need_red_cols if c not in red.columns]
    if miss_red:
        raise RuntimeError(f'Model execution message{miss_red}Model execution message{list(red.columns)}')

    def parse_market_from_filename(x):
        x = str(x)
        m = re.match('^\\s*([.][A-Za-z0-9]+)', x)
        return m.group(1) if m else None
    red['market'] = red['filename'].apply(parse_market_from_filename)
    red['start_date'] = pd.to_datetime(red['start_date'], errors='coerce').dt.normalize()
    red['end_date'] = pd.to_datetime(red['end_date'], errors='coerce').dt.normalize()
    red = red.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red['span_days'] = (red['end_date'] - red['start_date']).dt.days
    red = red.sort_values(['market', 'span_days'], ascending=[True, False]).drop_duplicates('market', keep='first')
    red_map = {r['market']: (r['start_date'], r['end_date']) for _, r in red.iterrows()}
    error_rows = []
    error_index_markets = []
    skipped = []
    markets_to_run = [m for m in red['market'].tolist() if m in data.columns]
    for country in tqdm(markets_to_run, desc='RF markets'):
        data_RV = data[[country]].copy()
        data_RV['RV_now'] = data_RV[country]
        data_RV = data_RV.drop(columns=[country])
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = data_RV['RV_now'].rolling(window=H, center=False).mean().shift(-(H - 1))
        ci_one = P_ci[P_ci['market'] == country].copy()
        if ci_one.empty:
            skipped.append((country, 'Model execution message'))
            continue
        ci_one = ci_one.set_index('date')[['P_corr', 'P_kendall', 'P_partial', 'P_tail']]
        ci_one = ci_one.rename(columns={'P_corr': 'CI_corr', 'P_kendall': 'CI_kendall', 'P_partial': 'CI_partial', 'P_tail': 'CI_tail'})
        data_RV = data_RV.join(ci_one, how='left')
        data_RV['CI_corr_D'] = data_RV['CI_corr'].shift(1)
        data_RV['CI_kendall_D'] = data_RV['CI_kendall'].shift(1)
        data_RV['CI_partial_D'] = data_RV['CI_partial'].shift(1)
        data_RV['CI_tail_D'] = data_RV['CI_tail'].shift(1)
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[FIT_START:]
        if data_RV.empty:
            skipped.append((country, 'Model execution message'))
            continue
        X_all = data_RV[['RV_D', 'RV_W', 'RV_M', 'CI_corr_D', 'CI_kendall_D', 'CI_partial_D', 'CI_tail_D']].copy()
        y_all = data_RV['RV_MA22'].copy()
        if ANCHOR_DATE in data_RV.index:
            anchor_pos = int(data_RV.index.get_loc(ANCHOR_DATE))
        else:
            anchor_pos = int(data_RV.index.searchsorted(ANCHOR_DATE))
            if anchor_pos >= len(data_RV.index):
                skipped.append((country, 'Model execution message'))
                continue
        window_len = anchor_pos
        if window_len <= H:
            skipped.append((country, f'Model execution message{window_len}）'))
            continue
        pred_start, pred_end = red_map[country]
        if pred_end < pred_start:
            skipped.append((country, 'Model execution message'))
            continue
        test_dates = data_RV.loc[pred_start:pred_end].index
        if len(test_dates) == 0:
            skipped.append((country, f'Model execution message{pred_start.date()}~{pred_end.date()}'))
            continue
        rf = RandomForestRegressor(n_estimators=200, max_depth=1, random_state=42, n_jobs=-1)
        predictions = []
        y_true_list = []
        pred_index = []
        for t in tqdm(test_dates, desc=f'RF {country}', leave=False):
            pos = int(data_RV.index.get_loc(t))
            train_last_label_pos = pos - H
            if train_last_label_pos < 0:
                predictions.append(np.nan)
                y_true_list.append(np.nan)
                pred_index.append(t)
                continue
            train_start_pos = train_last_label_pos - window_len + 1
            train_end_exclusive = train_last_label_pos + 1
            if train_start_pos < 0:
                predictions.append(np.nan)
                y_true_list.append(np.nan)
                pred_index.append(t)
                continue
            current_train_X = X_all.iloc[train_start_pos:train_end_exclusive].copy()
            current_train_y = y_all.iloc[train_start_pos:train_end_exclusive].copy()
            current_train_X = current_train_X.dropna()
            current_train_y = current_train_y.loc[current_train_X.index].dropna()
            current_train_X = current_train_X.loc[current_train_y.index]
            x_one = X_all.loc[[t]].dropna()
            y_one = y_all.loc[t] if t in y_all.index else np.nan
            if current_train_X.empty or current_train_y.empty or x_one.empty or pd.isna(y_one):
                predictions.append(np.nan)
                y_true_list.append(np.nan)
                pred_index.append(t)
                continue
            rf.fit(current_train_X, current_train_y)
            pred = rf.predict(x_one)[0]
            predictions.append(float(pred))
            y_true_list.append(float(y_one))
            pred_index.append(t)
        SUM_final = pd.DataFrame({'RV_True': y_true_list, 'RV_Predict': predictions}, index=pd.DatetimeIndex(pred_index, name='date')).dropna()
        if SUM_final.empty:
            skipped.append((country, 'Model execution message'))
            continue
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        eps = 1e-12
        pred_safe = np.clip(SUM_final['RV_Predict'].values, eps, None)
        true_safe = np.clip(SUM_final['RV_True'].values, eps, None)
        SUM_final['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        MSE = float(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2))
        MAE = float(np.mean(np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])))
        denom = np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2))
        Theil = float(np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / denom) if denom != 0 else np.nan
        error_rows.append([MSE, MAE, Theil])
        error_index_markets.append(country)
        model_outputs[str(country).strip().lstrip('.')] = SUM_final.copy()
    if len(error_rows) > 0:
        ERROR = pd.DataFrame(error_rows, columns=['MSE', 'MAE', 'Theil'], index=error_index_markets).T
    if len(skipped) > 0:
        skipped_df = pd.DataFrame(skipped, columns=['market', 'reason'])
    return model_outputs


def run_deep_learning(periods):
    model_outputs = {'LSTM': {}, 'CNN': {}, 'Transformer': {}}
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, Dropout, Input, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    warnings.filterwarnings('ignore')
    np.random.seed(42)
    tf.random.set_seed(42)
    RV_PATH = RV_FILE
    P_STORE_PATH = CENTRALITY_FILE
    red_PATH = 'in_memory_bull_periods'
    OUT_LSTM = 'LSTM'
    OUT_CNN = 'CNN'
    OUT_TR = 'Transformer'
    H = 22
    FIT_START = pd.to_datetime('2014-07-04').normalize()
    n_steps = 1
    EPOCHS = 200
    BATCH_SIZE = 32
    PLOT = False
    data = pd.read_csv(RV_PATH)
    data = data.dropna(how='all')
    if 'date' not in data.columns:
        raise RuntimeError("RV file must contain a 'date' column.")
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.tz_localize(None).dt.normalize()
    data = data.fillna(method='ffill')
    data = data.set_index('date').sort_index()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='any')
    data = data.loc[FIT_START:]
    red_df = periods.copy()
    need_cols = {'filename', 'start_date', 'end_date'}
    if not need_cols.issubset(set(red_df.columns)):
        raise RuntimeError(f'Model execution message{need_cols}Model execution message{list(red_df.columns)}')

    def extract_market_code(x: str):
        m = re.match('^\\s*([.][A-Za-z0-9]+)', str(x))
        return m.group(1) if m else None
    red_df['market'] = red_df['filename'].apply(extract_market_code)
    red_df['start_date'] = pd.to_datetime(red_df['start_date'], errors='coerce').dt.normalize()
    red_df['end_date'] = pd.to_datetime(red_df['end_date'], errors='coerce').dt.normalize()
    red_df = red_df.dropna(subset=['market', 'start_date', 'end_date']).copy()
    red_df['span_days'] = (red_df['end_date'] - red_df['start_date']).dt.days
    red_df = red_df.sort_values(['market', 'span_days'], ascending=[True, False]).drop_duplicates('market', keep='first')
    red_range = {r['market']: (r['start_date'], r['end_date']) for _, r in red_df.iterrows()}
    table_markets = list(dict.fromkeys(red_df['market'].tolist()))
    markets_to_run = [m for m in table_markets if m in data.columns]
    if len(markets_to_run) == 0:
        raise RuntimeError('Model execution message')
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'Model execution message{miss}Model execution message{list(P_all.columns)}')
    P_all['date'] = pd.to_datetime(P_all['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    P_all = P_all.dropna(subset=['date', 'market']).copy()
    P_ci = P_all.groupby(['date', 'market'])[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].mean().reset_index()

    def build_lstm_model(input_shape):
        model = Sequential([LSTM(64, activation='relu', input_shape=input_shape), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_cnn_model(input_shape):
        model = Sequential([Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape), MaxPooling1D(pool_size=1), Flatten(), Dense(50, activation='relu'), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        return model

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        res = x + inputs
        x = Dense(ff_dim, activation='relu')(res)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    def build_transformer_model(input_shape):
        inputs = Input(shape=input_shape)
        x = transformer_encoder(inputs, head_size=32, num_heads=4, ff_dim=64, dropout=0.0)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def make_h22_target(rv_now: pd.Series) -> pd.Series:
        return rv_now.rolling(window=H, center=False).mean().shift(-(H + 1))

    def create_dataset_with_dates(X_df: pd.DataFrame, y_s: pd.Series, time_steps=1):
        Xs, ys, y_dates = ([], [], [])
        for i in range(len(X_df) - time_steps):
            Xs.append(X_df.iloc[i:i + time_steps].values)
            ys.append(y_s.iloc[i + time_steps])
            y_dates.append(y_s.index[i + time_steps])
        return (np.array(Xs), np.array(ys), pd.DatetimeIndex(y_dates))

    def calc_theil(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1, 1)
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        denom = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
        if denom == 0:
            return np.nan
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)

    def save_mcs_and_metrics(model, X_test, y_test_scaled, scaler_y, date_index, out_dir, country):
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
        y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
        mcs = pd.DataFrame({'RV_True': y_true, 'RV_Predict': y_pred}, index=pd.DatetimeIndex(date_index, name='date')).dropna()
        if len(mcs) == 0:
            return None
        mcs['MSE'] = (mcs['RV_True'] - mcs['RV_Predict']) ** 2
        mcs['MAE'] = np.abs(mcs['RV_True'] - mcs['RV_Predict'])
        eps = 1e-12
        pred_safe = np.clip(mcs['RV_Predict'].values, eps, None)
        true_safe = np.clip(mcs['RV_True'].values, eps, None)
        mcs['Qlike'] = true_safe / pred_safe + np.log(pred_safe)
        model_outputs[out_dir][str(country).strip().lstrip('.')] = mcs.copy()
        mse = float(mean_squared_error(mcs['RV_True'], mcs['RV_Predict']))
        mae = float(mean_absolute_error(mcs['RV_True'], mcs['RV_Predict']))
        theil = calc_theil(mcs['RV_True'].values, mcs['RV_Predict'].values)
        return [mse, mae, theil]
    error_lstm = {}
    error_cnn = {}
    error_transformer = {}
    skipped = []
    for country in tqdm(markets_to_run, desc='Markets'):
        oos_start, oos_end = red_range[country]
        data_RV = data[[country]].copy()
        data_RV['RV_now'] = data_RV[country]
        data_RV = data_RV.drop(columns=[country])
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = make_h22_target(data_RV['RV_now'])
        ci_one = P_ci[P_ci['market'] == country].copy()
        if ci_one.empty:
            skipped.append((country, 'missing CI in P.csv'))
            continue
        ci_one = ci_one.set_index('date')[['P_corr', 'P_kendall', 'P_partial', 'P_tail']].rename(columns={'P_corr': 'CI_corr', 'P_kendall': 'CI_kendall', 'P_partial': 'CI_partial', 'P_tail': 'CI_tail'})
        data_RV = data_RV.join(ci_one, how='left')
        data_RV['CI_corr_D'] = data_RV['CI_corr'].shift(1)
        data_RV['CI_kendall_D'] = data_RV['CI_kendall'].shift(1)
        data_RV['CI_partial_D'] = data_RV['CI_partial'].shift(1)
        data_RV['CI_tail_D'] = data_RV['CI_tail'].shift(1)
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc[FIT_START:]
        X = data_RV[['RV_D', 'RV_W', 'RV_M', 'CI_corr_D', 'CI_kendall_D', 'CI_partial_D', 'CI_tail_D']].copy()
        y = data_RV['RV_MA22'].copy()
        X_train_df = X.loc[X.index < oos_start].copy()
        y_train_s = y.loc[y.index < oos_start].copy()
        X_test_df = X.loc[(X.index >= oos_start) & (X.index <= oos_end)].copy()
        y_test_s = y.loc[(y.index >= oos_start) & (y.index <= oos_end)].copy()
        if len(X_train_df) <= n_steps + 10 or len(X_test_df) <= n_steps + 1:
            skipped.append((country, f'too short train/test for red ({oos_start.date()}~{oos_end.date()})'))
            continue
        X_context = X_train_df.iloc[-n_steps:].copy()
        y_context = y_train_s.iloc[-n_steps:].copy()
        X_test_for_seq = pd.concat([X_context, X_test_df], axis=0)
        y_test_for_seq = pd.concat([y_context, y_test_s], axis=0)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train_df), index=X_train_df.index, columns=X_train_df.columns)
        y_train_scaled = pd.Series(scaler_y.fit_transform(y_train_s.values.reshape(-1, 1)).reshape(-1), index=y_train_s.index)
        X_test_scaled = pd.DataFrame(scaler_X.transform(X_test_for_seq), index=X_test_for_seq.index, columns=X_test_for_seq.columns)
        y_test_scaled = pd.Series(scaler_y.transform(y_test_for_seq.values.reshape(-1, 1)).reshape(-1), index=y_test_for_seq.index)
        X_train_seq, y_train_seq, _ = create_dataset_with_dates(X_train_scaled, y_train_scaled, n_steps)
        X_test_seq, y_test_seq, test_dates = create_dataset_with_dates(X_test_scaled, y_test_scaled, n_steps)
        keep_test = (test_dates >= oos_start) & (test_dates <= oos_end)
        X_test_seq = X_test_seq[keep_test]
        y_test_seq = y_test_seq[keep_test]
        test_dates = test_dates[keep_test]
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            skipped.append((country, 'empty sequences after build'))
            continue
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
        lstm_model = build_lstm_model((n_steps, X_train_seq.shape[2]))
        lstm_model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0)
        m_lstm = save_mcs_and_metrics(lstm_model, X_test_seq, y_test_seq, scaler_y, test_dates, OUT_LSTM, country)
        if m_lstm is not None:
            error_lstm[country] = m_lstm
        cnn_model = build_cnn_model((n_steps, X_train_seq.shape[2]))
        cnn_model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        cnn_pred_scaled = cnn_model.predict(X_test_seq, verbose=0)
        m_cnn = save_mcs_and_metrics(cnn_model, X_test_seq, y_test_seq, scaler_y, test_dates, OUT_CNN, country)
        if m_cnn is not None:
            error_cnn[country] = m_cnn
        transformer_model = build_transformer_model((n_steps, X_train_seq.shape[2]))
        transformer_model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        tr_pred_scaled = transformer_model.predict(X_test_seq, verbose=0)
        m_tr = save_mcs_and_metrics(transformer_model, X_test_seq, y_test_seq, scaler_y, test_dates, OUT_TR, country)
        if m_tr is not None:
            error_transformer[country] = m_tr
    if len(error_lstm) > 0:
        pass
    if len(error_cnn) > 0:
        pass
    if len(error_transformer) > 0:
        pass
    if len(skipped) > 0:
        pass
    return model_outputs


def calc_mse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {'MSE': np.nan}
    err = yp - yt
    mse = float(np.mean(err ** 2))
    return {'MSE': mse}


def qlike_explosive_mask(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    df = pd.concat([y_true.rename('y'), y_pred.rename('yhat')], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=bool)
    y = df['y'].astype(float).values
    yhat = df['yhat'].astype(float).values
    idx = df.index
    ok_pos = (y > EPS) & (yhat > EPS)
    ratio = np.full_like(y, np.nan, dtype=float)
    ratio[ok_pos] = y[ok_pos] / yhat[ok_pos]
    q = np.full_like(y, np.nan, dtype=float)
    valid_ratio = ok_pos & np.isfinite(ratio) & (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
    q[valid_ratio] = ratio[valid_ratio] - np.log(ratio[valid_ratio]) - 1.0
    explosive = np.ones_like(y, dtype=bool)
    ok_q = np.isfinite(q) & (q >= 0.0) & (q <= QLIKE_CAP)
    explosive[ok_q] = False
    return pd.Series(explosive, index=idx)


def qlike_with_shared_drop(y_true: pd.Series, y_pred: pd.Series, drop_dates: set) -> float:
    df = pd.concat([y_true.rename('y'), y_pred.rename('yhat')], axis=1).dropna()
    if df.empty:
        return np.nan
    if drop_dates:
        df = df.loc[~df.index.isin(drop_dates)]
    if df.empty:
        return np.nan
    y = df['y'].astype(float).values
    yhat = df['yhat'].astype(float).values
    ok_pos = (y > EPS) & (yhat > EPS)
    if not np.any(ok_pos):
        return np.nan
    ratio = y[ok_pos] / yhat[ok_pos]
    ok_ratio = np.isfinite(ratio) & (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
    if not np.any(ok_ratio):
        return np.nan
    ratio = ratio[ok_ratio]
    q = ratio - np.log(ratio) - 1.0
    ok_q = np.isfinite(q) & (q >= 0.0) & (q <= QLIKE_CAP)
    if not np.any(ok_q):
        return np.nan
    return float(np.mean(q[ok_q]))


def add_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean_row = df.mean(axis=0, skipna=True)
    q1_row = df.quantile(0.25, axis=0, interpolation='linear')
    q2_row = df.quantile(0.5, axis=0, interpolation='linear')
    q3_row = df.quantile(0.75, axis=0, interpolation='linear')
    df2 = df.copy()
    df2.loc['Mean'] = mean_row
    df2.loc['Q1'] = q1_row
    df2.loc['Q2'] = q2_row
    df2.loc['Q3'] = q3_row
    return df2


def prepare_mcs_matrix_relaxed(sum_pre: pd.DataFrame, min_valid_per_model: int=10, min_rows_for_mcs: int=5):
    info = {'initial_shape': tuple(sum_pre.shape), 'dropped_low_valid_models': [], 'dropped_const_models': [], 'final_shape': None, 'reason': None}
    if sum_pre.empty or sum_pre.shape[1] == 0:
        info['reason'] = 'The loss matrix is empty'
        return (None, info)
    valid_counts = sum_pre.notna().sum(axis=0)
    keep_cols = valid_counts[valid_counts >= min_valid_per_model].index.tolist()
    info['dropped_low_valid_models'] = [c for c in sum_pre.columns if c not in keep_cols]
    if len(keep_cols) < 2:
        info['reason'] = f'Fewer than two models with at least {min_valid_per_model} valid observations'
        return (None, info)
    x = sum_pre[keep_cols].copy()
    x = x.dropna(axis=0, how='any')
    if x.empty:
        info['reason'] = 'No common valid observations'
        return (None, info)
    if x.shape[1] < 2:
        info['reason'] = 'Fewer than two models after dropping missing observations'
        return (None, info)
    if x.shape[0] < min_rows_for_mcs:
        info['reason'] = f'Insufficient common observations ({x.shape[0]} < {min_rows_for_mcs})'
        return (None, info)
    variances = x.var(axis=0, skipna=True)
    non_const_cols = variances[variances > 0].index.tolist()
    info['dropped_const_models'] = [c for c in x.columns if c not in non_const_cols]
    x = x[non_const_cols]
    if x.shape[1] < 2:
        info['reason'] = 'Fewer than two models after removing constant columns'
        return (None, info)
    if x.shape[0] < min_rows_for_mcs:
        info['reason'] = f'Insufficient common observations after removing constant columns ({x.shape[0]} < {min_rows_for_mcs})'
        return (None, info)
    info['final_shape'] = tuple(x.shape)
    return (x, info)


def choose_one_model_to_drop_for_retry(raw_sum_pre: pd.DataFrame, current_cols: list):
    if len(current_cols) <= 2:
        return None
    sub = raw_sum_pre[current_cols].copy()
    valid_counts = sub.notna().sum(axis=0)
    min_valid = valid_counts.min()
    candidates = valid_counts[valid_counts == min_valid].index.tolist()
    if len(candidates) == 1:
        return candidates[0]
    var_map = {}
    for c in candidates:
        s = pd.to_numeric(sub[c], errors='coerce').dropna()
        if len(s) <= 1:
            var_map[c] = -1.0
        else:
            v = float(s.var())
            if np.isnan(v):
                v = -1.0
            var_map[c] = v
    drop_col = sorted(var_map.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
    return drop_col


def run_mcs_with_retry(sum_pre: pd.DataFrame, country_code: str, min_valid_per_model: int, min_rows_for_mcs: int, max_retry_drop: int, debug_logs: list):
    details = {'country': country_code, 'retry_count': 0, 'dropped_in_retry': [], 'fail_reason': None, 'final_shape': None, 'final_models': None}
    if sum_pre.empty or sum_pre.shape[1] < 2:
        details['fail_reason'] = 'Fewer than two available models'
        return (False, None, None, None, details)
    current_cols = list(sum_pre.columns)
    for _ in range(max_retry_drop + 1):
        tmp_sum = sum_pre[current_cols].copy()
        mcs_df, prep_info = prepare_mcs_matrix_relaxed(tmp_sum, min_valid_per_model=min_valid_per_model, min_rows_for_mcs=min_rows_for_mcs)
        if mcs_df is None:
            if len(current_cols) <= 2:
                details['fail_reason'] = f"Preparation failed with too few models to retry: {prep_info.get('reason')}"
                return (False, None, None, None, details)
            drop_col = choose_one_model_to_drop_for_retry(sum_pre, current_cols)
            if drop_col is None:
                details['fail_reason'] = f"Preparation failed and no model can be removed: {prep_info.get('reason')}"
                return (False, None, None, None, details)
            current_cols.remove(drop_col)
            details['retry_count'] += 1
            details['dropped_in_retry'].append(drop_col)
            debug_logs.append(f"[{country_code}] Preparation failed; retry without model: {drop_col} | Reason: {prep_info.get('reason')}")
            continue
        try:
            mcs = MCS(losses=mcs_df, size=0.9, method='R', reps=20000, block_size=5)
            mcs.compute()
            details['final_shape'] = tuple(mcs_df.shape)
            details['final_models'] = list(mcs_df.columns)
            return (True, mcs.pvalues, mcs.included, mcs_df, details)
        except Exception as e:
            if len(current_cols) <= 2:
                details['fail_reason'] = f'MCS failed with too few models to retry: {str(e)}'
                return (False, None, None, None, details)
            drop_col = choose_one_model_to_drop_for_retry(sum_pre, current_cols)
            if drop_col is None:
                details['fail_reason'] = f'MCS failed and no model can be removed: {str(e)}'
                return (False, None, None, None, details)
            current_cols.remove(drop_col)
            details['retry_count'] += 1
            details['dropped_in_retry'].append(drop_col)
            debug_logs.append(f'[{country_code}] MCS failed; retry without model: {drop_col} | Error: {str(e)}')
            continue
    details['fail_reason'] = f'Maximum retry count exceeded: {max_retry_drop}'
    return (False, None, None, None, details)


def qlike_series_with_rules(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q = np.full(shape=len(y_true), fill_value=np.nan, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return q
    yt = y_true[mask]
    yp = y_pred[mask]
    ok_pos = (yt > EPS) & (yp > EPS)
    ratio = np.full_like(yt, np.nan, dtype=float)
    ratio[ok_pos] = yt[ok_pos] / yp[ok_pos]
    ok_ratio = np.isfinite(ratio) & (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
    q_sub = np.full_like(yt, np.nan, dtype=float)
    q_sub[ok_ratio] = ratio[ok_ratio] - np.log(ratio[ok_ratio]) - 1.0
    ok_q = np.isfinite(q_sub) & (q_sub >= 0.0) & (q_sub <= QLIKE_CAP)
    q_masked = np.full_like(yt, np.nan, dtype=float)
    q_masked[ok_q] = q_sub[ok_q]
    q[mask] = q_masked
    return q


def load_prediction_pair(frame):
    dates = pd.to_datetime(frame.index, errors="coerce")
    y_true = pd.Series(pd.to_numeric(frame["RV_True"], errors="coerce").to_numpy(), index=dates)
    y_pred = pd.Series(pd.to_numeric(frame["RV_Predict"], errors="coerce").to_numpy(), index=dates)
    valid = dates.notna() & np.isfinite(y_true.to_numpy()) & np.isfinite(y_pred.to_numpy())
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) and np.mean(y_true.to_numpy() > THRESH) > 0.5:
        y_true = y_true * SCALE_IF_MOST_GT_THRESH
        y_pred = y_pred * SCALE_IF_MOST_GT_THRESH
    y_true = y_true[~y_true.index.duplicated(keep="last")].sort_index()
    y_pred = y_pred[~y_pred.index.duplicated(keep="last")].sort_index()
    return y_true, y_pred


def load_mcs_drop_dates(markets):
    data = pd.read_csv(MCS_DROP_DATES_FILE)
    required = {"market", "date"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing QLIKE exclusion columns: {sorted(required - set(data.columns))}")
    data = data[["market", "date"]].copy()
    data["market"] = data["market"].astype(str).str.strip().str.lstrip(".")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any():
        raise ValueError("The QLIKE exclusion file contains invalid dates")
    if data.duplicated(["market", "date"]).any():
        raise ValueError("The QLIKE exclusion file contains duplicate market-date rows")
    file_markets = set(data["market"])
    expected_markets = set(markets)
    if file_markets != expected_markets:
        difference = sorted(file_markets.symmetric_difference(expected_markets))
        raise ValueError(f"QLIKE exclusion markets do not match the forecast markets: {difference}")
    return {market: set(group["date"]) for market, group in data.groupby("market")}


def qlike_loss_series(frame, drop_dates):
    data = pd.DataFrame({
        "date": pd.to_datetime(frame.index, errors="coerce"),
        "y_true": pd.to_numeric(frame["RV_True"], errors="coerce").to_numpy(),
        "y_pred": pd.to_numeric(frame["RV_Predict"], errors="coerce").to_numpy(),
    }).dropna(subset=["date"])
    if data["date"].dt.tz is not None:
        data["date"] = data["date"].dt.tz_localize(None)
    data["date"] = data["date"].dt.normalize()
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    data = data.loc[~data["date"].isin(drop_dates)]
    return pd.Series(qlike_series_with_rules(data["y_true"].to_numpy(), data["y_pred"].to_numpy()), index=data["date"])


def validate_model_output(label, result, markets):
    if set(result) != markets:
        raise RuntimeError(f"{label}: forecast markets do not match the input data")
    for market, frame in result.items():
        if frame.empty or not {'RV_True', 'RV_Predict', 'MSE'}.issubset(frame.columns):
            raise RuntimeError(f"{label}: no usable forecast output for {market}")
        if frame.index.has_duplicates:
            raise RuntimeError(f"{label}: duplicate forecast dates for {market}")


def save_table(summaries, counts, model_rows, output_path, title, bold_mode="best"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = title
    sheet.merge_cells("A1:E1")
    sheet.merge_cells("G1:K1")
    sheet["A1"] = "MSE"
    sheet["G1"] = "QLIKE"
    for start in (1, 7):
        for offset, label in enumerate(("Index\\Statistic", "Mean", "Q1", "Median", "MCS(#)")):
            sheet.cell(2, start + offset, label)
    for row, (key, label, _) in enumerate(model_rows, start=3):
        for start, metric in ((1, "MSE"), (7, "QLIKE")):
            sheet.cell(row, start, label)
            for offset, statistic in enumerate(("Mean", "Q1", "Q2"), start=1):
                value = float(summaries[metric].loc[key, statistic])
                if not np.isfinite(value):
                    raise ValueError(f"Cannot export a missing {metric} {statistic} for {key}")
                sheet.cell(row, start + offset, value).number_format = "0.000"
            sheet.cell(row, start + 4, int(counts[metric][key])).number_format = "0"
    thin = Side(style="thin", color="000000")
    for row in sheet:
        for cell in row:
            cell.font = Font(name="Times New Roman", size=11, bold=False)
            cell.alignment = Alignment(horizontal="left" if cell.column in (1, 7) and cell.row > 1 else "center", vertical="center")
            if cell.column != 6:
                if cell.row == 1:
                    cell.border = Border(top=thin, bottom=thin)
                elif cell.row in (2, 18):
                    cell.border = Border(bottom=thin)
    for start, metric in ((1, "MSE"), (7, "QLIKE")):
        for row, (key, _, _) in enumerate(model_rows, start=3):
            for offset, stat in enumerate(("Mean", "Q1", "Q2", "MCS(#)"), start=1):
                bold = bold_mode == "har-ci" and key == "HAR-CI-new"
                if bold_mode == "best":
                    bold = counts[metric][key] == max(counts[metric].values()) if offset == 4 else np.isclose(summaries[metric].loc[key, stat], summaries[metric][stat].min(), rtol=0, atol=1e-12)
                if bold:
                    sheet.cell(row, start + offset).font = Font(name="Times New Roman", size=11, bold=True)
    for column in ("A", "G"):
        sheet.column_dimensions[column].width = 23
    for column in ("B", "C", "D", "E", "H", "I", "J", "K"):
        sheet.column_dimensions[column].width = 12
    sheet.column_dimensions["F"].width = 3
    sheet.freeze_panes = "B3"
    sheet.sheet_view.showGridLines = False
    sheet.print_options.horizontalCentered = True
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 1
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_area = "A1:K18"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


C03_MODELS = [
    ('HAR-RV', 'HAR-RV', run_har),
    ('HAR-CI-new', 'HAR-CI', run_har_ci),
    ('ks', 'HAR-KS', run_har_ks),
    ('mean', 'MC', run_mc),
    ('median', 'MDC', run_mdc),
    ('tmean', 'TMC', run_tmc),
    ('dm0.9', 'DMSPE(0.9)', run_dmspe_09),
    ('dm1', 'DMSPE(1)', run_dmspe_1),
    ('pca1', 'PCA1', run_pca1),
    ('PCA_BIC', 'PCABIC', run_pca_bic),
    ('pls', 'PLS', run_pls),
    ('ridge', 'Ridge', run_ridge),
    ('lasso', 'Lasso', run_lasso),
    ('ENET', 'ENET', run_enet),
    ('spillover-t', 'Spillover-Time', run_spillover_time),
    ('spillover-f', 'Spillover-Frequency', run_spillover_frequency),
]
C04_MODELS = [
    ('LogHAR', 'LogHAR-RV', run_loghar),
    ('LogHAR-CI', 'LogHAR-CI', run_loghar_ci),
    ('HExp-GlRV', 'Hexp-Gl', run_hexp),
    ('MVF-Log', 'MVF', run_mvf),
    ('Multivariate-LogHAR', 'Multivariate', run_multivariate),
]
C05_MODELS = [
    ('HAR-RV', 'HAR-RV', run_har),
    ('HAR-CI-new', 'HAR-CI', run_har_ci),
    ('bagging', 'Bagging', run_bagging),
    ('gradient', 'Gradient', run_gradient),
    ('RF_red', 'RF', run_rf),
    ('CNN', 'CNN', None),
    ('LSTM', 'LSTM', None),
    ('Transformer', 'Transformer', None),
]
MCS_ORDERS = {
    3: ['dm0.9', 'dm1', 'ENET', 'HAR-CI-new', 'HAR-RV', 'ks', 'lasso', 'mean', 'median', 'pca1', 'PCA_BIC', 'pls', 'ridge', 'spillover-f', 'spillover-t', 'tmean'],
    4: ['HExp-GlRV', 'LogHAR', 'LogHAR-CI', 'Multivariate-LogHAR', 'MVF-Log'],
    5: ['bagging', 'CNN', 'gradient', 'HAR-CI-new', 'HAR-RV', 'LSTM', 'RF_red', 'Transformer'],
}
RF_REFERENCE_MARKETS = {'AEX', 'AORD', 'BFX', 'BSESN', 'BVSP', 'FCHI', 'GDAXI', 'GSPTSE', 'IXIC', 'KS11'}
OUTPUTS = {
    3: ROOT / 'results/generated/Table_C_03_bull_market_category1.xlsx',
    4: ROOT / 'results/generated/Table_C_04_bull_market_category2.xlsx',
    5: ROOT / 'results/generated/Table_C_05_bull_market_category3.xlsx',
}


def compute_summary_drop_dates(predictions):
    excluded = {'CNN', 'Transformer', 'LSTM', 'HAR-RV', 'mean', 'median', 'tmean', 'dm0.9', 'dm1'}
    display_names = {'mean': 'Mean'}
    markets = sorted(predictions['HAR-RV'])
    dates = {market: set() for market in markets}
    for model, values in predictions.items():
        if display_names.get(model, model) in excluded:
            continue
        for market in markets:
            if market not in values:
                continue
            y_true, y_pred = load_prediction_pair(values[market])
            invalid = qlike_explosive_mask(y_true, y_pred)
            dates[market].update(invalid.index[invalid.to_numpy()])
    return dates


def summarize_table(predictions_by_metric, model_rows, summary_drop_dates, table_number):
    summaries = {}
    for metric, predictions in predictions_by_metric.items():
        markets = sorted(predictions['HAR-RV'])
        values = pd.DataFrame(index=markets, columns=list(predictions), dtype=float)
        for market in markets:
            for model in predictions:
                if market not in predictions[model]:
                    continue
                y_true, y_pred = load_prediction_pair(predictions[model][market])
                if metric == 'MSE':
                    loss = calc_mse(y_true.to_numpy(), y_pred.to_numpy())['MSE']
                else:
                    loss = qlike_with_shared_drop(y_true, y_pred, summary_drop_dates[market])
                values.loc[market, model] = loss
        values = values.dropna(how='all').dropna(subset=['HAR-RV'], how='all')
        relative = values.div(values['HAR-RV'].replace(0.0, np.nan), axis=0)
        relative = relative.where(~(np.abs(relative) > POST_SCALE_THRESHOLD), relative * POST_SCALE_FACTOR)
        statistics = add_summary_rows(relative).loc[['Mean', 'Q1', 'Q2']]
        if table_number == 4:
            original_baseline = statistics['LogHAR'].copy()
            statistics = statistics.div(original_baseline.replace(0.0, np.nan), axis=0)
            statistics['LogHAR'] = 1.0
        summaries[metric] = statistics[[key for key, _, _ in model_rows]].T
    return summaries


def count_mcs_memberships(predictions_by_metric, mcs_drop_dates, table_number):
    models = MCS_ORDERS[table_number]
    counts = {metric: dict.fromkeys(models, 0) for metric in ('MSE', 'QLIKE')}
    for metric in counts:
        predictions = predictions_by_metric[metric]
        for market in tqdm(sorted(predictions['HAR-RV'])):
            losses = pd.DataFrame()
            for model in models:
                if market not in predictions[model]:
                    continue
                frame = predictions[model][market]
                if metric == 'MSE':
                    series = pd.to_numeric(frame['MSE'], errors='coerce').reset_index(drop=True)
                else:
                    series = qlike_loss_series(frame, mcs_drop_dates[market])
                if series.notna().sum():
                    losses[model] = series
            logs = []
            success, _, included, used, details = run_mcs_with_retry(losses, market, 10, 5, 50, logs)
            if not success:
                raise RuntimeError(f"Table C.{table_number}, MCS {metric}, {market}: {details['fail_reason']}")
            for model in included:
                counts[metric][model] += 1
    return counts


def finish_table(table_number, predictions_by_metric, model_rows, mcs_drop_dates):
    summary_drop_dates = compute_summary_drop_dates(predictions_by_metric['QLIKE'])
    summaries = summarize_table(predictions_by_metric, model_rows, summary_drop_dates, table_number)
    counts = count_mcs_memberships(predictions_by_metric, mcs_drop_dates, table_number)
    save_table(summaries, counts, model_rows, OUTPUTS[table_number], f'Table C.{table_number}')


def main():
    for path in (PRICE_FILE, RV_FILE, RETURN_FILE, CENTRALITY_FILE, LOG_CENTRALITY_FILE, TCI_FILE, MCS_DROP_DATES_FILE):
        if not path.is_file():
            raise FileNotFoundError(f'Required input not found: {path}')
    all_markets = [str(c).strip().lstrip('.') for c in pd.read_csv(RV_FILE, nrows=0).columns if c != 'date']
    mcs_drop_dates = load_mcs_drop_dates(all_markets)
    periods = compute_bull_market_periods(PRICE_FILE, RV_FILE)
    periods['filename'] = periods['market']
    markets = set(periods['market'].str.lstrip('.'))
    c03_mse = {}
    for key, label, runner in tqdm(C03_MODELS):
        result = runner(periods)
        validate_model_output(label, result, markets)
        c03_mse[key] = result
    c03_qlike = dict(c03_mse)
    c03_qlike['HAR-CI-new'] = run_har_ci_qlike(periods)
    c03_qlike['spillover-t'] = run_spillover_time_qlike(periods)
    for key in ('HAR-CI-new', 'spillover-t'):
        validate_model_output(key + ' QLIKE', c03_qlike[key], markets)
    finish_table(3, {'MSE': c03_mse, 'QLIKE': c03_qlike}, C03_MODELS, mcs_drop_dates)
    c05_mse = {key: c03_mse[key] for key in ('HAR-RV', 'HAR-CI-new')}
    c05_qlike = {key: c03_qlike[key] for key in ('HAR-RV', 'HAR-CI-new')}
    del c03_mse, c03_qlike
    c04 = {'HAR-RV': run_har_c04(periods)}
    validate_model_output('HAR C.4 baseline', c04['HAR-RV'], markets)
    for key, label, runner in tqdm(C04_MODELS):
        result = runner(periods)
        validate_model_output(label, result, markets)
        c04[key] = result
    finish_table(4, {'MSE': c04, 'QLIKE': c04}, C04_MODELS, mcs_drop_dates)
    del c04
    for key, label, runner in tqdm(C05_MODELS[2:5]):
        selected_markets = RF_REFERENCE_MARKETS if key == 'RF_red' else markets
        selected_periods = periods.loc[periods['market'].str.lstrip('.').isin(selected_markets)].copy()
        result = runner(selected_periods)
        validate_model_output(label, result, selected_markets)
        c05_mse[key] = result
        c05_qlike[key] = result
    deep_outputs = run_deep_learning(periods)
    for key in ('CNN', 'LSTM', 'Transformer'):
        validate_model_output(key, deep_outputs[key], markets)
        c05_mse[key] = deep_outputs[key]
        c05_qlike[key] = deep_outputs[key]
    finish_table(5, {'MSE': c05_mse, 'QLIKE': c05_qlike}, C05_MODELS, mcs_drop_dates)


if __name__ == '__main__':
    main()

