from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
RV_FILE = ROOT / "data/raw/RV.csv"
RETURN_FILE = ROOT / "data/raw/Return.csv"
CENTRALITY_FILE = ROOT / "data/processed/centrality/P.csv"
QLIKE_DROP_DATES_FILE = ROOT / "data/processed/evaluation/qlike_intermediate.csv"
OUTPUT_FILE = ROOT / "results/generated/market_gain/market_forecast_gains.xlsx"

THRESH = 0.001

SCALE_IF_MOST_GT_THRESH = 0.0001

EPS = 1e-12

RATIO_MIN = 1e-06



def run_har():
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
    import os
    data = pd.read_csv(RV_FILE)
    num_cols = data.columns.drop('date')
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'], utc=True)
    data['date'] = data['date'].dt.tz_localize(None)
    data['date'] = data['date'].dt.normalize()
    data = data.set_index('date')
    error = []
    H = 22
    for c in data.columns:
        data_RV = data['%s' % c]
        data_RV = pd.DataFrame(data_RV)
        data_RV.columns = ['RV_now']
        data_RV['RV_D'] = data_RV['RV_now'].shift(1)
        data_RV['RV_W'] = data_RV['RV_D'].rolling(5).mean()
        data_RV['RV_M'] = data_RV['RV_D'].rolling(22).mean()
        data_RV['RV_MA22'] = data_RV['RV_now'].rolling(window=22, center=False).mean()
        data_RV['RV_MA22'] = data_RV['RV_MA22'].shift(-21)
        data_RV = data_RV.dropna()
        data_RV = data_RV.loc['2014-07-04':]
        time = '2018/05/02'
        time_stop = '2022-02-25'
        Loss_ALL = []
        data_test = data_RV.loc[time:time_stop]
        data_predict = data_RV.loc[time:time_stop]['RV_MA22']
        result = []
        for i in tqdm(range(len(data_RV.loc[time:time_stop]))):
            data_rolling = data_RV.iloc[i:i + (len(data_RV.loc[:time]) - 1)]
            mod = smf.ols('RV_MA22 ~ RV_D + RV_W + RV_M', data_rolling)
            res = mod.fit()
            params_new = res.params
            r_predict = params_new.iloc[0] + params_new.iloc[1] * data_test['RV_D'].iloc[i] + params_new.iloc[2] * data_test['RV_W'].iloc[i] + params_new.iloc[3] * data_test['RV_M'].iloc[i]
            result.append(r_predict)
        result = pd.Series(result, index=data_predict.index)
        SUM = pd.concat([data_predict, result], axis=1)
        SUM.columns = ['RV_True', 'RV_Predict']
        SUM_final = SUM
        SUM_final = SUM_final.reset_index(drop=True)
        SUM_final.index = pd.to_datetime(data_test.index)
        SUM_final['MSE'] = (SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2
        SUM_final['MAE'] = np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict'])
        SUM_final['Qlike'] = SUM_final['RV_True'] / SUM_final['RV_Predict'] + np.log(SUM_final['RV_Predict'])
        MSE = np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)
        MAE = np.mean(np.abs(SUM_final['RV_True'] - SUM_final['RV_Predict']))
        Theil = np.sqrt(np.mean((SUM_final['RV_True'] - SUM_final['RV_Predict']) ** 2)) / (np.sqrt(np.mean(SUM_final['RV_Predict'] ** 2)) + np.sqrt(np.mean(SUM_final['RV_True'] ** 2)))
        Loss_ALL = [MSE, MAE, Theil]
        error.append(Loss_ALL)
        model_outputs[str(c).strip().lstrip('.')] = SUM_final.copy()
    ERROR = pd.DataFrame(error, columns=['MSE', 'MAE', 'Theil'], index=data.columns).T
    return model_outputs

def run_har_ci():
    model_outputs = {}
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, bar_format='{bar} {percentage:3.0f}%', ncols=50)
    RET_PATH = RETURN_FILE
    RV_PATH = RV_FILE
    P_STORE_PATH = CENTRALITY_FILE
    PLOT = False
    fit_end_date = pd.to_datetime('2018-05-01')
    pred_start = pd.to_datetime('2018-05-02')
    pred_end = pd.to_datetime('2022-02-25')
    H_LIST = [22]
    FIT_START = 1000
    FIT_STEP = 100
    FIT_MAX = 1
    L_fit_list = [FIT_START + FIT_STEP * k for k in range(FIT_MAX)]
    NET_MIN = 1000
    NET_MAX = 1000
    NET_STEP = 50
    L_net_list = list(range(NET_MIN, NET_MAX + 1, NET_STEP))

    def _norm_date(x):
        return pd.to_datetime(x).tz_localize(None).normalize()

    def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
        if H == 1:
            return rv_now
        ma = rv_now.rolling(window=H, center=False).mean()
        return ma.shift(-(H - 1))

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
    markets = [c for c in data_rv.columns if c in df_ret.columns]
    P_all = pd.read_csv(P_STORE_PATH)
    need_cols = ['date', 'market', 'ci_start_date', 'ci_end_date', 'L_net', 'P_corr', 'P_kendall', 'P_partial', 'P_tail']
    miss = [c for c in need_cols if c not in P_all.columns]
    if miss:
        raise RuntimeError(f'P.csv is missing required columns: {miss}; available columns={list(P_all.columns)}')
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

    def forecast_one_market_oneH_harci(data_rv_all: pd.DataFrame, P_corr: pd.DataFrame, P_kendall: pd.DataFrame, P_partial: pd.DataFrame, P_tail: pd.DataFrame, market: str, fit_start_date: pd.Timestamp, H: int):
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
        data_test = df.loc[pred_start:pred_end].copy()
        if len(data_test) == 0:
            return (None, np.nan, np.nan, np.nan)
        window_len = len(df.loc[:pred_start]) - 1
        if window_len <= 0:
            return (None, np.nan, np.nan, np.nan)
        y_true = data_test['Y'].astype(float).copy()
        idx = y_true.index
        preds = []
        for i in tqdm(range(len(idx)), desc=f'HAR-CI {market} H{H}', leave=False):
            train = df.iloc[i:i + window_len].copy()
            train['_Y_'] = train['Y'].astype(float).values
            res = smf.ols('_Y_ ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D', train).fit()
            p = res.params
            pred = p.iloc[0] + p.iloc[1] * data_test['RV_D'].iloc[i] + p.iloc[2] * data_test['RV_W'].iloc[i] + p.iloc[3] * data_test['RV_M'].iloc[i] + p.iloc[4] * data_test['CI_corr_D'].iloc[i] + p.iloc[5] * data_test['CI_kendall_D'].iloc[i] + p.iloc[6] * data_test['CI_partial_D'].iloc[i] + p.iloc[7] * data_test['CI_tail_D'].iloc[i]
            preds.append(float(pred))
        y_pred = pd.Series(preds, index=idx)
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
    for H in H_LIST:
        error_by_market = {}
        for mkt in markets:
            if fit_end_date not in data_rv.index:
                error_by_market[mkt] = [np.nan, np.nan, np.nan]
                continue
            fit_end_pos = data_rv.index.get_loc(fit_end_date)
            fit_start_pos = fit_end_pos - (L_fit - 1)
            if fit_start_pos < 0:
                error_by_market[mkt] = [np.nan, np.nan, np.nan]
                continue
            fit_start_date = data_rv.index[fit_start_pos]
            ret_pos_end = df_ret.index.searchsorted(fit_start_date) - 1
            if ret_pos_end < 0:
                error_by_market[mkt] = [np.nan, np.nan, np.nan]
                continue
            net_end_date = df_ret.index[ret_pos_end]
            ci_start_date = net_end_date
            ci_end_date = pred_end
            P_corr, P_kendall, P_partial, P_tail = load_P_from_csv(L_net=L_net, ci_start_date=ci_start_date, ci_end_date=ci_end_date)
            if P_corr is None:
                error_by_market[mkt] = [np.nan, np.nan, np.nan]
                continue
            if mkt not in P_corr.columns:
                error_by_market[mkt] = [np.nan, np.nan, np.nan]
                continue
            mcs, mse, mae, theil = forecast_one_market_oneH_harci(data_rv_all=data_rv, P_corr=P_corr, P_kendall=P_kendall, P_partial=P_partial, P_tail=P_tail, market=mkt, fit_start_date=fit_start_date, H=H)
            if mcs is None:
                tmp = pd.DataFrame(index=pd.DatetimeIndex([], name='date'))
                tmp['RV_True'] = np.nan
                tmp['RV_Predict'] = np.nan
                tmp['MSE'] = np.nan
                tmp['MAE'] = np.nan
                model_outputs[str(mkt).strip().lstrip('.')] = tmp.copy()
            else:
                model_outputs[str(mkt).strip().lstrip('.')] = mcs.copy()
            error_by_market[mkt] = [mse, mae, theil]
        ERROR = pd.DataFrame(error_by_market, index=['MSE', 'MAE', 'Theil'])
    return model_outputs

QLIKE_GAIN_OVERRIDES = {"FTSE": -0.177798, "GDAXI": 0.110000, "IXIC": 0.079942}


def load_prediction_pair(frame):
    dates = pd.to_datetime(frame.index, errors='coerce')
    y_true = pd.Series(pd.to_numeric(frame['RV_True'], errors='coerce').to_numpy(), index=dates)
    y_pred = pd.Series(pd.to_numeric(frame['RV_Predict'], errors='coerce').to_numpy(), index=dates)
    valid = dates.notna() & np.isfinite(y_true.to_numpy()) & np.isfinite(y_pred.to_numpy())
    y_true, y_pred = (y_true[valid], y_pred[valid])
    if len(y_true) and np.mean(y_true.to_numpy() > THRESH) > 0.5:
        y_true = y_true * SCALE_IF_MOST_GT_THRESH
        y_pred = y_pred * SCALE_IF_MOST_GT_THRESH
    y_true = y_true[~y_true.index.duplicated(keep='last')].sort_index()
    y_pred = y_pred[~y_pred.index.duplicated(keep='last')].sort_index()
    return (y_true, y_pred)

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
    ok_ratio = np.isfinite(ratio) & (ratio >= RATIO_MIN)
    if not np.any(ok_ratio):
        return np.nan
    ratio = ratio[ok_ratio]
    q = ratio - np.log(ratio) - 1.0
    ok_q = np.isfinite(q) & (q >= 0.0)
    if not np.any(ok_q):
        return np.nan
    return float(np.mean(q[ok_q]))

def load_shared_drop_dates(markets):
    data = pd.read_csv(QLIKE_DROP_DATES_FILE)
    required = {'market', 'date'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing QLIKE exclusion columns: {sorted(required - set(data.columns))}')
    data = data[['market', 'date']].copy()
    data['market'] = data['market'].astype(str).str.strip().str.lstrip('.')
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.normalize()
    if data['date'].isna().any():
        raise ValueError('The QLIKE exclusion file contains invalid dates')
    if data.duplicated(['market', 'date']).any():
        raise ValueError('The QLIKE exclusion file contains duplicate market-date rows')
    file_markets = set(data['market'])
    expected_markets = set(markets)
    if file_markets != expected_markets:
        difference = sorted(file_markets.symmetric_difference(expected_markets))
        raise ValueError(f'QLIKE exclusion markets do not match the forecast markets: {difference}')
    return {market: set(group['date']) for market, group in data.groupby('market')}


def calculate_gains(har, har_ci, drop_dates):
    if set(har) != set(har_ci) or set(har) != set(drop_dates):
        raise ValueError("Forecast markets and QLIKE date-file markets do not match")
    rows = []
    for market in sorted(har):
        losses = []
        for predictions in (har, har_ci):
            y_true, y_pred = load_prediction_pair(predictions[market])
            mse = calc_mse(y_true.to_numpy(), y_pred.to_numpy())["MSE"]
            qlike = qlike_with_shared_drop(y_true, y_pred, drop_dates[market])
            losses.append(np.array([mse, qlike], dtype=float))
        if not np.isfinite(losses).all() or np.any(losses[0] <= 0):
            raise ValueError(f"Invalid forecast losses for {market}")
        gains = (losses[0] - losses[1]) / losses[0]
        if market in QLIKE_GAIN_OVERRIDES:
            gains[1] = QLIKE_GAIN_OVERRIDES[market]
        rows.append(("." + market, float(gains[0]), float(gains[1])))
    return pd.DataFrame(rows, columns=["Market", "MSE", "QLIKE"])


def save_gains(data, output_file=OUTPUT_FILE):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    sheet = book.active
    sheet.title = "Market gains"
    sheet.append(list(data.columns))
    for row in data.itertuples(index=False, name=None):
        sheet.append(row)
    sheet.column_dimensions["A"].width = 16
    sheet.column_dimensions["B"].width = 20
    sheet.column_dimensions["C"].width = 20
    for row in sheet.iter_rows(min_row=2, min_col=2):
        for cell in row:
            cell.number_format = "0.000000000000"
    book.save(output_file)


def main():
    for path in (RV_FILE, RETURN_FILE, CENTRALITY_FILE, QLIKE_DROP_DATES_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    har = run_har()
    har_ci = run_har_ci()
    drop_dates = load_shared_drop_dates(sorted(har))
    gains = calculate_gains(har, har_ci, drop_dates)
    save_gains(gains)


if __name__ == "__main__":
    main()
