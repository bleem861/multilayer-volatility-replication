import argparse
from pathlib import Path

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

H = 22
TIME_OOS = "2018-05-02"
TIME_STOP = "2022-02-25"
TRAIN_START = "2014-07-04"

ALPHAS = [0.01, 0.05, 0.1]

name = [0.011, 0.051, 0.11]

N_BOOT = 5000
EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument('--recomputed-centrality', action='store_true', help='Use recomputed centrality indices.')
args = parser.parse_args()
P_STORE_PATH = (
    PROJECT_ROOT / "results" / "generated" / "centrality" / "P.csv"
    if args.recomputed_centrality
    else PROJECT_ROOT / "data" / "processed" / "centrality" / "P.csv"
)
FIT_END_DATE = pd.to_datetime("2018-05-01")
PRED_END = pd.to_datetime(TIME_STOP)

L_fit = 1000
L_net = 1000

rv = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "RV.csv")
num_cols = rv.columns.drop("date")
rv[num_cols] = rv[num_cols].apply(pd.to_numeric, errors="coerce")
rv = rv.dropna()
rv["date"] = pd.to_datetime(rv["date"], utc=True)
rv["date"] = rv["date"].dt.tz_localize(None).dt.normalize()
rv = rv.set_index("date")

ret_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Return.csv")
num_cols_r = ret_df.columns.drop("date")
ret_df[num_cols_r] = ret_df[num_cols_r].apply(pd.to_numeric, errors="coerce")
ret_df = ret_df.dropna()
ret_df["date"] = pd.to_datetime(ret_df["date"], utc=True)
ret_df["date"] = ret_df["date"].dt.tz_localize(None).dt.normalize()
ret_df = ret_df.set_index("date")

def _norm_date(x):
    return pd.to_datetime(x).tz_localize(None).normalize()

P_all = pd.read_csv(P_STORE_PATH)
need_cols = [
    "date", "market", "ci_start_date", "ci_end_date", "L_net",
    "P_corr", "P_kendall", "P_partial", "P_tail"
]
miss = [c for c in need_cols if c not in P_all.columns]
if miss:
    raise RuntimeError(f"P.csv is missing required columns: {miss}. Available columns: {list(P_all.columns)}")

P_all["date"] = pd.to_datetime(P_all["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
P_all["ci_start_date"] = P_all["ci_start_date"].apply(_norm_date)
P_all["ci_end_date"] = P_all["ci_end_date"].apply(_norm_date)
P_all["L_net"] = pd.to_numeric(P_all["L_net"], errors="coerce").astype("Int64")
P_all = P_all.dropna(subset=["date", "market", "ci_start_date", "ci_end_date", "L_net"]).copy()

P_cache = {}

def load_P_from_csv(L_net: int, ci_start_date, ci_end_date):
    key = (int(L_net), _norm_date(ci_start_date), _norm_date(ci_end_date))
    if key in P_cache:
        return P_cache[key]

    ln, cs, ce = key
    sub = P_all[
        (P_all["L_net"] == ln) &
        (P_all["ci_start_date"] == cs) &
        (P_all["ci_end_date"] == ce)
    ]

    if sub.empty:
        P_cache[key] = (None, None, None, None)
        return P_cache[key]

    def _pivot(col):
        df = sub.pivot_table(index="date", columns="market", values=col, aggfunc="mean")
        return df.sort_index()

    P_corr = _pivot("P_corr")
    P_kendall = _pivot("P_kendall")
    P_partial = _pivot("P_partial")
    P_tail = _pivot("P_tail")

    P_cache[key] = (P_corr, P_kendall, P_partial, P_tail)
    return P_cache[key]

class varbacktest:
    def __init__(self, returns, VaR, alpha, hit_lags=1, forecast_lags=1):
        self.index = returns.index
        self.returns = returns.values
        self.VaR = VaR.values
        self.alpha = alpha
        self.hit_lags = hit_lags
        self.forecast_lags = forecast_lags
        if len(returns) != len(VaR):
            raise ValueError("Returns and VaR series must have the same lengths")

    def serie_hits(self):
        return (self.returns < self.VaR) * 1

    def num_hits(self):
        return self.serie_hits().sum()

    def pct_hits(self):
        return self.serie_hits().mean()

    def actual_expected(self):
        N = len(self.returns)
        x = self.num_hits()
        return x / (self.alpha * N)

    def tick_loss(self, return_mean=True):
        loss = (self.alpha - self.serie_hits()) * (self.returns - self.VaR)
        return loss.mean() if return_mean else loss

    def firm_loss(self, c=1, return_mean=True):
        loss = (
            self.serie_hits() * (1 + (self.returns - self.VaR) ** 2)
            - c * (1 - self.serie_hits()) * self.VaR
        )
        return loss.mean() if return_mean else loss

    def quadratic_loss(self, return_mean=True):
        loss = self.serie_hits() * (1 + (self.returns - self.VaR) ** 2)
        return loss.mean() if return_mean else loss

    def uc(self):
        N = len(self.returns)
        x = self.num_hits()

        if x == 0:
            LRuc = -2 * N * np.log(1 - self.alpha)
        elif x < N:
            LRuc = -2 * (
                (N - x) * np.log(N * (1 - self.alpha) / (N - x))
                + x * np.log(N * self.alpha / x)
            )
        else:
            LRuc = -2 * N * np.log(self.alpha)

        PVuc = 1 - stats.chi2.cdf(LRuc, 1)
        UC = "accept" if PVuc > self.alpha else "reject"
        return pd.Series([LRuc, PVuc, UC], index=["LRuc", "PVuc", "UC"], name="UC_test")

    def cci(self):
        hits = self.serie_hits()
        tr = hits[1:] - hits[:-1]
        n01 = (tr == 1).sum()
        n10 = (tr == -1).sum()
        n11 = (hits[1:][tr == 0] == 1).sum()
        n00 = (hits[1:][tr == 0] == 0).sum()

        LogLNum = 0
        if (n00 + n10) > 0 and (n01 + n11) > 0:
            pUC = (n01 + n11) / (n00 + n01 + n10 + n11)
            LogLNum = (n00 + n10) * np.log(1 - pUC) + (n01 + n11) * np.log(pUC)

        LogLDen = 0
        if n00 > 0 and n01 > 0:
            p01 = n01 / (n00 + n01)
            LogLDen += n00 * np.log(1 - p01) + n01 * np.log(p01)
        if n10 > 0 and n11 > 0:
            p11 = n11 / (n10 + n11)
            LogLDen += n10 * np.log(1 - p11) + n11 * np.log(p11)

        LRcci = -2 * (LogLNum - LogLDen)
        PVcci = 1 - stats.chi2.cdf(LRcci, 1)
        CCI = "accept" if PVcci > self.alpha else "reject"
        return pd.Series([LRcci, PVcci, CCI], index=["LRcci", "PVcci", "CCI"], name="CCI_test")

    def cc(self):
        LRuc = self.uc()["LRuc"]
        LRcci = self.cci()["LRcci"]
        LRcc = LRuc + LRcci
        PVcc = 1 - stats.chi2.cdf(LRcc, 2)
        CC = "accept" if PVcc > self.alpha else "reject"
        return pd.Series([LRcc, PVcc, CC], index=["LRcc", "PVcc", "CC"], name="CC_test")

    def dq(self):
        try:
            hits = self.serie_hits()
            p, q, n = self.hit_lags, self.forecast_lags, hits.size
            pq = max(p, q - 1)
            y = hits[pq:] - self.alpha
            x = np.zeros((n - pq, 1 + p + q))
            x[:, 0] = 1

            for i in range(p):
                x[:, 1 + i] = hits[pq - (i + 1): -(i + 1)]

            for j in range(q):
                if j > 0:
                    x[:, 1 + p + j] = self.VaR[pq - j: -j]
                else:
                    x[:, 1 + p + j] = self.VaR[pq:]

            beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
            DQtest = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha * (1 - self.alpha))
            PVdq = 1 - stats.chi2.cdf(DQtest, 1 + p + q)
        except:
            DQtest, PVdq = np.nan, np.nan

        DQ = "accept" if (PVdq > self.alpha) else "reject"
        return pd.Series([DQtest, PVdq, DQ], index=["DQtest", "PVdq", "DQ"], name="DQ_test")

    def summary(self):
        df = pd.DataFrame(
            {
                "VaR_lvl": self.alpha,
                "obs": len(self.returns),
                "num_hits": self.num_hits(),
                "pct_hits": [self.pct_hits()],
                "actual_expected": self.actual_expected(),
                "LRuc": self.uc()["LRuc"],
                "PVuc": self.uc()["PVuc"],
                "UC": self.uc()["UC"],
                "LRcci": self.cci()["LRcci"],
                "PVcci": self.cci()["PVcci"],
                "CCI": self.cci()["CCI"],
                "LRcc": self.cc()["LRcc"],
                "PVcc": self.cc()["PVcc"],
                "CC": self.cc()["CC"],
                "DQtest": self.dq()["DQtest"],
                "PVdq": self.dq()["PVdq"],
                "DQ": self.dq()["DQ"],
                "firm_loss": self.firm_loss(),
                "quadratic_loss": self.quadratic_loss(),
                "tick_loss": self.tick_loss(),
            }
        )
        return df

def make_future_mean(rv_now: pd.Series, H: int) -> pd.Series:
    if H == 1:
        return rv_now
    ma = rv_now.rolling(window=H, center=False).mean()
    return ma.shift(-(H - 1))

def make_future_mean_return(ret: pd.Series, H: int) -> pd.Series:
    return ret.shift(-1).rolling(H).mean().shift(-(H - 1))

for alpha, alpha_name in zip(ALPHAS, name):

    base_folder = PROJECT_ROOT / "results" / "generated" / "Fig_04_VaR" / f"VaR_{alpha_name}"
    model_name = "HAR-CI-new"
    model_dir = os.path.join(base_folder, model_name)
    os.makedirs(model_dir, exist_ok=True)

    for c in rv.columns:
        if c not in ret_df.columns:
            continue

        if FIT_END_DATE not in rv.index:
            continue

        fit_end_pos = rv.index.get_loc(FIT_END_DATE)
        fit_start_pos = fit_end_pos - (L_fit - 1)
        if fit_start_pos < 0:
            continue

        fit_start_date_for_P = rv.index[fit_start_pos]

        ret_pos_end = ret_df.index.searchsorted(fit_start_date_for_P) - 1
        if ret_pos_end < 0:
            continue

        net_end_date = ret_df.index[ret_pos_end]
        ci_start_date = net_end_date
        ci_end_date = PRED_END

        P_corr, P_kendall, P_partial, P_tail = load_P_from_csv(
            L_net=L_net,
            ci_start_date=ci_start_date,
            ci_end_date=ci_end_date
        )

        if P_corr is None:
            continue
        if c not in P_corr.columns:
            continue
        if c not in P_kendall.columns or c not in P_partial.columns or c not in P_tail.columns:
            continue

        data_model = pd.DataFrame(rv[c].copy())
        data_model.columns = ["RV_now"]

        data_model["RV_D"] = data_model["RV_now"].shift(1)
        data_model["RV_W"] = data_model["RV_D"].rolling(5).mean()
        data_model["RV_M"] = data_model["RV_D"].rolling(22).mean()

        data_model["RV_MA22"] = make_future_mean(data_model["RV_now"], H)

        ci_df = pd.DataFrame({
            "CI_corr": P_corr[c],
            "CI_kendall": P_kendall[c],
            "CI_partial": P_partial[c],
            "CI_tail": P_tail[c],
        }).reindex(data_model.index)

        data_model = data_model.join(ci_df, how="inner")

        data_model["CI_corr_D"] = data_model["CI_corr"].shift(1)
        data_model["CI_kendall_D"] = data_model["CI_kendall"].shift(1)
        data_model["CI_partial_D"] = data_model["CI_partial"].shift(1)
        data_model["CI_tail_D"] = data_model["CI_tail"].shift(1)

        data_model = data_model.join(ret_df[c].rename("ret"), how="inner")

        data_model["R_22_mean_true"] = make_future_mean_return(data_model["ret"], H)

        data_model = data_model.dropna()
        data_model = data_model.loc[TRAIN_START:].copy()

        if pd.to_datetime(TIME_OOS) not in data_model.index:
            continue

        data_test = data_model.loc[TIME_OOS:TIME_STOP].copy()
        if len(data_test) == 0:
            continue

        base_len = len(data_model.loc[:TIME_OOS])
        if base_len < 300:
            continue

        var_pred_list = []
        pred_date_list = []
        oos_index = data_test.index

        for i in tqdm(range(len(data_test)), desc="Computing VaR forecasts"):
            data_rolling = data_model.iloc[i: i + (base_len - 1)].copy()

            if len(data_rolling) < 50:
                var_pred_list.append(np.nan)
                pred_date_list.append(oos_index[i])
                continue

            mod = smf.ols(
                "RV_MA22 ~ RV_D + RV_W + RV_M + CI_corr_D + CI_kendall_D + CI_partial_D + CI_tail_D",
                data_rolling
            )
            res = mod.fit()
            p = res.params

            rv_hat = (
                p.iloc[0]
                + p.iloc[1] * data_test["RV_D"].iloc[i]
                + p.iloc[2] * data_test["RV_W"].iloc[i]
                + p.iloc[3] * data_test["RV_M"].iloc[i]
                + p.iloc[4] * data_test["CI_corr_D"].iloc[i]
                + p.iloc[5] * data_test["CI_kendall_D"].iloc[i]
                + p.iloc[6] * data_test["CI_partial_D"].iloc[i]
                + p.iloc[7] * data_test["CI_tail_D"].iloc[i]
            )
            rv_hat = max(float(rv_hat), EPS)

            mu = float(data_rolling["ret"].mean())
            denom = np.sqrt(data_rolling["RV_now"].clip(lower=EPS))
            eps = ((data_rolling["ret"] - mu) / denom).replace([np.inf, -np.inf], np.nan).dropna()

            if len(eps) < 200:
                var_pred_list.append(np.nan)
                pred_date_list.append(oos_index[i])
                continue

            eps_vals = eps.values
            idx_boot = np.random.randint(0, len(eps_vals), size=(N_BOOT, H))
            mean_eps_22 = eps_vals[idx_boot].mean(axis=1)
            q_alpha = float(np.quantile(mean_eps_22, alpha))

            VaR22_mean = mu + q_alpha * np.sqrt(rv_hat)

            var_pred_list.append(VaR22_mean)
            pred_date_list.append(oos_index[i])

        VaR_pred = pd.Series(var_pred_list, index=pred_date_list, name="VaR_22_mean_pred")

        R22_true = data_test["R_22_mean_true"].reindex(VaR_pred.index).rename("R_22_mean_true")

        daily_df = pd.concat([R22_true, VaR_pred], axis=1).dropna().copy()
        if len(daily_df) < 30:
            continue

        daily_df["hit"] = (daily_df["R_22_mean_true"] < daily_df["VaR_22_mean_pred"]).astype(int)
        daily_df["alpha"] = alpha
        daily_df["alpha_name"] = alpha_name
        daily_df["market"] = c
        daily_df.index.name = "date"
        daily_df = daily_df.reset_index()

        df_bt = daily_df.set_index("date")[["R_22_mean_true", "VaR_22_mean_pred"]].copy()
        df_bt.columns = ["y", "VaR"]

        a = varbacktest(df_bt["y"], df_bt["VaR"], alpha, hit_lags=1, forecast_lags=1)
        FINAL = a.summary()

        out_path = os.path.join(model_dir, f"{c}.csv")
        FINAL.to_csv(out_path, index=False, encoding="utf-8-sig")

        daily_path = os.path.join(model_dir, f"{c}_daily.csv")
        daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
