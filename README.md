# Koukou and his friends' brilliant ideas

#散户情绪与股市变化成正相关，本项目旨在构建数理经济模型，让散户投资有依靠。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp, log
from scipy.stats import norm

# ---------- 数据 ----------
df = pd.read_csv("gme_HLOC.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
px = df["GME.Close"].astype(float)

# ---------- 参数 ----------
r = 0.01         # 无风险利率
borrow = 0.20    # 借券费率（年化）
MMR = 0.30       # 维持保证金率
Q_short0 = 10_000
t0_short = "2021-01-11"
t1_flip  = "2021-01-25"
t2_cover = "2021-01-27"
t3_re_short = "2021-02-24"
end_date = "2021-03-10"

K = 100.0
iv = 3.0
T_days = 30

# ---------- Black-Scholes ----------
def bsm_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def margin_call_trigger(equity, short_val, mmr):
    mv = abs(short_val)
    return equity < mmr * mv

dates = px.loc[t0_short:end_date].index

# ---------- 策略1：持续做空 ----------
def run_S1():
    cash, Q = 0.0, Q_short0
    short_cost_accrued = 0.0
    pnl_series = []
    final_equity = None
    for d in dates:
        S = px.loc[d]
        short_cost_accrued += (borrow / 252) * (S * Q)
        short_val = -Q * S
        equity = cash + short_val - short_cost_accrued
        forced = margin_call_trigger(equity, short_val, MMR)
        if forced or d >= pd.to_datetime(t2_cover):
            cash += short_val  # 回补平仓
            final_equity = cash - short_cost_accrued
            pnl_series.append((d, final_equity))
            break
        pnl_series.append((d, equity))
    s = pd.Series(dict(pnl_series), name="S1_PnL")
    if final_equity is not None:
        s = s.reindex(dates).ffill().fillna(final_equity)
    return s

# ---------- 策略2：反手做多 ----------
def run_S2():
    cash, Q = 0.0, Q_short0
    pnl_series = []
    contracts = 100
    premium0, t_start = None, None
    for d in dates:
        S = px.loc[d]
        if d < pd.to_datetime(t1_flip):
            pnl_series.append((d, cash - Q * S))
        elif d == pd.to_datetime(t1_flip):
            cash += -(-Q) * S  # 平空
            T = max((pd.to_datetime(t1_flip) + pd.Timedelta(days=T_days) - d).days, 0) / 252
            premium0 = bsm_call(S, K, T, r, iv)
            t_start = d
            cash -= premium0 * contracts
            pnl_series.append((d, cash))
        else:
            T = max((t_start + pd.Timedelta(days=T_days) - d).days, 0) / 252
            price = bsm_call(S, K, T, r, iv)
            pnl_series.append((d, cash + (price - premium0) * contracts))
    return pd.Series(dict(pnl_series), name="S2_PnL")

# ---------- 策略3：2/24 起再建空 ----------
def run_S3():
    cash, Q = 0.0, 0
    pnl_series = []
    steps = 5
    add_each = Q_short0 // steps
    for d in dates:
        S = px.loc[d]
        if d >= pd.to_datetime(t3_re_short) and Q < Q_short0:
            Q += add_each
        short_val = -Q * S
        pnl_series.append((d, cash + short_val))
    return pd.Series(dict(pnl_series), name="S3_PnL")

# ---------- 执行 ----------
s1 = run_S1()
s2 = run_S2()
s3 = run_S3()
out = pd.concat([s1, s2, s3], axis=1)

# ---------- 策略指标 ----------
def strategy_metrics(series: pd.Series, name=""):
    s = series.dropna()
    if s.empty:
        return {"Strategy": name, "Final PnL": np.nan, "Max Drawdown": np.nan}
    final_pnl = s.iloc[-1]
    cummax = s.cummax()
    drawdown = (s - cummax)
    max_dd = drawdown.min()
    return {
        "Strategy": name,
        "Final PnL": round(final_pnl, 2),
        "Max Drawdown": round(max_dd, 2),
        "Peak Date": cummax.idxmax().strftime("%Y-%m-%d"),
        "End Date": s.index[-1].strftime("%Y-%m-%d")
    }

metrics = pd.DataFrame([
    strategy_metrics(s1, "S1: 持续做空"),
    strategy_metrics(s2, "S2: 反手做多"),
    strategy_metrics(s3, "S3: 再建空")
])

print(metrics)

# ---------- 可视化 ----------
plt.figure(figsize=(10, 6))
out.plot(ax=plt.gca())
plt.title("PnL Comparison of Three Strategies")
plt.ylabel("PnL (USD)")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.show()





