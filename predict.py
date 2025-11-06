
import os, json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def make_summary():
    df = pd.read_csv("facts.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 週次（月曜）に揃える
    wk = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON")})
    df = wk.merge(df, on="date", how="left")
    df["orders"] = df["orders"].ffill()
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # 8週先を“ざっくり”逐次予測
    future_weeks = 8
    history = df.copy()
    cur_date = history["date"].max() + pd.offsets.Week(weekday=0)
    preds = []

    for _ in range(future_weeks):
        ma4 = float(np.mean(history["orders"].tail(4))) if len(history) >= 4 else float(np.mean(history["orders"]))
        target_week = int(pd.Timestamp(cur_date).isocalendar().week)
        same_week = history[history["weekofyear"] == target_week]
        seasonal = float(same_week["orders"].iloc[-52]) if len(same_week) >= 52 else ma4

        yhat = 0.6 * seasonal + 0.4 * ma4
        band = max(0.15 * yhat, 1.0)  # ±15%帯
        low, high = yhat - band, yhat + band

        preds.append({"ds": pd.Timestamp(cur_date), "yhat": yhat, "yhat_lower": low, "yhat_upper": high})

        # 次週へ（予測値を履歴に接続）
        history = pd.concat([
            history,
            pd.DataFrame({"date":[pd.Timestamp(cur_date)], "orders":[yhat], "weekofyear":[target_week]})
        ], ignore_index=True)
        cur_date = cur_date + pd.offsets.Week(weekday=0)

    last = preds[-1]
    now_utc = datetime.now(timezone.utc).isoformat()

    summary = {
        "run_id": now_utc,
        "period": pd.to_datetime(last["ds"]).strftime("%Y-W%U"),
        "orders_forecast": int(round(last["yhat"])),
        "orders_low": int(round(last["yhat_lower"])),
        "orders_high": int(round(last["yhat_upper"])),
        "top_drivers": ["前年同週（あれば）","直近4週平均"],
        "updated_at": now_utc,
        "source_url": ""
    }
    return summary

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    summary = make_summary()
    pd.DataFrame([summary]).to_json("reports/forecast_summary.json", orient="records", force_ascii=False)
    pd.DataFrame([summary]).to_csv("reports/forecast_summary.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False))
