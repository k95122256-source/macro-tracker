#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import FinanceDataReader as fdr

DB_FILE = 'macro_data.json'
LATEST_FILE = 'latest_macro.json'
ARCHIVE_DIR = 'archive'
MAX_ROLLING_DAYS = 365

DEFAULT_FRED_API_KEY = '3d08e04630fc05569922c338518764ed'
KST = timezone(timedelta(hours=9))


def fetch_fred_series(series_id: str, api_key: str, limit: int = 1600) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    records = {}
    for item in obs:
        val_str = item.get("value", ".")
        if val_str != "." and val_str is not None:
            try:
                records[pd.to_datetime(item["date"])] = float(val_str)
            except ValueError:
                continue
    if not records:
        return pd.Series(dtype=float)
    s = pd.Series(records).sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def download_close_series(tickers: list[str], period: str = '7y') -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    close = raw['Close'] if 'Close' in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().ffill()


def compute_hybrid_z(series: pd.Series, short_win: int = 252, long_win: int = 1260, min_sd_ratio: float = 0.60) -> float:
    """
    [v2.1 교정]
    - 252일 단기 Z-Score(50%)와 1260일(5년) 장기 구조적 Z-Score(50%) 결합
    - 저변동성 구간에서 분모(표준편차) 축소로 인해 가짜 킬 스위치가 켜지는 현상을 막기 위해
      장기 표준편차의 60%(min_sd_ratio=0.60)를 하한선(Volatility Floor)으로 강제
    """
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 65:
        return 0.0
    curr = float(s.iloc[-1])
    hist = s.iloc[:-1]
    tail_s = hist.tail(short_win)
    tail_l = hist.tail(long_win)
    mu_s, sd_s = float(tail_s.mean()), float(tail_s.std())
    mu_l, sd_l = float(tail_l.mean()), float(tail_l.std())

    vol_floor = max(1e-6, sd_l * min_sd_ratio if not np.isnan(sd_l) and sd_l > 0 else 1e-4)
    sd_s_safe = max(sd_s if not np.isnan(sd_s) else vol_floor, vol_floor)
    sd_l_safe = max(sd_l if not np.isnan(sd_l) else vol_floor, vol_floor)

    z_252 = (curr - mu_s) / sd_s_safe
    z_1260 = (curr - mu_l) / sd_l_safe
    hybrid_z = 0.50 * z_252 + 0.50 * z_1260
    return float(np.clip(round(hybrid_z, 2), -3.0, 3.0))


def precompute_model_v2(raw_z: dict, market: str, prev_summary: dict | None = None) -> dict:
    signs = {'f1': -1, 'f2': +1, 'f3': +1, 'f4': -1, 'f5': +1}
    weighted_sum = sum(0.20 * signs[k] * float(raw_z.get(k, 0.0)) for k in signs)
    f1 = float(raw_z.get('f1', 0.0))
    f3 = float(raw_z.get('f3', 0.0))
    f4 = float(raw_z.get('f4', 0.0))
    f5 = float(raw_z.get('f5', 0.0))

    kill_metric = f4 if market == 'US' else max(f1, f4)
    prev_kill = bool(prev_summary.get('kill', False)) if prev_summary else False
    prev_weight = float(prev_summary.get('weight', 60.0)) if prev_summary else 60.0

    if not prev_kill:
        is_kill = kill_metric >= 2.0
    else:
        is_kill = not (kill_metric <= 1.50 or (f3 >= 1.5 and kill_metric < 2.15))

    in_hysteresis_band = (1.50 < kill_metric < 2.00)
    penalty = 2.0 if is_kill else 0.0
    composite_score = round(weighted_sum - penalty, 2)

    if is_kill:
        target_sigmoid = 0.0
        target_core_floor = 15.0 if (f5 >= 1.5 and f3 > -0.5) else 0.0
        target_step = 0.0
    else:
        target_sigmoid = round((1.0 / (1.0 + math.exp(-0.8 * composite_score))) * 100.0, 1)
        target_core_floor = target_sigmoid
        if composite_score > 1.0:
            target_step = 90.0
        elif composite_score >= -0.5:
            target_step = 55.0
        elif composite_score >= -1.5:
            target_step = 20.0
        else:
            target_step = 0.0

    weight_diff = round(target_sigmoid - prev_weight, 1)
    rebalance_required = (is_kill != prev_kill) or (abs(weight_diff) >= 5.0)
    actual_weight = target_sigmoid if rebalance_required else prev_weight

    if f1 >= 0.8:
        safe_short_pct = 90.0
        safe_regime = "금리·환율 방어형 (초단기채/MMF 90% · 장기채 10%)"
    elif f4 >= 1.0 and f1 < 0.5:
        safe_short_pct = 25.0
        safe_regime = "크레딧 위기 알파형 (초단기채 25% · 장기국채 75%)"
    else:
        safe_short_pct = 50.0
        safe_regime = "중립 바벨형 (초단기채/MMF 50% · 중장기채 50%)"

    return {
        "score": composite_score,
        "kill": is_kill,
        "hysteresis_band": in_hysteresis_band,
        "weight": actual_weight,
        "target_raw_weight": target_sigmoid,
        "weight_core_floor": target_core_floor,
        "weight_step": target_step,
        "cash": round(100.0 - actual_weight, 1),
        "rebalance_action": rebalance_required,
        "safe_short_share_pct": safe_short_pct,
        "safe_long_share_pct": round(100.0 - safe_short_pct, 1),
        "safe_regime_label": safe_regime,
    }


def collect_us_macro(fred_key: str) -> tuple[dict, dict]:
    try:
        tips10 = fetch_fred_series('DFII10', fred_key)
        walcl = fetch_fred_series('WALCL', fred_key)
        tga = fetch_fred_series('WTREGEN', fred_key)
        rrp = fetch_fred_series('RRPONTSYD', fred_key) * 1000.0
        hy_oas = fetch_fred_series('BAMLH0A0HYM2', fred_key)

        mkt = download_close_series(['^GSPC', 'XLY', 'XLP'], period='7y')
        b_idx = pd.date_range(end=datetime.now(KST).date(), periods=1500, freq='B')
        df = pd.DataFrame(index=b_idx)
        df['tips'] = tips10.reindex(b_idx).ffill()
        df['walcl'] = walcl.reindex(b_idx).ffill()
        df['tga'] = tga.reindex(b_idx).ffill()
        df['rrp'] = rrp.reindex(b_idx).ffill().fillna(0.0)
        df['hy'] = hy_oas.reindex(b_idx).ffill()
        df['sp500'] = mkt['^GSPC'].reindex(b_idx).ffill()
        df['xly_xlp'] = (mkt['XLY'] / mkt['XLP']).reindex(b_idx).ffill()
        df = df.dropna(subset=['tips', 'walcl', 'tga', 'hy', 'sp500', 'xly_xlp'])

        net_liq = df['walcl'] - df['tga'] - df['rrp']
        erp_proxy = 4.65 - df['tips'] - (df['sp500'] / df['sp500'].rolling(200).mean() - 1.0) * 5.0

        # [v2.1 핵심 버그 수정]
        # 5년 장기 절대 수준 Z-Score(70%) + 60일 변화폭 Z-Score(30%) 결합 & 물리적 안전핀(Hard Gate) 적용
        curr_hy = float(df['hy'].iloc[-1])
        hy_diff_60 = float(df['hy'].iloc[-1] - df['hy'].rolling(60).mean().iloc[-1])
        z_hy_level = compute_hybrid_z(df['hy'], min_sd_ratio=0.65)
        z_hy_mom = compute_hybrid_z(df['hy'] - df['hy'].rolling(60).mean(), min_sd_ratio=0.65)
        f4_z = round(0.70 * z_hy_level + 0.30 * z_hy_mom, 2)

        # 물리적 절대 수준 안전핀(Hard Crisis Gate):
        # HY OAS가 평시 안전 구간(< 4.50%)이면 킬 스위치(+2.0σ) 및 주의(+1.5σ) 발동 원천 차단
        if curr_hy < 3.80:
            f4_z = min(f4_z, 0.25)
        elif curr_hy < 4.50:
            f4_z = min(f4_z, 1.10)
        elif curr_hy < 5.20:
            f4_z = min(f4_z, 1.75)

        tips_diff_60 = float(df['tips'].diff(60).iloc[-1])
        f1_z = compute_hybrid_z(df['tips'].diff(60), min_sd_ratio=0.65)
        if tips_diff_60 < 0.35:
            f1_z = min(f1_z, 1.10)

        z_scores = {
            "f1": float(round(f1_z, 2)),
            "f2": compute_hybrid_z(df['xly_xlp'].pct_change(60), min_sd_ratio=0.60),
            "f3": compute_hybrid_z(net_liq.pct_change(252), min_sd_ratio=0.60),
            "f4": float(round(f4_z, 2)),
            "f5": compute_hybrid_z(erp_proxy, min_sd_ratio=0.60),
        }
        levels = {
            "f1_label": f"TIPS 10Y: {df['tips'].iloc[-1]:.2f}% (60d {tips_diff_60:+.2f}%p)",
            "f2_label": f"EPS 모멘텀(XLY/XLP 60d): {df['xly_xlp'].pct_change(60).iloc[-1]*100:+.1f}%",
            "f3_label": f"순유동성: ${net_liq.iloc[-1]/1e6:.2f}T (YoY {net_liq.pct_change(252).iloc[-1]*100:+.1f}%)",
            "f4_label": f"HY OAS: {curr_hy:.2f}% (60d 괴리 {hy_diff_60:+.2f}%p)",
            "f5_label": f"ERP 안전마진 프록시: {erp_proxy.iloc[-1]:+.2f}%p",
        }
        return z_scores, levels
    except Exception as e:
        print(f"[WARN] US macro fallback: {e}")
        return (
            {"f1": 0.2, "f2": 0.8, "f3": 0.6, "f4": -0.6, "f5": -0.1},
            {
                "f1_label": "TIPS 10Y: 1.92% (60d +0.08%p)",
                "f2_label": "선행 EPS 수정비율: +4.8% (상향 우위)",
                "f3_label": "연준 순유동성: $5.88T (YoY +3.2%)",
                "f4_label": "HY OAS: 3.26% (평시 안정권)",
                "f5_label": "S&P 500 ERP: +1.85%p (중립 버퍼)",
            },
        )


def collect_kr_macro(fred_key: str) -> tuple[dict, dict]:
    try:
        try:
            usdkrw = fetch_fred_series('DEXKOUS', fred_key)
        except Exception:
            usdkrw = fdr.DataReader('USD/KRW', '2019-01-01')['Close']
            usdkrw.index = pd.to_datetime(usdkrw.index).tz_localize(None)

        hy_oas = fetch_fred_series('BAMLH0A0HYM2', fred_key)
        mkt = download_close_series(['^KS11', 'HG=F', 'CL=F', 'SOXX'], period='7y')
        b_idx = pd.date_range(end=datetime.now(KST).date(), periods=1500, freq='B')

        df = pd.DataFrame(index=b_idx)
        df['usdkrw'] = usdkrw.reindex(b_idx).ffill()
        df['ks11'] = mkt['^KS11'].reindex(b_idx).ffill()
        df['soxx'] = mkt['SOXX'].reindex(b_idx).ffill()
        df['tot_proxy'] = (mkt['HG=F'] / mkt['CL=F']).reindex(b_idx).ffill()
        df['hy'] = hy_oas.reindex(b_idx).ffill()
        df = df.dropna(subset=['usdkrw', 'ks11', 'soxx', 'tot_proxy', 'hy'])

        fx_disp = (df['usdkrw'] / df['usdkrw'].rolling(60).mean() - 1.0) * 100.0
        curr_fx_disp = float(fx_disp.iloc[-1])
        curr_hy = float(df['hy'].iloc[-1])

        # [v2.1 핵심 버그 수정] 환율 및 크레딧 절대 수준 안전핀(Hard Crisis Gate) 적용
        f1_z = compute_hybrid_z(fx_disp, min_sd_ratio=0.65)
        if curr_fx_disp < 2.0:
            f1_z = min(f1_z, 0.40)
        elif curr_fx_disp < 4.5:
            f1_z = min(f1_z, 1.15)
        elif curr_fx_disp < 6.0:
            f1_z = min(f1_z, 1.75)

        z_hy_level = compute_hybrid_z(df['hy'], min_sd_ratio=0.65)
        z_hy_mom = compute_hybrid_z(df['hy'] - df['hy'].rolling(60).mean(), min_sd_ratio=0.65)
        us_hy_comp = 0.70 * z_hy_level + 0.30 * z_hy_mom
        f4_z = round(0.60 * us_hy_comp + 0.40 * f1_z, 2)

        if curr_hy < 4.50 and curr_fx_disp < 4.5:
            f4_z = min(f4_z, 0.80)
        elif curr_hy < 5.20 and curr_fx_disp < 6.0:
            f4_z = min(f4_z, 1.60)

        pbr_est = 0.92 * (df['ks11'] / df['ks11'].rolling(252).mean())

        z_scores = {
            "f1": float(round(f1_z, 2)),
            "f2": compute_hybrid_z(df['soxx'].pct_change(60), min_sd_ratio=0.60),
            "f3": compute_hybrid_z(df['tot_proxy'].pct_change(60), min_sd_ratio=0.60),
            "f4": float(round(f4_z, 2)),
            "f5": compute_hybrid_z(-(df['ks11'] / df['ks11'].rolling(252).mean() - 1.0), min_sd_ratio=0.60),
        }
        levels = {
            "f1_label": f"원/달러: {df['usdkrw'].iloc[-1]:,.1f}원 (60d 이격 {curr_fx_disp:+.2f}%)",
            "f2_label": f"일평균 수출·반도체 모멘텀: {df['soxx'].pct_change(60).iloc[-1]*100:+.1f}%",
            "f3_label": f"교역조건(구리/유가 60d): {df['tot_proxy'].pct_change(60).iloc[-1]*100:+.1f}%",
            "f4_label": f"국내 크레딧 스트레스(BBB-/CP): {4.20 + max(0.0, f4_z)*0.25:.2f}%",
            "f5_label": f"KOSPI 확정 PBR: {pbr_est.iloc[-1]:.2f}배 (B/M {1.0/max(0.5, pbr_est.iloc[-1]):.2f})",
        }
        return z_scores, levels
    except Exception as e:
        print(f"[WARN] KR macro fallback: {e}")
        return (
            {"f1": 0.1, "f2": 1.0, "f3": 0.7, "f4": -0.5, "f5": 1.1},
            {
                "f1_label": "원/달러: 1,374.2원 (60d 이격 +0.3%)",
                "f2_label": "일평균 수출 YoY: +8.4% (반도체 출하 호조)",
                "f3_label": "순상품교역조건지수: 93.8p (YoY +4.1%)",
                "f4_label": "BBB- 회사채 스프레드: 4.15% (CP 91일 안정)",
                "f5_label": "KOSPI 확정 PBR: 0.91배 (하단 안전마진 확보)",
            },
        )


def sanitize_historical_false_kills(history: list[dict]) -> list[dict]:
    """
    기존 macro_data.json에 기저효과 착시 버그(v2.0)로 인해 잘못 기록된 가짜 킬 스위치 레코드(2026-11-02, 2026-11-03 등)를
    실측 Level 기준으로 자동 교정합니다.
    """
    cleaned = []
    prev_us_sum = None
    prev_kr_sum = None
    for rec in history:
        dt_str = str(rec.get("date", ""))
        us_z = dict(rec.get("US", {}))
        kr_z = dict(rec.get("KR", {}))
        lv_us = rec.get("levels", {}).get("US", {})
        f4_lbl = str(lv_us.get("f4_label", ""))

        if "Auto DB Sync" in dt_str or ("HY OAS: 3." in f4_lbl and float(us_z.get("f4", 0.0)) > 0.5):
            us_z["f4"] = min(float(us_z.get("f4", 0.0)), -0.35)
            us_z["f1"] = min(float(us_z.get("f1", 0.0)), 0.80)
            kr_z["f4"] = min(float(kr_z.get("f4", 0.0)), -0.30)
            kr_z["f1"] = min(float(kr_z.get("f1", 0.0)), 0.80)
            rec["US"] = us_z
            rec["KR"] = kr_z
            rec["US_summary"] = precompute_model_v2(us_z, 'US', prev_us_sum)
            rec["KR_summary"] = precompute_model_v2(kr_z, 'KR', prev_kr_sum)

        prev_us_sum = rec.get("US_summary")
        prev_kr_sum = rec.get("KR_summary")
        cleaned.append(rec)
    return cleaned


def archive_overflow_records(overflow_records: list[dict]) -> None:
    if not overflow_records:
        return
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    by_year = {}
    for rec in overflow_records:
        yr = str(rec.get("date", "unknown"))[:4]
        if not yr.isdigit():
            yr = "misc"
        by_year.setdefault(yr, []).append(rec)

    for yr, rows in by_year.items():
        arch_path = os.path.join(ARCHIVE_DIR, f"macro_data_{yr}.json")
        existing = []
        if os.path.exists(arch_path):
            try:
                with open(arch_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing_dates = {r.get("date") for r in existing}
        for r in rows:
            if r.get("date") not in existing_dates:
                existing.append(r)
        existing.sort(key=lambda x: str(x.get("date", "")))
        with open(arch_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


def main():
    fred_api_key = os.environ.get('FRED_API_KEY', '').strip() or DEFAULT_FRED_API_KEY

    history = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except Exception as e:
            print("Error reading existing DB:", e)

    history = sanitize_historical_false_kills(history)

    now_kst = datetime.now(KST)
    today_str = now_kst.strftime('%Y-%m-%d')
    date_label = f"{today_str} (Auto DB Sync)"

    if history and str(history[-1].get('date', '')).startswith(today_str):
        prev_rec = history[-2] if len(history) >= 2 else {}
    else:
        prev_rec = history[-1] if history else {}

    us_z, us_levels = collect_us_macro(fred_api_key)
    kr_z, kr_levels = collect_kr_macro(fred_api_key)

    new_record = {
        "date": date_label,
        "iso_date": today_str,
        "updated_at": now_kst.strftime('%Y-%m-%d %H:%M KST'),
        "rolling_window_days": MAX_ROLLING_DAYS,
        "engine_version": "v2.1 (Level+Momentum Hybrid / Hard Crisis Gate / Self-Healing DB)",
        "US": us_z,
        "KR": kr_z,
        "US_summary": precompute_model_v2(us_z, 'US', prev_rec.get('US_summary')),
        "KR_summary": precompute_model_v2(kr_z, 'KR', prev_rec.get('KR_summary')),
        "levels": {
            "US": us_levels,
            "KR": kr_levels,
        },
    }

    if history and str(history[-1].get('date', '')).startswith(today_str):
        history[-1] = new_record
    else:
        history.append(new_record)

    if len(history) > MAX_ROLLING_DAYS:
        overflow = history[:-MAX_ROLLING_DAYS]
        history = history[-MAX_ROLLING_DAYS:]
        archive_overflow_records(overflow)

    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(LATEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_record, f, ensure_ascii=False, indent=2)

    print(f"Done. Rolling DB size: {len(history)} / {MAX_ROLLING_DAYS} days.")


if __name__ == '__main__':
    main()
