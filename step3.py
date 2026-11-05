"""STEP 3. 제출 파일 만들기: 제출샘플.xlsx 복사본에 testA(과제 1)와 testB(과제 2 baseline)를 채운다.

testB baseline (간단한 규칙 기반)
- 선박마다 시간순으로 정렬한 시간대별 특징이 "그 선박의 평소 값"에서 얼마나 벗어났는지
  robust z-score(중앙값·MAD 기준)로 계산하고, 평균을 0~1 이상항적 점수로 만든다.
- 사용 신호: 이동거리, 최대 이동 간격, 꺾임 각도, 선 조각 수, 정지 비율,
  직전 시간 대비 위치 점프(m), 직전 기록과의 시간 간격(h)
- 선박별 최고 점수 시각 = 사고시각. 최고 점수 상위 FLAG_RATIO 비율의 선박만 사고로 표시하고
  나머지는 rfid_id만 남긴다 (사고 없음).
- 사고로 표시한 선박은 사고시각 이전 TRACK_HOURS 시간 동안의 시각별 점수를 시각1, 시각2...로 기록한다.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

# ===== 설정 =====
FEATURE_DIR = Path(r"C:\ship_pj\pre_output")                       # test_XX.parquet 폴더
PRED_A = Path(r"C:\ship_pj\output\testA_submission.csv")           # STEP 2 결과
SAMPLE = Path(r"C:\ship_pj\제출샘플.xlsx")                          # 대회 제출샘플
OUT = Path(r"C:\ship_pj\output\submission.xlsx")                   # 최종 제출 파일
FLAG_RATIO = 0.10     # 사고로 표시할 선박 비율 (리더보드 보면서 조정)
TRACK_HOURS = 24      # 사고로 표시한 선박에 기록할 사고 전 추적 시간
# ================

SIGNALS = ["distance_total_m", "step_max_m", "turn_mean_deg", "n_parts",
           "duplicate_point_ratio", "jump_m", "gap_h"]
R = 6371008.8


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_test(feature_dir=FEATURE_DIR):
    cols = ["rfid_id", "part_dt", "rfid_time", "duplicate_record", "center_lon", "center_lat",
            "distance_total_m", "step_max_m", "turn_mean_deg", "n_parts", "duplicate_point_ratio"]
    files = sorted(Path(feature_dir).glob("test_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{feature_dir}에 test_*.parquet 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df[~df["duplicate_record"].astype(bool)].drop(columns="duplicate_record")


def hourly_scores(df):
    """선박·시각별 이상항적 점수 (0~1)."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["part_dt"].astype(str), format="%Y%m%d", errors="coerce") + \
        pd.to_timedelta(pd.to_numeric(d["rfid_time"], errors="coerce"), unit="h")
    d = d.dropna(subset=["ts"])
    # 같은 선박·시각 기록이 여러 개면 하나로
    d = (d.groupby(["rfid_id", "ts"], as_index=False)
           .agg({"center_lon": "median", "center_lat": "median", "distance_total_m": "sum",
                 "step_max_m": "max", "turn_mean_deg": "mean", "n_parts": "sum",
                 "duplicate_point_ratio": "mean"})
           .sort_values(["rfid_id", "ts"]))
    prev = d.groupby("rfid_id")[["center_lon", "center_lat", "ts"]].shift()
    d["jump_m"] = hav(prev["center_lon"], prev["center_lat"], d["center_lon"], d["center_lat"])
    d["gap_h"] = (d["ts"] - prev["ts"]).dt.total_seconds() / 3600

    g = d.groupby("rfid_id")[SIGNALS]
    med = g.transform("median")
    mad = (d[SIGNALS] - med).abs().groupby(d["rfid_id"]).transform("median")
    floor = (d[SIGNALS] - d[SIGNALS].median()).abs().median() + 1e-9   # MAD가 0인 선박 대비
    z = ((d[SIGNALS] - med).abs() / (1.4826 * mad.clip(lower=floor, axis=1))).clip(upper=5)
    d["score"] = (z.mean(axis=1, skipna=True) / 5).fillna(0).round(4)
    d["time_str"] = d["ts"].dt.strftime("%Y%m%d %H")
    return d[["rfid_id", "ts", "time_str", "score"]]


def build_testB(scores, ids):
    """선박별 1행: 사고 여부, 사고시각, 점수, 사고 전 추적 시각별 점수."""
    best = scores.loc[scores.groupby("rfid_id")["score"].idxmax()].set_index("rfid_id")
    n_flag = int(round(len(best) * FLAG_RATIO))
    flagged = set(best["score"].sort_values(ascending=False).index[:n_flag])
    rows = {}
    for rid in ids:
        if rid not in flagged:
            rows[rid] = []
            continue
        acc = best.loc[rid]
        win = scores[(scores.rfid_id == rid) & (scores.ts < acc["ts"]) &
                     (scores.ts >= acc["ts"] - pd.Timedelta(hours=TRACK_HOURS))]
        cells = [acc["time_str"], float(acc["score"])]
        for t, s in zip(win["time_str"], win["score"]):
            cells += [t, float(s)]
        rows[rid] = cells
    return rows, len(flagged)


def sheet_ids(ws):
    return [str(r[0].value).strip() for r in ws.iter_rows(min_row=2) if r[0].value is not None]


def fill_workbook(pred_a, rows_b, sample, out):
    shutil.copy(sample, out)
    wb = load_workbook(out)
    if len(wb.worksheets) < 2:
        wb.create_sheet("testB")
    ws_a, ws_b = wb.worksheets[0], wb.worksheets[1]
    ws_a.title, ws_b.title = "testA", "testB"

    # testA: 샘플의 선박 순서 유지 (샘플이 비어 있으면 예측 순서대로 작성)
    ids_a = sheet_ids(ws_a) or list(pred_a.index)
    fallback = pred_a["예측 조업형태"].mode()[0]
    ws_a.delete_rows(2, ws_a.max_row)
    for c, h in enumerate(["rfid_id", "예측 조업형태", "예측조업형태 확률"], 1):
        ws_a.cell(1, c, h)
    miss_a = 0
    for i, rid in enumerate(ids_a, 2):
        if rid in pred_a.index:
            lab, p = pred_a.at[rid, "예측 조업형태"], float(pred_a.at[rid, "예측조업형태 확률"])
        else:
            lab, p, miss_a = fallback, 0.25, miss_a + 1
        ws_a.cell(i, 1, int(rid) if rid.isdigit() else rid)
        ws_a.cell(i, 2, lab)
        ws_a.cell(i, 3, p)

    # testB
    ids_b = sheet_ids(ws_b) or ids_a
    ws_b.delete_rows(1, ws_b.max_row)
    longest = max([len(v) for v in rows_b.values()] + [2])
    header = ["rfid_id", "사고시각", "이상항적 점수"]
    for k in range(1, (longest - 2) // 2 + 1):
        header += [f"시각{k}", f"시각{k} 이상항적 점수"]
    ws_b.append(header)
    miss_b = 0
    for rid in ids_b:
        cells = rows_b.get(rid)
        if cells is None:
            cells, miss_b = [], miss_b + 1
        ws_b.append([int(rid) if rid.isdigit() else rid] + cells)
    wb.save(out)
    print(f"저장: {out}")
    print(f"testA {len(ids_a):,}척 (예측 없어 기본값 {miss_a}척) | "
          f"testB {len(ids_b):,}척 (데이터 없어 사고없음 처리 {miss_b}척)")


def main():
    pred_a = pd.read_csv(PRED_A, dtype={"rfid_id": str}).set_index("rfid_id")
    test = load_test()
    test["rfid_id"] = test["rfid_id"].astype(str).str.strip()
    scores = hourly_scores(test)
    ids = sorted(test["rfid_id"].unique())
    rows_b, n_flag = build_testB(scores, ids)
    print(f"testB: 선박 {len(ids):,}척 중 {n_flag:,}척을 사고로 표시 (FLAG_RATIO={FLAG_RATIO})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fill_workbook(pred_a, rows_b, SAMPLE, OUT)


if __name__ == "__main__":
    main()