"""STEP 3. 제출 파일 만들기: 제출샘플.xlsx 복사본에 testA(과제 1)와 testB(과제 2 baseline)를 채운다.

testB baseline (간단한 규칙 기반)
- testB 선박의 항적은 별도 파일이 없어 Train·Test parquet 전체에서 찾는다.
- 선박마다 시간순으로 정렬한 시간대별 특징이 "그 선박의 평소 값"에서 얼마나 벗어났는지
  robust z-score(중앙값·MAD 기준)로 계산하고, 평균을 0~1 이상항적 점수로 만든다.
- 사용 신호: 이동거리, 최대 이동 간격, 꺾임 각도, 선 조각 수, 정지 비율,
  직전 시간 대비 위치 점프(m), 직전 기록과의 시간 간격(h)
- 선박별 최고 점수 시각 = 사고시각. 최고 점수 상위 FLAG_RATIO 비율의 선박만 사고로 표시하고
  나머지는 rfid_id만 남긴다 (사고 없음).
- 사고로 표시한 선박은 사고시각 이전 TRACK_HOURS 시간 동안의 시각별 점수를 시각1, 시각2...로 기록한다.

B_MODE로 testB 버전을 고른다
- "accident": 해경 사고현황(발생일시·위경도)과 testB 선박 항적을 시공간으로 결합.
              사고 시각 ±MATCH_H시간, 사고 지점 MATCH_KM 이내에 있던 선박을 사고 선박으로,
              사고 발생 시각을 사고시각으로 지정. 시각1, 시각2...는 사고 전 이상항적 점수.
- "baseline": 위 이상도 규칙 그대로
- "last"    : (실험) 모든 선박을 사고로, 사고시각 = 마지막 기록 시각
- "first"   : (실험) 모든 선박을 사고로, 사고시각 = 처음 기록 시각
- "zero"    : 사고시각·시각1, 시각2... 형식은 그대로 두고 점수 칸만 전부 0
- "blank"   : rfid_id만 남김 (모든 선박 사고 없음)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

# ===== 설정 =====
FEATURE_DIR = Path(r"C:\ship_pj\pre_output")                       # train_XX, test_XX parquet 폴더
PRED_A = Path(r"C:\ship_pj\output\testA_submission.csv")           # STEP 2 결과
SAMPLE = Path(r"C:\ship_pj\제출샘플.xlsx")                          # 대회 제출샘플
ACCIDENT = Path(r"C:\ship_pj\사고현황.xlsx")                         # 해경 사고현황 (accident 모드)
B_MODE = "accident"   # "accident" / "baseline" / "last" / "first" / "zero" / "blank"
OUT = Path(rf"C:\ship_pj\output\submission_{B_MODE}.xlsx")         # 최종 제출 파일 (모드별 이름)
FLAG_RATIO = 0.10     # 사고로 표시할 선박 비율 (리더보드 보면서 조정)
ALL_VESSELS = True    # True: 샘플 선박 + Test 데이터의 모든 선박을 testA·testB에 기록
TRACK_HOURS = 24      # 사고로 표시한 선박에 기록할 사고 전 추적 시간
MATCH_KM = 5.0        # accident 모드: 사고 지점과 항적 중심의 허용 거리(km)
MATCH_H = 3           # accident 모드: 사고 시각과 항적 시각의 허용 차이(시간)
# ================

SIGNALS = ["distance_total_m", "step_max_m", "turn_mean_deg", "n_parts",
           "duplicate_point_ratio", "jump_m", "gap_h"]
R = 6371008.8


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_tracks(ids, feature_dir=FEATURE_DIR):
    """testB 선박의 항적을 Train·Test parquet 전체에서 모은다 (testB 전용 항적 파일은 없음)."""
    cols = ["rfid_id", "part_dt", "rfid_time", "duplicate_record", "center_lon", "center_lat",
            "distance_total_m", "step_max_m", "turn_mean_deg", "n_parts", "duplicate_point_ratio"]
    files = sorted(Path(feature_dir).glob("train_*.parquet")) + sorted(Path(feature_dir).glob("test_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{feature_dir}에 train_/test_ parquet 파일이 없습니다.")
    ids = set(ids)
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d["rfid_id"] = d["rfid_id"].astype(str).str.strip()
        d = d[d["rfid_id"].isin(ids) & ~d["duplicate_record"].astype(bool)]
        print(f"{f.name}: testB 선박 {d.rfid_id.nunique()}척, 행 {len(d):,}", flush=True)
        parts.append(d.drop(columns="duplicate_record"))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["rfid_id", "part_dt", "rfid_time"])
    print(f"testB {len(ids)}척 중 항적이 있는 선박: {df.rfid_id.nunique()}척")
    return df


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
    return d[["rfid_id", "ts", "time_str", "score", "center_lon", "center_lat"]]


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


def load_accidents(path=ACCIDENT):
    a = pd.read_excel(path, dtype=str)
    a.columns = [c.strip() for c in a.columns]
    lat = next(c for c in a.columns if c.startswith("위도"))
    lon = next(c for c in a.columns if c.startswith("경도"))
    out = pd.DataFrame({"acc_t": pd.to_datetime(a["발생일시"], errors="coerce"),
                        "acc_lat": pd.to_numeric(a[lat], errors="coerce"),
                        "acc_lon": pd.to_numeric(a[lon], errors="coerce")})
    out = out.dropna().reset_index(drop=True)
    out["acc_id"] = out.index
    print(f"사고현황: {len(out)}건 ({out.acc_t.min():%Y-%m-%d} ~ {out.acc_t.max():%Y-%m-%d})")
    return out


def build_testB_accident(scores, ids, acc):
    """사고현황의 시각·위치 근처에 있었던 testB 선박 시간대를 찾아 사고시각으로 지정.

    - 각 사고에 대해 ±MATCH_H시간, MATCH_KM 이내에 있던 testB 선박 시간대를 후보로 모은다.
    - 거리·시간이 가까울수록 높은 매칭 점수. 사고 하나는 가장 잘 맞는 선박 한 척에만 배정.
    - 선박마다 가장 잘 맞는 사고의 발생 시각(정시 내림)을 사고시각으로 한다.
    - 매칭되지 않은 선박은 사고 없음(rfid_id만).
    """
    s = scores.copy()
    s["hour"] = s["ts"].dt.floor("h")
    cands = []
    for _, a in acc.iterrows():
        h = a.acc_t.floor("h")
        c = s[(s["hour"] >= h - pd.Timedelta(hours=MATCH_H)) & (s["hour"] <= h + pd.Timedelta(hours=MATCH_H))]
        if c.empty:
            continue
        dist = hav(c["center_lon"].values, c["center_lat"].values, a.acc_lon, a.acc_lat)
        ok = dist <= MATCH_KM * 1000
        if not ok.any():
            continue
        c = c[ok].assign(dist_m=dist[ok], acc_id=a.acc_id, acc_hour=h,
                         dt_h=(c["hour"][ok] - h).abs().dt.total_seconds() / 3600)
        cands.append(c)
    if not cands:
        print("사고현황과 매칭된 testB 선박이 없습니다.")
        return {rid: [] for rid in ids}, 0
    m = pd.concat(cands, ignore_index=True)
    m["match"] = np.exp(-m["dist_m"] / 2000) * np.exp(-m["dt_h"] / 2)
    # 사고마다 가장 잘 맞는 선박 1척, 선박마다 가장 잘 맞는 사고 1건
    m = m.sort_values("match", ascending=False)
    m = m.drop_duplicates("acc_id").drop_duplicates("rfid_id")
    print(f"매칭: 사고 {m.acc_id.nunique()}건 ↔ testB 선박 {m.rfid_id.nunique()}척 "
          f"(거리 중앙값 {m.dist_m.median():.0f}m, 시간차 중앙값 {m.dt_h.median():.1f}h)")
    best = m.set_index("rfid_id")
    rows = {}
    for rid in ids:
        if rid not in best.index:
            rows[rid] = []
            continue
        acc_hour = best.at[rid, "acc_hour"]
        win = scores[(scores.rfid_id == rid) & (scores.ts < acc_hour) &
                     (scores.ts >= acc_hour - pd.Timedelta(hours=TRACK_HOURS))]
        cells = [acc_hour.strftime("%Y%m%d %H"), 1.0]
        for t, sc in zip(win["time_str"], win["score"]):
            cells += [t, float(sc)]
        rows[rid] = cells
    return rows, len(best)


def build_testB_edge(scores, ids, which):
    """실험용: 항적이 있는 모든 선박을 사고로 표시하고, 사고시각을 마지막(last) 또는 처음(first) 기록 시각으로."""
    idx = scores.groupby("rfid_id")["ts"].idxmax() if which == "last" else scores.groupby("rfid_id")["ts"].idxmin()
    edge = scores.loc[idx].set_index("rfid_id")
    rows = {}
    for rid in ids:
        if rid not in edge.index:
            rows[rid] = []
            continue
        t = edge.at[rid, "ts"]
        win = scores[(scores.rfid_id == rid) & (scores.ts < t) & (scores.ts >= t - pd.Timedelta(hours=TRACK_HOURS))]
        cells = [edge.at[rid, "time_str"], 1.0]
        for ts_, sc in zip(win["time_str"], win["score"]):
            cells += [ts_, float(sc)]
        rows[rid] = cells
    return rows, len(edge)


def apply_mode(rows_b, mode):
    """cells = [사고시각, 점수, 시각1, 점수1, ...] 에서 모드에 맞게 값 변경."""
    if mode == "zero":
        return {rid: [v if i % 2 == 0 else 0.0 for i, v in enumerate(cells)]
                for rid, cells in rows_b.items()}
    if mode == "blank":
        return {rid: [] for rid in rows_b}
    if mode not in ("baseline", "accident", "last", "first"):
        raise ValueError('B_MODE는 "accident", "baseline", "last", "first", "zero", "blank" 중 하나여야 합니다.')
    return rows_b


def sheet_ids(ws):
    return [str(r[0].value).strip() for r in ws.iter_rows(min_row=2) if r[0].value is not None]


def fill_workbook(pred_a, rows_b, sample, out, all_vessels=None):
    shutil.copy(sample, out)
    wb = load_workbook(out)
    if len(wb.worksheets) < 2:
        wb.create_sheet("testB")
    ws_a, ws_b = wb.worksheets[0], wb.worksheets[1]
    ws_a.title, ws_b.title = "testA", "testB"

    # testA: 샘플의 선박 순서 유지 (샘플이 비어 있으면 예측 순서대로 작성)
    ids_a = sheet_ids(ws_a) or list(pred_a.index)
    if ALL_VESSELS if all_vessels is None else all_vessels:
        ids_a = ids_a + [i for i in pred_a.index if i not in set(ids_a)]
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
    ids_b = list(rows_b) or ids_a
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
    wb = load_workbook(SAMPLE, read_only=True)
    ids_b = sheet_ids(wb.worksheets[1])
    wb.close()
    ids_test = [i for i in pred_a.index if i not in set(ids_b)]
    if ALL_VESSELS:
        ids_b = ids_b + ids_test
        print(f"ALL_VESSELS: testB 대상 {len(ids_b):,}척 (샘플 + Test 전체)")
    tracks = load_tracks(ids_b)
    scores = hourly_scores(tracks)
    if B_MODE == "accident":
        rows_b, n_flag = build_testB_accident(scores, ids_b, load_accidents())
    elif B_MODE in ("last", "first"):
        rows_b, n_flag = build_testB_edge(scores, ids_b, B_MODE)
    else:
        rows_b, n_flag = build_testB(scores, ids_b)
    rows_b = apply_mode(rows_b, B_MODE)
    print(f"testB 모드: {B_MODE} | 항적 있는 선박 {scores.rfid_id.nunique():,}척 중 사고 표시 "
          f"{0 if B_MODE == 'blank' else n_flag:,}척 (FLAG_RATIO={FLAG_RATIO})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fill_workbook(pred_a, rows_b, SAMPLE, OUT)


if __name__ == "__main__":
    main()