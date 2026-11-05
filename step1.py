"""STEP 1. parquet에 위치 컬럼(center_lon, center_lat)을 붙여 새 폴더에 저장한다.

- 원본 CSV의 geom에서 중심 좌표만 빠르게 계산하고, 팀원이 남긴 source_row(1부터 시작)로 붙인다.
- 팀원 원본 parquet은 수정하지 않는다. 결과는 OUT_DIR에 같은 파일 이름으로 저장된다.
- 경로는 아래 설정만 실제 위치에 맞게 바꾸면 된다.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ===== 설정: 실제 경로로 바꾸기 =====
TRAIN_CSV = r"C:\ship_pj\data\raw\vpass_{m}월 데이터.csv"          # Train 원본 CSV
TEST_CSV = r"C:\ship_pj\testData\vpass_{m}월 데이터.csv"   # Test 원본 CSV
FEATURE_DIR = Path(r"C:\ship_pj\parquet")                           # 팀원 parquet 폴더
OUT_DIR = Path(r"C:\ship_pj\pre_output")                                   # 위치가 붙은 결과 폴더
MONTHS = range(1, 7)
# ====================================

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def center(wkt):
    """항적의 모든 점을 감싸는 사각형의 중심 (팀원 코드의 center와 같은 정의)."""
    if not isinstance(wkt, str):
        return (np.nan, np.nan)
    a = np.array(NUM.findall(wkt), dtype=float)
    if a.size < 2:
        return (np.nan, np.nan)
    a = a[: a.size // 2 * 2].reshape(-1, 2)   # (경도, 위도) 쌍
    if (np.abs(a[:, 0]) > 180).any() or (np.abs(a[:, 1]) > 90).any():
        return (np.nan, np.nan)
    lo, hi = a.min(0), a.max(0)
    return tuple((lo + hi) / 2)


def add_location(csv_path, parquet_path, out_path):
    parts, start = [], 1
    for ch in pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str,
                          usecols=["rfid_id", "geom"], chunksize=100_000):
        c = np.array([center(w) for w in ch["geom"].values])
        parts.append(pd.DataFrame({
            "source_row": np.arange(start, start + len(ch)),
            "rfid_id_csv": ch["rfid_id"].str.strip().values,
            "center_lon": c[:, 0], "center_lat": c[:, 1],
        }))
        start += len(ch)
    loc = pd.concat(parts, ignore_index=True)

    df = pd.read_parquet(parquet_path)
    df = df.drop(columns=[c for c in ["center_lon", "center_lat"] if c in df.columns])
    merged = df.merge(loc, on="source_row", how="left", validate="one_to_one")

    mismatch = int((merged["rfid_id"] != merged["rfid_id_csv"]).sum())
    print(f"{Path(parquet_path).name}: 행 {len(df):,} | ID 불일치 {mismatch} | "
          f"위치 결측 {int(merged['center_lon'].isna().sum()):,}", flush=True)
    if mismatch:
        raise ValueError("행 번호가 맞지 않습니다. 팀원이 사용한 것과 같은 버전의 원본 CSV인지 확인하세요.")
    merged.drop(columns="rfid_id_csv").to_parquet(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, pattern in [("train", TRAIN_CSV), ("test", TEST_CSV)]:
        for m in MONTHS:
            pq_path = FEATURE_DIR / f"{split}_{m:02d}.parquet"
            if not pq_path.exists():
                print(f"건너뜀 (파일 없음): {pq_path}")
                continue
            add_location(pattern.format(m=m), pq_path, OUT_DIR / pq_path.name)


if __name__ == "__main__":
    main()