"""STEP 2. 선박 단위 집계 → 5-fold 교차검증(LightGBM) → Test A 예측 → 제출 파일 저장.

1월 데이터 실험에서 가장 좋았던 구성(모양 + 시간대 + 세분화 위치 특징)을 그대로 사용한다.
- 입력: STEP 1 결과 폴더의 train_XX.parquet, test_XX.parquet (center_lon, center_lat 포함)
- 출력: OUT_DIR/testA_submission.csv (+ openpyxl이 있으면 .xlsx), OUT_DIR/cv_report.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:                     # lightgbm이 없을 때만 sklearn으로 대체
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LGBM = False

# ===== 설정 =====
FEATURE_DIR = Path(r"C:\ship_pj\pre_output")
OUT_DIR = Path(r"C:\ship_pj\output")
SEED = 42
N_FOLDS = 5
IDLE_M = 100          # 시간당 이동거리가 이 값(m) 미만이면 정박/정지로 본다
# ================

SHAPE = ["n_points", "n_parts", "duplicate_point_ratio", "distance_total_m", "step_mean_m",
         "step_max_m", "step_std_m", "extent_width_m", "extent_height_m",
         "turn_mean_deg", "turn_max_deg"]
LOC = ["center_lon", "center_lat"]
BASE = ["rfid_id", "part_dt", "hour", "duplicate_record"]


def load(split, feature_dir=FEATURE_DIR):
    cols = BASE + SHAPE + LOC + (["fishing_code"] if split == "train" else [])
    files = sorted(Path(feature_dir).glob(f"{split}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{feature_dir}에 {split}_*.parquet 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    before = len(df)
    df = df[~df["duplicate_record"].astype(bool)].drop(columns="duplicate_record")
    print(f"{split}: 파일 {len(files)}개, 행 {before:,} → 완전중복 제외 {len(df):,}, 선박 {df.rfid_id.nunique():,}")
    return df


def aggregate(d):
    """시간대별 행 → 선박당 1행. Train/Test 모두 반드시 이 함수 하나로 처리한다."""
    d = d.copy()
    d["idle"] = d["distance_total_m"] < IDLE_M
    g = d.groupby("rfid_id")

    # 1) 모양 특징 요약 (날마다 패턴이 흔들려서 평균 외 분산·분위도 함께)
    agg = g[SHAPE].agg(["mean", "std", "median", "max", "min"])
    agg.columns = ["_".join(c) for c in agg.columns]

    # 2) 움직일 때만의 패턴
    mv = d[~d.idle].groupby("rfid_id")[SHAPE].median().add_suffix("_moving_median")

    # 3) 시간대 활동 비율 (hour 평균 대신)
    band = pd.cut(d["hour"], [-1, 5, 11, 17, 23], labels=["dawn", "morning", "afternoon", "night"])
    hr = pd.crosstab(d["rfid_id"], band, normalize="index").add_prefix("ratio_")

    # 4) 활동량
    extra = pd.DataFrame({"n_records": g.size(), "n_days": g["part_dt"].nunique(),
                          "idle_ratio": g["idle"].mean()})
    extra["records_per_day"] = extra["n_records"] / extra["n_days"]

    # 5) 위치: 전체 중앙값·범위, 정박 위치(모항 추정), 이동 위치(어장 추정), 10%/90% 분위
    loc = g[LOC].median()
    loc["lon_range"] = g["center_lon"].max() - g["center_lon"].min()
    loc["lat_range"] = g["center_lat"].max() - g["center_lat"].min()
    port = d[d.idle].groupby("rfid_id")[LOC].median().add_prefix("idle_")
    ground = d[~d.idle].groupby("rfid_id")[LOC].median().add_prefix("moving_")
    q = g[LOC].quantile([0.1, 0.9]).unstack()
    q.columns = [f"{a}_q{int(b * 100)}" for a, b in q.columns]

    X = agg.join([mv, hr, extra, loc, port, ground, q])
    for c in ["ratio_dawn", "ratio_morning", "ratio_afternoon", "ratio_night"]:
        if c not in X:
            X[c] = 0.0
    return X.sort_index(axis=1)


def make_model():
    if USE_LGBM:
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.03,
                                  num_leaves=31, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                          validation_fraction=0.15, random_state=SEED)


def fit(model, X_tr, y_tr, X_va, y_va):
    if USE_LGBM:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X_tr, y_tr)
    return model


def run(train_raw, test_raw, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = aggregate(train_raw)
    y = train_raw.groupby("rfid_id")["fishing_code"].first().loc[X.index]
    n_labels = train_raw.groupby("rfid_id")["fishing_code"].nunique()
    if (n_labels > 1).any():
        raise ValueError(f"라벨이 여러 개인 선박 {(n_labels > 1).sum()}척 → 처리 방법을 먼저 정하세요.")
    X_test = aggregate(test_raw).reindex(columns=X.columns)
    print(f"학습 선박 {len(X):,} / 테스트 선박 {len(X_test):,} / 특징 {X.shape[1]}개 / "
          f"{'LightGBM' if USE_LGBM else 'HistGradientBoosting (lightgbm 없음)'}")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    classes = sorted(y.unique())
    oof = pd.Series(index=y.index, dtype=object)
    test_proba = np.zeros((len(X_test), len(classes)))
    fold_scores, base_scores = [], []

    for k, (tr, va) in enumerate(folds, 1):
        model = fit(make_model(), X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        assert list(model.classes_) == classes
        oof.iloc[va] = model.predict(X.iloc[va])
        test_proba += model.predict_proba(X_test) / N_FOLDS
        fold_scores.append(f1_score(y.iloc[va], oof.iloc[va], average="micro"))
        base_scores.append(f1_score(y.iloc[va], [y.iloc[tr].mode()[0]] * len(va), average="micro"))
        print(f"fold {k}: micro-F1 {fold_scores[-1]:.4f} (기준선 {base_scores[-1]:.4f})", flush=True)

    report = [
        f"기준선(최다 클래스) micro-F1: {np.mean(base_scores):.4f}",
        f"모델 micro-F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}",
        f"모델 macro-F1: {f1_score(y, oof, average='macro'):.4f}",
        "", "혼동행렬 (행=실제, 열=예측)",
        pd.DataFrame(confusion_matrix(y, oof, labels=classes), index=classes, columns=classes).to_string(),
        "", classification_report(y, oof, digits=3),
    ]
    text = "\n".join(report)
    print("\n" + text)
    (out_dir / "cv_report.txt").write_text(text, encoding="utf-8")

    # 특징 중요도 (발표 자료용): 전체 데이터로 한 번 더 학습
    if USE_LGBM:
        full = make_model().set_params(n_estimators=200)
        full.fit(X, y)
        imp = pd.Series(full.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp.to_csv(out_dir / "feature_importance.csv", encoding="utf-8-sig", header=["importance"])

    sub = pd.DataFrame({
        "rfid_id": X_test.index,
        "예측 조업형태": np.array(classes)[test_proba.argmax(1)],
        "예측조업형태 확률": test_proba.max(1).round(4),
    })
    assert sub["rfid_id"].is_unique and np.allclose(test_proba.sum(1), 1)
    sub.to_csv(out_dir / "testA_submission.csv", index=False, encoding="utf-8-sig")
    try:
        sub.to_excel(out_dir / "testA_submission.xlsx", sheet_name="testA", index=False)
    except ImportError:
        print("openpyxl이 없어 xlsx는 건너뜀 (pip install openpyxl)")
    print(f"\n제출 파일 저장: {out_dir / 'testA_submission.csv'} ({len(sub):,}척)")
    print(sub["예측 조업형태"].value_counts())
    return sub


if __name__ == "__main__":
    run(load("train"), load("test"))