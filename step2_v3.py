"""STEP 2 v3. 과제 1 개선 실험 + Test A 예측.

기존 STEP 2(모양 + 시간대 + 위치 집계)에 세 가지를 더하고, 같은 fold로 나란히 비교한다.
  1) 월별 활동 비율        : 1~6월 각 달이 그 선박 기록에서 차지하는 비율
  2) 이웃 선박(KNN) 비율   : 모항(정지 위치)이 가까운 학습 선박 K척의 조업형태 비율 + 평균 거리
                            fold 안에서 학습 선박만 이웃으로 쓰고, 학습 선박 자신은 이웃에서 제외 (누수 방지)
  2-b) KNN 확장 (v3)     : 어장(움직일 때 위치) 기준 이웃도 추가, 이웃 수 5/10/30/50
  4) seed 평균 (v3)       : 최종 구성은 LightGBM을 여러 seed로 학습해 확률 평균
  3) 클래스별 확률 보정     : 교차검증 예측 확률에 클래스 가중치를 곱해 micro-F1 최대가 되는 값을 찾아 Test에 적용

출력 (OUT_DIR)
  - experiment_report.txt    : 실험별 교차검증 점수 비교 + 최종 구성의 혼동행렬
  - testA_submission.csv     : 가장 좋은 구성으로 만든 Test A 예측 (STEP 3이 그대로 사용)
"""
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore", module="lightgbm")
warnings.filterwarnings("ignore", message=".*eval_set.*")

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LGBM = False

# ===== 설정 =====
FEATURE_DIR = Path(r"C:\ship_pj\pre_output")      # train_XX, test_XX parquet (center_lon/lat 포함)
OUT_DIR = Path(r"C:\ship_pj\output")
SEED = 42
N_FOLDS = 5
IDLE_M = 100            # 시간당 이동거리가 이 값(m) 미만이면 정지로 본다
KNN_KS = (10, 30)              # 기본 KNN: 모항 기준 이웃 수
KNN_KS_EXT = (5, 10, 30, 50)   # 확장 KNN: 모항 + 어장 기준 이웃 수
N_SEEDS = 3                    # 최종 구성의 seed 평균 개수 (1이면 끔)
EXPERIMENTS = ["+월별+KNN", "+월별+KNN확장"]   # v2 최고 구성과 확장 버전 비교
# ================

SHAPE = ["n_points", "n_parts", "duplicate_point_ratio", "distance_total_m", "step_mean_m",
         "step_max_m", "step_std_m", "extent_width_m", "extent_height_m",
         "turn_mean_deg", "turn_max_deg"]
LOC = ["center_lon", "center_lat"]
BASE = ["rfid_id", "part_dt", "hour", "duplicate_record"]


# ---------- 데이터 ----------
def load(split, feature_dir=FEATURE_DIR):
    cols = BASE + SHAPE + LOC + (["fishing_code"] if split == "train" else [])
    files = sorted(Path(feature_dir).glob(f"{split}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{feature_dir}에 {split}_*.parquet 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df["rfid_id"] = df["rfid_id"].astype(str).str.strip()
    df = df[~df["duplicate_record"].astype(bool)].drop(columns="duplicate_record")
    print(f"{split}: 파일 {len(files)}개, 행 {len(df):,}, 선박 {df.rfid_id.nunique():,}")
    return df


def aggregate(d):
    """시간대별 행 → 선박당 1행. Train/Test 모두 이 함수 하나로 처리."""
    d = d.copy()
    d["idle"] = d["distance_total_m"] < IDLE_M
    g = d.groupby("rfid_id")

    agg = g[SHAPE].agg(["mean", "std", "median", "max", "min"])
    agg.columns = ["_".join(c) for c in agg.columns]
    mv = d[~d.idle].groupby("rfid_id")[SHAPE].median().add_suffix("_moving_median")
    band = pd.cut(d["hour"], [-1, 5, 11, 17, 23], labels=["dawn", "morning", "afternoon", "night"])
    hr = pd.crosstab(d["rfid_id"], band, normalize="index").add_prefix("ratio_")
    extra = pd.DataFrame({"n_records": g.size(), "n_days": g["part_dt"].nunique(),
                          "idle_ratio": g["idle"].mean()})
    extra["records_per_day"] = extra["n_records"] / extra["n_days"]

    loc = g[LOC].median()
    loc["lon_range"] = g["center_lon"].max() - g["center_lon"].min()
    loc["lat_range"] = g["center_lat"].max() - g["center_lat"].min()
    port = d[d.idle].groupby("rfid_id")[LOC].median().add_prefix("idle_")
    ground = d[~d.idle].groupby("rfid_id")[LOC].median().add_prefix("moving_")
    q = g[LOC].quantile([0.1, 0.9]).unstack()
    q.columns = [f"{a}_q{int(b * 100)}" for a, b in q.columns]

    # (신규) 월별 활동 비율
    month = pd.to_numeric(d["part_dt"].astype(str).str[4:6], errors="coerce")
    mon = pd.crosstab(d["rfid_id"], month, normalize="index")
    mon = mon.reindex(columns=range(1, 7), fill_value=0.0)
    mon.columns = [f"month_{m}" for m in mon.columns]

    X = agg.join([mv, hr, extra, loc, port, ground, q, mon])
    for c in ["ratio_dawn", "ratio_morning", "ratio_afternoon", "ratio_night"]:
        if c not in X:
            X[c] = 0.0
    for c in [f"month_{m}" for m in range(1, 7)]:
        X[c] = X[c].fillna(0.0)
    return X.sort_index(axis=1)


MONTH_COLS = [f"month_{m}" for m in range(1, 7)]


# ---------- KNN 이웃 특징 ----------
def home_coords(X):
    """모항 좌표: 정지 위치 중앙값, 없으면 전체 위치 중앙값."""
    lat = X["idle_center_lat"].fillna(X["center_lat"])
    lon = X["idle_center_lon"].fillna(X["center_lon"])
    return np.radians(np.c_[lat.values, lon.values])


def ground_coords(X):
    """어장 좌표: 움직일 때 위치 중앙값, 없으면 전체 위치 중앙값."""
    lat = X["moving_center_lat"].fillna(X["center_lat"])
    lon = X["moving_center_lon"].fillna(X["center_lon"])
    return np.radians(np.c_[lat.values, lon.values])


def knn_features(X_ref, y_ref, X_query, classes, exclude_self=False,
                 coords=home_coords, ks=KNN_KS, prefix="knn"):
    """X_ref(정답 있는 학습 선박)를 이웃 후보로, X_query 선박마다 이웃 조업형태 비율·평균 거리(km)."""
    tree = BallTree(coords(X_ref), metric="haversine")
    kmax = max(ks) + (1 if exclude_self else 0)
    dist, idx = tree.query(coords(X_query), k=min(kmax, len(X_ref)))
    if exclude_self:                       # 자기 자신(가장 가까운 0거리 1개) 제외
        dist, idx = dist[:, 1:], idx[:, 1:]
    yv = np.asarray(y_ref)
    out = {}
    for k in ks:
        lab = yv[idx[:, :k]]
        for c in classes:
            out[f"{prefix}{k}_{c}"] = (lab == c).mean(1)
        out[f"{prefix}{k}_dist_km"] = dist[:, :k].mean(1) * 6371.0
    return pd.DataFrame(out, index=X_query.index)


# ---------- 모델 ----------
def make_model(seed=SEED):
    if USE_LGBM:
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.03,
                                  num_leaves=31, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed)


def fit(model, X_tr, y_tr, X_va, y_va):
    if USE_LGBM:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X_tr, y_tr)
    return model


def feature_set(X, name):
    cols = [c for c in X.columns if c not in MONTH_COLS]
    if "월별" in name:
        cols += MONTH_COLS
    return cols


def add_knn(name, ref, yref, Xq, classes, exclude_self):
    """실험 이름에 맞는 KNN 특징 묶음."""
    if "KNN확장" in name:
        a = knn_features(ref, yref, Xq, classes, exclude_self, home_coords, KNN_KS_EXT, "knnH")
        b = knn_features(ref, yref, Xq, classes, exclude_self, ground_coords, KNN_KS_EXT, "knnG")
        return a.join(b)
    return knn_features(ref, yref, Xq, classes, exclude_self)


def run_experiment(name, X, y, X_test, folds, classes, seeds=(SEED,)):
    """한 구성의 교차검증 OOF 확률과 Test 확률(fold 모델 × seed 평균)."""
    cols = feature_set(X, name)
    use_knn = "KNN" in name
    oof = np.zeros((len(X), len(classes)))
    test_p = np.zeros((len(X_test), len(classes)))
    for tr, va in folds:
        X_tr, X_va, X_te = X.iloc[tr][cols], X.iloc[va][cols], X_test[cols]
        if use_knn:
            ref, yref = X.iloc[tr], y.iloc[tr]
            X_tr = X_tr.join(add_knn(name, ref, yref, ref, classes, True))
            X_va = X_va.join(add_knn(name, ref, yref, X.iloc[va], classes, False))
            X_te = X_te.join(add_knn(name, ref, yref, X_test, classes, False))
        for sd in seeds:
            m = fit(make_model(sd), X_tr, y.iloc[tr], X_va, y.iloc[va])
            assert list(m.classes_) == classes
            oof[va] += m.predict_proba(X_va) / len(seeds)
            test_p += m.predict_proba(X_te) / (len(folds) * len(seeds))
    return oof, test_p


def micro(y, proba, classes, w=None):
    p = proba * (np.ones(len(classes)) if w is None else w)
    return f1_score(y, np.array(classes)[p.argmax(1)], average="micro")


def tune_weights(y, oof, classes):
    """클래스별 확률 가중치를 좌표 탐색으로 찾는다 (첫 클래스는 1로 고정)."""
    w = np.ones(len(classes))
    grid = np.round(np.arange(0.5, 2.51, 0.1), 2)
    best = micro(y, oof, classes, w)
    for _ in range(3):
        for j in range(1, len(classes)):
            for v in grid:
                w2 = w.copy()
                w2[j] = v
                s = micro(y, oof, classes, w2)
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, test_raw = load("train"), load("test")
    X = aggregate(train_raw)
    y = train_raw.groupby("rfid_id")["fishing_code"].first().loc[X.index]
    if (train_raw.groupby("rfid_id")["fishing_code"].nunique() > 1).any():
        raise ValueError("라벨이 여러 개인 선박이 있습니다.")
    X_test = aggregate(test_raw).reindex(columns=X.columns)
    classes = sorted(y.unique())
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    print(f"학습 {len(X):,}척 / 테스트 {len(X_test):,}척 / {'LightGBM' if USE_LGBM else 'HistGB'}")

    base = np.mean([f1_score(y.iloc[va], [y.iloc[tr].mode()[0]] * len(va), average="micro")
                    for tr, va in folds])
    lines = [f"기준선(최다 클래스) micro-F1: {base:.4f}", "",
             f"{'실험':<12}{'micro-F1':>10}{'+확률보정':>12}  보정 가중치"]
    results = {}
    for name in EXPERIMENTS:
        oof, test_p = run_experiment(name, X, y, X_test, folds, classes)
        raw = micro(y, oof, classes)
        w, tuned = tune_weights(y, oof, classes)
        results[name] = dict(oof=oof, test=test_p, raw=raw, w=w, tuned=tuned)
        wtxt = ", ".join(f"{c}={v:.1f}" for c, v in zip(classes, w))
        lines.append(f"{name:<12}{raw:>10.4f}{tuned:>12.4f}  {wtxt}")
        print(lines[-1], flush=True)

    # 최종 구성 선택: 보정 전 점수 기준 (보정은 같은 데이터로 찾아서 다소 낙관적이므로 0.003 이상 오를 때만 채택)
    best_name = max(results, key=lambda n: results[n]["raw"])
    if N_SEEDS > 1:
        seeds = tuple(SEED + i for i in range(N_SEEDS))
        oof, test_p = run_experiment(best_name, X, y, X_test, folds, classes, seeds)
        raw = micro(y, oof, classes)
        w, tuned = tune_weights(y, oof, classes)
        lines.append(f"{best_name + f' (seed {N_SEEDS}개 평균)':<12}{raw:>10.4f}{tuned:>12.4f}")
        print(lines[-1], flush=True)
        if raw >= results[best_name]["raw"]:
            results[best_name] = dict(oof=oof, test=test_p, raw=raw, w=w, tuned=tuned)
            best_name_note = f"{best_name} + seed {N_SEEDS}개 평균"
        else:
            best_name_note = best_name
    else:
        best_name_note = best_name
    r = results[best_name]
    use_w = r["tuned"] - r["raw"] >= 0.003
    w = r["w"] if use_w else np.ones(len(classes))
    oof_pred = np.array(classes)[(r["oof"] * w).argmax(1)]
    lines += ["", f"최종 구성: {best_name_note} | 확률보정 {'적용' if use_w else '미적용'}",
              "", "혼동행렬 (행=실제, 열=예측)",
              pd.DataFrame(confusion_matrix(y, oof_pred, labels=classes),
                           index=classes, columns=classes).to_string(),
              "", classification_report(y, oof_pred, digits=3)]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "experiment_report.txt").write_text(text, encoding="utf-8")

    p = r["test"] * w
    p = p / p.sum(1, keepdims=True)
    sub = pd.DataFrame({"rfid_id": X_test.index,
                        "예측 조업형태": np.array(classes)[p.argmax(1)],
                        "예측조업형태 확률": p.max(1).round(4)})
    sub.to_csv(OUT_DIR / "testA_submission.csv", index=False, encoding="utf-8-sig")
    print(f"\n제출용 예측 저장: {OUT_DIR / 'testA_submission.csv'} ({len(sub):,}척)")
    print(sub["예측 조업형태"].value_counts())


if __name__ == "__main__":
    main()