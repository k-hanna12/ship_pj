# 어선 항적 전처리

대회 ZIP의 월별 CSV를 직접 읽어 항적 특징 Parquet과 품질 보고서를 만든다.
원본 ZIP을 수정하거나 CSV 전체를 압축 해제하지 않는다. 모델 학습·선박별 집계는 다음 단계다.

## 이 컴퓨터에서 실행

PowerShell에서 이 저장소 폴더로 이동한 후 실행한다. 아래 ZIP 경로는 실제 파일 위치로 바꾼다.
현재 데스크톱의 Python 3.12와 작업 폴더의 `.python-deps`를 사용한다.
기본 Python 3.15 환경이나 기존 `main.py`는 변경하지 않았다.

```powershell
# 1. 훈련·테스트 각각 1월 첫 1,000행
.\run_preprocess.ps1 --zip 'C:\data\문제 데이터_수정_1103_v2.zip' --split all --months 1 --max-rows 1000 --output data/features_sample

# 2. 1월 훈련 전체
.\run_preprocess.ps1 --zip 'C:\data\문제 데이터_수정_1103_v2.zip' --split train --months 1 --output data/features_january

# 3. 나머지 월과 테스트: 같은 출력 폴더에서 이미 완료한 월은 제외해 실행
.\run_preprocess.ps1 --zip 'C:\data\문제 데이터_수정_1103_v2.zip' --split train --months 2 3 4 5 6 --output data/features_january
.\run_preprocess.ps1 --zip 'C:\data\문제 데이터_수정_1103_v2.zip' --split test --output data/features_january
```

월별 파일명에 연도가 없으므로, 출력 폴더는 데이터 버전마다 별도로 사용한다.
이미 존재하는 월별 출력은 덮어쓰지 않는다. 실패한 실행은 `.partial.parquet`이 남을 수 있으므로
원인을 확인한 후 새 출력 폴더에서 재실행한다. 출력은 `data/` 아래 두며 Git에 포함하지 않는다.

## 팀원 컴퓨터에서 실행

검증한 Python 3.12 가상환경에서 설치하고 일반 Python 명령으로 실행할 수 있다.
플랫폼에 맞는 바이너리 패키지를 사용한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-preprocess.txt
.\.venv\Scripts\python.exe preprocess.py --zip '원본ZIP경로' --split train --months 1 --max-rows 1000 --output data/features_sample
.\.venv\Scripts\python.exe -m unittest -v test_preprocess
```

## 출력과 해석

- `train_01.parquet` / `test_01.parquet`: 원본 레코드별 특징. 원본 행 수를 유지한다.
- `*.quality.json`: 처리 범위, 완료 여부, 날짜, 라벨 분포, 결측·중복·항적 오류, 선박별 라벨.
- `summary.json`: 해당 출력 폴더의 보고서 통합, 월간 라벨 불일치, Train/Test 선박 ID 겹침.
- `source_row`: 헤더를 제외한 원본 CSV 레코드 순번(1부터). 따옴표 안 줄바꿈이 있으면 물리적 줄 번호와 다르다.
- `complete_file=false`: `--max-rows` 등으로 제한한 일부 처리 결과다. 전체 데이터 통계로 해석하지 않는다.
- `timestamp`: 원본 날짜와 시간대 결합. 공식 시간대는 미확인이라 시간대를 임의 지정하지 않는다.
- Train의 라벨은 `fishing_code`, Test는 같은 컬럼에 null을 저장한다.
- `parse_ok`는 WKT 파싱 여부, `feature_ok`는 좌표 범위를 통과해 공간 특징을 생성했는지 여부다.
- `geometry_status`: `ok`, `missing`, `parse_failed`, `empty`, `unsupported_type`, `invalid_coordinates`.
- 잘못된 ID·날짜/시간, 파일 월과 실제 날짜의 불일치, 중복 키·동일 레코드를 표시하고 보존한다.
- 동일 레코드 중복은 기본 컬럼의 원문 값 기준이다. 중복 검사는 같은 CSV의 청크 사이까지 수행한다.
- 라벨 불일치를 자동으로 교정하지 않는다. 예측 전 처리 기준을 합의해야 한다.

### 첫 특징 목록

| 컬럼 | 의미 |
| --- | --- |
| n_points / n_parts / n_segments | 좌표 수 / 선분 수 / 선분 내부 인접 좌표 쌍 수 |
| duplicate_point_ratio | 선분 내부 인접 좌표 쌍 중 완전 동일한 좌표 비율. 정지시간 비율이 아님 |
| distance_total_m | 선분 내부 거리 합계. 서로 끊긴 선분 사이 거리를 합하지 않음 |
| step_mean_m / step_max_m / step_std_m | 선분 내부 좌표 간 거리 통계. 속도가 아님 |
| extent_width_m / extent_height_m | 좌표 범위의 중심 위도·경도에서 측정한 폭·높이 |
| turn_mean_deg / turn_max_deg | 연속된 두 유효 이동 구간의 방향 차이 절댓값. 정지 구간과 선분 경계 제외 |
| year / month / hour | 실제 날짜·시간에서 파생. 잘못된 값은 null |

거리·방향은 WGS84 경위도라는 가정으로 `pyproj.Geod`를 사용한다. 공식 좌표계 확인이 필요하다.
전 지구 범위 검사만 수행하므로 한반도 해역 범위의 오류나 경위도 뒤바뀜을 모두 탐지하지는 못한다.
범위 폭은 날짜변경선을 가로지르는 항적에 적합하지 않다. 원본 좌표를 확인한 뒤 확장한다.
좌표별 시각이 없어 순간속도·가속도·정지시간을 계산하지 않는다.

### 모델 담당자에게

식별·추적 컬럼(`rfid_id`, `source_file`, `source_row`, `split`)과 정답을 입력 특징에 넣지 않는다.
오류·중복 행은 자동 제거하지 않았으므로 품질 보고서를 확인해 학습 정책을 결정한다.
Test 행은 누락시키지 말고 특징이 없는 경우의 예측 방식을 정한다.
같은 선박의 모든 월을 같은 분할에 두고, 결측 대체 등 학습형 변환은 Train에만 맞춘다.
과제 1의 검증·최종 예측은 선박별 단위로 평가한다.
원본 Train 첫 행은 2026년, Test 첫 행은 2025년이므로 파일 월만 보고 연도를 덮어쓰지 않는다.
`year` 등 날짜 특징을 모델에 사용할지는 연도 차이에 따른 일반화 위험을 검토해 결정한다.

메모리 사용을 줄이기 위해 기본 10,000행씩 읽고 중복 키는 임시 SQLite에 기록한다.
디스크 여유가 512MiB 미만이면 청크 완료 시 중단한다. 이 검사는 디스크 부족을 완전히 방지하는 예약 기능은 아니다.
ZIP 목록·일부 행 확인은 전체 ZIP 무결성 검사가 아니다. 월별 CSV를 끝까지 읽을 때 ZIP CRC도 검사된다.

## 결과를 표로 확인하기

훈련 1~6월이 모두 완료된 뒤 다음 명령으로 월별 전체 CSV를 생성할 수 있다.
같은 Python 환경에서 실행하며, 이 컴퓨터에서는 앞서 안내한 데스크톱 Python과 `.python-deps`를 사용한다.

```powershell
python export_features_csv.py --feature-dir data/features_january --output-dir data
```

`data/train_01_features.csv`부터 `data/train_06_features.csv`까지 UTF-8 BOM으로 저장한다.
한 번에 10,000행씩 변환하고 CSV의 열과 전체 행 수를 Parquet과 비교한다.
이미 존재하는 CSV는 열·행 수를 검사한 후 유지한다. 다른 데이터 버전에는 새 출력 폴더를 사용한다.
CSV는 시트를 지원하지 않는다. 이 파일들은 전처리 확인용이며 최종 제출 엑셀과 구분한다.

전처리 결과는 모델 입력 특징이다. 제출 예시의 예측 조업형태·확률·사고시각·이상점수는
분류·이상탐지 모델 단계를 실행한 뒤 생성한다. `fishing_code`는 Train의 실제 정답이며 예측값이 아니다.

설치한 Python 환경에서 아래 코드를 실행하면 전체 파일을 메모리에 올리지 않고 일부 행을 확인할 수 있다.

```python
import pyarrow.parquet as pq

frame = next(pq.ParquetFile("data/features_january/train_01.parquet").iter_batches(batch_size=20)).to_pandas()
print(frame[["rfid_id", "timestamp", "fishing_code", "n_points", "distance_total_m", "turn_mean_deg", "geometry_status"]].to_string(index=False))
# 필요하면 로컬 HTML 표로 저장한다. data/는 Git에서 제외된다.
frame.to_html("data/features_preview.html", index=False)
```

### 다음 단계

1. 전체 월별 처리가 끝나면 `summary.json`과 월별 `*.quality.json`에서 라벨 불일치·중복·오류를 확인한다.
2. 같은 선박의 모든 월을 같은 학습/검증 분할에 두고, 선박별 집계 또는 예측 확률 집계 방식을 정한다.
3. 과제 1 기준 모델을 학습하고 선박별 micro-F1을 측정한 뒤 Test A 예측을 만든다.
4. 과제 2는 사고정보와 시간순 특징을 연결해 이상점수와 예측 사고시각을 만든다.
5. 최신 제출샘플의 Test A/B 선박 ID와 시트·컬럼에 맞춰 최종 엑셀 파일을 작성한다.

코드와 실행 문서만 공유한다. 원본 ZIP/CSV, 특징 Parquet, 품질 JSON, 실행 로그와 HTML 미리보기는
`data/` 아래 보관하며 GitHub에 올리지 않는다. 현재 저장소의 `main.py`는 기존 예시로 유지한다.
