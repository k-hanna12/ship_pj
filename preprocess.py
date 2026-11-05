"""Read monthly CSVs directly from the competition ZIP; never extract raw CSVs."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Geod
from shapely import from_wkt

GEOD = Geod(ellps="WGS84")
NUMERIC = ["n_points", "n_parts", "n_segments", "duplicate_point_ratio",
           "distance_total_m", "step_mean_m", "step_max_m", "step_std_m",
           "extent_width_m", "extent_height_m", "turn_mean_deg", "turn_max_deg"]
SCHEMA = pa.schema([
    ("source_file", pa.string()), ("source_row", pa.int64()),
    ("split", pa.string()), ("rfid_id", pa.string()),
    ("part_dt", pa.string()), ("rfid_time", pa.string()),
    ("timestamp", pa.timestamp("ns")), ("year", pa.float64()),
    ("month", pa.float64()), ("hour", pa.float64()),
    ("fishing_code", pa.string()), ("geometry_status", pa.string()),
    ("parse_ok", pa.bool_()), ("feature_ok", pa.bool_()),
    ("invalid_id", pa.bool_()), ("invalid_datetime", pa.bool_()),
    ("month_mismatch", pa.bool_()), ("duplicate_key", pa.bool_()),
    ("duplicate_record", pa.bool_()),
] + [(name, pa.float64()) for name in NUMERIC])


def geometry_features(wkt):
    result = dict.fromkeys(NUMERIC, np.nan)
    result.update(geometry_status="missing", parse_ok=False, feature_ok=False)
    if not wkt or not wkt.strip():
        return result
    geom = from_wkt(wkt, on_invalid="ignore")
    if geom is None:
        result["geometry_status"] = "parse_failed"
        return result
    result["parse_ok"] = True
    if geom.geom_type not in ("LineString", "MultiLineString"):
        result["geometry_status"] = "unsupported_type"
        return result
    if geom.is_empty:
        result["geometry_status"] = "empty"
        return result
    parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    coords = [np.asarray(part.coords)[:, :2] for part in parts]
    xy = np.concatenate(coords)
    result.update(n_points=float(len(xy)), n_parts=float(len(parts)),
                  n_segments=float(sum(max(0, len(c) - 1) for c in coords)))
    if not np.isfinite(xy).all() or (np.abs(xy[:, 0]) > 180).any() or (np.abs(xy[:, 1]) > 90).any():
        result["geometry_status"] = "invalid_coordinates"
        return result
    distances, turns = [], []
    duplicates = 0
    for c in coords:
        if len(c) < 2:
            continue
        az, _, d = GEOD.inv(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1])
        distances.extend(d)
        duplicates += int(np.all(c[1:] == c[:-1], axis=1).sum())
        # Zero-length steps have no heading; do not infer a turn across them.
        adjacent = (d[:-1] > 0) & (d[1:] > 0)
        turns.extend(np.abs((np.diff(az)[adjacent] + 180) % 360 - 180))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo + hi) / 2
    width = GEOD.inv(lo[0], center[1], hi[0], center[1])[2]
    height = GEOD.inv(center[0], lo[1], center[0], hi[1])[2]
    result.update(
        geometry_status="ok", feature_ok=True,
        duplicate_point_ratio=duplicates / len(distances) if distances else np.nan,
        distance_total_m=float(np.sum(distances)),
        step_mean_m=float(np.mean(distances)) if distances else np.nan,
        step_max_m=float(np.max(distances)) if distances else np.nan,
        step_std_m=float(np.std(distances)) if distances else np.nan,
        extent_width_m=abs(float(width)), extent_height_m=abs(float(height)),
        turn_mean_deg=float(np.mean(turns)) if turns else np.nan,
        turn_max_deg=float(np.max(turns)) if turns else np.nan,
    )
    return result


def members(archive, split, months):
    for info in archive.infolist():
        match = re.search(r"vpass_(\d+)월 데이터\.csv$", info.filename)
        if not match:
            continue
        kind = "test" if "테스트셋/" in info.filename else "train"
        month = int(match[1])
        if split in (kind, "all") and (not months or month in months):
            yield info, kind, month


def summarize(output):
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("*.quality.json"))]
    labels, ids = defaultdict(set), defaultdict(set)
    for report in reports:
        ids[report["split"]].update(report["vessel_ids"])
        for vessel, codes in report["vessel_labels"].items():
            labels[vessel].update(codes)
    summary = {
        "files": [{k: r[k] for k in ["source_file", "split", "rows", "complete_file", "output_bytes", "counts"]} for r in reports],
        "train_vessels": len(ids["train"]), "test_vessels": len(ids["test"]),
        "train_test_overlap_ids": sorted(ids["train"] & ids["test"]),
        "inconsistent_train_labels": {k: sorted(v) for k, v in labels.items() if len(v) > 1},
        "scope": "Only completed feature files in this output directory; partial samples are explicitly marked.",
        "quarantines": [{"source_file": r["source_file"], "input_rows": r.get("input_rows", r["rows"]),
                         "output_rows": r["rows"], "quarantined_count": r.get("quarantined_count", 0),
                         "quarantined_rows": r.get("quarantined_rows", []),
                         "quarantine_file": r.get("quarantine_file")}
                        for r in reports if r.get("quarantined_count")],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def process(archive, info, split, month, output, chunk_size, max_rows, encoding,
            quarantine_rows=()):
    stem = f"{split}_{month:02d}"
    target = output / f"{stem}.parquet"
    report_path = output / f"{stem}.quality.json"
    partial = output / f"{stem}.partial.parquet"
    db_path = output / f"{stem}.partial.sqlite"
    quarantine_path = output / f"{stem}.quarantine.jsonl"
    quarantine_rows = set(quarantine_rows)
    if any(p.exists() for p in [target, report_path, partial, db_path, quarantine_path]):
        raise FileExistsError(f"Output already exists for {stem}. Use a new --output directory.")
    report = {"source_file": info.filename, "split": split, "source_uncompressed_bytes": info.file_size,
              "source_crc32": info.CRC, "rows": 0, "counts": {}, "geometry_status": {},
              "class_rows": {}, "vessel_labels": {}, "vessel_ids": [],
              "date_min": None, "date_max": None, "complete_file": False,
              "coordinate_assumption": "WGS84 longitude/latitude; official CRS still needs confirmation",
              "duplicate_policy": "Flag and preserve all rows; checks are within this CSV, across chunks."}
    report.update(input_rows=0, quarantined_rows=[], quarantined_count=0,
                  quarantine_file=quarantine_path.name if quarantine_rows else None)
    if quarantine_rows:
        report["duplicate_policy"] = "Quarantine explicitly approved source rows; flag and preserve other rows."
    counts, statuses, classes = Counter(), Counter(), Counter()
    vessel_labels, vessels = defaultdict(set), set()
    observed_end = False
    try:
        with closing(sqlite3.connect(db_path)) as db, pq.ParquetWriter(partial, SCHEMA, compression="zstd") as writer:
            db.execute("CREATE TABLE keys (v TEXT PRIMARY KEY) WITHOUT ROWID")
            db.execute("CREATE TABLE records (v BLOB PRIMARY KEY) WITHOUT ROWID")
            with archive.open(info) as raw:
                reader = pd.read_csv(raw, encoding=encoding, dtype=str, keep_default_na=False,
                                     chunksize=chunk_size, on_bad_lines="error")
                for chunk in reader:
                    remaining = max_rows - report["input_rows"] if max_rows else len(chunk)
                    if remaining <= 0:
                        break
                    chunk = chunk.iloc[:remaining]
                    source_rows = pd.Series(range(report["input_rows"] + 1,
                                                  report["input_rows"] + len(chunk) + 1), index=chunk.index)
                    report["input_rows"] += len(chunk)
                    required = {"part_dt", "rfid_id", "rfid_time", "geom"}
                    if split == "train":
                        required.add("fishing_code")
                    if not required.issubset(chunk.columns):
                        raise ValueError(f"Missing columns: {required - set(chunk.columns)}")
                    rejected = source_rows.isin(quarantine_rows)
                    if rejected.any():
                        # Preserve all parsed fields, including unnamed extras, without truncation.
                        mode = "a" if report["quarantined_rows"] else "x"
                        with quarantine_path.open(mode, encoding="utf-8", newline="\n") as handle:
                            for idx in chunk.index[rejected]:
                                record = {"source_file": info.filename, "source_row": int(source_rows.loc[idx]),
                                          "reason": "user_approved_structurally_broken_record",
                                          "fields": chunk.loc[idx].to_dict()}
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                report["quarantined_rows"].append(int(source_rows.loc[idx]))
                        chunk = chunk.loc[~rejected]
                    row_numbers = source_rows.loc[chunk.index].tolist()
                    extra = [c for c in chunk.columns if c not in required]
                    if any(chunk[c].str.strip().ne("").any() for c in extra):
                        raise ValueError(f"Unexpected nonempty columns: {extra}")
                    dates = pd.to_datetime(chunk.part_dt, format="%Y%m%d", errors="coerce", exact=True)
                    dates = dates.where(chunk.part_dt.str.fullmatch(r"\d{8}"))
                    hours = pd.to_numeric(chunk.rfid_time, errors="coerce")
                    valid_hour = hours.between(0, 23) & hours.mod(1).eq(0)
                    timestamps = pd.Series(pd.NaT, index=chunk.index, dtype="datetime64[ns]")
                    valid_time = dates.notna() & valid_hour
                    timestamps.loc[valid_time] = dates.loc[valid_time] + pd.to_timedelta(hours.loc[valid_time], unit="h")
                    output_rows = []
                    for pos, row in enumerate(chunk.to_dict("records")):
                        date, hour, stamp = dates.iloc[pos], hours.iloc[pos], timestamps.iloc[pos]
                        vessel = row["rfid_id"].strip()
                        label = row.get("fishing_code", "").strip() or None
                        invalid_id = not bool(vessel)
                        invalid_time = pd.isna(stamp)
                        date_valid = not pd.isna(date)
                        key = json.dumps([vessel, date.strftime("%Y%m%d") if date_valid else row["part_dt"],
                                          int(hour) if valid_hour.iloc[pos] else row["rfid_time"]])
                        digest = hashlib.blake2b(json.dumps([row.get(c, "") for c in sorted(required)], ensure_ascii=False).encode(), digest_size=16).digest()
                        duplicate_key = db.execute("INSERT OR IGNORE INTO keys VALUES (?)", (key,)).rowcount == 0
                        duplicate_record = db.execute("INSERT OR IGNORE INTO records VALUES (?)", (digest,)).rowcount == 0
                        flags = dict(invalid_id=invalid_id, invalid_datetime=bool(invalid_time),
                                     month_mismatch=bool(date_valid and date.month != month),
                                     duplicate_key=duplicate_key, duplicate_record=duplicate_record)
                        counts.update(k for k, v in flags.items() if v)
                        features = geometry_features(row["geom"])
                        statuses[features["geometry_status"]] += 1
                        if features["n_points"] <= 2:
                            counts["short_track_le_2_points"] += 1
                        if split == "train" and label is None:
                            counts["missing_label"] += 1
                        if vessel:
                            vessels.add(vessel)
                            if label:
                                vessel_labels[vessel].add(label)
                        if label:
                            classes[label] += 1
                        output_rows.append(dict(
                            source_file=info.filename, source_row=row_numbers[pos],
                            split=split, rfid_id=vessel, part_dt=row["part_dt"], rfid_time=row["rfid_time"],
                            timestamp=stamp, year=float(date.year) if date_valid else np.nan,
                            month=float(date.month) if date_valid else np.nan,
                            hour=float(hour) if valid_hour.iloc[pos] else np.nan,
                            fishing_code=label, **flags, **features))
                    if output_rows:
                        writer.write_table(pa.Table.from_pandas(pd.DataFrame(output_rows), schema=SCHEMA, preserve_index=False))
                    db.commit()
                    report["rows"] += len(chunk)
                    valid_dates = dates.dropna()
                    if not valid_dates.empty:
                        lo, hi = valid_dates.min().isoformat(), valid_dates.max().isoformat()
                        report["date_min"] = min(report["date_min"] or lo, lo)
                        report["date_max"] = max(report["date_max"] or hi, hi)
                    print(f"{stem}: {report['rows']:,} output / {report['input_rows']:,} input rows (CSV bytes read: {raw.tell() / info.file_size:.1%})", flush=True)
                    if shutil.disk_usage(output).free < 512 * 1024**2:
                        raise OSError("Less than 512 MiB free; stopped without deleting source data.")
                    if max_rows and report["input_rows"] >= max_rows:
                        break
                else:
                    observed_end = True
        if observed_end and set(report["quarantined_rows"]) != quarantine_rows:
            raise ValueError("Some requested quarantine source rows were not found")
        report["quarantined_count"] = len(report["quarantined_rows"])
        if report["input_rows"] != report["rows"] + report["quarantined_count"]:
            raise ValueError("Input/output/quarantine row counts do not reconcile")
        report.update(counts=dict(counts), geometry_status=dict(statuses), class_rows=dict(classes),
                      vessel_labels={k: sorted(v) for k, v in vessel_labels.items()},
                      vessel_ids=sorted(vessels), complete_file=observed_end,
                      output_bytes=partial.stat().st_size)
        report["class_vessels"] = dict(Counter(code for codes in vessel_labels.values() for code in codes))
        report["inconsistent_labels"] = {k: sorted(v) for k, v in vessel_labels.items() if len(v) > 1}
        partial.rename(target)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        # This is our temporary duplicate-index file, never the competition data.
        if db_path.exists():
            db_path.unlink()
    summarize(output)
    print(f"Saved {target} ({report['rows']:,} rows; complete={observed_end})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/features"))
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--months", type=int, nargs="+")
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means the full file; limit is per CSV")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--quarantine-rows", type=int, nargs="+", default=[],
                        help="Approved original CSV record numbers to preserve separately; one CSV only")
    parser.add_argument("--list", action="store_true", help="List CSV members without extracting")
    args = parser.parse_args()
    if args.chunksize < 1 or args.max_rows < 0:
        parser.error("chunksize must be positive and max-rows must be nonnegative")
    with zipfile.ZipFile(args.zip) as archive:
        selected = list(members(archive, args.split, args.months))
        if not selected:
            parser.error("No matching monthly CSVs")
        if args.quarantine_rows and (len(selected) != 1 or min(args.quarantine_rows) < 1):
            parser.error("quarantine-rows requires one selected CSV and positive source record numbers")
        if args.list:
            for info, split, month in selected:
                print(f"{split} month={month}: {info.file_size:,} bytes | {info.filename}")
            return
        args.output.mkdir(parents=True, exist_ok=True)
        for info, split, month in selected:
            process(archive, info, split, month, args.output, args.chunksize, args.max_rows, args.encoding,
                    args.quarantine_rows)


if __name__ == "__main__":
    main()
