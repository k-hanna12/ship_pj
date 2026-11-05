"""Export completed training feature files as Excel-readable UTF-8 BOM CSVs."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq


def check_csv(path, expected_rows, expected_columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != expected_columns:
            raise ValueError(f"CSV columns do not match: {path}")
        count = 0
        for row in reader:
            if len(row) != len(expected_columns):
                raise ValueError(f"Malformed CSV record: {path}, row {count + 1}")
            count += 1
    if count != expected_rows:
        raise ValueError(f"CSV row count {count} != {expected_rows}: {path}")


def export_csv(source, target):
    source, target = Path(source), Path(target)
    parquet = pq.ParquetFile(source)
    expected = parquet.metadata.num_rows
    columns = parquet.schema_arrow.names
    if target.exists():
        check_csv(target, expected, columns)
        print(f"Verified existing {target}: {expected:,} rows", flush=True)
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.csv")
    count = 0
    # Exclusive creation preserves any incomplete earlier export for inspection.
    with partial.open("x", encoding="utf-8-sig", newline="") as handle:
        for batch in parquet.iter_batches(batch_size=10000):
            if shutil.disk_usage(target.parent).free < 512 * 1024**2:
                raise OSError("Less than 512 MiB free; CSV export stopped.")
            frame = batch.to_pandas()
            frame.to_csv(handle, index=False, header=count == 0,
                         date_format="%Y-%m-%d %H:%M:%S")
            count += len(frame)
        if count == 0:
            csv.writer(handle).writerow(columns)
    check_csv(partial, expected, columns)
    partial.rename(target)
    print(f"Saved {target}: {count:,} rows", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_january"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    sources = []
    for month in range(1, 7):
        source = args.feature_dir / f"train_{month:02d}.parquet"
        report = json.loads(source.with_suffix(".quality.json").read_text(encoding="utf-8"))
        if not report.get("complete_file") or pq.ParquetFile(source).metadata.num_rows != report["rows"]:
            raise ValueError(f"Training month {month} is incomplete or inconsistent")
        sources.append(source)
    results = []
    for source in sources:
        target = args.output_dir / f"{source.stem}_features.csv"
        rows = export_csv(source, target)
        results.append({"file": target.name, "rows": rows, "bytes": target.stat().st_size})
    (args.output_dir / "csv_export_complete.json").write_text(
        json.dumps({"complete": True, "files": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
