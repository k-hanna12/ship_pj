import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from preprocess import GEOD, geometry_features, process, SCHEMA


class PreprocessingChecks(unittest.TestCase):
    def test_disconnected_parts_and_repeats(self):
        result = geometry_features("MULTILINESTRING ((126 35, 126 35, 126.01 35), (129 35, 129.01 35))")
        expected = GEOD.inv(126, 35, 126.01, 35)[2] + GEOD.inv(129, 35, 129.01, 35)[2]
        self.assertAlmostEqual(result["distance_total_m"], expected, places=6)
        self.assertAlmostEqual(result["duplicate_point_ratio"], 1 / 3)
        self.assertTrue(np.isnan(result["turn_mean_deg"]))
        stopped = geometry_features("LINESTRING (126 35, 126 35)")
        self.assertEqual(stopped["distance_total_m"], 0)
        self.assertTrue(stopped["feature_ok"])

    def test_invalid_geometry_and_turn(self):
        for wkt, status in [("", "missing"), ("garbage", "parse_failed"),
                            ("MULTILINESTRING EMPTY", "empty"),
                            ("LINESTRING (200 35, 201 35)", "invalid_coordinates"),
                            ("POINT (126 35)", "unsupported_type")]:
            self.assertEqual(geometry_features(wkt)["geometry_status"], status)
        turn = geometry_features("LINESTRING (126 35, 126.01 35, 126.01 35.01)")
        self.assertAlmostEqual(turn["turn_mean_deg"], 90, delta=0.1)

    def test_streaming_preserves_ids_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            csv = pd.DataFrame([
                ["20260101", "001", "0", "LINESTRING (126 35, 126.01 35)", "A"],
                ["20260101", "001", "0", "LINESTRING (126 35, 126.01 35)", "A"],
                ["20260102", "001", "24", "bad", "B"],
            ], columns=["part_dt", "rfid_id", "rfid_time", "geom", "fishing_code"])
            archive_path = root / "input.zip"
            with zipfile.ZipFile(archive_path, "w") as z:
                z.writestr("train/vpass_1월 데이터.csv", csv.to_csv(index=False))
            with zipfile.ZipFile(archive_path) as z:
                process(z, z.infolist()[0], "train", 1, root, 1, 0, "utf-8-sig")
            table = pq.read_table(root / "train_01.parquet")
            self.assertEqual(table.schema, SCHEMA)
            data = table.to_pandas()
            self.assertEqual(data.rfid_id.tolist(), ["001"] * 3)
            self.assertEqual(data.duplicate_record.tolist(), [False, True, False])
            self.assertEqual(data.source_row.tolist(), [1, 2, 3])
            self.assertTrue(data.invalid_datetime.iloc[2])
            report = json.loads((root / "train_01.quality.json").read_text(encoding="utf-8"))
            self.assertTrue(report["complete_file"])
            self.assertEqual(report["inconsistent_labels"], {"001": ["A", "B"]})


if __name__ == "__main__":
    unittest.main()
