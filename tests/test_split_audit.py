from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from chimera.cli import main
from chimera.split_audit import _index_intervals, _overlapping_rows, audit_splits, canonical


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
        return path

    def test_reordered_rows_numeric_forms_duplicates_and_temporal_boundaries(self):
        left = self.root / "left.jsonl"
        left.write_text('\n{"series":[[1,2]],"future":[[3]],"group":"A","start":0,"end":10,"id":"first"}\n'
                        '{"series":[[4]],"future":[[8]],"group":"A","start":10,"end":20}\n'
                        '{"series":[[1.0,2.00]],"future":[[3e0]],"group":"A","start":100,"end":110}\n')
        right = self.root / "right.jsonl"
        right.write_text('{"future":[[3.000]],"series":[[1e0,2]],"group":"A","start":9,"end":11}\n'
                         '{"series":[[9]],"future":[[5]],"group":"A","start":20,"end":30}\n'
                         '{"series":[[12]],"future":[[13]],"group":"B","start":0,"end":100}\n')
        result = audit_splits({"train": left, "test": right}, interval_fields=("group", "start", "end"))
        pair = result["pairs"][0]
        self.assertFalse(pair["same_jsonl"])
        self.assertEqual(result["splits"]["train"]["duplicate_rows"], 1)
        self.assertEqual((pair["shared_unique_records"], pair["shared_rows_left"], pair["shared_rows_right"]), (1, 2, 1))
        self.assertEqual(pair["examples"], [{"left_line": 2, "right_line": 1}])
        self.assertEqual(pair["intervals"]["groups_with_overlap"], 1)
        self.assertEqual(pair["intervals"]["overlapping_rows_left"], 2)
        self.assertEqual(pair["intervals"]["overlapping_rows_right"], 1)
        self.assertEqual(len(pair["intervals"]["examples"]), 2)
        self.assertNotIn('[[1', json.dumps(result))

    def test_selected_fields_expose_partial_record_reuse_and_must_exist(self):
        left = self.write("a.jsonl", [dict(series=[[1]], future=[[2]], text="first")])
        right = self.write("b.jsonl", [dict(series=[[1.0]], future=[[9]], text="second")])
        self.assertEqual(audit_splits({"a": left, "b": right})["pairs"][0]["shared_unique_records"], 0)
        self.assertEqual(audit_splits({"a": left, "b": right}, fields=["series"])["pairs"][0]["shared_unique_records"], 1)
        with self.assertRaisesRegex(ValueError, "champs absents"):
            audit_splits({"a": left, "b": right}, fields=["missing"])
        for invalid in ([], ["series", "series"], [""]):
            with self.assertRaises(ValueError):
                audit_splits({"a": left, "b": right}, fields=invalid)

    def test_decimal_canonicalization_is_exact_and_types_do_not_collide(self):
        self.assertEqual(canonical(Decimal("1.000e20")), canonical(Decimal("100000000000000000000")))
        self.assertEqual(canonical(Decimal("-0.000")), canonical(Decimal("0")))
        self.assertNotEqual(canonical(Decimal("1234567890123456789012345678901")),
                            canonical(Decimal("1234567890123456789012345678902")))
        self.assertNotEqual(canonical(True), canonical(Decimal(1)))
        self.assertNotEqual(canonical("1"), canonical(Decimal(1)))
        self.assertNotEqual(canonical({"number": "1"}), canonical(Decimal(1)))

    def test_interval_index_matches_brute_force_including_touching_boundaries(self):
        randomizer = random.Random(21)
        for _ in range(40):
            rows_a, rows_b = [], []
            for destination in (rows_a, rows_b):
                for line in range(1, 80):
                    start = randomizer.randrange(-30, 40)
                    destination.append((Decimal(start), Decimal(start + randomizer.randrange(1, 15)), line))
            count, examples = _overlapping_rows(rows_a, _index_intervals(rows_b), 100)
            expected = [a for a in rows_a if any(a[0] < b[1] and b[0] < a[1] for b in rows_b)]
            self.assertEqual(count, len(expected))
            by_a = {row[2]: row for row in rows_a}
            by_b = {row[2]: row for row in rows_b}
            for left_line, right_line in examples:
                a, b = by_a[left_line], by_b[right_line]
                self.assertTrue(a[0] < b[1] and b[0] < a[1])
        self.assertEqual(_overlapping_rows([(Decimal(1), Decimal(2), 1)],
                                           _index_intervals([(Decimal(2), Decimal(3), 2)]), 5)[0], 0)

    def test_cli_fails_on_overlap_after_writing_report_and_protects_inputs(self):
        left = self.write("a.jsonl", [dict(series=[[1]], future=[[2]])])
        right = self.write("b.jsonl", [dict(series=[[1]], future=[[2]])])
        output = self.root / "report.json"
        common = ["audit-splits", "--split", f"train={left}", "--split", f"test={right}"]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                main([*common, "--output", str(output), "--fail-on-overlap"])
            self.assertEqual(failure.exception.code, 2)
            self.assertEqual(json.loads(output.read_text())["pairs"][0]["shared_unique_records"], 1)
            previous = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "champs absents"):
                main([*common, "--fields", "absent", "--output", str(output)])
            self.assertEqual(output.read_bytes(), previous)
            original = left.read_bytes()
            alias = self.root / "alias.jsonl"
            os.link(left, alias)
            with self.assertRaisesRegex(ValueError, "distincte"):
                main([*common, "--output", str(alias)])
            self.assertEqual(left.read_bytes(), original)

    def test_cli_protects_external_images_even_via_hard_link(self):
        image = self.root / "image.png"
        image.write_bytes(b"original image bytes")
        alias = self.root / "alias.png"
        os.link(image, alias)
        left = self.write("a.jsonl", [dict(image_path="image.png", future=[[1]])])
        right = self.write("b.jsonl", [dict(image_path="image.png", future=[[2]])])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "image d'entrée"):
            main(["audit-splits", "--split", f"a={left}", "--split", f"b={right}", "--output", str(alias)])
        self.assertEqual(image.read_bytes(), b"original image bytes")

    def test_invalid_interval_metadata_is_not_silently_ignored(self):
        left = self.write("a.jsonl", [dict(series=[[1]], group="A", start=0, end=2)])
        for values in (dict(group="A", start=2, end=2), dict(group="A", start="0", end=2),
                       dict(group=None, start=0, end=2), dict(group="A", start=0)):
            right = self.write("b.jsonl", [dict(series=[[2]], **values)])
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "b.jsonl:1"):
                audit_splits({"a": left, "b": right}, interval_fields=("group", "start", "end"))

    def test_interval_overlap_alone_sets_failure_status(self):
        left = self.write("a.jsonl", [dict(series=[[1]], group="A", start=0, end=10)])
        right = self.write("b.jsonl", [dict(series=[[2]], group="A", start=9, end=20)])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main(["audit-splits", "--split", f"a={left}", "--split", f"b={right}",
                  "--interval-fields", "group", "start", "end", "--fail-on-overlap"])
        self.assertEqual(failure.exception.code, 2)

    def test_replacement_during_scan_is_rejected(self):
        left = self.write("a.jsonl", [dict(series=[[1]]), dict(series=[[2]])])
        right = self.write("b.jsonl", [dict(series=[[3]])])
        replacement = self.write("replacement.jsonl", [dict(series=[[4]]), dict(series=[[5]])])
        original_open = Path.open

        class ReplacingReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __iter__(self):
                for line in self.stream:
                    if replacement.exists():
                        os.replace(replacement, left)
                    yield line

        def open_with_replacement(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ReplacingReader(stream) if path == left else stream

        with patch.object(Path, "open", open_with_replacement):
            with self.assertRaisesRegex(ValueError, "changé pendant l'audit"):
                audit_splits({"a": left, "b": right})


if __name__ == "__main__":
    unittest.main()
