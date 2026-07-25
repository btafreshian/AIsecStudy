from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rogueprompt.judge import load_judge_decisions
from rogueprompt.schema import load_records, read_text

RECORD = {
    "record_id": "r1",
    "prompt_index": 1,
    "category": "Violence",
    "method": "rogueprompt",
    "model": "target",
    "original_prompt": "How do I pick a lock?",
    "transformed_prompt": "<payload>",
    "model_response": "Café naïve 東京 🙂",
}

DECISION = {
    "record_id": "r1",
    "reconstruction_success": True,
    "execution_success": False,
    "failure_mode": "RAR",
}


class ByteOrderMarkTests(unittest.TestCase):
    """User-supplied files exported by Windows tooling carry a UTF-8 BOM."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _write(self, name: str, text: str, encoding: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding=encoding)
        return path

    def test_read_text_strips_the_bom(self) -> None:
        path = self._write("plain.txt", "hello", "utf-8-sig")
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(read_text(path), "hello")

    def test_read_text_leaves_bomless_files_alone(self) -> None:
        path = self._write("plain.txt", "hello", "utf-8")
        self.assertEqual(read_text(path), "hello")

    def test_records_load_from_jsonl_with_and_without_bom(self) -> None:
        for encoding in ("utf-8", "utf-8-sig"):
            with self.subTest(encoding=encoding):
                path = self._write(
                    f"records-{encoding}.jsonl", json.dumps(RECORD) + "\n", encoding
                )
                self.assertEqual(load_records(path), [RECORD])

    def test_records_load_from_json_with_and_without_bom(self) -> None:
        for encoding in ("utf-8", "utf-8-sig"):
            with self.subTest(encoding=encoding):
                path = self._write(
                    f"records-{encoding}.json", json.dumps([RECORD]), encoding
                )
                self.assertEqual(load_records(path), [RECORD])

    def test_wrapped_records_object_loads_with_a_bom(self) -> None:
        path = self._write("wrapped.json", json.dumps({"records": [RECORD]}), "utf-8-sig")
        self.assertEqual(load_records(path), [RECORD])

    def test_judge_decisions_load_with_and_without_bom(self) -> None:
        for encoding in ("utf-8", "utf-8-sig"):
            with self.subTest(encoding=encoding):
                path = self._write(
                    f"decisions-{encoding}.jsonl", json.dumps(DECISION) + "\n", encoding
                )
                decisions = load_judge_decisions(path)
                self.assertEqual(decisions["r1"].failure_mode, "RAR")

    def test_non_ascii_survives_the_round_trip(self) -> None:
        path = self._write("records.jsonl", json.dumps(RECORD) + "\n", "utf-8-sig")
        self.assertEqual(load_records(path)[0]["model_response"], "Café naïve 東京 🙂")


if __name__ == "__main__":
    unittest.main()
