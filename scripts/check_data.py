"""Check that data/ still matches the code that generated it.

The prompt files are the artifact the paper reports on, so they are checked in
rather than generated at run time. That only holds if they still regenerate
byte-for-byte from the current transforms: a change to a wrapper string or to
the serialization would otherwise leave data/ describing prompts nobody ran.

Run directly, or through the tests workflow:

    python scripts/check_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rogueprompt import transforms as T  # noqa: E402
from rogueprompt.versions import BASELINE_TEMPLATES  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

GENERATORS = {
    "rogueprompt_original": T.generate_rogueprompt,
    "rogueprompt_no_rot13": T.generate_no_rot13_prompt,
    "rogueprompt_no_splitting": T.generate_no_splitting_prompt,
    "rogueprompt_no_vigenere": T.generate_no_vigenere_prompt,
    "rogueprompt_rot13_only": T.generate_rot13_only_prompt,
    "rogueprompt_splitting_only": T.generate_splitting_only_prompt,
    "rogueprompt_vigenere_only": T.generate_vigenere_only_prompt,
}

SHARED_COLUMNS = ("category", "source", "forbidden_prompt")


def _load(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def main() -> int:
    failures: list[str] = []
    source = _load("source_prompts")
    by_index = {row["index"]: row for row in source}
    print(f"source_prompts: {len(source)} rows")

    for name, generate in GENERATORS.items():
        rows = _load(name)
        stale = [r["index"] for r in rows if generate(r["forbidden_prompt"]) != r["jailbroken_prompt"]]
        drift = [
            r["index"]
            for r in rows
            if any(r[c] != by_index[r["index"]][c] for c in SHARED_COLUMNS)
        ]
        status = "ok" if not (stale or drift) else "STALE"
        print(f"  {name:28} n={len(rows):4} {status}")
        if stale:
            failures.append(f"{name}: {len(stale)} rows do not regenerate: {stale[:5]}")
        if drift:
            failures.append(f"{name}: {len(drift)} rows drift from source_prompts: {drift[:5]}")

    baselines = _load("baseline_prompts")
    columns = tuple(c for c in baselines[0] if c.startswith("jailbroken_prompt"))
    if columns != BASELINE_TEMPLATES:
        failures.append(f"baseline columns {columns} != declared {BASELINE_TEMPLATES}")
    else:
        print(f"  {'baseline_prompts':28} n={len(baselines):4} ok")

    round_trips = 0
    for row in source:
        prompt = T.generate_rogueprompt(row["forbidden_prompt"])
        expected = unicodedata.normalize("NFC", row["forbidden_prompt"])
        if T.reconstruct_rogueprompt(prompt) != expected:
            failures.append(f"round trip failed for index {row['index']}")
        else:
            round_trips += 1
    print(f"round trips: {round_trips}/{len(source)}")

    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
