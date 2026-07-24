# RoguePrompt

This repository accompanies **RoguePrompt: Dual-Layer Encoding for Self-Reconstruction to Circumvent LLM Moderation** (under review; author, venue, and preprint details withheld for anonymous review).

It contains the prompt construction code, the prompt JSON files used for the paper-side comparisons, and a small evaluation protocol for user-supplied records.

## Contents

- `rogueprompt/transforms.py`: RoguePrompt and ablation transformations.
- `rogueprompt/schema.py`, `rogueprompt/scorers.py`, `rogueprompt/aggregate.py`, `rogueprompt/judge.py`, `rogueprompt/cli.py`: schema checks, staged scoring, aggregation, judge-request helpers, and CLI.
- `data/source_prompts.json`: 313 source prompts with category and source metadata.
- `data/rogueprompt_*.json`: full method and six ablation prompt sets.
- `data/baseline_prompts.json`: five baseline prompt columns used for comparison.
- `ETHICS.md`: responsible-use notes.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Prompt Transformations

The full method splits a source prompt into even- and odd-indexed word streams, encrypts the odd stream with the Vigenere key `RESEARCH`, places reconstruction instructions around the split text, and applies an outer ROT13 layer.

```python
from rogueprompt.transforms import generate_rogueprompt

prompt = generate_rogueprompt("source prompt text")
```

The six ablations correspond to the paper's leave-one-component-out and single-component rows:

- `rogueprompt_no_rot13.json`: Vigenere + split.
- `rogueprompt_no_splitting.json`: ROT13 + Vigenere.
- `rogueprompt_no_vigenere.json`: ROT13 + split.
- `rogueprompt_rot13_only.json`: ROT13 only.
- `rogueprompt_splitting_only.json`: splitting only.
- `rogueprompt_vigenere_only.json`: Vigenere only.

## Paper Map

| Paper component | Repository files |
| --- | --- |
| Full method | `data/rogueprompt_original.json` |
| Ablation study | `data/rogueprompt_no_rot13.json`, `data/rogueprompt_no_splitting.json`, `data/rogueprompt_no_vigenere.json`, `data/rogueprompt_rot13_only.json`, `data/rogueprompt_splitting_only.json`, `data/rogueprompt_vigenere_only.json` |
| Baselines | `data/baseline_prompts.json` |
| Scoring protocol | `rogueprompt/schema.py`, `rogueprompt/scorers.py`, `rogueprompt/aggregate.py`, `rogueprompt/judge.py`, `rogueprompt-evaluate` |
| Reported rates | Recomputed by applying the evaluator to a completed evaluation record set |

## Data Files

`data/source_prompts.json`

Rows: 313

Fields:

- `index`: integer prompt identifier.
- `category`: policy/category label used for grouping.
- `source`: upstream source label.
- `forbidden_prompt`: source prompt used to generate prompt variants.

Each RoguePrompt variant file has 313 rows:

- `index`
- `category`
- `source`
- `forbidden_prompt`
- `jailbroken_prompt`

`data/baseline_prompts.json` also has 313 rows:

- `index`
- `category`
- `source`
- `forbidden_prompt`
- `jailbroken_prompt_pair`
- `jailbroken_prompt_pap_authority_endorsement`
- `jailbroken_prompt_auto_payload_splitting`
- `jailbroken_prompt_disemvowel`
- `jailbroken_prompt_base64_raw`

## Evaluation Protocol

The evaluator works with JSON or JSONL records supplied by the user. Required input fields are:

- `record_id`
- `prompt_index`
- `category`
- `method`
- `model`
- `original_prompt`
- `transformed_prompt`

The scorer produces bypass, reconstruction, execution, and failure-mode fields. Failure modes use the paper labels `DPF`, `PR`, `RAR`, and `OTH`.

```bash
rogueprompt-evaluate validate path/to/evaluation_records.jsonl
rogueprompt-evaluate score path/to/evaluation_records.jsonl --summary-json summary.json
```

Create classification-only judge requests for an external LLM judge:

```bash
rogueprompt-evaluate judge-requests path/to/evaluation_records.jsonl --output judge_requests.jsonl
```

## Verification

Run these checks before publishing:

```bash
python -m compileall -q rogueprompt
```

Also confirm that all seven RoguePrompt variant files have 313 records and matching source metadata.

## Citation

If you use this repository, please cite the accompanying paper and this repository. Citation details are withheld during anonymous review and will be added on acceptance.
