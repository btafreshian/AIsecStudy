# RoguePrompt

This repository accompanies **RoguePrompt: Dual-Layer Encoding for Self-Reconstruction to Circumvent LLM Moderation**. It builds the transformed prompts,
ships the prompt sets used in the paper, and scores separately collected model
responses. RoguePrompt splits a forbidden prompt into two span streams, Vigenère-
encrypts one, ROT13s the assembled payload, and wraps it in natural-language
reconstruction instructions.

**See the paper for the method, threat model, evaluation design, and results.**
This README only covers running the code.

## Responsible Use

RoguePrompt studies transformation-based moderation bypass. The code and prompt
files are intended for controlled research, paper review, and independent
analysis by readers working under their own institutional and provider rules.

The evaluator and judge-request helpers classify completed records. They are
meant to make the paper's scoring protocol easier to inspect and rerun on
separately collected data.

## Contents

- `rogueprompt/transforms.py`: RoguePrompt and its six ablation transformations.
- `rogueprompt/schema.py`, `rogueprompt/scorers.py`, `rogueprompt/aggregate.py`, `rogueprompt/judge.py`, `rogueprompt/cli.py`: schema checks, staged scoring, aggregation, judge-request helpers, and CLI.
- `rogueprompt/lexical.py`: the regular-expression and lexical checks of Section 5.2.
- `rogueprompt/semantic.py`: the embedding-similarity signal of Section 5.2.
- `rogueprompt/versions.py`: the per-component versions logs record.
- `data/source_prompts.json`: 313 StrongREJECT source prompts with category and source metadata.
- `data/rogueprompt_*.json`: full method and six ablation prompt sets.
- `data/baseline_prompts.json`: five baseline prompt columns used for comparison.
- `NOTICE`: third-party data licensing and attribution (StrongREJECT)
- `LICENSE`: MIT license for the project

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . # add '.[embeddings]', '.[judge]', or '.[all]' for optional extras
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e . # add '.[embeddings]', '.[judge]', or '.[all]' for optional extras
```
## Run

**Generate a transformed prompt.** `generate_rogueprompt` is the full method; the
six ablations have one generator each (`generate_no_rot13_prompt`, etc.).

```python
from rogueprompt.transforms import generate_rogueprompt
prompt = generate_rogueprompt("source prompt text")
```

**Run the tests** (includes the paper's worked demo vector):

```bash
python -m unittest discover -s tests -v
```

**Score response records.** Supply JSON/JSONL records with at least `record_id`,
`prompt_index`, `category`, `method`, `model`, `original_prompt`, and
`transformed_prompt`; include observable status/error fields (`status_code`,
`finish_reason`, `block_reason`, …) for accurate block detection. Scoring needs a
judge — without `--judge-endpoint` or `--judge-decisions` it stops on the first
response it cannot label (`--labels-only` leaves them unlabeled instead). The
labeling procedure, failure modes (`BI`/`DPF`/`PR`/`RAR`/`OTH`), and block rules
are defined in Section 5.2 of the paper.


```bash
rogueprompt-evaluate validate records.jsonl        # check record format
rogueprompt-evaluate versions                       # frozen component versions (Section 4.5)

# inline judge (OpenAI-compatible endpoint; key from env, never a flag)
export ROGUEPROMPT_JUDGE_API_KEY=...
rogueprompt-evaluate score records.jsonl --similarity jina \
    --judge-endpoint https://your-deployment/v1 --judge-model llama-3.3-70b-instruct \
    --summary-json summary.json

# or judge offline, then read the replies back (mutually exclusive with the above)
rogueprompt-evaluate judge-requests records.jsonl --output requests.jsonl --similarity jina
rogueprompt-evaluate score records.jsonl --judge-decisions decisions.jsonl --summary-json summary.json
```

`--similarity jina` matches the paper (needs `[embeddings]`); the default `auto`
falls back to a `difflib` approximation that does not reproduce the paper's setup.

Full flag reference: `rogueprompt-evaluate <command> --help`. Labeling rationale
and failure modes: paper &sect;5.2 and Algorithm 2.


## Paper Map

| Paper component | Repository files |
| --- | --- |
| Full method | `data/rogueprompt_original.json` |
| Ablation study | `data/rogueprompt_no_rot13.json`, `data/rogueprompt_no_splitting.json`, `data/rogueprompt_no_vigenere.json`, `data/rogueprompt_rot13_only.json`, `data/rogueprompt_splitting_only.json`, `data/rogueprompt_vigenere_only.json` |
| Baselines | `data/baseline_prompts.json` |
| Scoring protocol | `rogueprompt/schema.py`, `rogueprompt/scorers.py`, `rogueprompt/aggregate.py`, `rogueprompt/judge.py`, `rogueprompt-evaluate` |
| Rule-based checks (Section 5.2) | `rogueprompt/lexical.py` |
| Run provenance (Section 4.5) | `rogueprompt/versions.py`, `rogueprompt-evaluate versions` |

### Prompt Transformations

The full method runs seven steps:

1. Normalize the source prompt to Unicode NFC.
2. Cut it into spans. A span is one run of non-whitespace plus the whitespace after it; any whitespace at the very start becomes its own span, since it has no word to attach to. The spans concatenate back to the input exactly.
3. Partition the spans into an even and an odd stream by position, counting from zero.
4. Write each stream out as `len(span):span` entries joined by `|`. Lengths are in Unicode code points  after normalization.
5. Encrypt the serialized odd stream with the Vigenere key `RESEARCH`. Only ASCII letters are shifted; case is preserved and the key advances only on the letters it shifts, so the length prefixes and separators survive intact.
6. Assemble `EVEN=<serialized-even>;ODD=<encrypted-serialized-odd>;KEY=<key>;ORDER=0-even`.
7. ROT13 the whole payload. The reconstruction instructions around it stay in plain text.

```python
from rogueprompt.transforms import generate_rogueprompt

prompt = generate_rogueprompt("source prompt text")
```

The six ablations correspond to the paper's leave-one-component-out and single-component rows:

| Variant (Table 4) | File | Kept components |
| --- | --- | --- |
| No ROT13 (Vig+Split) | `rogueprompt_no_rot13.json` | Vigenère + split |
| No Splitting (Vig+ROT13) | `rogueprompt_no_splitting.json` | Vigenère + ROT13 |
| No Vigenère (ROT13+Split) | `rogueprompt_no_vigenere.json` | ROT13 + split |
| ROT13 only | `rogueprompt_rot13_only.json` | ROT13 |
| Splitting only | `rogueprompt_splitting_only.json` | split |
| Vigenère only | `rogueprompt_vigenere_only.json` | Vigenère |


### Data

- `data/source_prompts.json` — 313 StrongREJECT prompts (`index`, `category`,
  `source`, `forbidden_prompt`).
- `data/rogueprompt_*.json` — full method + six ablation prompt sets, each adding
  a `jailbroken_prompt` column. Files map to the paper's ablation rows (Table 4).
- `data/baseline_prompts.json` — the five baselines, one `jailbroken_prompt_*`
  column each.


## Attribution

The source prompts are the StrongREJECT benchmark (Souly et al., 2024),
313 prompts across six categories. See NOTICE for licensing and citation.

Baseline transformations follow Zeng et al. (PAP), Kang et al. (auto payload
splitting), and StrongREJECT (disemvowel, base64 raw).


## AI Use Disclosure
Code in this repository was written by the authors; Claude Code was used to refactor it. The authors reviewed all AI-assisted output and take full responsibility for it. This matches the paper's Usage of Generative AI statement.
