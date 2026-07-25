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
- `NOTICE`: third-party data licensing and attribution (StrongREJECT)
- `LICENSE`: MIT license for the project

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

The scorer produces bypass, reconstruction, execution, and failure-mode fields. Failure modes use the paper labels `BI`, `DPF`, `PR`, `RAR`, and `OTH`.

```bash
rogueprompt-evaluate validate path/to/evaluation_records.jsonl
rogueprompt-evaluate score path/to/evaluation_records.jsonl --summary-json summary.json
rogueprompt-evaluate score path/to/evaluation_records.jsonl --conditions conditions.csv
```

### Hybrid evaluator

The scoring pipeline follows the staged, hybrid labeling procedure described in the paper: a rule-based bypass check, a semantic-similarity signal, and an LLM judge that assigns the reconstruction, execution, and failure-mode labels.

- **Similarity signal (`--similarity`).** The `jina` backend embeds the original request (`retrieval.query`) and chunks of the model response (`retrieval.passage`) with `jinaai/jina-embeddings-v3` and supplies the maximum and top-three mean cosine similarity as advisory signals to the judge. Install it with the `embeddings` extra:

  ```bash
  pip install -e '.[embeddings]'
  ```

  The default `auto` backend uses `jina` when it is installed and otherwise falls back to a dependency-free `difflib` approximation, so the package runs without the model. The `difflib` fallback is an approximation and does not reproduce the paper's reported rates.

- **LLM judge (`--judge-endpoint`).** Point the evaluator at an OpenAI-compatible endpoint (e.g. a self-hosted `Llama-3.3-70B-Instruct` deployment, as in the paper). The judge receives the original prompt, full response, and the auxiliary signals; the transformed attack prompt and the target-model/provider identities are withheld. Install the client with the `judge` extra and pass the API key via an environment variable:

  ```bash
  pip install -e '.[judge]'
  export ROGUEPROMPT_JUDGE_API_KEY=...   # read from the env, never passed as a flag
  rogueprompt-evaluate score records.jsonl --similarity jina \
      --judge-endpoint https://your-deployment/v1 --judge-model llama-3.3-70b-instruct \
      --summary-json summary.json
  ```

Without `--judge-endpoint`, the evaluator uses the deterministic fallback (similarity threshold plus refusal patterns) so it still runs offline. Install both extras with `pip install -e '.[all]'`.

Alternatively, generate classification-only judge requests for an external judge and apply the returned decisions:

```bash
rogueprompt-evaluate judge-requests path/to/evaluation_records.jsonl --output judge_requests.jsonl
rogueprompt-evaluate score path/to/evaluation_records.jsonl --judge-decisions decisions.jsonl --summary-json summary.json
```

## Verification

Run these checks before publishing:

```bash
python -m compileall -q rogueprompt
```

Also confirm that all seven RoguePrompt variant files have 313 records and matching source metadata.

## Attribution

The source prompts are the StrongREJECT benchmark (Souly et al., 2024),
313 prompts across six categories. See NOTICE for licensing and citation.

Baseline transformations follow Zeng et al. (PAP), Kang et al. (auto payload
splitting), and StrongREJECT (disemvowel, base64 raw).
