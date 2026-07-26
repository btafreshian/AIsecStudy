# Changelog

Component versions are declared in `rogueprompt/versions.py` and are what
Section 4.5 asks a log to record. They move independently of the distribution
version in `pyproject.toml`.

The prompt files in `data/` were generated under `wrapper` 1.0, `key` 1.0,
`serialization` 1.0, `baseline_template` 1.0, and `parser` 1.0, and none of
those components has changed since.

## evaluator

### 4.0

- Both judge paths now finish through the same `FinalizeLabel` step
  (Algorithm 2 line 15). Previously `score --judge-endpoint` used the judge's
  categorical outcome verbatim while `score --judge-decisions` re-derived one
  when the judge left it null, so the same judge reply could produce
  `failure_mode=None` on one path and `RAR` on the other. The derived label is
  the paper-aligned behaviour; the integrated path was the deviation.
- `HybridEvaluator` is the single scoring entry point and delegates the staged
  rules to `score_record`. The block-to-BI rule and "execution implies
  reconstruction" each had three implementations and now have one.
- Every scored record carries the similarity signals under the same names the
  judge prompt uses (`max_similarity`, `top3_mean_similarity`, `num_chunks`).
  They were previously attached only on some branches and under a second set
  of names.
- The component digest now covers the package code reachable from the named
  objects, not just the objects themselves. Editing a helper such as
  `lexical._normalize_reason` or `scorers.coerce_bool` changes labels, and
  those edits previously left `code_id` unchanged.
- `FAILURE_MODES` is ordered as Table 2 lists the labels. This changes the
  column order of aggregate output, not any label.
- `scorers._bool_or_none` is now `scorers.coerce_bool`: two other modules
  import it, so it was never private in practice.

### 3.0

Moved the Section 5.2 regex and lexical checks into the code. Block detection
reads the provider status and error fields instead of trusting a supplied
boolean, and the judge receives the regex signals next to the similarity ones.

### 2.0

Dropped the offline similarity/word-count labeler. Section 5.2 states the
continuous signals "did not independently determine a label", and the paper
defines no offline stand-in for the judge.

### 1.0

Initial staged scorer.

## wrapper, key, serialization, baseline_template, parser

### 1.0

Initial release; unchanged since. These components fix prompt construction, so
a change to any of them invalidates the correspondence between `data/` and the
prompts the paper reports on.
