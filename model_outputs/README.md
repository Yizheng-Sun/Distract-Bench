# Model Outputs

This directory contains corrupt-side model outputs for the 506 public
Distract-Bench samples.

## Files

```text
model_outputs/
  corrupt_results/
    <model>/
      <run_id>_samples_logical_robustness_bench.jsonl
  corruption_reference/
    <model>/
      <run_id>_samples_logical_robustness_bench_ref_judged.jsonl
  corrupt_results_summary.json
  corruption_reference_summary.json
```

Each corrupt-result JSONL file has 506 rows, sorted by the public numeric sample
id:

- `final_id`: public Distract-Bench sample id, `"1"` through `"506"`.
- `doc_id`: same public sample id as an integer.
- `input_media`: rewritten to `data/images/<final_id>.png`.
- `target`: gold answer from `data/questions.jsonl`.
- `filtered_resps`: model output after the evaluation harness's response
  extraction/filtering.
- `resps`: raw response when available.
- `llm_as_judge_eval.question_id` and `submission.question_id`: rewritten to
  the public numeric sample id.

The actual images are not included in this GitHub repository.

## Corruption Reference Judgments

`model_outputs/corruption_reference/` contains LLM-judged
distractor-reference labels for the corrupt-side model outputs. Each per-model
JSONL file also has 506 rows sorted by public `final_id`. These labels can be
regenerated with `scripts/judge_distractor_reference.py`; see the root README
for the API command.

- `verdict`: `yes` if the model explicitly refers to the injected distractor,
  `no` otherwise. One retained source judgment has `parse_error`.
- `distraction`, `edit_instruction`, and `trap_answer`: copied from
  `data/edit_instructions.jsonl`.

Aggregate per-model DRR/HFR CSVs and the original reference report are copied
alongside the per-model files.
