# Distract-Bench DRR Report

_Generated: 2026-05-23 12:22_

DRR is the fraction of corrupt-side Distract-Bench outputs whose reasoning or answer explicitly refers to the injected semantic distractor. This report also joins DRR with Distract-Bench correctness to separate harmless/recoverable reference from harmful reference.

## Metrics

- `DRR = referenced / joined samples`.
- `RRR = Reference Recovery Rate = P(Distract-Bench correct | referenced)`.
- `HFR = Harmful Reference Rate = P(referenced and Distract-Bench wrong)` over joined samples.
- `failure_given_reference = P(Distract-Bench wrong | referenced) = 1 - RRR`.

## At a glance

- Models evaluated: 8
- Joined DRR/correctness samples: 4,047
- Referenced distractor: 1,110 (27.4%)
- Referenced and correct: 663 (59.7% of referenced)
- Referenced and wrong: 447 (11.0% HFR)
- Pooled Distract-Bench accuracy on joined samples: 72.8%

## Reference Outcome Matrix

| Reference status | Distract-Bench correct | Distract-Bench wrong | Total |
|---|---|---|---|
| Referenced | 663 | 447 | 1110 |
| Not referenced | 2282 | 655 | 2937 |

## Per-model Outcome-aware DRR

Sorted by HFR descending. High DRR alone means the model mentions or uses the distractor; high HFR means those references coincide with wrong Distract-Bench answers.

| Model | n | Distract-Bench acc | Ref / DRR | Ref+correct / RRR | Ref+wrong / HFR | P(wrong\|ref) | P(correct\|no-ref) | P(ref\|wrong) |
|---|---|---|---|---|---|---|---|---|
| Fancy-MLLM__R1-Onevision-7B | 506 | 61.3% | 127 / 25.1% | 50 / 39.4% | 77 / 15.2% | 60.6% | 68.6% | 39.3% |
| Qwen__Qwen3-VL-8B-Thinking | 506 | 81.0% | 197 / 38.9% | 132 / 67.0% | 65 / 12.8% | 33.0% | 90.0% | 67.7% |
| ydeng9__OpenVLThinker-7B-v1.2 | 506 | 69.4% | 134 / 26.5% | 77 / 57.5% | 57 / 11.3% | 42.5% | 73.7% | 36.8% |
| Osilly__Vision-R1-7B | 506 | 66.0% | 96 / 19.0% | 40 / 41.7% | 56 / 11.1% | 58.3% | 71.7% | 32.6% |
| MMR1__MMR1-7B-RL | 506 | 75.3% | 159 / 31.4% | 104 / 65.4% | 55 / 10.9% | 34.6% | 79.8% | 44.0% |
| FanqingM__MM-Eureka-Qwen-7B | 506 | 71.9% | 111 / 21.9% | 61 / 55.0% | 50 / 9.9% | 45.0% | 76.7% | 35.2% |
| Qwen__Qwen2.5-VL-7B-Instruct | 505 | 73.7% | 96 / 19.0% | 51 / 53.1% | 45 / 8.9% | 46.9% | 78.5% | 33.8% |
| Qwen__Qwen3-VL-8B-Instruct | 506 | 83.6% | 190 / 37.5% | 148 / 77.9% | 42 / 8.3% | 22.1% | 87.0% | 50.6% |

## Quality Flags

- `Qwen__Qwen2.5-VL-7B-Instruct`: 1 DRR parse/error verdicts excluded.

## Companion Files

- `per_model_rates.csv` — raw DRR per model
- `drr_outcomes_per_model.csv` — DRR joined with Distract-Bench correctness
