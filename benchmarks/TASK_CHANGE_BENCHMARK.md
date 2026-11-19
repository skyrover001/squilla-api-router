# Task-change intent benchmark

Offline, deterministic benchmark for New / Continue / Redo / Task Changed.
It does not call the router HTTP service or any model provider.

## Run

```powershell
python benchmarks\benchmark_task_change.py --repeat 3 --sweep
python benchmarks\benchmark_task_change_hybrid.py --repeat 3
```

Outputs:

- console summary;
- `benchmarks/task_change_benchmark_latest.json`;
- `benchmarks/task_change_hybrid_latest.json`;
- deterministic decision SHA-256 for repeated runs.

## Compared strategies

- `current_rule`: existing production keyword + Jaccard implementation.
- `semantic_v1`: explicit rules plus local
  `BAAI/bge-small-zh-v1.5` cosine similarity against a rolling task goal.

The semantic strategy in this benchmark is experimental. It is not connected
to the running router.

## Dataset

`task_change_cases.jsonl` currently contains 53 curated cases covering:

- explicit Continue / Redo / New Task;
- short follow-ups;
- tool-result continuation;
- repeated prompts;
- same-domain but different tasks;
- cross-domain task changes;
- system prompt or tool-schema changes.

## Metrics

- accuracy and macro F1;
- confusion matrix;
- `false_split_rate`: same-task turns incorrectly split;
- `missed_change_rate`: real task changes incorrectly kept;
- semantic similarity medians for same-task and changed-task cases;
- threshold sensitivity sweep.

The sweep is in-sample diagnostic only. Do not copy its winning threshold into
production without a separate holdout set.

## Initial result

As of the initial 53-case baseline:

| Strategy | Accuracy | Macro F1 | False split | Missed change |
|---|---:|---:|---:|---:|
| current rule | 79.25% | 0.631 | 27.59% | 12.50% |
| BGE semantic v1 | 67.92% | 0.569 | 31.03% | 33.33% |

The best in-sample BGE threshold combination reached 81.13% accuracy with zero
false splits, but missed 41.67% of real task changes. That trade-off is not
acceptable for production.

The initial result therefore rejects raw BGE cosine similarity as a standalone
task-boundary classifier. The next candidate must combine stronger
continuation/task-shift signals or use a small supervised task-boundary model,
and it must be validated on a holdout set.

## Independent holdout and hybrid candidate

`task_change_holdout.jsonl` contains 52 cases that are not used to train the
hybrid classifier. The candidate keeps explicit/session/tool rules
deterministic and trains a small binary logistic model only for ambiguous user
turns.

| Strategy on 52 holdout cases | Accuracy | Macro F1 | False split | Missed change |
|---|---:|---:|---:|---:|
| current rule | 65.38% | 0.552 | 45.45% | 26.67% |
| BGE semantic v1 | 71.15% | 0.597 | 13.64% | 40.00% |
| rules + small classifier (`hybrid_v2`) | 92.31% | 0.705 | 18.18% | 0.00% |

The hybrid candidate is promising but still experimental:

- it eliminates missed task changes in this holdout;
- it still splits 18.18% of same-task turns;
- 28 ambiguous training cases are too few for production confidence;
- three repeated runs produced the same decision SHA-256.

Do not enable `hybrid_v2` in the router until it is validated on substantially
more real multi-turn sessions and its model artifacts have a stable versioned
format.
