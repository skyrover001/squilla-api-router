"""Train and package the local task-boundary model for safe routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from squilla_api_router.task_change_policy import (
    DEFAULT_BGE_DIR,
    DEFAULT_MODEL_PATH,
    hard_rule_decision,
    numeric_features,
    semantic_similarities,
)
from squilla_api_router.v4_runtime.bge_onnx import OnnxBGE


DEFAULT_TRAIN = ROOT / "benchmarks" / "task_change_train_v2.jsonl"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train(
    train_path: Path,
    output_path: Path,
    bge_dir: Path,
) -> dict[str, Any]:
    all_cases = _load_cases(train_path)
    cases = [
        case
        for case in all_cases
        if hard_rule_decision(case) is None
        and case["expected"] in {"continue", "redo", "task_changed"}
    ]
    encoder = OnnxBGE(bge_dir)
    numeric: list[list[float]] = []
    texts: list[str] = []
    labels: list[int] = []
    for case in cases:
        goal_similarity, anchor_similarity = semantic_similarities(encoder, case)
        numeric.append(
            numeric_features(
                case,
                goal_similarity=goal_similarity,
                anchor_similarity=anchor_similarity,
            )
        )
        texts.append(str(case["latest"]))
        labels.append(1 if case["expected"] == "task_changed" else 0)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=1,
        max_features=512,
        sublinear_tf=True,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(texts)
    matrix = np.hstack(
        [np.asarray(numeric, dtype=np.float64), text_matrix.toarray()]
    )
    model = LogisticRegression(
        C=0.8,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
        max_iter=2000,
    )
    model.fit(matrix, np.asarray(labels, dtype=np.int64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    artifact = {
        "version": "task_change_v1",
        "training_dataset_sha256": training_hash,
        "training_cases": len(all_cases),
        "ambiguous_training_cases": len(cases),
        "hard_threshold": 0.45,
        "soft_threshold": 0.20,
        "model": model,
        "vectorizer": vectorizer,
    }
    joblib.dump(artifact, output_path)
    artifact_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        artifact_sha + "\n",
        encoding="ascii",
    )
    return {
        "output": str(output_path),
        "training_cases": len(all_cases),
        "ambiguous_training_cases": len(cases),
        "training_dataset_sha256": training_hash,
        "artifact_sha256": artifact_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--bge-dir", type=Path, default=DEFAULT_BGE_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.train, args.output, args.bge_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
