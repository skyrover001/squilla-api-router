"""ONNX-INT8 backend for BGE encoder.

Public surface intentionally mirrors the small slice of the
`sentence_transformers.SentenceTransformer` API used by the router:
`encode(texts, batch_size=..., show_progress_bar=..., convert_to_numpy=True)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from squilla_api_router.v4_runtime.onnx_session import (
    create_cpu_inference_session,
)

_DEFAULT_MAX_LENGTH = 510


class OnnxBGE:
    """Lazy, pickle-safe ONNX BGE wrapper."""

    def __init__(self, model_dir: str | Path, max_length: int = _DEFAULT_MAX_LENGTH):
        self.model_dir = str(Path(model_dir))
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._session: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._session is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(
                str(Path(self.model_dir) / "tokenizer.json")
            )
            tokenizer.enable_truncation(max_length=self.max_length)
            pad_token = "[PAD]"
            tokenizer.enable_padding(
                pad_id=tokenizer.token_to_id(pad_token) or 0,
                pad_token=pad_token,
            )
            self._tokenizer = tokenizer
            self._session = create_cpu_inference_session(
                Path(self.model_dir) / "model.onnx"
            )
        return self._tokenizer, self._session

    def encode(
        self,
        texts: str | list[str],
        *,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **_kwargs: Any,
    ) -> np.ndarray[Any, Any]:
        if isinstance(texts, str):
            texts = [texts]

        tokenizer, session = self._ensure_loaded()
        outputs: list[np.ndarray[Any, Any]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer.encode_batch(batch)
            ort_inputs = {
                "input_ids": np.asarray([enc.ids for enc in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [enc.attention_mask for enc in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [enc.type_ids for enc in encoded], dtype=np.int64
                ),
            }
            last_hidden = session.run(None, ort_inputs)[0]
            cls = last_hidden[:, 0, :]
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            outputs.append((cls / np.maximum(norms, 1e-12)).astype(np.float32))

        return np.asarray(np.concatenate(outputs, axis=0), dtype=np.float32)

    def __getstate__(self) -> dict[str, object]:
        return {"model_dir": self.model_dir, "max_length": self.max_length}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_dir = state["model_dir"]
        self.max_length = state["max_length"]
        self._tokenizer = None
        self._session = None

