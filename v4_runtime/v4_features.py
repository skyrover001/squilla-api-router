"""V4 router feature extraction helpers for the Phase 3 inference path.

Three logical channels added to the legacy v4 baseline:
  * BGE × 3 segments (current_user, history_user, prev_assistant) → PCA(64) each
  * 12-dim assistant handcrafted features (refusal/clarification/usage stats)
  * History-user concatenation helper

Phase 3's 390-dim online assembly lives in the inference package.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from squilla_api_router.v4_runtime.bge_onnx import OnnxBGE

__all__ = [
    "extract_assistant_handcrafted",
    "extract_continuation_features",
    "extract_reasoning_features",
    "make_history_user_text",
    "BGEChannelExtractor",
]


# ---------------------------------------------------------------------------
# Channel: assistant handcrafted features (12 dims)
# ---------------------------------------------------------------------------

_RE_CLAR = re.compile(
    r"(?:能否|请\s*提供|需要(?:更多|具体).{0,8}信息"
    r"|could you (?:clarify|provide)|please (?:specify|provide)|clarify which)",
    re.I,
)
_RE_REFUSAL = re.compile(
    r"(?:I cannot|I can't help|对不起.{0,5}无法|抱歉.{0,5}不能"
    r"|作为(?:AI|大语言模型))",
    re.I,
)
_RE_SELF_DOUBT = re.compile(
    r"(?:我不(?:确定|清楚)|可能(?:不太|不一定)"
    r"|not sure|might not be|I'm not entirely)",
    re.I,
)
_RE_CODE_INLINE = re.compile(r"`[^`]{4,}`")
_RE_NUMBERED_LIST = re.compile(r"^\s*\d+[\.、]\s", re.M)
_RE_CONTINUATION = re.compile(
    r"(?:请继续|继续|接着|续写|展开一下|再说|more|continue|go on|carry on|next)",
    re.I,
)
_RE_REASONING = re.compile(
    r"(?:why|compare|trade[ -]?off|analy[sz]e|architecture|reasoning|design"
    r"|解释|原因|对比|分析|架构|设计|权衡)",
    re.I,
)


def _zh_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = sum(1 for c in text if "一" <= c <= "鿿")
    return zh / max(len(text), 1)


def _normalize_log_usage(
    usage: dict[str, Any] | None,
    key: str,
    divisor: float = 10.0,
) -> float:
    value = (usage or {}).get(key, 0) or 0
    return float(np.log1p(max(value, 0)) / divisor)


def extract_assistant_handcrafted(prev_assistant_text: str | None,
                                   prev_assistant_usage: dict[str, Any] | None,
                                   current_user_text: str) -> np.ndarray[Any, Any]:
    """Return a 12-dim float32 vector of assistant signal features.

    Layout:
      0:  has_prev_asst              (0/1)
      1:  has_clarification_question (0/1)
      2:  has_refusal                (0/1)
      3:  self_doubt                 (0/1)
      4:  has_code_block             (0/1)
      5:  has_steps_list             (0/1)
      6:  log_output_tokens          (log1p / 10, soft-bounded)
      7:  log_reasoning_tokens       (log1p / 10)
      8:  log_duration_ms            (log1p / 10)
      9:  ans_user_ratio             (clip [0, 1])
      10: zh_ratio                   ([0, 1])
      11: cached_token_ratio         ([0, 1])
    """
    if prev_assistant_text is None:
        return np.zeros(12, dtype=np.float32)
    t = prev_assistant_text
    u = prev_assistant_usage or {}
    return np.array([
        1.0,
        float(_RE_CLAR.search(t) is not None),
        float(_RE_REFUSAL.search(t) is not None),
        float(_RE_SELF_DOUBT.search(t) is not None),
        float("```" in t or _RE_CODE_INLINE.search(t) is not None),
        float(_RE_NUMBERED_LIST.search(t) is not None),
        np.log1p(u.get("output_tokens", 0) or 0) / 10.0,
        np.log1p(u.get("reasoning_tokens", 0) or 0) / 10.0,
        np.log1p(u.get("duration_ms", 0) or 0) / 10.0,
        min(len(t) / max(len(current_user_text), 1), 5.0) / 5.0,
        _zh_char_ratio(t),
        (u.get("cached_tokens", 0) or 0) / max(u.get("input_tokens", 1) or 1, 1),
    ], dtype=np.float32)


def extract_continuation_features(
    prev_assistant_usage: dict[str, Any] | None,
    current_user_text: str,
) -> np.ndarray[Any, Any]:
    """Return a 2-dim float32 vector for short continuation prompts.

    Layout:
      0: has_continuation_cue     (0/1)
      1: prev_output_tokens_log   (log1p / 10)
    """
    text = (current_user_text or "").strip()
    is_short = len(text) <= 24
    has_cue = bool(text) and is_short and _RE_CONTINUATION.search(text) is not None
    return np.array([
        float(has_cue),
        _normalize_log_usage(prev_assistant_usage, "output_tokens"),
    ], dtype=np.float32)


def extract_reasoning_features(
    prev_assistant_usage: dict[str, Any] | None,
    current_user_text: str,
) -> np.ndarray[Any, Any]:
    """Return a 5-dim float32 vector for reasoning-heavy prompt cues.

    Layout:
      0: has_reasoning_cue         (0/1)
      1: question_density          (clipped [0, 1])
      2: prompt_length_log         (log1p / 10)
      3: prev_reasoning_tokens_log (log1p / 10)
      4: prev_duration_ms_log      (log1p / 10)
    """
    text = (current_user_text or "").strip()
    qmarks = text.count("?") + text.count("？")
    return np.array([
        float(_RE_REASONING.search(text) is not None),
        min(qmarks / max(len(text), 1) * 20.0, 1.0),
        float(np.log1p(len(text)) / 10.0),
        _normalize_log_usage(prev_assistant_usage, "reasoning_tokens"),
        _normalize_log_usage(prev_assistant_usage, "duration_ms"),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: history user text concatenation
# ---------------------------------------------------------------------------

_HISTORY_SEP = "\n[SEP]\n"


def make_history_user_text(prior_user_turns: list[str], max_turns: int = 4,
                           max_chars: int = 1500) -> str:
    """Concatenate up to max_turns prior user turns, oldest→newest, [SEP]-separated.

    BGE tokenizer has a 512-token limit. Empirically 1500 chars is a safe
    upper bound (zh ~750 tokens at worst, en ~375). If the result exceeds
    max_chars, truncate from the front (drop oldest turns first).
    """
    if not prior_user_turns:
        return ""
    selected = list(prior_user_turns[-max_turns:])  # oldest→newest of the window
    text = _HISTORY_SEP.join(selected)
    while len(text) > max_chars and len(selected) > 1:
        selected = selected[1:]   # drop the oldest
        text = _HISTORY_SEP.join(selected)
    if len(text) > max_chars:
        # only one turn left and still too long: hard truncate from the front
        text = text[-max_chars:]
    return text


# ---------------------------------------------------------------------------
# Channel: BGE × 3 segments + shared PCA(64)
# ---------------------------------------------------------------------------

class BGEChannelExtractor:
    """Shared BGE encoder + shared PCA(64) for three text segments.

    Each call to transform_one runs the BGE encoder three times on
    [current_user, history_user, prev_assistant] (None → empty string).
    PCA is fitted once on the union of all three text types.

    Output shape: (192,) = concat of 3 × PCA(64).
    """

    def __init__(self, bge_model_name: str = "BAAI/bge-small-zh-v1.5",
                 pca_dim: int = 64, seed: int = 42,
                 backend: str = "onnx",
                 onnx_model_dir: str | None = None) -> None:
        self.bge_model_name = bge_model_name
        self.pca_dim = pca_dim
        self.seed = seed
        self.backend = backend
        self.onnx_model_dir = onnx_model_dir
        if backend != "onnx" or not onnx_model_dir:
            raise ValueError("Phase 3 runtime requires an ONNX BGE model directory")
        self._bge: OnnxBGE | None = None
        self.pca: Any | None = None
        self.fitted = False

    def _ensure_bge(self) -> OnnxBGE:
        if self._bge is None:
            if self.onnx_model_dir is None:
                raise RuntimeError("ONNX BGE model directory is unavailable")
            self._bge = OnnxBGE(self.onnx_model_dir)
        return self._bge

    def _encode_triplet(
        self,
        current_user: str | None,
        history_user: str | None,
        prev_assistant: str | None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform_one().")
        bge = self._ensure_bge()
        texts = [current_user or "", history_user or "", prev_assistant or ""]
        raw = bge.encode(texts, batch_size=3, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)   # (3, 512)
        if self.pca is None:
            raise RuntimeError("BGE PCA projection is unavailable")
        reduced = np.asarray(self.pca.transform(raw))                # (3, k)
        if reduced.shape[1] < self.pca_dim:
            pad = np.zeros((reduced.shape[0], self.pca_dim - reduced.shape[1]),
                           dtype=reduced.dtype)
            reduced = np.concatenate([reduced, pad], axis=1)
        return reduced.astype(np.float32), raw

    def transform_triplet(self, current_user: str | None, history_user: str | None,
                          prev_assistant: str | None) -> tuple[
                              np.ndarray[Any, Any], np.ndarray[Any, Any]
                          ]:
        reduced, raw = self._encode_triplet(
            current_user,
            history_user,
            prev_assistant,
        )
        return (
            np.concatenate(reduced, axis=0).astype(np.float32),
            raw.reshape(-1).astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> BGEChannelExtractor:
        state: dict[str, Any] = joblib.load(path)
        ex = cls(
            state["bge_model_name"],
            state["pca_dim"],
            state["seed"],
            state.get("backend", "onnx"),
            state.get("onnx_model_dir"),
        )
        ex.pca = state["pca"]
        ex.fitted = state["fitted"]
        return ex

