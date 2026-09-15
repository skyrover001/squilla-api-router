from __future__ import annotations

from typing import Any

import numpy as np

from squilla_api_router.v4_runtime.inference.types import FeatureBundle, HeadOutputs


def _softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return np.asarray(exps / np.sum(exps), dtype=np.float64)


def _get_session_input_name(mlp_session: Any) -> str:
    inputs = mlp_session.get_inputs()
    if not inputs:
        raise ValueError("ONNX MLP session exposes no inputs")
    return str(inputs[0].name)


def run_heads(
    bundle: FeatureBundle,
    main_model: Any,
    aux_model: Any,
    mlp_session: Any,
    mlp_scaler: Any,
    temperature: float,
) -> HeadOutputs:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")

    p_main = np.asarray(
        main_model.predict(bundle.features_390[None, :], num_threads=1)[0],
        dtype=np.float64,
    )
    p_aux = None
    if aux_model is not None:
        p_aux = np.asarray(
            aux_model.predict(bundle.features_390[None, :], num_threads=1)[0],
            dtype=np.float64,
        )

    scaled = mlp_scaler.transform(bundle.raw_bge_1536[None, :]).astype(np.float32)
    input_name = _get_session_input_name(mlp_session)
    logits = np.asarray(
        mlp_session.run(None, {input_name: scaled})[0][0], dtype=np.float64
    )
    calibrated = _softmax(logits / temperature)
    return HeadOutputs(
        p_main_lgbm=p_main,
        p_aux_lgbm=p_aux,
        logits_mlp=logits,
        p_mlp_calibrated=calibrated,
    )

