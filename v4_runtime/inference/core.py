from __future__ import annotations

from typing import Any

import numpy as np

from squilla_api_router.v4_runtime.inference.artifacts import InferenceArtifacts
from squilla_api_router.v4_runtime.inference.ensemble import fuse_probabilities
from squilla_api_router.v4_runtime.inference.features import build_feature_bundle
from squilla_api_router.v4_runtime.inference.heads import run_heads
from squilla_api_router.v4_runtime.inference.postprocess import apply_postprocess
from squilla_api_router.v4_runtime.inference.types import (
    FeatureBundle,
    FinalDecision,
    HeadOutputs,
    InferenceRequest,
    InferenceResult,
)
from squilla_api_router.v4_runtime.policy import ROUTE_CLASSES


class InferenceCore:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        alpha: np.ndarray[Any, Any],
        temperature: float,
        main_model: Any,
        aux_model: Any,
        mlp_session: Any,
        mlp_scaler: Any,
        v3_extractor: Any,
        bge_extractor: Any,
    ) -> None:
        self.config = config
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.temperature = float(temperature)
        self.main_model = main_model
        self.aux_model = aux_model
        self.mlp_session = mlp_session
        self.mlp_scaler = mlp_scaler
        self.v3_extractor = v3_extractor
        self.bge_extractor = bge_extractor

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        config: dict[str, Any],
        *,
        use_aux_head: bool,
    ) -> InferenceCore:
        artifacts = InferenceArtifacts.load(model_dir)
        loaded = artifacts.load_runtime_objects(
            config=config,
            use_aux_head=use_aux_head,
        )
        return cls(
            config=config,
            alpha=np.asarray(artifacts.manifest["per_class_alpha"], dtype=np.float64),
            temperature=float(artifacts.manifest["temperature"]),
            main_model=loaded["main_model"],
            aux_model=loaded["aux_model"],
            mlp_session=loaded["mlp_session"],
            mlp_scaler=loaded["mlp_scaler"],
            v3_extractor=loaded["v3_extractor"],
            bge_extractor=loaded["bge_extractor"],
        )

    def predict(self, request: InferenceRequest) -> InferenceResult:
        bundle = self._build_features(request)
        outputs = self._run_heads(bundle)
        fused = self._fuse(outputs)
        decision = self._postprocess(fused, outputs.p_aux_lgbm, request)
        intermediates = {
            "bge_channels_used": bundle.bge_channels_used,
            "asst_signal_present": bundle.asst_signal_present,
        }
        return InferenceResult(
            decision=decision,
            probabilities={
                route_class: float(fused[idx])
                for idx, route_class in enumerate(ROUTE_CLASSES)
            },
            aux_decision_probs=self._aux_probs_dict(outputs.p_aux_lgbm),
            intermediates=intermediates,
        )

    def _build_features(self, request: InferenceRequest) -> FeatureBundle:
        return build_feature_bundle(
            request=request,
            v3_extractor=self.v3_extractor,
            bge_extractor=self.bge_extractor,
        )

    def _run_heads(self, bundle: FeatureBundle) -> HeadOutputs:
        return run_heads(
            bundle=bundle,
            main_model=self.main_model,
            aux_model=self.aux_model,
            mlp_session=self.mlp_session,
            mlp_scaler=self.mlp_scaler,
            temperature=self.temperature,
        )

    def _fuse(self, outputs: HeadOutputs) -> np.ndarray[Any, Any]:
        return fuse_probabilities(
            outputs.p_main_lgbm,
            outputs.p_mlp_calibrated,
            self.alpha,
        )

    def _postprocess(
        self,
        fused_probs: np.ndarray[Any, Any],
        aux_probs: np.ndarray[Any, Any] | None,
        request: InferenceRequest,
    ) -> FinalDecision:
        return apply_postprocess(
            fused_probs=fused_probs,
            aux_probs=self._aux_probs_dict(aux_probs),
            request=request,
            config=self.config,
        )

    @staticmethod
    def _aux_probs_dict(
        aux_probs: np.ndarray[Any, Any] | None,
    ) -> dict[str, float] | None:
        if aux_probs is None:
            return None
        aux_probs = np.asarray(aux_probs, dtype=np.float64)
        return {
            "initial": float(aux_probs[0]),
            "maintain": float(aux_probs[1]),
            "upgrade": float(aux_probs[2]),
            "downgrade": float(aux_probs[3]),
        }

