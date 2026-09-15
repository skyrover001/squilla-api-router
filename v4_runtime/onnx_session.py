from pathlib import Path
from typing import Any

ONNX_INTRA_OP_NUM_THREADS = 4
ONNX_INTER_OP_NUM_THREADS = 1


def create_cpu_inference_session(model_path: str | Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = ONNX_INTRA_OP_NUM_THREADS
    options.inter_op_num_threads = ONNX_INTER_OP_NUM_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
