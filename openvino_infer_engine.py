"""
Drop-in replacement for rapidocr_onnxruntime's OrtInferSession that runs
inference through OpenVINO Runtime instead of ONNXRuntime.

We reuse RapidOCR's own (already correct, tested) detection post-processing
and CTC decoding logic - only the actual neural network execution is swapped
to OpenVINO. This avoids re-implementing the tricky DB box-decoding /
CTC-decoding math from scratch, while still giving genuine OpenVINO-accelerated
inference for every model call.
"""
from pathlib import Path
import numpy as np
try:
    from openvino.runtime import Core
except ImportError:
    from openvino import Core

_ov_core = Core()


class OpenVINOInferSession:
    def __init__(self, config):
        model_path = Path(config.get("model_path"))
        # model_path from rapidocr's config points at the .onnx file;
        # we look for the converted .xml (OpenVINO IR) sitting next to it.
        ir_path = model_path.with_suffix(".xml")
        if not ir_path.exists():
            raise FileNotFoundError(
                f"OpenVINO IR not found at {ir_path}. Convert it first with: "
                f"ovc {model_path} --output_model {model_path.stem}"
            )

        model = _ov_core.read_model(ir_path)
        self.compiled_model = _ov_core.compile_model(model, "CPU")
        self.output_ports = self.compiled_model.outputs
        self._input_names = [inp.get_any_name() for inp in self.compiled_model.inputs]
        self._output_names = [out.get_any_name() for out in self.output_ports]

        # Character list metadata doesn't survive ONNX->IR conversion, so we
        # read it from a sidecar .txt file extracted before conversion.
        char_path = model_path.parent / f"{model_path.stem}_characters.txt"
        self._characters = None
        if char_path.exists():
            self._characters = char_path.read_text(encoding="utf-8").splitlines()

    def __call__(self, input_content: np.ndarray):
        result = self.compiled_model([input_content])
        # Match onnxruntime's session.run() return shape: a list of arrays
        # in output order.
        return [result[port] for port in self.output_ports]

    def get_input_names(self):
        return self._input_names

    def get_output_names(self):
        return self._output_names

    def get_character_list(self, key: str = "character"):
        if self._characters is None:
            raise KeyError(
                f"No character list found for this model. Expected a "
                f"sidecar file with the '{key}' metadata extracted before "
                f"conversion."
            )
        return self._characters

    def have_key(self, key: str = "character"):
        return self._characters is not None
