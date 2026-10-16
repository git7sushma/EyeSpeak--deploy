"""
OpenVINO-accelerated OCR for EyeSpeak.

Reads text printed on packaging using PP-OCRv4 detection + recognition
models (plus a small angle-classification model), converted to OpenVINO
IR and run through OpenVINO Runtime.

We reuse RapidOCR's own tested pre/post-processing (text-box decoding,
CTC decoding) and only swap the neural network inference itself from
ONNXRuntime to OpenVINO - see openvino_infer_engine.py for how that's done.
"""
import os
import numpy as np
from openvino_infer_engine import OpenVINOInferSession

import rapidocr_onnxruntime.ch_ppocr_det.text_detect as _det_mod
import rapidocr_onnxruntime.ch_ppocr_rec.text_recognize as _rec_mod
import rapidocr_onnxruntime.ch_ppocr_cls.text_cls as _cls_mod

# Must happen before RapidOCR() is instantiated below.
_det_mod.OrtInferSession = OpenVINOInferSession
_rec_mod.OrtInferSession = OpenVINOInferSession
_cls_mod.OrtInferSession = OpenVINOInferSession

from rapidocr_onnxruntime import RapidOCR

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_models")

print("Loading OCR models with OpenVINO...")
_ocr = RapidOCR(
    det_model_path=os.path.join(_MODEL_DIR, "ch_PP-OCRv4_det_infer.onnx"),
    rec_model_path=os.path.join(_MODEL_DIR, "ch_PP-OCRv4_rec_infer.onnx"),
    cls_model_path=os.path.join(_MODEL_DIR, "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
)
print("OCR ready — running on OpenVINO Runtime.")

# Below this confidence, or shorter than this, a detected text line is
# treated as noise rather than a real label.
OCR_CONFIDENCE_THRESHOLD = 0.4
OCR_MIN_LENGTH = 3
# Below this AVERAGE confidence across all kept lines, treat the whole
# reading as unreliable and fall back rather than speak garbled text.
OCR_OVERALL_CONFIDENCE_THRESHOLD = 0.75


import re

# Signals that a line is company/address footer text rather than product
# name or description - these are very consistent across real packaging
# regardless of font size, which font-height alone doesn't reliably catch.
_ADDRESS_PATTERNS = [
    r'\b(inc\.?|ltd\.?|corp\.?|llc|co\.)\b',          # company suffixes
    r'\b[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\b',         # Canada-style postal code
    r'\bboul(evard)?\b|\bstreet\b|\bave(nue)?\b|\brd\.?\b|\bblvd\b|\broad\b',  # street words
]
_address_regex = re.compile('|'.join(_ADDRESS_PATTERNS), re.IGNORECASE)


def _is_address_line(text):
    return bool(_address_regex.search(text))


def read_label_text(image):
    """
    Run OCR on a PIL image and return the main product text, combined in
    top-to-bottom, left-to-right reading order - or None if nothing legible
    was found. Company/address footer text is filtered out both by font
    size (fine print is smaller than the product name/description) and by
    recognizable address patterns (postal codes, "Inc.", street words),
    since address blocks don't always end up smaller in practice on
    curved or wrapped labels.
    """
    result, _elapse = _ocr(np.array(image))
    if not result:
        return None

    candidates = [
        (box, text.strip(), conf) for (box, text, conf) in result
        if conf >= OCR_CONFIDENCE_THRESHOLD and len(text.strip()) >= OCR_MIN_LENGTH
    ]
    if not candidates:
        return None

    def box_height(box):
        ys = [point[1] for point in box]
        return max(ys) - min(ys)

    max_height = max(box_height(box) for box, _text, _conf in candidates)
    good_lines = [
        (box, text, conf) for (box, text, conf) in candidates
        if box_height(box) >= 0.45 * max_height and not _is_address_line(text)
    ]
    if not good_lines:
        return None

    # If overall recognition confidence is too low, this photo likely had
    # motion blur, glare, or a difficult curved/reflective surface - better
    # to fall back to the CLIP category guess than speak garbled text.
    avg_confidence = sum(conf for _box, _text, conf in good_lines) / len(good_lines)
    if avg_confidence < OCR_OVERALL_CONFIDENCE_THRESHOLD:
        return None

    # Sort top-to-bottom, then left-to-right, using each box's top-left
    # corner - so multi-line labels read in a sensible order rather than
    # whatever order the model happened to detect them in.
    good_lines.sort(key=lambda item: (round(item[0][0][1] / 15), item[0][0][0]))

    combined = " ".join(text for (_box, text, _conf) in good_lines)
    return combined
